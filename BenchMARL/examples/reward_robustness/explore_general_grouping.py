"""Research harness (offline): general, task-agnostic ways to find reward-equivalence groups under noise.

Run on streams recorded with collect_stream.py (pursuit16, spread5, ...). Gaussian noise with KNOWN sigma.

Methods
  none / global / oracle   references (oracle = epsilon-oracle on true rewards, --oracle_eps)
  cov                      persistent (identity) groups from cross-agent second moments of noisy rewards:
                           E[(r~_i - r~_j)^2] - 2 sigma^2 is an unbiased estimate of E[(r_i - r_j)^2]
                           (independent noise), certified top-down (UCB <= eps^2)
  rm                       reward model f(o,a,o') on noisy rewards, kernel grouping (ungated + skill-gated)
  rm_aux / rm_ema          + auxiliary dynamics loss / EMA weights (did not speed up learning)
  rm_cert                  per-bin certification of predicted-reward cells (too data-hungry)
  rm_rule                  rule-level certification of the prediction rule (pooled over merged pairs)
  certified                CERTIFIED CONSENSUS: candidate rules {merge agents with close predictions,
                           same but only within identity clusters}; each rule is certified from the
                           same-step pairs it merges (lagged, out of sample); the certified rule that
                           merges most is used, otherwise no correction. Identity clusters only RESTRICT
                           merges (they cannot see rare situational events).

Options: --teams k stacks k independent environments per row (k persistent teams), --team_offset shifts
team j's rewards by j*offset (role-dependent reward level), --z confidence level, --decay memory.

Metrics (per agent): r^_ti = a*m_t + b*d_ti + e, m_t = true step mean, d_ti = true within-step deviation.
  snr_common = a*std(m)/std(e) (shared objective), beta = b (credit retention: 1 kept, 0 erased),
  snr_within = b*std(d)/std(e) (individual credit).
"""
import argparse
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from offline_grouping_benchmark import load_stream  # noqa: E402

from benchmarl.reward_robustness import apply_consensus, Assignment, GroupData, perturb_rewards
from benchmarl.reward_robustness.clustering import epsilon_labels
from benchmarl.reward_robustness.groupers import assignment_from_predictions
from benchmarl.reward_robustness.task_spec import get_task_spec


# ----------------------------------------------------------------------------- metrics
class Acc:
    def __init__(self):
        self.r, self.c = [], []
        self.z = []

    def add(self, true, corr):
        a = true.shape[-1]
        self.r.append(true.reshape(-1, a).double())
        self.c.append(corr.reshape(-1, a).double())

    def summary(self):
        """Per-agent decomposition: r^_ti = a * m_t + b * d_ti + const + e_ti, with m_t the true
        step mean and d_ti = r_ti - m_t the agent's true within-step deviation (orthogonal)."""
        r, c = torch.cat(self.r), torch.cat(self.c)
        m = r.mean(1, keepdim=True).expand_as(r)
        d = r - m
        mc = m - m.mean()
        a = (c * mc).sum() / (mc * mc).sum().clamp(min=1e-12)
        b = (c * d).sum() / (d * d).sum().clamp(min=1e-12)
        e = c - a * m - b * d
        e = e - e.mean()
        noise = e.std()
        return {
            "a": a.item(),
            "beta": b.item(),
            "noise": noise.item(),
            "snr_common": (a * m[:, 0].std() / noise).item(),
            "snr_within": (b * d.std() / noise).item(),
            "mse": ((c - r) ** 2).mean().item(),
        }


# ----------------------------------------------------------------------------- persistent groups
class CovGrouper:
    """Persistent (identity-based) groups from cross-agent second moments of noisy rewards.

    For independent zero-mean noise, E[(r~_i - r~_j)^2] = E[(r_i - r_j)^2] + 2 sigma^2, so
    delta_ij = E[(r~_i - r~_j)^2] - 2 sigma^2 estimates the (time-averaged) squared reward
    difference without bias (including constant offsets between agents). Agents are merged (average linkage) only if the upper
    confidence bound of the pooled delta between two clusters is <= eps^2.
    Statistics come from previous batches only (out of sample).
    """

    def __init__(self, n_agents, eps, sigma=None, z=2.0):
        self.a, self.eps, self.sigma, self.z = n_agents, eps, sigma, z
        self.n = 0
        self.s1 = np.zeros(n_agents)
        self.s2 = np.zeros((n_agents, n_agents))
        self.labels = np.arange(n_agents)

    def update(self, reward):
        x = reward.reshape(-1, self.a).double().numpy()
        self.n += x.shape[0]
        self.s1 += x.sum(0)
        self.s2 += x.T @ x
        self._recluster()

    def sigma2(self, D):
        if self.sigma is not None:
            return self.sigma ** 2
        # estimate: smallest pairwise difference variance / 2 (valid if some pair is equal)
        off = D[~np.eye(self.a, dtype=bool)]
        return float(np.sort(off)[: max(1, len(off) // 10)].mean() / 2)

    def _recluster(self):
        M = self.s2 / self.n  # uncentred second moments: mean offsets between agents count too
        diag = np.diag(M)
        D = diag[:, None] + diag[None, :] - 2 * M  # E[(r~_i - r~_j)^2] = E[(r_i - r_j)^2] + 2 sigma^2
        s2 = self.sigma2(D)
        delta = D - 2 * s2
        se_pair = 2 * s2 * np.sqrt(2.0 / self.n)  # SE of one pairwise difference variance
        # Top-down certification along an average-linkage dendrogram of the point estimates:
        # accept a node if every member's mean delta to the other members has UCB <= eps^2
        # (pooling over k-1 pairs shrinks the SE by ~sqrt(k-1)); otherwise recurse.
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform

        dist = np.clip(delta, 0, None)
        np.fill_diagonal(dist, 0)
        tree = to_tree(linkage(squareform((dist + dist.T) / 2, checks=False), "average"))

        def certify(node):
            members = node.pre_order()
            k = len(members)
            if k == 1:
                return [members]
            sub = delta[np.ix_(members, members)]
            loo = (sub.sum(1) - np.diag(sub)) / (k - 1)
            if np.all(loo + self.z * se_pair / np.sqrt(k - 1) <= self.eps ** 2):
                return [members]
            return certify(node.get_left()) + certify(node.get_right())

        lab = np.empty(self.a, dtype=np.int64)
        for k, cl in enumerate(certify(tree)):
            lab[cl] = k
        self.labels = lab

    def assign(self, shape):
        if self.n == 0:
            return Assignment()
        lab = torch.as_tensor(self.labels).expand(*shape)
        return Assignment(labels=lab.clone())


class BinCertifier:
    """Certify cells (bins of predicted reward, width eps) as reward-homogeneous.

    For agents in the same cell at the same step, E[(r~_i - r~_j)^2] = E[(r_i - r_j)^2] + 2 sigma^2,
    so delta_cell = mean pair sq. difference - 2 sigma^2 estimates the within-step squared reward
    difference (Assumption 1) without bias. Statistics are collected out of sample (predictions of the
    model before it trains on the batch) and decayed so they track the evolving model.
    A cell is used for grouping only if UCB(delta) <= eps^2.
    """

    def __init__(self, eps, sigma, z=2.0, decay=0.5):
        self.eps, self.sigma, self.z, self.decay = eps, sigma, z, decay
        self.ss, self.npairs, self.dof = {}, {}, {}

    def bins(self, pred):
        return torch.floor(pred / self.eps).long()

    def certified(self, b):
        ok = torch.zeros_like(b, dtype=torch.bool)
        s2 = self.sigma ** 2
        for k in b.unique().tolist():
            if self.dof.get(k, 0) < 2:
                continue
            delta = self.ss[k] / self.npairs[k] - 2 * s2
            se = 2 * s2 * np.sqrt(2.0 / self.dof[k])
            if delta + self.z * se <= self.eps ** 2:
                ok |= b == k
        return ok

    def update(self, pred, reward):
        b = self.bins(pred).reshape(-1, pred.shape[-1])
        x = reward.reshape(-1, pred.shape[-1]).double()
        for k in list(self.ss):
            self.ss[k] *= self.decay; self.npairs[k] *= self.decay; self.dof[k] *= self.decay
        for k in b.unique().tolist():
            m = (b == k).double()
            n = m.sum(1)
            sx, sxx = (x * m).sum(1), (x * x * m).sum(1)
            pair_ss = n * sxx - sx ** 2          # sum_{i<j} (x_i - x_j)^2 per row
            rows = n >= 2
            self.ss[k] = self.ss.get(k, 0.0) + pair_ss[rows].sum().item()
            self.npairs[k] = self.npairs.get(k, 0.0) + (n * (n - 1) / 2)[rows].sum().item()
            self.dof[k] = self.dof.get(k, 0.0) + (n - 1)[rows].sum().item()


class RuleCertifier:
    """Certify a grouping *rule* from the pairs it merges (pooled over all merged pairs).

    delta = mean over merged same-step pairs of (r~_i - r~_j)^2 - 2 sigma^2 is an unbiased estimate
    of the mean squared reward difference of merged pairs; certification (UCB <= eps^2) bounds the
    RMS bias of the consensus by eps. Stats come from the previous batches' proposals (lagged,
    out of sample for the model that made them) with exponential decay.
    """

    def __init__(self, eps, sigma, z=2.0, decay=0.5):
        self.eps, self.sigma, self.z, self.decay = eps, sigma, z, decay
        self.ss = self.npairs = self.dof = 0.0

    def stats(self):
        s2 = self.sigma ** 2
        delta = self.ss / self.npairs - 2 * s2
        return delta, self.z * 2 * s2 * np.sqrt(2.0 / self.dof)

    @property
    def ok(self):
        if self.dof < 2:
            return False
        delta, margin = self.stats()
        return delta + margin <= self.eps ** 2

    def update(self, labels, reward):
        a = labels.shape[-1]
        lab = labels.reshape(-1, a)
        x = reward.reshape(-1, a).double()
        self.ss *= self.decay; self.npairs *= self.decay; self.dof *= self.decay
        for k in lab.unique().tolist():
            if k < 0:
                continue
            m = (lab == k).double()
            n = m.sum(1)
            sx, sxx = (x * m).sum(1), (x * x * m).sum(1)
            rows = n >= 2
            self.ss += (n * sxx - sx ** 2)[rows].sum().item()
            self.npairs += (n * (n - 1) / 2)[rows].sum().item()
            self.dof += (n - 1)[rows].sum().item()


# ----------------------------------------------------------------------------- reward models
class Net(nn.Module):
    def __init__(self, obs_shape, act_dim, cnn, aux):
        super().__init__()
        self.cnn, self.aux = cnn, aux
        if cnn:
            h, w, c = obs_shape
            self.enc = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
            feat = 32 * h * w
        else:
            dim = int(np.prod(obs_shape))
            self.enc = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            feat = 128
        self.out_dim = int(np.prod(obs_shape))
        self.head = nn.Sequential(nn.Linear(2 * feat + act_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        if aux:  # dynamics head: predict o' from (o, a) -> noise-free targets for the encoder
            self.dyn = nn.Sequential(nn.Linear(feat + act_dim, 256), nn.ReLU(), nn.Linear(256, self.out_dim))

    def encode(self, o):
        return self.enc(o.permute(0, 3, 1, 2) if self.cnn else o.flatten(1))

    def forward(self, o, a, o2):
        zo, zn = self.encode(o), self.encode(o2)
        r = self.head(torch.cat([zo, zn, a], -1)).squeeze(-1)
        dyn = self.dyn(torch.cat([zo, a], -1)) if self.aux else None
        return r, dyn


class RewardModel:
    def __init__(self, obs_shape, act_dim, cnn, aux=False, ema=0.0, lr=1e-3, bs=512, aux_w=1.0):
        torch.manual_seed(0)
        self.net = Net(obs_shape, act_dim, cnn, aux)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.ema = ema
        self.avg = [p.detach().clone() for p in self.net.parameters()] if ema > 0 else None
        self.bs, self.aux_w, self.n_updates, self.skill_z = bs, aux_w, 0, None

    @torch.no_grad()
    def predict(self, d):
        o, a, o2 = d.obs.flatten(0, 2), d.action.flatten(0, 2), d.next_obs.flatten(0, 2)
        if self.avg is not None:
            saved = [p.detach().clone() for p in self.net.parameters()]
            for p, q in zip(self.net.parameters(), self.avg):
                p.copy_(q)
        out = torch.cat([self.net(o[i:i + 65536], a[i:i + 65536], o2[i:i + 65536])[0]
                         for i in range(0, o.shape[0], 65536)])
        if self.avg is not None:
            for p, q in zip(self.net.parameters(), saved):
                p.copy_(q)
        return out.reshape(d.reward.shape)

    def update(self, d):
        if self.n_updates:
            f = self.predict(d)
            dd = (d.reward - d.reward.mean()) ** 2 - (d.reward - f) ** 2
            self.skill_z = (dd.mean() / (dd.std() / dd.numel() ** 0.5)).item()
        o, a, o2 = d.obs.flatten(0, 2), d.action.flatten(0, 2), d.next_obs.flatten(0, 2)
        y = d.reward.flatten()
        perm = torch.randperm(y.numel())
        for s in range(0, y.numel(), self.bs):
            idx = perm[s:s + self.bs]
            r, dyn = self.net(o[idx], a[idx], o2[idx])
            loss = ((r - y[idx]) ** 2).mean()
            if dyn is not None:
                loss = loss + self.aux_w * ((dyn - o2[idx].flatten(1)) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.avg is not None:
                with torch.no_grad():
                    for p, q in zip(self.net.parameters(), self.avg):
                        q.mul_(self.ema).add_(p.detach(), alpha=1 - self.ema)
        self.n_updates += 1


# ----------------------------------------------------------------------------- main
def composite(d, k=2, offset=0.0):
    """Stack k independent environments side by side: k persistent teams per row.
    Team j's rewards are shifted by j * offset (role-dependent reward level)."""
    b = d.reward.shape[0] // k
    cat = lambda x: torch.cat([x[j * b:(j + 1) * b] for j in range(k)], dim=2)
    a = d.reward.shape[2]
    shift = torch.cat([torch.full((a,), j * offset) for j in range(k)])
    true = cat(d.true_reward) + shift
    return GroupData(cat(d.obs), cat(d.next_obs), cat(d.action), true.clone(), cat(d.mask), true)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--task", default="pursuit")
    p.add_argument("--sigma", type=float, default=3.5)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--teams", type=int, default=1, help="stack k independent envs per row")
    p.add_argument("--team_offset", type=float, default=0.0)
    p.add_argument("--methods", nargs="+", default=["none", "oracle", "global", "cov", "cov_est"])
    p.add_argument("--oracle_eps", type=float, default=0.05)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--z", type=float, default=2.0)
    args = p.parse_args()

    n_act = 5
    stream = load_stream(args.stream, n_act)
    if args.teams > 1:
        stream = [composite(d, args.teams, args.team_offset) for d in stream]
    shape = stream[0].reward.shape
    n_agents = shape[-1]
    cnn = len(stream[0].obs.shape) == 6
    print(f"task={args.task} sigma={args.sigma} eps={args.eps} teams={args.teams} agents={n_agents} "
          f"batches={len(stream)}", flush=True)

    for method in args.methods:
        t0 = time.time()
        acc, gated_acc, traj = Acc(), Acc(), []
        cov = CovGrouper(n_agents, args.eps, sigma=None if method.endswith("_est") else args.sigma, z=args.z)
        rm = None
        certifier = BinCertifier(args.eps, args.sigma)
        rule = RuleCertifier(args.eps, args.sigma, z=args.z, decay=args.decay)
        rules = {k: RuleCertifier(args.eps, args.sigma, z=args.z, decay=args.decay) for k in ["pred", "pred_in_id"]}
        diag = []
        grouped_frac = []
        if method.startswith(("rm", "combo", "certified")):
            kw = dict(aux="aux" in method, ema=0.99 if "ema" in method else 0.0)
            rm = RewardModel(stream[0].obs.shape[3:], stream[0].action.shape[-1], cnn, **kw)
        for i, d in enumerate(stream):
            torch.manual_seed(i)
            d.reward = perturb_rewards(d.true_reward, "normal", stdev=args.sigma)
            gated = None
            if method == "none":
                a = Assignment()
            elif method == "oracle":
                a = Assignment(labels=epsilon_labels(d.true_reward, args.oracle_eps))
            elif method == "global":
                a = Assignment(labels=torch.zeros_like(d.reward, dtype=torch.long))
            elif method.startswith("cov"):
                a = cov.assign(shape)
            elif method == "certified":
                # candidate rules: identity clusters (cov, self-certified), prediction rule, prediction rule
                # within identity clusters; use the certified rule that merges the most agents
                # identity clusters only *restrict* merges: they cannot see situational events, and rare
                # large events are invisible in reward moments, so merges must be vetted by the model
                cands = []
                id_lab = cov.assign(shape).labels
                if rm.n_updates:
                    pred = rm.predict(d)
                    props = {"pred": epsilon_labels(pred, args.eps)}
                    if id_lab is not None:
                        sub = torch.full_like(id_lab, -1)
                        for g in id_lab.unique().tolist():
                            m = id_lab == g
                            sub = torch.where(m, epsilon_labels(pred, args.eps, m) + g * n_agents, sub)
                        props["pred_in_id"] = sub
                    for name, lab in props.items():
                        if rules[name].ok:
                            cands.append((name, lab))
                        rules[name].update(lab, d.reward)
                def size(lab):
                    l = lab.reshape(-1, n_agents)
                    return (l.unsqueeze(-1) == l.unsqueeze(-2)).float().sum(-1).mean().item()
                if cands:
                    name, lab = max(cands, key=lambda c: size(c[1]))
                    a = Assignment(labels=lab)
                    grouped_frac.append(name)
                else:
                    a = Assignment()
                    grouped_frac.append("-")
            elif method.startswith("rm") and method.endswith("rule"):
                if rm.n_updates:
                    pred = rm.predict(d)
                    proposal = epsilon_labels(pred, args.eps)       # merge agents with close predictions
                    a = Assignment(labels=proposal) if rule.ok else Assignment()
                    grouped_frac.append(float(rule.ok))
                    rule.update(proposal, d.reward)                 # certify the rule for later batches
                    truth = RuleCertifier(args.eps, 1e-9, decay=0.0)
                    truth.update(proposal, d.true_reward)
                    est, margin = rule.stats()
                    diag.append((i, round(truth.ss / max(truth.npairs, 1), 3), round(est, 3), round(margin, 3)))
                else:
                    a = Assignment()
            elif method.startswith("rm") and method.endswith("cert"):
                if rm.n_updates:
                    pred = rm.predict(d)
                    b = certifier.bins(pred)
                    ok = certifier.certified(b)
                    a = Assignment(labels=torch.where(ok, b - b.min(), torch.full_like(b, -1)))
                    grouped_frac.append(ok.float().mean().item())
                    certifier.update(pred, d.reward)  # out of sample for the current model
                else:
                    a = Assignment()
            elif method.startswith("rm"):
                if rm.n_updates:
                    pred = rm.predict(d)
                    a = assignment_from_predictions(pred, d.mask, mode="kernel", epsilon=args.eps)
                    gated = a if (rm.skill_z or -1) >= 3 else Assignment()
                else:
                    a = gated = Assignment()
            elif method.startswith("combo"):
                base = cov.assign(shape)
                a = base
                if rm.n_updates and (rm.skill_z or -1) >= 3 and base.labels is not None:
                    pred = rm.predict(d)
                    lab = base.labels
                    sub = torch.full_like(lab, -1)
                    for g in lab.unique().tolist():
                        m = lab == g
                        sub = torch.where(m, epsilon_labels(pred, args.eps, m) + g * n_agents, sub)
                    a = Assignment(labels=sub)
            corr = apply_consensus(d.reward, a, d.mask)
            if i % 2 == 1:
                acc.add(d.true_reward, corr)
                if gated is not None:
                    gated_acc.add(d.true_reward, apply_consensus(d.reward, gated, d.mask))
                one = Acc()
                one.add(d.true_reward, corr)
                s = one.summary()
                traj.append((i, round(s["beta"], 2), None if rm is None or rm.skill_z is None else round(rm.skill_z, 1))
                            if not method.startswith("cov") else (i, len(set(cov.labels.tolist()))))
            # updates after grouping (out of sample)
            cov.update(d.reward)
            if rm is not None:
                rm.update(d)
        s = acc.summary()
        line = (f"{method:10s} snr_common={s['snr_common']:5.2f} snr_within={s['snr_within']:5.2f} "
                f"beta={s['beta']:5.2f} noise={s['noise']:5.2f} mse={s['mse']:6.3f}")
        if method.startswith(("cov", "combo", "certified")):
            line += f" groups={cov.labels.tolist()}"
        if gated_acc.r:
            g = gated_acc.summary()
            line += f" | gated: snr_common={g['snr_common']:.2f} snr_within={g['snr_within']:.2f} beta={g['beta']:.2f}"
        print(line + f" ({time.time() - t0:.0f}s)", flush=True)
        if method.startswith(("rm", "combo")):
            print(f"           per-batch (iter, beta, skill_z): {traj}", flush=True)
        if diag:
            print(f"           (iter, true delta of proposal, est. delta (decayed), z*SE): {diag}", flush=True)
        if grouped_frac:
            print(f"           certified rule per batch: {[g if isinstance(g, str) else round(g, 2) for g in grouped_frac]}", flush=True)
        if method.startswith("cov"):
            print(f"           per-batch (iter, n_groups used): {traj}", flush=True)


if __name__ == "__main__":
    main()
