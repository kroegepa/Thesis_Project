"""Extensions of certified consensus (offline, known noise std per agent).

Variants (combine with '+', e.g. 'ivw+pseudo'):
  base       certified spatial consensus (prediction rule, optionally within identity clusters)
  ivw        inverse-variance weighted group means (weights 1/s_j^2) instead of equal weights
  pseudo     reward model as a pseudo-agent: c <- (1 - mu_b) c + mu_b f, with mu_b certified per
             predicted-reward bin b from E[(r~ - f)^2] - s^2 (unbiased for the model's squared error);
             mu_b = min(MSE-optimal weight, eps / sqrt(UCB_b)) so the added RMS bias is <= eps
  tempW      spatio-temporal: average c over the same agent's +-W//2 neighbouring steps, rule certified
             from (r~_t - r~_t')^2 - 2 s^2 over merged temporal pairs
  tempWvet   same, but only neighbours whose predicted reward is eps-close are merged
  warmK      reward model warm-started on K clean batches (even batches), then trained on noisy rewards
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from explore_general_grouping import Acc, composite, CovGrouper, load_stream, RewardModel, RuleCertifier  # noqa: E402

from benchmarl.reward_robustness import apply_consensus, Assignment
from benchmarl.reward_robustness.clustering import epsilon_labels


class StatCertifier:
    """Decayed mean/SE of a per-sample statistic (unbiased for a squared bias)."""

    def __init__(self, z, decay):
        self.z, self.decay = z, decay
        self.n = self.s = self.ss = 0.0

    def update(self, stat):
        self.n = self.n * self.decay + stat.numel()
        self.s = self.s * self.decay + stat.sum().item()
        self.ss = self.ss * self.decay + (stat ** 2).sum().item()

    def ucb(self):
        if self.n < 20:
            return float("inf")
        m = self.s / self.n
        return m + self.z * np.sqrt(max(self.ss / self.n - m * m, 0) / self.n)


def temporal_average(c, valid_nb, h):
    """c [B,T,A]; valid_nb[k] [B,T,A] bool for offset k in [-h, h] \\ {0}."""
    num, den = c.clone(), torch.ones_like(c)
    for k, ok in valid_nb.items():
        shifted = torch.roll(c, shifts=-k, dims=1)
        num += torch.where(ok, shifted, torch.zeros_like(c))
        den += ok.float()
    return num / den


def run(variant, args, stream, n_agents, scales):
    feats = set(variant.split("+"))
    temporal = next((f for f in feats if f.startswith("temp")), None)
    win = int(temporal[4:].replace("vet", "")) if temporal else 0
    vet = temporal is not None and temporal.endswith("vet")
    warm = next((int(f[4:]) for f in feats if f.startswith("warm")), 0)
    h = win // 2
    s2 = torch.as_tensor((args.sigma * scales) ** 2, dtype=torch.float32)  # known per-agent noise variances
    cnn = len(stream[0].obs.shape) == 6
    rm = RewardModel(stream[0].obs.shape[3:], stream[0].action.shape[-1], cnn)
    if warm:
        for d in stream[0::2][:warm]:
            clean = type(d)(d.obs, d.next_obs, d.action, d.true_reward.clone(), d.mask, d.true_reward)
            for _ in range(10):
                rm.update(clean)
    cov = CovGrouper(n_agents, args.eps, sigma=args.sigma * scales, z=args.z)
    rules = {k: RuleCertifier(args.eps, z=args.z, decay=args.decay) for k in ["pred", "pred_in_id"]}
    tcert = StatCertifier(args.z, args.decay)
    bins = {}
    acc, late, log = Acc(), Acc(), []
    for i, d in enumerate(stream):
        torch.manual_seed(i)
        d.reward = d.true_reward + torch.randn_like(d.true_reward) * torch.sqrt(s2)
        c = d.reward.clone()
        used = "-"
        if rm.n_updates:
            pred = rm.predict(d)
            # ---- spatial certified consensus
            id_lab = cov.assign(d.reward.shape).labels
            props = {"pred": epsilon_labels(pred, args.eps)}
            if id_lab is not None and len(set(cov.labels.tolist())) < n_agents:
                sub = torch.full_like(id_lab, -1)
                for g in id_lab.unique().tolist():
                    m = id_lab == g
                    sub = torch.where(m, epsilon_labels(pred, args.eps, m) + g * n_agents, sub)
                props["pred_in_id"] = sub
            cands = [(k, lab) for k, lab in props.items() if rules[k].ok]
            var_c = s2.expand_as(c).clone()
            if cands:
                def size(lab):
                    l2 = lab.reshape(-1, n_agents)
                    return (l2.unsqueeze(-1) == l2.unsqueeze(-2)).float().sum(-1).mean().item()
                used, lab = max(cands, key=lambda kv: size(kv[1]))
                same = (lab.unsqueeze(-1) == lab.unsqueeze(-2)) & (lab >= 0).unsqueeze(-1)
                w = 1.0 / s2 if "ivw" in feats else torch.ones_like(s2)
                W = same.float() * w  # row i: weight of agent j
                c = apply_consensus(d.reward, Assignment(weights=W), d.mask)
                var_c = (W ** 2 * s2).sum(-1) / W.sum(-1) ** 2  # variance of the weighted mean
            for k, lab in props.items():
                rules[k].update(lab, d.reward, s2)
            # ---- reward model as pseudo-agent, certified per predicted-reward bin
            if "pseudo" in feats:
                b = torch.floor(pred / args.eps).long()
                mu = torch.zeros_like(c)
                for k in b.unique().tolist():
                    cert = bins.get(k)
                    if cert is None:
                        continue
                    ucb = cert.ucb()
                    if not np.isfinite(ucb):
                        continue
                    m = b == k
                    d2 = max(cert.s / max(cert.n, 1), 1e-6)
                    mse_opt = var_c / (var_c + d2)                       # inverse-variance combination
                    budget = 1.0 if ucb <= args.eps ** 2 else args.eps / np.sqrt(ucb)
                    mu = torch.where(m, torch.clamp(mse_opt, max=budget), mu)
                c = (1 - mu) * c + mu * pred
                for k in b.unique().tolist():  # certify model bias per bin on this batch (out of sample)
                    m = b == k
                    stat = ((d.reward - pred) ** 2 - s2.expand_as(c))[m]
                    bins.setdefault(k, StatCertifier(args.z, args.decay)).update(stat)
            # ---- spatio-temporal consensus, certified temporal merge rule
            if win:
                nb = {}
                stats = []
                for k in [k for k in range(-h, h + 1) if k != 0]:
                    ok = torch.zeros_like(c, dtype=torch.bool)
                    lo, hi = max(0, -k), c.shape[1] - max(0, k)
                    ok[:, lo:hi] = True
                    if vet:
                        ok &= (torch.roll(pred, -k, 1) - pred).abs() <= args.eps
                    nb[k] = ok
                    if k > 0:
                        diff = torch.roll(d.reward, -k, 1) - d.reward
                        stats.append((diff ** 2 - 2 * s2)[ok])
                if tcert.ucb() <= args.eps ** 2:
                    c = temporal_average(c, nb, h)
                    used += "+T"
                tcert.update(torch.cat(stats))
        if i % 2 == 1:
            acc.add(d.true_reward, c)
            if i >= 13:
                late.add(d.true_reward, c)
        log.append(used)
        cov.update(d.reward)
        rm.update(d)
    return acc.summary(), log, late.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--sigma", type=float, default=3.5)
    p.add_argument("--eps", type=float, default=0.5)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.5)
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--teams", type=int, default=1)
    p.add_argument("--team_offset", type=float, default=0.0)
    p.add_argument("--variants", nargs="+", default=["base"])
    args = p.parse_args()
    stream = load_stream(args.stream, 5)
    if args.teams > 1:
        stream = [composite(d, args.teams, args.team_offset) for d in stream]
    n_agents = stream[0].reward.shape[-1]
    scales = np.random.default_rng(123).uniform(0.5, 1.5, n_agents) if args.hetero else np.ones(n_agents)
    for v in args.variants:
        t0 = time.time()
        s, log, sl = run(v, args, stream, n_agents, scales)
        print(f"{v:18s} snr_common={s['snr_common']:5.2f} snr_within={s['snr_within']:5.2f} beta={s['beta']:5.2f} "
              f"noise={s['noise']:5.2f} | iter>=13: common={sl['snr_common']:.2f} within={sl['snr_within']:.2f} "
              f"beta={sl['beta']:.2f} noise={sl['noise']:.2f} ({time.time() - t0:.0f}s)\n                   used: {log}", flush=True)


if __name__ == "__main__":
    main()
