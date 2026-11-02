"""Offline benchmark of grouping methods on a recorded stream of training batches.

Policy learning is expensive and noisy, so this script isolates the part GR²-MARL is
about: how well each grouping method recovers the true rewards from perturbed ones.

The stream (see ``collect_stream.py``) holds the clean rewards of consecutive training
iterations of a policy trained on clean rewards. For every noise level, each method
processes the batches *in order*, exactly as during training: out-of-sample methods
group a batch with parameters fitted on earlier batches, then update. The warm-up
classifier is trained on the clean rewards of the even batches. All metrics are
computed on the odd batches only.

Metrics (pooled over the evaluated batches), where "event" is the highest oracle
level (captures in Pursuit):

* ``corr``: correlation of corrected and true rewards
* ``signal``: fraction of the event-vs-rest reward gap preserved,
  ``(E[r^|event] - E[r^|other]) / (E[r|event] - E[r|other])``
* ``contrast``: the same gap but *within* timesteps that contain an event, i.e. how
  much more the capturing agents receive than the non-capturing agents of the same
  step. It is 0 when the event reward is shared by everyone (global grouping
  destroys credit assignment).
* ``noise_other``: std of ``r^ - r`` on non-event samples
* ``snr``: ``contrast * gap / noise_other`` (how detectable the credited event reward is)
* ``pair_precision`` / ``pair_recall``: pairwise grouping precision / recall vs. the
  oracle partition

Reward-model *families* train one model and evaluate several ways of turning its
predictions into groups (k-means, ε-gap, kernel, residual, replacement), plus the
in-sample variant (the same model applied after updating on the evaluated batch).

Example::

    python offline_grouping_benchmark.py --stream data/pursuit16 --sigma 2.5 3.5 \\
        --methods none oracle global gmm_direct warmup gmm rm_next rm
"""

import argparse
import glob
import json
import time
from collections import defaultdict

import torch

from benchmarl.reward_robustness import (
    apply_consensus,
    Assignment,
    get_task_spec,
    GroupData,
    GroupingConfig,
    make_grouper,
    oracle_labels,
    perturb_rewards,
)
from benchmarl.reward_robustness.clustering import gmm_labels
from benchmarl.reward_robustness.groupers import assignment_from_predictions
from benchmarl.reward_robustness.metrics import event_contrast_sums, pairwise_grouping_scores

# Single methods: name -> (grouping_method, GroupingConfig overrides)
METHODS = {
    "none": ("none", {}),
    "oracle": ("oracle", {}),
    "global": ("global", {}),
    "gmm_direct": ("gmm_direct", {}),  # groups = GMM clusters of the current noisy rewards
    "warmup": ("warmup", {}),
    "warmup_next": ("warmup", {"use_next_obs": True}),
    "gmm": ("gmm", {"gmm_sort": False, "out_of_sample": False}),  # original No-Warmup
    "gmm_fixed": ("gmm", {"gmm_sort": True, "out_of_sample": True}),
}

# Reward-model families: name -> (GroupingConfig overrides, variants)
# variant -> (mode, kwargs, in_sample)
_ALL_VARIANTS = {
    "kmeans": ("kmeans", {}, False),
    "kmeans_insample": ("kmeans", {}, True),
    "eps0.5": ("epsilon", {"epsilon": 0.5}, False),
    "eps1": ("epsilon", {"epsilon": 1.0}, False),
    "eps2": ("epsilon", {"epsilon": 2.0}, False),
    "kernel0.5": ("kernel", {"epsilon": 0.5}, False),
    "kmeans_resid": ("kmeans", {"residual": True}, False),
    "global_resid": ("global", {"residual": True}, False),
    "only": ("replace", {}, False),
}
FAMILIES = {
    "rm_next": ({"use_next_obs": True}, _ALL_VARIANTS),
    "rm": (
        {"use_next_obs": False},
        {k: _ALL_VARIANTS[k] for k in ["kmeans", "kmeans_insample", "only"]},
    ),
}


def load_stream(path, n_actions, max_batches=None):
    files = sorted(glob.glob(f"{path}/batch_*.pt"))[:max_batches]
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files in {path}")
    stream = []
    for f in files:
        d = torch.load(f)
        action = d["action"]
        if not torch.is_floating_point(action):
            action = torch.nn.functional.one_hot(action.long(), n_actions).float()
        stream.append(
            GroupData(
                obs=d["obs"].float(),
                next_obs=d["next_obs"].float(),
                action=action,
                reward=d["reward"].clone(),
                mask=torch.ones_like(d["reward"], dtype=torch.bool),
                true_reward=d["reward"],
            )
        )
    return stream


class Accumulator:
    def __init__(self, event_level: int):
        self.event_level = event_level
        self.true, self.corr, self.levels = [], [], []
        self.pair = defaultdict(list)
        self.contrast = [0.0, 0.0]

    def add(self, true, corrected, levels, pred_labels):
        self.true.append(true.flatten())
        self.corr.append(corrected.flatten())
        self.levels.append(levels.flatten())
        num, den, _ = event_contrast_sums(true, corrected, levels == self.event_level)
        self.contrast[0] += num
        self.contrast[1] += den
        if pred_labels is not None:
            for k, v in pairwise_grouping_scores(levels, pred_labels).items():
                self.pair[k].append(v)

    def summary(self):
        t, c, lv = torch.cat(self.true), torch.cat(self.corr), torch.cat(self.levels)
        event = lv == self.event_level
        gap = t[event].mean() - t[~event].mean()
        err = c - t
        contrast = self.contrast[0] / self.contrast[1] if self.contrast[1] else float("nan")
        out = {
            "corr": torch.corrcoef(torch.stack([t, c]))[0, 1].item(),
            "signal": ((c[event].mean() - c[~event].mean()) / gap).item(),
            "contrast": contrast,
            "noise_other": err[~event].std().item(),
            "noise_event": err[event].std().item() if event.sum() > 1 else float("nan"),
            "bias_event": err[event].mean().item(),
            "bias_other": err[~event].mean().item(),
            "mse": (err**2).mean().item(),
            "event_frac": event.float().mean().item(),
            "snr": (contrast * gap / err[~event].std()).item(),
        }
        for k, v in self.pair.items():
            out[k] = sum(v) / len(v)
        return out


def _perturb(data, args, sigma, i):
    torch.manual_seed(args.seed * 1000 + i)  # identical noise for every method
    if args.noise == "normal":
        data.reward = perturb_rewards(data.true_reward, "normal", stdev=sigma)
    else:
        data.reward = perturb_rewards(
            data.true_reward, "bernoulli", flip_prob=sigma, outlier_value=args.outlier
        )


def _cfg(args, method, overrides):
    cfg = GroupingConfig(method=method, num_groups=args.num_groups, **overrides)
    if method in ("warmup", "gmm"):
        cfg.epochs = args.classifier_epochs
    else:
        cfg.epochs = args.reward_model_epochs
    if args.noise == "bernoulli":
        cfg.reward_model_loss = "l1"  # conditional median: robust to outliers
    return cfg


def run_method(name, args, sigma, stream, spec):
    method, overrides = METHODS[name]
    grouper = None
    if method not in ("none", "gmm_direct"):
        grouper = make_grouper(_cfg(args, method, overrides), spec, device=args.device)
    acc = Accumulator(args.num_groups - 1)
    if grouper is not None and grouper.needs_warmup:
        for data in stream[0::2][: args.warmup_batches]:
            grouper.warmup_update(data)
    for i, data in enumerate(stream):
        _perturb(data, args, sigma, i)
        if grouper is not None:
            assignment, _ = grouper(data)
        elif method == "gmm_direct":
            assignment = Assignment(labels=gmm_labels(data.reward, args.num_groups, data.mask))
        else:
            assignment = Assignment()
        if i % 2 == 1:
            corrected = apply_consensus(data.reward, assignment, data.mask, method=args.consensus)
            levels = oracle_labels(data.true_reward, spec, args.num_groups, data.mask)
            acc.add(data.true_reward, corrected, levels, assignment.labels)
    return {name: acc.summary()}


def run_family(name, args, sigma, stream, spec):
    overrides, variants = FAMILIES[name]
    grouper = make_grouper(_cfg(args, "reward_model", overrides), spec, device=args.device)
    accs = {v: Accumulator(args.num_groups - 1) for v in variants}
    need_insample = any(ins for _, _, ins in variants.values())
    for i, data in enumerate(stream):
        _perturb(data, args, sigma, i)
        evaluate = i % 2 == 1
        pred_out = grouper.predict_rewards(data) if evaluate and grouper.n_updates else None
        grouper.update(data)
        pred_in = grouper.predict_rewards(data) if evaluate and need_insample else None
        if not evaluate:
            continue
        levels = oracle_labels(data.true_reward, spec, args.num_groups, data.mask)
        for v, (mode, kw, in_sample) in variants.items():
            pred = pred_in if in_sample else pred_out
            if pred is None:
                assignment = Assignment()
            else:
                assignment = assignment_from_predictions(
                    pred, data.mask, mode=mode, num_groups=args.num_groups, **kw
                )
            corrected = apply_consensus(data.reward, assignment, data.mask, method=args.consensus)
            accs[v].add(data.true_reward, corrected, levels, assignment.labels)
    return {f"{name}/{v}": acc.summary() for v, acc in accs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--task", default="pursuit")
    p.add_argument("--n_actions", type=int, default=5)
    p.add_argument("--noise", default="normal", choices=["normal", "bernoulli"])
    p.add_argument("--sigma", type=float, nargs="+", default=[2.5],
                   help="Gaussian stdev, or flip probability for --noise bernoulli")
    p.add_argument("--outlier", type=float, default=10.0)
    p.add_argument("--methods", nargs="+", default=list(METHODS) + list(FAMILIES))
    p.add_argument("--num_groups", type=int, default=3)
    p.add_argument("--warmup_batches", type=int, default=10)
    p.add_argument("--classifier_epochs", type=int, default=35)
    p.add_argument("--reward_model_epochs", type=int, default=5)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    args.consensus = "mean" if args.noise == "normal" else "median"

    spec = get_task_spec(args.task)
    stream = load_stream(args.stream, args.n_actions, args.max_batches)
    keys = ["corr", "signal", "contrast", "noise_other", "snr", "pair_precision", "pair_recall"]
    results = {}
    for sigma in args.sigma:
        for name in args.methods:
            t0 = time.time()
            runner = run_family if name in FAMILIES else run_method
            out = runner(name, args, sigma, stream, spec)
            for variant, summary in out.items():
                results[f"{variant}@{sigma}"] = summary
                vals = " ".join(f"{k}={summary.get(k, float('nan')):.3f}" for k in keys)
                print(f"[{args.noise} {sigma}] {variant:22s} {vals} ({time.time() - t0:.0f}s)",
                      flush=True)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
