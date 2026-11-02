"""Offline benchmark of grouping methods on a recorded stream of training batches.

Policy learning is expensive and noisy, so this script isolates the part GR²-MARL is
about: how well each grouping method recovers the true rewards from perturbed ones.

The stream (see ``collect_stream.py``) holds the clean rewards of consecutive training
iterations of a policy trained on clean rewards. For every noise setting, each method
processes the batches *in order*, exactly as during training (out-of-sample methods
group a batch with parameters fitted on earlier batches, then update). The warm-up
method is trained on clean rewards of the even batches; all metrics are computed on
the odd batches only.

Metrics (pooled over the evaluated batches), with "event" = highest oracle level
(captures in Pursuit):

* ``corr``: correlation of corrected and true rewards (single summary number)
* ``signal``: fraction of the event reward gap preserved,
  ``(E[r^|event] - E[r^|other]) / (E[r|event] - E[r|other])`` (1 = unbiased, 0 = erased)
* ``noise_other`` / ``noise_event``: std of ``r^ - r`` on non-event / event samples
* ``contrast``: the same gap but *within* timesteps that contain an event, i.e. how much
  more the capturing agents receive than the non-capturing agents of the same step
  (0 when the event reward is shared by everyone, as with global grouping)
* ``snr``: ``contrast * gap / noise_other`` (how detectable the credited event reward is)
* ``prec`` / ``rec``: pairwise grouping precision / recall vs. the oracle partition

Example::

    python offline_grouping_benchmark.py --stream data/pursuit16 --task pursuit \
        --sigma 1.5 2.5 3.5 --methods none oracle global warmup gmm_paper rm_kmeans_next
"""

import argparse
import glob
import json
import time
from collections import defaultdict

import torch

from benchmarl.reward_robustness import (
    apply_consensus,
    get_task_spec,
    GroupData,
    GroupingConfig,
    make_grouper,
    oracle_labels,
    perturb_rewards,
)
from benchmarl.reward_robustness.metrics import event_contrast_sums, pairwise_grouping_scores

# name -> (grouping_method, GroupingConfig overrides)
METHODS = {
    "none": ("none", {}),
    "oracle": ("oracle", {}),
    "global": ("global", {}),
    "warmup": ("warmup", {}),
    "warmup_next": ("warmup", {"use_next_obs": True}),
    # Original No-Warmup: unsorted GMM labels, groups computed after fitting the same batch
    "gmm_paper": ("gmm", {"gmm_sort": False, "out_of_sample": False}),
    "gmm_sorted": ("gmm", {"gmm_sort": True, "out_of_sample": False}),
    "gmm_fixed": ("gmm", {"gmm_sort": True, "out_of_sample": True}),
    # Warm-up free reward-model groupers
    "rm_kmeans": ("reward_model", {"epochs": 5}),
    "rm_kmeans_next": ("reward_model", {"epochs": 5, "use_next_obs": True}),
    "rm_kmeans_next_insample": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "out_of_sample": False},
    ),
    "rm_eps_next": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "reward_model_assign": "epsilon", "epsilon": 1.0},
    ),
    "rm_kernel_next": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "reward_model_assign": "kernel", "epsilon": 0.5},
    ),
    "rm_kmeans_resid_next": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "reward_model_residual": True},
    ),
    "rm_global_resid_next": (
        "reward_model",
        {
            "epochs": 5,
            "use_next_obs": True,
            "reward_model_assign": "global",
            "reward_model_residual": True,
        },
    ),
    "rm_only_next": ("reward_model_only", {"epochs": 5, "use_next_obs": True}),
    # Sensitivity
    "rm_kmeans_next_e35": ("reward_model", {"epochs": 35, "use_next_obs": True}),
    "rm_kmeans_next_e1": ("reward_model", {"epochs": 1, "use_next_obs": True}),
    "rm_eps05_next": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "reward_model_assign": "epsilon", "epsilon": 0.5},
    ),
    "rm_eps2_next": (
        "reward_model",
        {"epochs": 5, "use_next_obs": True, "reward_model_assign": "epsilon", "epsilon": 2.0},
    ),
}


def load_stream(path, n_actions):
    files = sorted(glob.glob(f"{path}/batch_*.pt"))
    if not files:
        raise FileNotFoundError(f"No batch_*.pt files in {path}")
    for f in files:
        d = torch.load(f)
        action = d["action"]
        if not torch.is_floating_point(action):
            action = torch.nn.functional.one_hot(action.long(), n_actions).float()
        yield GroupData(
            obs=d["obs"].float(),
            next_obs=d["next_obs"].float(),
            action=action,
            reward=d["reward"].clone(),
            mask=torch.ones_like(d["reward"], dtype=torch.bool),
            true_reward=d["reward"],
        )


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
        signal = (c[event].mean() - c[~event].mean()) / gap
        err = c - t
        out = {
            "corr": torch.corrcoef(torch.stack([t, c]))[0, 1].item(),
            "signal": signal.item(),
            "noise_other": err[~event].std().item(),
            "noise_event": err[event].std().item() if event.sum() > 1 else float("nan"),
            "bias_event": err[event].mean().item(),
            "bias_other": err[~event].mean().item(),
            "mse": (err**2).mean().item(),
            "event_frac": event.float().mean().item(),
        }
        contrast = self.contrast[0] / self.contrast[1] if self.contrast[1] else float("nan")
        out["contrast"] = contrast
        out["snr"] = (contrast * gap / err[~event].std()).item()
        for k, v in self.pair.items():
            out[k] = sum(v) / len(v)
        return out


def run_method(name, args, sigma):
    method, overrides = METHODS[name]
    spec = get_task_spec(args.task)
    cfg = GroupingConfig(method=method, num_groups=args.num_groups, **overrides)
    if args.noise == "bernoulli" and method.startswith("reward_model"):
        cfg.reward_model_loss = "l1"  # conditional median: robust to outliers
    grouper = None if method == "none" else make_grouper(cfg, spec, device=args.device)
    consensus = "mean" if args.noise == "normal" else "median"
    acc = Accumulator(event_level=args.num_groups - 1)
    stream = list(load_stream(args.stream, args.n_actions))[: args.max_batches]
    if grouper is not None and grouper.needs_warmup:
        for data in stream[0::2][: args.warmup_batches]:
            grouper.warmup_update(data)
    timeline = []
    for i, data in enumerate(stream):
        torch.manual_seed(args.seed * 1000 + i)  # same noise for every method
        if args.noise == "normal":
            data.reward = perturb_rewards(data.true_reward, "normal", stdev=sigma)
        else:
            data.reward = perturb_rewards(
                data.true_reward, "bernoulli", flip_prob=sigma, outlier_value=args.outlier
            )
        if grouper is None:
            corrected, labels = data.reward, None
        else:
            assignment, _ = grouper(data)
            corrected = apply_consensus(data.reward, assignment, data.mask, method=consensus)
            labels = assignment.labels
        if i % 2 == 1:
            levels = oracle_labels(data.true_reward, spec, args.num_groups, data.mask)
            acc.add(data.true_reward, corrected, levels, labels)
            step = Accumulator(event_level=args.num_groups - 1)
            step.add(data.true_reward, corrected, levels, labels)
            s = step.summary()
            timeline.append({"iter": i, "corr": s["corr"], "signal": s["signal"]})
    return acc.summary(), timeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream", required=True)
    p.add_argument("--task", default="pursuit")
    p.add_argument("--n_actions", type=int, default=5)
    p.add_argument("--noise", default="normal", choices=["normal", "bernoulli"])
    p.add_argument("--sigma", type=float, nargs="+", default=[2.5],
                   help="Gaussian stdev, or flip probability for --noise bernoulli")
    p.add_argument("--outlier", type=float, default=10.0)
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--num_groups", type=int, default=3)
    p.add_argument("--warmup_batches", type=int, default=10)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = {}
    for sigma in args.sigma:
        for name in args.methods:
            t0 = time.time()
            summary, timeline = run_method(name, args, sigma)
            results[f"{name}@{sigma}"] = {"summary": summary, "timeline": timeline}
            keys = ["corr", "signal", "contrast", "noise_other", "snr", "pair_precision", "pair_recall"]
            vals = " ".join(f"{k}={summary.get(k, float('nan')):.3f}" for k in keys)
            print(f"[{args.noise} {sigma}] {name:26s} {vals} ({time.time() - t0:.0f}s)", flush=True)
            if args.out:
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
