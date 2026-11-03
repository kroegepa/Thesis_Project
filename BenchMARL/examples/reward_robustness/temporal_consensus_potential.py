"""Potential of spatio-temporal consensus: per-agent centred moving average (window w) of the
consensus-corrected rewards, for fixed groupings. Metrics as in explore.py (per-agent decomposition)."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from explore_general_grouping import Acc, load_stream  # noqa: E402
from benchmarl.reward_robustness import apply_consensus, Assignment
from benchmarl.reward_robustness.clustering import epsilon_labels

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="folder containing recorded streams (spread5, pursuit16)")
args = p.parse_args()
for stream, sigma in [("spread5", 2.5), ("spread5", 3.5), ("pursuit16", 3.5)]:
    data = load_stream(f"{args.data}/{stream}", 5)[10:]
    print(f"== {stream} sigma={sigma}")
    for gname in ["none", "oracle", "global"]:
        row = f"   {gname:7s}"
        for w in [1, 3, 5, 9]:
            acc = Acc()
            for i, d in enumerate(data):
                torch.manual_seed(i)
                noisy = d.true_reward + sigma * torch.randn_like(d.true_reward)
                lab = {"none": None, "oracle": epsilon_labels(d.true_reward, 0.05),
                       "global": torch.zeros_like(noisy, dtype=torch.long)}[gname]
                c = apply_consensus(noisy, Assignment(labels=lab), d.mask)
                if w > 1:  # centred moving average along time (per env, per agent)
                    k = torch.ones(1, 1, w) / w
                    x = c.permute(0, 2, 1).reshape(-1, 1, c.shape[1])
                    x = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (w // 2, w // 2), mode="replicate"), k)
                    c = x.reshape(c.shape[0], c.shape[2], c.shape[1]).permute(0, 2, 1)
                acc.add(d.true_reward, c)
            s = acc.summary()
            row += f" | w={w}: common {s['snr_common']:.2f} within {s['snr_within']:.2f} beta {s['beta']:.2f}"
        print(row, flush=True)
