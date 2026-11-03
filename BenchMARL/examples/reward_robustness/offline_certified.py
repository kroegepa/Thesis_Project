"""Run the package CertifiedGrouper offline on a recorded stream (collect_stream.py); pooled + late metrics."""
import argparse, sys, time
import torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from explore_general_grouping import Acc, load_stream
from benchmarl.reward_robustness import apply_consensus, get_task_spec, GroupingConfig, make_grouper

p = argparse.ArgumentParser()
p.add_argument("--stream", required=True); p.add_argument("--task", default="pursuit")
p.add_argument("--sigma", type=float, default=3.5); p.add_argument("--known", type=int, default=1)
p.add_argument("--warm", type=int, default=0); p.add_argument("--late", type=int, default=13)
p.add_argument("--temporal", type=int, default=0); p.add_argument("--ivw", type=int, default=0)
p.add_argument("--hetero", type=int, default=0); p.add_argument("--teams", type=int, default=1)
p.add_argument("--team_offset", type=float, default=0.0)
args = p.parse_args()
stream = load_stream(args.stream, 5)
import numpy as np
from explore_general_grouping import composite
if args.teams > 1:
    stream = [composite(d, args.teams, args.team_offset) for d in stream]
n_agents = stream[0].reward.shape[-1]
scale = torch.as_tensor(np.random.default_rng(123).uniform(0.5, 1.5, n_agents) if args.hetero else np.ones(n_agents), dtype=torch.float32)
torch.manual_seed(0)
cfg = GroupingConfig(method="certified", use_next_obs=True, epochs=1,
                     certify_noise_std=args.sigma if args.known else None,
                     certify_temporal_window=args.temporal, certify_ivw=bool(args.ivw))
g = make_grouper(cfg, get_task_spec(args.task))
for d in stream[0::2][: args.warm]:  # clean warm start of the proposal model
    clean = type(d)(d.obs, d.next_obs, d.action, d.true_reward.clone(), d.mask, d.true_reward)
    for _ in range(10):
        g.update(clean)
acc, late, used = Acc(), Acc(), []
t0 = time.time()
for i, d in enumerate(stream):
    torch.manual_seed(i)
    d.reward = d.true_reward + args.sigma * scale * torch.randn_like(d.true_reward)
    a, logs = g(d)
    c = apply_consensus(d.reward, a, d.mask)
    used.append(int(logs["grouper/certified_rule"]) + (10 if logs.get("grouper/temporal_active") else 0))
    if i % 2 == 1:
        acc.add(d.true_reward, c)
        if i >= args.late:
            late.add(d.true_reward, c)
for name, ac in [("all", acc), (f"iter>={args.late}", late)]:
    s = ac.summary()
    print(f"known={args.known} warm={args.warm} T={args.temporal} ivw={args.ivw} hetero={args.hetero} teams={args.teams} [{name}] snr_common={s['snr_common']:.2f} snr_within={s['snr_within']:.2f} "
          f"beta={s['beta']:.2f} noise={s['noise']:.2f}")
print(f"   rule per batch (0 none, 1 pred, 2 pred_in_id; +10 temporal): {used}  ({time.time() - t0:.0f}s)")
