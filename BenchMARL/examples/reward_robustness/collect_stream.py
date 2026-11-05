"""Train MAPPO on clean rewards and save every collected batch (compact) for offline grouper evaluation."""
import argparse, os, time, torch
from benchmarl.algorithms import MappoConfig
from benchmarl.environments import PettingZooTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models import CnnConfig, MlpConfig

p = argparse.ArgumentParser()
p.add_argument("--task", default="pursuit")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--out", required=True)
p.add_argument("--frames", type=int, default=3000)
p.add_argument("--n_agents", type=int, default=16)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

class Saver(Callback):
    def on_batch_collected(self, batch):
        it = self.experiment.n_iters_performed
        g = list(self.experiment.group_map)[0]
        obs = batch[g]["observation"]; nxt = batch["next", g, "observation"]
        if args.task == "pursuit":
            assert torch.equal(obs, obs.round()) and obs.max() < 256
            obs, nxt = obs.to(torch.uint8), nxt.to(torch.uint8)
        act = batch[g]["action"]
        torch.save({"obs": obs.clone(), "next_obs": nxt.clone(), "action": act.clone(),
                    "reward": batch["next", g, "reward"].squeeze(-1).clone()},
                   f"{args.out}/batch_{it:04d}.pt")
        r = batch["next", g, "reward"]
        print(f"[iter {it}] catch-frac={(r > 2).float().mean().item():.4f} mean_r={r.mean().item():.4f}", flush=True)

cfg = ExperimentConfig.get_from_yaml()
cfg.loggers = []; cfg.create_json = False; cfg.evaluation = False; cfg.render = False
cfg.max_n_iters = args.iters; cfg.max_n_frames = None
cfg.on_policy_collected_frames_per_batch = args.frames
cfg.on_policy_n_minibatch_iters = 8; cfg.on_policy_minibatch_size = 400; cfg.lr = 3e-4
cfg.save_folder = args.out
if args.task == "pursuit":
    task = PettingZooTask.PURSUIT.get_from_yaml()
    if args.n_agents == 8:
        task.config.update(n_pursuers=8, n_evaders=30, x_size=16, y_size=16)
    model = CnnConfig.get_from_yaml()
else:
    task = PettingZooTask.SIMPLE_SPREAD.get_from_yaml(); task.config.update(N=args.n_agents)
    model = MlpConfig.get_from_yaml()
exp = Experiment(task=task, algorithm_config=MappoConfig.get_from_yaml(), model_config=model,
                 seed=0, config=cfg, callbacks=[Saver()])
t = time.time(); exp.run(); print("done in", time.time() - t)
