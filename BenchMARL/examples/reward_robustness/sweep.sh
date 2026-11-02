#!/usr/bin/env bash
# Full-scale GR²-MARL sweep (hydra multirun): methods x noise levels x training seeds.
#
# Reviewers asked for independent training seeds, so every cell is run with
# SEEDS (default 5). Each (method, sigma, seed) is a separate training run; the
# evaluation episodes within a run are NOT seeds.
#
# Usage:  bash sweep.sh pursuit16 [extra hydra overrides]   (or pursuit8 | spread5 | spread8)
# Override e.g. METHODS="none,oracle,reward_model" SIGMAS="3.5" SEEDS="0,1,2" bash sweep.sh pursuit16
set -euo pipefail
cd "$(dirname "$0")/../.."

SETTING=${1:-pursuit16}
METHODS=${METHODS:-none,oracle,warmup,gmm,reward_model,reward_model_only}
SIGMAS=${SIGMAS:-0.0,1.5,2.5,3.5}
SEEDS=${SEEDS:-0,1,2,3,4}
ALGO=${ALGO:-mappo}

case "$SETTING" in
  pursuit16) TASK="task=pettingzoo/pursuit task.n_pursuers=16 task.n_evaders=60 task.x_size=23 task.y_size=23"
             MODEL="model=layers/cnn model@critic_model=layers/cnn" ;;
  pursuit8)  TASK="task=pettingzoo/pursuit task.n_pursuers=8 task.n_evaders=30 task.x_size=16 task.y_size=16"
             MODEL="model=layers/cnn model@critic_model=layers/cnn" ;;
  spread5)   TASK="task=pettingzoo/simple_spread task.N=5" ; MODEL="" ;;
  spread8)   TASK="task=pettingzoo/simple_spread task.N=8" ; MODEL="" ;;
  *) echo "unknown setting $SETTING"; exit 1 ;;
esac

# Settings reported in the paper (Sec. 5.2): 6000 frames/iteration, 1.5M frames,
# evaluation every 20 iterations on 10 episodes.
python benchmarl/run.py -m \
  algorithm="$ALGO" $TASK $MODEL \
  seed="$SEEDS" \
  experiment.reward_perturbation=True \
  experiment.reward_perturbation_stdev="$SIGMAS" \
  experiment.grouping_method="$METHODS" \
  experiment.on_policy_collected_frames_per_batch=6000 \
  experiment.max_n_frames=1500000 \
  experiment.evaluation_interval=120000 \
  experiment.evaluate_grouping=True \
  experiment.grouping_use_next_obs=True \
  experiment.loggers="[csv,wandb]" \
  "${@:2}"   # extra hydra overrides, e.g. hydra/launcher=submitit_slurm
