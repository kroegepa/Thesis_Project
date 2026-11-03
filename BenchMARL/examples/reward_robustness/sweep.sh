#!/usr/bin/env bash
# Full-scale GR²-MARL sweep (hydra multirun): methods x noise levels x training seeds.
#
# Reviewers asked for independent training seeds, so every cell is run with
# SEEDS (default 5). Each (method, sigma, seed) is a separate training run; the
# evaluation episodes within a run are NOT seeds.
#
# Usage:  bash sweep.sh pursuit16 [extra hydra overrides]   (or pursuit8 | spread5 | spread8)
# e.g.    SIGMAS="3.5" SEEDS="0,1,2" bash sweep.sh pursuit16 hydra/launcher=submitit_slurm
set -euo pipefail
set -f  # $COMMON is word-split on purpose; do not glob-expand e.g. [csv,wandb]
cd "$(dirname "$0")/../.."

SETTING=${1:-pursuit16}
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
COMMON="algorithm=$ALGO $TASK $MODEL seed=$SEEDS
  experiment.reward_perturbation=True experiment.reward_perturbation_stdev=$SIGMAS
  experiment.on_policy_collected_frames_per_batch=6000 experiment.max_n_frames=1500000
  experiment.evaluation_interval=120000 experiment.evaluate_grouping=True
  experiment.loggers=[csv,wandb]"

# 1) Baselines
python benchmarl/run.py -m $COMMON experiment.grouping_method=none,oracle "${@:2}"

# 1b) The paper's learned instances (35-epoch classifiers, as in the paper), with and
#     without the next observation o' as grouping input
python benchmarl/run.py -m $COMMON \
  experiment.grouping_method=warmup,gmm experiment.grouping_epochs=35 \
  experiment.grouping_use_next_obs=False,True "${@:2}"

# 2) Warm-up-free reward-model instances (see notes/paper_review_and_revision_plan.md, Sec. 3):
#    kernel consensus on predicted rewards, o' as input, 1 epoch per iteration (more epochs
#    only overfit the noisy targets), with and without the out-of-sample skill gate; plus
#    the reward-model-only baseline (Romoff et al.-style, as asked by Reviewer 2).
python benchmarl/run.py -m $COMMON \
  experiment.grouping_method=reward_model,reward_model_only \
  experiment.grouping_use_next_obs=True experiment.grouping_epochs=1 \
  experiment.reward_model_assign=kernel experiment.grouping_epsilon=0.5 \
  experiment.reward_model_min_skill_z=0,3 "${@:2}"

# 2b) Certified consensus: known noise std vs noise-variance-free
python benchmarl/run.py -m $COMMON \
  experiment.grouping_method=certified \
  experiment.grouping_use_next_obs=True experiment.grouping_epochs=1 \
  'experiment.certify_noise_std=${experiment.reward_perturbation_stdev},null' "${@:2}"

# 2c) Certified consensus with extensions (known noise std): temporal merging, IVW, and with/without
#     a 2-iteration clean warm-up of the proposal model
python benchmarl/run.py -m $COMMON \
  experiment.grouping_method=certified \
  experiment.grouping_use_next_obs=True experiment.grouping_epochs=1 \
  'experiment.certify_noise_std=${experiment.reward_perturbation_stdev}' \
  experiment.certify_temporal_window=5 experiment.certify_ivw=True \
  experiment.certify_warmup=False,True experiment.warmup_iters=2 "${@:2}"

# 3) Robust global aggregation baseline (Byzantine-style global median, Reviewer 3)
python benchmarl/run.py -m $COMMON \
  experiment.grouping_method=global experiment.consensus=median "${@:2}"
