# Grouped-Reward Robust MARL (GR²-MARL)

Reward perturbation and grouped-consensus reward correction for BenchMARL.
Every iteration, `RewardRobustness.process` (called from the experiment loop):

1. perturbs every agent's reward with the kernel K (`reward_perturbation_*`);
2. groups the agents of the task's agent group (`grouping_method`);
3. replaces each agent's reward with a consensus over its group (`consensus`).

Evaluation always uses the true environment rewards.

| module | content |
|---|---|
| `task_spec.py` | per-task agent group, oracle thresholds, grouping-network preset (`TASK_SPECS`) |
| `perturbation.py` | Gaussian and Bernoulli reward kernels |
| `consensus.py` | mean / median / trimmed-mean / soft (weighted) / residual consensus |
| `clustering.py` | thresholds, sorted GMM, 1-D k-means, ε-gap clustering |
| `models.py` | `TransitionNet`: `(o, a[, o']) -> logits or reward` (CNN or MLP) |
| `groupers.py` | grouping methods (see below) |
| `metrics.py` | permutation-invariant grouping metrics and reward-error diagnostics |
| `robustness.py` | glue used by `Experiment` |

## Grouping methods

| `grouping_method` | needs clean rewards | description |
|---|---|---|
| `none` | – | No-Correction baseline |
| `oracle` | always | levels of the true reward (upper bound) |
| `global` | no | all agents in one group (ablation; `consensus=median` gives a Byzantine-style global median) |
| `warmup` | warm-up only | classifier on oracle labels, trained for `warmup_iters` iterations on clean rewards, then frozen |
| `gmm` | no | classifier on GMM clusters of the noisy rewards (paper "No-Warmup") |
| `reward_model` | no | regress the noisy rewards on `(o, a[, o'])`, group on the predictions |
| `reward_model_only` | no | reward-estimation baseline: replace rewards with the reward model |

The learned groupers default to `grouping_out_of_sample=True`: the groups of
iteration k come from parameters fitted on iterations < k, so they do not depend on
the current noise realisation. See the docstring of `groupers.py` for why this
matters.

`reward_model` safeguards:
- `reward_model_min_skill_z` (default 3): rewards are only corrected once the model
  beats a constant predictor out of sample. An untrained model predicts nearly the
  same reward for everyone, and similarity grouping would then act like global
  grouping and erase the credit of rare reward events.
- `reward_model_heads > 1`: a bootstrapped ensemble that leaves agents with uncertain
  predictions ungrouped. In our tests the heads agreed even on missed rare events, so
  this did not help.

Diagnostics with `evaluate_grouping=True` (they use the true rewards, analysis only):
- `reward_robustness/event_contrast`: within-timestep credit of the highest reward
  level; 0 means the event reward is shared by everyone;
- `pair_precision` / `pair_recall` against the oracle partition;
- `level*_bias`, MSE before and after correction;
- `grouper/skill_z`.

## Migrating from the old flags

| old configuration | new configuration |
|---|---|
| `pof_enable=False` | `grouping_method=none` |
| `pof_enable=True oracle_grouping=True` | `grouping_method=oracle` |
| `pof_enable=True train_grouping_warmup=True` | `grouping_method=warmup` |
| `pof_enable=True train_grouping_no_warmup=True` | `grouping_method=gmm` (add `grouping_gmm_sort=False grouping_out_of_sample=False` to reproduce the original behaviour exactly) |
| median consensus was implied by `reward_perturbation_type!=normal` | `consensus=auto` (same behaviour) or set explicitly |
| Bernoulli outliers hard-coded (±10, last commit ±3) | `reward_perturbation_outlier_value` (default 10) |

## Examples

```bash
# Pursuit, 16 pursuers, sigma = 3.5, warm-up-free reward-model grouping with diagnostics
# (recommended settings from the offline benchmark: o' as input, 1 epoch per iteration,
#  kernel consensus with epsilon = 0.5, out-of-sample skill gate)
python benchmarl/run.py algorithm=mappo task=pettingzoo/pursuit \
    model=layers/cnn model@critic_model=layers/cnn \
    experiment.reward_perturbation=True experiment.reward_perturbation_stdev=3.5 \
    experiment.grouping_method=reward_model experiment.grouping_use_next_obs=True \
    experiment.grouping_epochs=1 experiment.reward_model_assign=kernel \
    experiment.grouping_epsilon=0.5 experiment.reward_model_min_skill_z=3 \
    experiment.evaluate_grouping=True

# Bernoulli corruption, oracle grouping (median consensus is chosen automatically)
python benchmarl/run.py algorithm=mappo task=pettingzoo/pursuit \
    model=layers/cnn model@critic_model=layers/cnn \
    experiment.reward_perturbation=True experiment.reward_perturbation_type=bernoulli \
    experiment.reward_perturbation_flip_prob=0.1 experiment.grouping_method=oracle
```

Offline evaluation of groupers on a recorded training stream:
`examples/reward_robustness/` (`collect_stream.py`, then `offline_grouping_benchmark.py`).

## Adding a task

Add a `TaskSpec` to `TASK_SPECS` in `task_spec.py`: the agent group to correct, the
reward thresholds the oracle uses (or `None` for GMM levels), and the encoder.
