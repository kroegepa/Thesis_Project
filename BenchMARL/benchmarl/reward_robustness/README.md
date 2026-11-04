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
| `certification.py` | statistics for certified consensus (rule/temporal certification, identity clusters, noise-variance estimators) |
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
| `certified` | no | **certified consensus**: reward-model proposals used only once the noisy data certify them (below) |

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

## Certified consensus (`grouping_method=certified`)

Groupings are only applied once the noisy data certify them. With additive, zero-mean
noise that is independent across agents and steps:

    E[(r~_i - r~_j)^2] = E[(r_i - r_j)^2] + s_i^2 + s_j^2

The noise shape does not matter, and per-agent variances s_i^2 are allowed. Each iteration:

1. **Propose.** The reward model, fitted on *earlier* batches, proposes merging agents of
   a step whose predicted rewards are ε-close. A second proposal does the same only
   within persistent identity clusters (roles/teams found from cross-agent statistics).
2. **Certify.** Each proposal rule is scored on later batches by the mean of the
   statistic above over the pairs it merges. A rule is used only if the upper confidence
   bound (`certify_z` SEs) is ≤ ε². Among certified rules, the one that merges most is
   applied; otherwise rewards are left unchanged. With high probability, the RMS bias of
   the consensus is then ≤ ε.
3. **Identity clusters only restrict merges.** At high noise, rare single-step reward
   events are invisible in reward statistics, so merges must be vetted by the model.

Noise variance:

- `certify_noise_std=<std>`: known noise. To tie it to the simulated perturbation, use
  `experiment.certify_noise_std='${experiment.reward_perturbation_stdev}'`.
- `certify_noise_std=null` (default): σ is estimated (`certify_noise_estimator`).
  - `replicates` (default) assumes **homoscedastic noise** (the same σ for every agent,
    any distribution) and a true reward that is a **deterministic function of the local
    transition** (o, a, o').
    - Samples with identical transitions then differ only by noise. Their within-class
      variance (ANOVA pure error) is unbiased for σ².
    - With continuous observations, it falls back to nearest neighbours in standardised
      (o, a, o') space and extrapolates their squared differences to distance 0
      (Tong & Wang, 2005).
    - The estimate is pooled over all iterations. The certifiers store raw squared
      differences; the noise floor 2σ̂² is subtracted when certifying, and its SE is
      added to the margin.
    - Unlike the temporal estimator, rare one-step reward events do not inflate σ̂², and
      temporal merging is available.
    - Precision matters: certifying ε against a noise floor of 2σ² needs σ² to about
      ε²/(2σ²) relative precision (1% at σ=3.5, ε=0.5). Pursuit has ~10⁴ exact
      replicates per iteration.
    - If rewards depend on something outside the local transition (e.g. a hidden team
      offset), σ̂² is inflated and certification becomes optimistic. With roles, use
      `grouping_use_agent_id=True` so replicates are per agent.
  - `temporal`: per-agent variances from temporal differences of the model residuals,
    identity clusters from lag-1 statistics, and the reward model must beat a constant
    out of sample (`certify_skill_gate`). Single-step reward events that the model has
    not yet learned are indistinguishable from noise, so this mode certifies somewhat
    early. In the offline tests it kept 0.77–0.85 of within-step credit in Pursuit, vs
    0.95 with known σ.

Extensions:

- `certify_ivw=True`: inverse-variance weighted group means (weights 1/sᵢ²), the best
  linear unbiased estimate when agents' noise levels differ. Weights use per-agent
  estimates from earlier batches, so the mean stays unbiased.
- `certify_temporal_window=5` (known σ or `replicates`): also average each agent's corrected
  reward over its own neighbouring steps (same episode) whose predicted rewards are
  ε-close. The temporal rule is certified like the spatial one, from
  (r̃_{i,t+k} − r̃_{i,t})² − 2sᵢ². This gives large gains for persistent rewards, and is
  rejected automatically for spiky ones until the model explains the spikes.
- `certify_warmup=True warmup_iters=2`: fit the proposal model on a few clean iterations
  first. Certification then starts almost immediately.
- `certify_replicate_consensus=True`: before the certified merges, average the noisy
  rewards of all samples in the batch (any agent, any step) with identical replicate keys.
  Under local determinism this has zero bias, so it applies from the first iteration,
  before anything is certified.
  - Pursuit: mostly co-located pursuers taking the same action; it removes 1–34% of the
    noise variance with the transition key, 4–54% with the o' key.
  - No-op for continuous observations.
- `certify_replicate_key=next_obs`: replicates on o' only. This assumes the reward is a
  function of o', which is stronger, and gives ~2× more replicates early, so σ̂² is
  precise sooner (Pursuit warm start: certified from iteration 6 instead of 10). In
  Pursuit it is not exactly true (RMS consensus bias up to 0.08).

Offline results (Pursuit σ=3.5, within-step SNR pooled over a 20-iteration stream; no
correction 0.14, spatial oracle 0.54): certified 0.16 → + 2 clean batches 0.43 →
+ 2 clean batches + temporal window 5 **0.74**. SimpleSpread σ=2.5 (shared-objective SNR;
none 0.20, oracle 0.42): certified 0.38 → + temporal 0.52.

σ-free with `replicates`: identical to known σ without a warm start. Pursuit 0.16 with
β 0.96 (temporal estimator: β 0.85), and 0.80 after iteration 13 with temporal window 5.
SimpleSpread 0.38, and 0.52 with temporal. With a clean warm start, certification waits
for a precise σ̂² (iteration 10 instead of 1 in Pursuit at σ=3.5).

Recommended: `grouping_use_next_obs=True grouping_epochs=1`. ε (`grouping_epsilon`) is the
tolerance of Assumption 1 in reward units. It is also an "erasure budget": events with
rate p and size Δ may be averaged away if p·Δ² ≤ ε².

Logged (`reward_robustness/grouper/*`):
- `certified_rule` (0 none, 1 `pred`, 2 `pred_in_id`);
- `pred_delta` / `pred_margin` (estimate and z·SE);
- `identity_clusters`, `noise_var_mean`, `skill_z`;
- `noise_var_replicates` / `_se`, `noise_replicate_dof` (replicate estimator).

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
