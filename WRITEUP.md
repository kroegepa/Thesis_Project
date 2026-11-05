# GR²-MARL revision: what was done, what worked, how to use it

This write-up summarises the work on this branch for the AAMAS'26 submission
"Consensus-Based Individual Reward Robustness in Multi-Agent Reinforcement Learning".
The main goal was the reviewers' main criticism: **groupings that work without a clean
warm-up period.**

The document covers:
1. what changed in the code;
2. everything that was tried, with results;
3. what worked best;
4. how to use the code;
5. limitations and what is still missing for the paper.

Detailed notes, including every table, are in
[`notes/paper_review_and_revision_plan.md`](notes/paper_review_and_revision_plan.md) (§2–3).
The new theory statements (propositions, assumptions, proofs) are in
[`notes/theory_additions.md`](notes/theory_additions.md).

---

## 0. Summary

- **Why the paper's No-Warmup failed.** It grouped agents with clusters of the *current
  noisy* rewards, so the groups followed the noise (selection bias). On top of that, GMM
  labels switched meaning between iterations. Fixing both is not enough: clustering
  noisy rewards gives mostly noise.
- **What works without clean data: certified consensus** (`grouping_method=certified`).
  1. A reward model fitted only on noisy rewards *proposes* merges.
  2. A merge rule is applied only once the noisy data *certify* it: an upper confidence
     bound on the mean squared true-reward difference of the merged pairs must be ≤ ε².
     This holds because E[(r̃ᵢ − r̃ⱼ)²] = E[(rᵢ − rⱼ)²] + σᵢ² + σⱼ² for independent
     zero-mean noise.
  3. If certified, the RMS bias of the consensus is ≤ ε with high probability. If not,
     rewards are left unchanged, so the method falls back to No-Correction instead of
     doing harm.
- **σ does not need to be known.** Assume the noise variance is the same for every
  agent, and the true reward is a deterministic function of the agent's local transition
  (o, a, o'). Then σ² is estimated from replicated transitions. This is now the default.
  Offline it matches known σ.
- **Certified temporal merging gives the largest gain.** It also averages each agent's
  reward over its own neighbouring steps, again only where certified. A window of 9
  steps is recommended.
- **Best offline results** (Pursuit, 16 pursuers, Gaussian σ = 3.5; within-step SNR,
  higher is better):

  | | all 20 iterations | after the cold start (it. ≥ 13) |
  |---|---|---|
  | no correction | 0.14 | 0.16 |
  | oracle grouping, paper's 3 reward levels (true rewards, same step) | 0.42 | 0.49 |
  | certified, σ-free, temporal window 9 + replicate consensus, no clean data | 0.18 | **0.91** |
  | certified, temporal window 9, 2 clean warm-up iterations, σ known | **0.84** | 0.97 |
  | certified, temporal window 9, 2 clean warm-up iterations, σ-free (o′ key + replicate consensus) | 0.25 | **1.04** |

  Credit retention (β, the share of individual within-step credit kept) stays at
  0.86–0.95. The certified pipeline beats same-step oracle grouping once certified,
  because it also averages over time.
- **The cold start remains.** Without clean data, the reward model needs ~10–13
  iterations of noisy data before anything can be certified. Until then the method is
  No-Correction. This is about 5% of a 250-iteration run.
- **Not done yet: end-to-end multi-seed training runs.** All results above are
  *offline replays* of recorded training streams. The short CPU pilots were dominated
  by seed variance. `BenchMARL/examples/reward_robustness/sweep.sh` is ready for the
  cluster.

---

## 1. What changed in the code

**Refactor.** The project-specific code (`pof_methods`, `pof_*` flags) was rewritten
as the package `BenchMARL/benchmarl/reward_robustness/`:

| module | content |
|---|---|
| `perturbation.py` | Gaussian and Bernoulli reward kernels |
| `consensus.py` | mean / median / trimmed mean / soft-weighted / residual consensus, alive mask |
| `clustering.py` | thresholds, sorted GMM, 1-D k-means, ε-gap clustering |
| `groupers.py` | all grouping methods (`none`, `oracle`, `global`, `warmup`, `gmm`, `reward_model`, `reward_model_only`, `certified`) |
| `certification.py` | certification statistics, identity clusters, noise-variance estimators, temporal merging, replicate consensus |
| `models.py` | `TransitionNet` (o, a[, o′]) → reward or logits (CNN or MLP) |
| `metrics.py` | permutation-invariant grouping metrics, reward-error diagnostics |
| `task_spec.py` | per-task settings (agent group, oracle thresholds, network preset) |
| `robustness.py` | glue called by `Experiment` |

Also changed:
- the experiment loop was split into helpers;
- `wandb` is imported lazily, so it is optional;
- checkpoints include the grouper state;
- there are 49 unit tests (`BenchMARL/test/test_reward_robustness.py`);
- the old `pof_*` flags map to the new keys (migration table in the
  [package README](BenchMARL/benchmarl/reward_robustness/README.md)).

**Bugs found in the original code that affect reported results.** Details are in notes §2.

| issue | effect | fix |
|---|---|---|
| GMM components unsorted: label ids changed meaning between batches | No-Warmup classifier trained on inconsistent targets | `grouping_gmm_sort=True` |
| No-Warmup grouped the batch it was trained on (in-sample, 35 epochs) | groups followed the noise: biased consensus | `grouping_out_of_sample=True` |
| Fig. 3 used F1 of cluster *ids* | near-zero F1 even for perfect partitions | permutation-invariant pairwise precision/recall, ARI |
| Bernoulli outliers changed from ±10 to ±3 in the last commit | check Fig. 4 | `reward_perturbation_outlier_value` (default 10) |
| alive mask unused (`goup_mask` typo) | dead agents counted in group means (Battle) | fixed |
| warm-up budget 150k frames in the config vs 60k in the paper | report the real cost | documented |
| "10 seeds" were 10 evaluation episodes of one training run | Reviewer 2 is right | the sweep uses ≥ 5 training seeds |
| training-episode returns never reset (root `done` flag) | training-curve logs only | fixed |

---

## 2. How results were measured

Policy learning is slow and noisy, so grouping methods were compared **offline**:
- `collect_stream.py` records the batches of a MAPPO run on clean rewards: Pursuit
  with 16 pursuers (3000 frames per iteration), and SimpleSpread with 5 agents;
- the batches are replayed under noise through each method, in order, exactly as
  during training. Out-of-sample methods group a batch with parameters fitted on earlier
  batches.

The metrics use the true rewards (analysis only). Corrected rewards are decomposed per
agent as r̂ₜᵢ = a·mₜ + β·dₜᵢ + e:
- mₜ is the true step-mean reward and dₜᵢ the agent's true within-step deviation;
- **β (credit retention)**: 1 keeps individual credit, 0 erases it (global grouping);
- **common SNR** = a·sd(m)/sd(e): signal-to-noise of the shared objective;
- **within SNR** = β·sd(d)/sd(e): signal-to-noise of individual credit (Pursuit
  captures). This is what matters most for credit assignment.

The earlier benchmark (notes §3.2) used a related "capture contrast / SNR" metric. The
numbers in §3.1 below use that metric and are not comparable with later tables.

"Pooled" means over the 20-iteration stream, including the cold start. "Late" means
iterations ≥ 13. Each stream is a single recording, so differences of ±0.02 are within
noise.

"Oracle" groups agents of a step by their true rewards, in one of two ways:
- the paper's oracle (3 reward levels): Pursuit within SNR 0.42 pooled / 0.49 late;
  SimpleSpread common SNR 0.42;
- the stricter exact-equal-reward oracle of the exploration harness: Pursuit 0.54 pooled.

---

## 3. Everything that was tried

### 3.1 Reward model on noisy rewards (first round)

Idea: E[r̃ | o, a, o′] = E[r | o, a, o′] under zero-mean noise. A regressor trained on
corrupted rewards is therefore consistent. Group agents whose *predicted* rewards are
close, out of sample.

Pursuit σ = 3.5, contrast-SNR metric (none 1.45, oracle 4.38):

| method | SNR | verdict |
|---|---|---|
| GMM on the current noisy rewards (No-Warmup limit) | 1.31 | **below no correction**: selection bias |
| paper No-Warmup classifier (unsorted, in-sample) | 0.80 | worse still |
| No-Warmup with sorted labels, out of sample | 1.28 | fixes help, still below no correction |
| warm-up classifier, input (o, a) (paper) | 2.72 | needs clean data |
| warm-up classifier, input (o, a, o′) | **3.95** | **o′ helps every learned grouper** |
| reward model (o, a, o′), k-means K = 3 | 2.37 | middle cluster mixes events |
| reward model (o, a, o′), ε-gap ε = 0.5 | 3.07 | |
| reward model (o, a, o′), soft kernel ε = 0.5 | **3.32** | 76% of oracle, no clean data |
| reward model, global + residual | 3.00 | |
| reward-model-only (Romoff-style baseline) | 7.00 | flattered by the metric: systematic model error never averages out |
| bootstrapped ensemble gate (5 heads) | 3.33 | heads agree even on missed events: no gain |

Findings:
- **Train the reward model for 1 epoch per iteration.** 5 or 35 epochs overfit the
  noisy targets.
- **Gate on skill.** Correct only once the model beats a constant out of sample (skill
  z ≥ 3).
- **Ungated grouping is dangerous early.** An untrained model predicts the same reward
  for everyone, so similarity grouping becomes global grouping and erases capture
  credit (contrast ≈ 0 for the first ~13 iterations).
- **The end-to-end pilots were inconclusive.** They used 8 pursuers, σ = 2.5,
  40 iterations and 1–2 seeds. Oracle ≫ no correction, but warm-up-free variants were
  within seed variance.

### 3.2 Feasibility checks for finding groups under noise

| idea | result | verdict |
|---|---|---|
| reward-free clustering of (o, Δo, a), K = 64–1024 | 0% of captures in capture-dominated clusters | fails: a capture is a configuration, not a region of input space |
| reward model trained on replay of all past data | gate opens at the same iteration | no gain |
| locality prior (5×5 neighbourhood input) | gate 2–3 iterations earlier | modest, task-specific |
| **small clean budget**: reward model warm-started on k clean batches | k = 2 (6k frames): SNR 3.02 (70% of oracle) | **best value per clean frame** |
| warm-up classifier frozen after k clean batches | needs ≥ 5 batches | |

### 3.3 Certified consensus (known σ)

Three ideas carry it:
1. **Cross-agent second moments are noise-free.** E[(r̃ᵢ − r̃ⱼ)²] − σᵢ² − σⱼ² is unbiased
   for the squared true-reward difference: no features, no clean data.
2. **Certify rules, not groups.** Average this over the same-step pairs a proposal
   rule merges, on later batches. Apply the rule only if UCB ≤ ε². This gives RMS
   consensus bias ≤ ε w.h.p.
3. **Proposals vs. restrictions.** The reward model proposes merges. Persistent
   identity clusters (roles, teams, from the same statistic over time) can only
   *restrict* them, because rare large events are invisible in reward moments at high σ.

Pooled results (exploration harness, known σ):

| setting | method / oracle / none | β |
|---|---|---|
| SimpleSpread σ = 2.5 (common SNR) | 0.38 / 0.42 / 0.20 | – |
| SimpleSpread σ = 3.5 | 0.24 / 0.30 / 0.14 | – |
| roles (two hidden teams, offset +1), σ = 2.5 | 0.25 / 0.34 / 0.16 | **0.98** (global and reward model: 0.03) |
| Pursuit σ = 3.5 (within SNR; exact-equal oracle) | 0.16 / 0.54 / 0.14 | 0.95 |

**ε is an erasure budget.** Events of rate p and size Δ may be averaged away if
p·Δ² ≤ ε², so ε must be set relative to the reward scale. Certification needs
~32·(σ/ε)⁴ merged-pair contrasts; z = 1 was the best trade-off.

### 3.4 Dropping the known-σ assumption

| approach | result | verdict |
|---|---|---|
| per-agent σᵢ from temporal differences of model residuals, identity from lag-1 statistics (noise-free) | SimpleSpread 0.30 vs 0.38; Pursuit β 0.77–0.85 vs 0.95 | works for persistent rewards; **leaks credit** on transient events: unexplained one-step events are indistinguishable from noise |
| homoscedasticity alone | provably insufficient: white reward variation and noise give identical data | – |
| **homoscedastic noise + reward deterministic in (o, a, o′): replicate estimator** | Pursuit: ANOVA pure error over identical transitions, unbiased (−0.3% ± 1.0% per batch). SimpleSpread: k-NN intercept (Tong & Wang 2005), −0.5% bias | **default now: matches known σ** |

Key design choices for the replicate estimator:
- **Pool all iterations.** The noise is stationary.
- **Subtract 2σ̂² at query time, with its SE in quadrature.** A lower-confidence shift
  was too conservative: at σ = 3.5, ε = 0.5, a 1% error in σ² uses the whole ε² budget.

Results (pooled, late in brackets):

| | σ known | σ-free, replicates | σ-free, temporal estimator |
|---|---|---|---|
| Pursuit σ = 3.5, within SNR | 0.16 (0.28), β 0.97 | 0.16 (0.52), **β 0.96** | 0.17 (0.51), β 0.85 |
| SimpleSpread σ = 2.5, common SNR | 0.38 | **0.38** | 0.22 |
| SimpleSpread σ = 2.5 + temporal 5 | 0.52 | **0.52** | n/a |

Stress tests (SimpleSpread σ = 2.5):
- **Per-agent noise levels** (not homoscedastic): the estimator returns the agent
  average (0.62 vs true 0.63). Results are the same as known σ.
- **Hidden team offset** (not locally deterministic): σ̂² is inflated by only 1–2%, and
  credit retention stays at β 0.98. With roles tied to agent identity, use
  `grouping_use_agent_id=True`.

### 3.5 Extensions of certified consensus

| extension | result | verdict |
|---|---|---|
| **certified temporal merging** (average over own neighbouring steps whose predictions are ε-close; certified on (r̃ₜ₊ₖ − r̃ₜ)² − 2σ²) | Pursuit late 0.52 → 0.80 (window 5); SimpleSpread 0.38 → 0.52 | **largest gain**; rejected automatically where rewards are spiky |
| **wider temporal window** | Pursuit late 0.80 → 0.88 → 0.92 (w = 5/9/13); SimpleSpread late 0.81 → 1.01 → 1.13; roles 1.10 → 1.42 (w = 5 → 9) | **use 9**; 13 shows the first credit loss (β 0.90) |
| **clean warm start** (`certify_warmup`, 2 iterations) | known σ: certified from iteration 1; Pursuit pooled 0.74 (w = 5) / 0.84 (w = 9) | best practical system if a few clean batches exist |
| inverse-variance weighting (per-agent noise levels) | 0.47 → 0.51; with temporal merging 0.68 | small gain, only with heteroscedastic noise |
| **replicate consensus**: average over identical transitions anywhere in the batch (zero bias, no certification needed) | removes 1–34% of noise variance early (Pursuit); with window 9, late 0.88 → 0.91 | small, free gain; no-op for continuous observations |
| **o′-only replicate key** (reward a function of o′) | ~2× the replicates early; σ-free warm start certified from iteration 6 instead of 10 | helps the warm-start case; slight bias (RMS 0.08) in Pursuit |
| certified shrinkage toward the consensus | Pursuit 0.16 → 0.22, but β 0.70 | spends the erasure budget |
| nested multi-granularity | no gain | finest level has the least evidence |
| prioritised proposal training | never certified | oversamples noise |
| reward model as a pseudo-agent | erased the shared signal in SimpleSpread | spends the budget on model bias |
| **residual rules** (`certify_residual`: merge on equal model *errors*) | Pursuit 0.74 → 0.53, β 0.86 → 0.79 | **negative**: models under-predict rare events (RMS error 1.5–3.8), so whole-step residual merges erase event credit |
| calibrating predictions (isotonic) | no gain | predictions are not systematically compressed |
| cross-batch replicate history | 0.41 vs 0.41 of samples with a replicate | transitions rarely recur across batches |
| per-agent σᵢ from within-agent replicates | ~10 dof per agent | infeasible |
| model-free proposals (near-identical transitions) | only exact replicates have zero difference | nothing beyond replicate consensus |
| k-NN consensus in transition space (continuous observations) | neighbours differ 10× more than within-step credit | smears credit |
| **certifier memory** (`certify_decay`) | 0.75 / 0.9 delay certification (it. 15 / 18 vs 13; late 0.80 → 0.29 / 0.22); 0.25 no gain | **keep 0.5** |
| cold-start tuning: 3 epochs, lr 3e-4 / 3e-3, width 512, auxiliary dynamics loss, EMA, replay | first certification 10–13 or never | **data-limited, not optimisation-limited** |

### 3.6 Bernoulli corruption (the paper's second kernel)

Each reward is replaced by ±10 with probability 0.1. This is not mean-preserving:
E[r̃] = 0.9·r. Pursuit, σ-free; within SNR pooled (late), β:

| none | oracle mean | oracle median | global median | certified mean | certified median |
|---|---|---|---|---|---|
| 0.14 (0.16), 0.90 | 0.41 (0.47), 0.90 | **0.86 (0.89)**, 1.01 | 0.00, **β 0** | 0.15 (0.26), 0.84 | 0.16 (0.29), 0.87 |

- **Certification stays valid.** The replicate σ² is unbiased, and there are no false
  certifications.
- **It is slow under heavy tails.** The pair statistic is ~3× noisier, so certification
  starts at iteration 14 in Pursuit and flickers in SimpleSpread.
- **Use `consensus=median`** inside certified groups (the default `consensus=auto` does
  this for Bernoulli).
- **Global median looks strong on the shared signal but erases all individual credit.**
  This is the caveat for Reviewer 3's Byzantine-style baseline.

---

## 4. What worked best

| situation | configuration | offline result (Pursuit σ = 3.5 within SNR, pooled / late) |
|---|---|---|
| **no clean data, σ unknown (default)** | `certified` + temporal window 9 + replicate consensus | 0.18 / **0.91**, β 0.91 (none 0.14 / 0.16) |
| a few clean iterations, σ known | `certified` + temporal window 9 + `certify_warmup=True warmup_iters=2`, known σ | **0.84** / 0.97 |
| a few clean iterations, σ unknown | same, σ-free, + `certify_replicate_key=next_obs` + replicate consensus | 0.25 / **1.04** |
| Bernoulli / outlier corruption | `certified` + `consensus=median` | 0.16 / 0.29 (none 0.14 / 0.16) |
| SimpleSpread σ = 2.5 (common SNR), no clean data | `certified` + temporal window 9 | 0.54 / **1.01** (none 0.20, oracle 0.42) |

Why these work:
- The reward model is trained only on noisy rewards (consistent).
- It is used out of sample (no selection bias).
- Its proposals are applied only when certified (safe).
- Spatial and temporal merges multiply the noise reduction where rewards persist.
- σ comes from replicated transitions.

Once certified, the spatio-temporal pipeline is well above same-step oracle grouping
(0.49 late with the paper's oracle).

---

## 5. How to use the code

### Install

```bash
pip install "torchrl~=0.7.0" "pettingzoo[mpe,sisl]==1.24.3" scikit-learn hydra-core
pip install -e BenchMARL
```

The default loggers are `[csv,wandb]`. Without `wandb` installed, pass
`experiment.loggers=[csv]`.

### Recommended run (no clean data, σ unknown)

```bash
cd BenchMARL
python benchmarl/run.py algorithm=mappo task=pettingzoo/pursuit \
    model=layers/cnn model@critic_model=layers/cnn \
    experiment.reward_perturbation=True experiment.reward_perturbation_stdev=3.5 \
    experiment.grouping_method=certified \
    experiment.grouping_use_next_obs=True experiment.grouping_epochs=1 \
    experiment.certify_temporal_window=9 experiment.certify_replicate_consensus=True \
    experiment.evaluate_grouping=True
```

Variants (append to the command above):

| goal | overrides |
|---|---|
| σ known (e.g. sensor spec) | `'experiment.certify_noise_std=${experiment.reward_perturbation_stdev}'` |
| use 2 clean warm-up iterations | `experiment.certify_warmup=True experiment.warmup_iters=2` |
| more replicates (reward a function of o′) | `experiment.certify_replicate_key=next_obs` |
| roles / teams tied to agent index | `experiment.grouping_use_agent_id=True` |
| Bernoulli corruption | `experiment.reward_perturbation_type=bernoulli experiment.reward_perturbation_flip_prob=0.1` (median consensus is chosen automatically) |
| SimpleSpread | `task=pettingzoo/simple_spread task.N=5` and drop the two `model=` arguments |
| tolerance | `experiment.grouping_epsilon=0.5`: in reward units; it is also the erasure budget |

Baselines: `experiment.grouping_method=` one of:
- `none` (No-Correction);
- `oracle`;
- `global`, or `global` with `experiment.consensus=median` (Byzantine-style);
- `reward_model_only` (Romoff-style);
- `warmup` (the paper's Warmup);
- `gmm` (the paper's No-Warmup, fixed);
- `reward_model` (the first-round reward model grouping).

### Certified-consensus options

| key | default | meaning |
|---|---|---|
| `grouping_epsilon` | 0.5 | tolerance ε (RMS bias bound when certified) |
| `certify_z` | 1.0 | z of the upper confidence bound |
| `certify_decay` | 0.5 | per-iteration decay of the certification statistics (keep) |
| `certify_noise_std` | null | known noise std; null estimates it |
| `certify_noise_estimator` | replicates | `replicates` (homoscedastic + local determinism) or `temporal` (per agent, with skill gate) |
| `certify_replicate_key` | transition | `transition` (o, a, o′) or `next_obs` (o′ only) |
| `certify_replicate_consensus` | False | average over identical transitions first (zero bias, from iteration 0) |
| `certify_temporal_window` | 0 | odd window for certified temporal merging; **9 recommended** |
| `certify_ivw` | False | inverse-variance weights (heteroscedastic noise) |
| `certify_warmup` | False | fit the proposal model on `warmup_iters` clean iterations first |
| `certify_residual` | False | residual rules (did not help; ablation only) |
| `certify_skill_gate` | True | skill gate for the `temporal` estimator |

### What gets logged

These are under `reward_robustness/grouper/*`; diagnostics need `evaluate_grouping=True`.
- `certified_rule`: index into (none, `pred`, `pred_in_id`, then the residual rules if
  enabled);
- `pred_delta` / `pred_margin`: estimate and z·SE of the certification statistic;
- `temporal_active`, `temporal_delta`, `temporal_margin`;
- `noise_var_replicates`, `noise_var_replicates_se`, `noise_replicate_dof`;
- `replicated_frac`, `identity_clusters`, `skill_z`;
- with diagnostics: `reward_robustness/event_contrast`, `pair_precision`,
  `pair_recall`, `ari`, `level*_bias`, and MSE before/after.

### Full paper sweep (cluster)

```bash
cd BenchMARL/examples/reward_robustness
SIGMAS="0.0,1.5,2.5,3.5" SEEDS="0,1,2,3,4" bash sweep.sh pursuit16 hydra/launcher=submitit_slurm
```

Settings: `pursuit16 | pursuit8 | spread5 | spread8`. The sweep covers:
- the baselines;
- the paper's instances, with and without o′;
- the reward-model groupers;
- certified with known σ, σ-free replicate and temporal estimators;
- certified extensions: window 9, replicate consensus, IVW, ± warm-up;
- the temporal-window ablation (0/5/9/13);
- the global-median baseline.

Each cell runs 5 independent training seeds.

### Offline evaluation (fast, no RL)

```bash
cd BenchMARL/examples/reward_robustness
python collect_stream.py --task pursuit --n_agents 16 --iters 20 --out /path/to/pursuit16
python offline_certified.py --stream /path/to/pursuit16 --task pursuit --sigma 3.5 --known 0 --temporal 9 --rep_consensus 1
```

`offline_certified.py` options:
- noise source: `--known 0|1`, `--estimator`, `--rep_key`, `--rep_consensus`;
- certified pipeline: `--temporal`, `--warm`, `--residual`, `--decay`, `--eps`, `--ivw`;
- noise model: `--noise bernoulli --flip`, `--hetero`;
- baselines and consensus: `--method none|oracle|global`, `--consensus mean|median`;
- roles test: `--teams 2 --team_offset 1`.

`offline_grouping_benchmark.py` runs the first-round comparison (§3.1).

### Tests

```bash
cd BenchMARL && python -m pytest test/test_reward_robustness.py -q
```

---

## 6. Limitations and what the paper still needs

1. **End-to-end evidence.** Run `sweep.sh` with ≥ 5 training seeds. Report CIs over
   seeds (rliable IQM or bootstrap) and tests against No-Correction. The offline gains
   are large, but no claim about returns can be made yet.
2. **Cold start.** Without clean data nothing is certified for the first ~10–13
   iterations; the method is then No-Correction. It is data-limited (§3.5). Report it
   as a cost, together with the 2-clean-iteration option.
3. **Assumptions behind σ-free certification.** Homoscedastic noise, and a reward that
   is a deterministic function of (o, a, o′). If the reward depends on something
   outside the transition, σ̂² is inflated and certification becomes optimistic.
4. **Heavy-tailed corruption** slows certification considerably (§3.6).
5. **ε is an erasure budget.** Set it relative to the reward scale. Very rare events
   (p·Δ² ≤ ε²) may be averaged away. This does happen at the very start of training:
   - captures are ~0.05% of samples, so merging everyone can legitimately certify;
   - one smoke run had capture contrast 0 at iteration 2;
   - the offline known-σ runs show the same blip at iteration 2.

   A relative tolerance or a minimum event-count rule would prevent it.
6. **Anytime-valid bounds.** Certification is re-tested every iteration with an
   adaptively chosen rule. A formal guarantee should use time-uniform confidence
   sequences instead of the decayed UCB.
7. **Not tested:**
   - critic-implied rewards as proposals (needs online runs);
   - symmetry-canonicalised replicates;
   - deconvolution with a discrete reward support;
   - a relative tolerance that protects rare events.

The revision plan, mapping every reviewer comment to a fix (baselines, seeds,
heterogeneous rewards, theory corrections, writing), is in §4 of
[`notes/paper_review_and_revision_plan.md`](notes/paper_review_and_revision_plan.md).
The LaTeX-ready theory additions are in [`notes/theory_additions.md`](notes/theory_additions.md):
- bias under noise-independent groupings;
- the selection-bias proposition;
- reward-model identifiability;
- median under Bernoulli corruption;
- residual consensus;
- σ-free certification (A7).
