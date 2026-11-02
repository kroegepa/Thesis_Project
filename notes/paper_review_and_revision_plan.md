# GR²-MARL: review, response to reviewers, and revision plan

This document covers four things: (1) my own review of the AAMAS'26 submission
"Consensus-Based Individual Reward Robustness in MARL", (2) what the code showed about
some of the reported results, (3) the warm-up-free grouping methods implemented in this
branch and how well they work, and (4) a prioritised revision plan that maps every
reviewer comment to a concrete fix.

---

## 1. Review

**Summary.** The paper formalises cooperative MARL where each agent's reward is
corrupted independently (RP-POSG). It proposes GR²-MARL: partition agents into groups
with (approximately) equal true rewards and replace each observed reward with a
consensus statistic (mean or median) over the group. The analysis covers bias under
oracle, dissimilar and mis-specified groupings, and variance under Gaussian noise.
Experiments use Pursuit and SimpleSpread with MAPPO/IPPO under oracle, warm-up and
no-warm-up groupings.

**Strengths**
- The problem (individual, not global, reward corruption) is relevant and well
  motivated, and the mechanism is simple and algorithm-agnostic.
- The oracle results clearly show the potential variance reduction, and the
  limitations of learning groups from noisy rewards are reported honestly.

**Weaknesses (in order of importance)**
1. **The practically relevant variant does not work.** The strong results need an oracle
   or clean rewards during warm-up. The only warm-up-free variant (No-Warmup) is at best
   on par with, and in Pursuit worse than, no correction. All three reviewers flag this.
   Section 2 shows that the failure is largely an implementation and design problem, not
   an inherent limitation.
2. **The theory assumes exogenous groupings, but the algorithm's groupings are not.**
   Prop. 1 and Lemma 1 condition on G. That is fine when G does not depend on the
   current noise (oracle, frozen warm-up model). The No-Warmup grouping, however, is
   computed from the same noisy rewards it then averages, so it groups agents with
   similar *noise* (selection bias). Reviewer 3 makes exactly this point.
3. **Thin empirical protocol.** There is one baseline, no significance tests, and an
   unclear number of independent runs. In the code, the "ten seeds" are 10 evaluation
   episodes of a single training run (`evaluation_episodes: 10`, one `seed`). The
   environments have near-symmetric rewards.
4. **Formal precision.** Several statements are imprecise or slightly wrong (details
   in §4.2), and there are unresolved `??` references.
5. **Positioning and novelty.** The consensus step is classical. The genuinely new parts
   are the problem formulation and how to obtain groupings without clean data. The
   second part is currently the weakest and should become the core contribution.

**Recommendation as it stands:** weak reject. The idea is reasonable, but the realistic
setting is unsolved and the evidence is thin. With the changes below (a working
warm-up-free grouper with a noise-independence argument, more seeds and baselines, a
heterogeneous-reward environment, and cleaned-up theory), it could be a solid accept.

---

## 2. What the code revealed (affects reported results)

| # | Issue | Effect on the paper |
|---|-------|---------------------|
| 1 | **GMM label switching.** `fuzzy_grouping` fitted a new GMM per batch, and sklearn's component order is arbitrary, so label "2" meant different reward levels in different iterations. The No-Warmup classifier was trained cumulatively on these inconsistent targets. | No-Warmup groupings were partly random. Fixed: components are sorted by mean (`grouping_gmm_sort`). |
| 2 | **In-sample grouping.** No-Warmup trained the classifier for 35 epochs on the *current* batch, then predicted groups for that same batch. That memorises the noise, so groups follow the noisy rewards (selection bias, weakness 2). | Biased consensus rewards. This is the likely main cause of No-Warmup being worse than no correction. Fixed: `grouping_out_of_sample: True` groups each batch with parameters fitted on earlier batches only. |
| 3 | **Fig. 3 metric.** Weighted F1 compared predicted cluster *ids* with oracle ids. For clustering-based groupers the ids are arbitrary, so F1 can be ~0 even for a perfect partition. The spiky 0–0.6 pattern of the pink curve looks like this artefact. | Fig. 3 should be recomputed with permutation-invariant, per-timestep metrics (pairwise precision/recall, ARI). Now logged under `reward_robustness/*` with `evaluate_grouping=True`. |
| 4 | **Bernoulli magnitude.** The paper says ±10. The last commit changed the code to ±3 (it was ±10 in commit `9b91dec`). | Make sure Fig. 4 used ±10. Now a config key, `reward_perturbation_outlier_value` (default 10). |
| 5 | **Alive mask unused** (`goup_mask` typo). Dead agents were included in group means. | No effect on Pursuit/SimpleSpread (no deaths). Affects Battle. Fixed. |
| 6 | **Warm-up budget.** The paper says T_w = 60 000 frames, but the config uses 50 iterations × 3 000 frames = 150 000 frames (the commit "longer warmup" changed 10 → 50). | State the budget actually used. Report it as a cost (Reviewer 1). |
| 7 | **"Seeds".** Evaluation logs `episode_reward_seed_{i}` for 10 evaluation *episodes* of one run. | Reviewer 2's concern is justified. Use ≥5 training seeds and report what the shaded areas are. |
| 8 | **Noisy training-episode returns** were recomputed with the root `done` flag, which is never true after auto-reset, so cumulative sums never reset. | Training-curve logs only (evaluation was correct). Now the true episode return is logged. |
| 9 | **Pursuit frames per batch.** The paper says 6 000 frames per iteration, the config says 3 000. | Commit the exact launch configs used for the paper. |

---

## 3. Warm-up-free grouping: what is implemented and why

### 3.1 Key idea: learn a reward model from the noisy rewards, then group on its predictions

Under Assumption 2 (zero-mean noise, independent of the transition):

    argmin_f E[(f(o,a,o') - r~)^2]  =  E[r~ | o,a,o']  =  E[r | o,a,o'].

So **a reward regressor trained on corrupted rewards is consistent**: the noise only
adds variance to the regression targets, not bias. This is the standard result for
regression with noisy labels, and it is the identifiability argument Reviewer 3 asks
for. GMM clustering of noisy rewards has no such guarantee.

`grouping_method=reward_model`:

1. Fit `f(o_i, a_i[, o'_i]) -> r~_i` online with MSE. For non-zero-mean kernels such as
   Bernoulli, use L1: the conditional median still recovers the true reward when the
   flip probability is below 0.5.
2. **Out-of-sample:** the groups of iteration k use f fitted on iterations < k. The
   grouping is then independent of the current noise, so Prop. 1 and Lemma 1 hold as
   stated (conditioning on G is legitimate).
3. Group agents whose *predicted* rewards are similar:
   - `kmeans`: K clusters over all predictions, same K as the paper;
   - `epsilon`: per-timestep ε-gap clustering, a direct implementation of Assumption 1;
   - `kernel`: soft weights exp(−(f_i − f_j)²/2ε²).
4. Consensus is still taken over the **observed** rewards. The model only needs to rank
   rewards into levels, not predict them exactly, so model error enters only through
   misgrouping (Prop. 3), not as direct bias.

Optional **residual consensus** (`reward_model_residual=True`):
`r^_i = f_i + C({r~_j − f_j : j ∈ G})`. This only needs the *residuals* to be
ε-similar, which relaxes Assumption 1. With a perfect model, any grouping (even
global) is unbiased. That gives a principled answer to Reviewer 3's point about
heterogeneous, role-differentiated rewards.

**Baseline the reviewers asked for:** `reward_model_only` replaces rewards by f. This is
a reward-estimation baseline in the spirit of Romoff et al. (2018, ref. 23), using the
same network.

### 3.2 Offline grouping benchmark

`examples/reward_robustness/collect_stream.py` records the batches of a MAPPO run on
clean Pursuit rewards (16 pursuers). `offline_grouping_benchmark.py` then replays that
stream under noise through every grouper, in order and exactly as during training.

Instantaneous MSE is a misleading criterion: in Pursuit, ~99% of rewards are about
−0.1. Global grouping, or predicting −0.1 everywhere, gets a low MSE while erasing the
capture reward that drives learning. The benchmark therefore reports:
- **signal**: the fraction of the capture-vs-rest reward gap that survives correction
  (1 = unbiased, 0 = erased);
- **noise**: the residual noise std;
- **SNR** = signal × gap / noise;
- pairwise grouping **precision/recall** against the oracle partition.

**Setup.**
- Pursuit, 16 pursuers / 60 evaders (the paper's large setting).
- A 20-iteration stream of MAPPO batches collected on clean rewards (capture rate rises
  from 0.05% to ~2.5% of agent-steps).
- Identical noise for every method. Learned groupers are updated on every batch;
  metrics are computed on the odd batches. The warm-up classifier is trained on the
  clean even batches (10 batches).
- Budget deviations: classifiers use 10 epochs (not 35) and reward models 5 epochs.
- Full table: `notes/offline_benchmark_results.md`. Selected rows, SNR (higher is
  better), with within-step capture contrast in parentheses:

| grouping | needs clean data | σ=1.5 | σ=2.5 | σ=3.5 |
|---|---|---|---|---|
| none (No-Correction) | – | 3.35 (1.00) | 2.02 (1.00) | 1.45 (1.00) |
| oracle | always | 10.14 (1.00) | 6.11 (1.00) | 4.38 (1.00) |
| global (Fig. 5 ablation) | no | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| GMM on current noisy rewards (No-Warmup limit) | no | 3.46 (0.92) | 1.85 (0.85) | 1.31 (0.85) |
| GMM classifier, original No-Warmup (unsorted, in-sample) | no | – | – | 0.80 (0.20) |
| GMM classifier, sorted labels + out-of-sample | no | – | – | 1.28 (0.26) |
| warmup, input (o, a) (paper) | warm-up | – | 3.73 (0.64) | 2.72 (0.65) |
| warmup, input (o, a, o') | warm-up | – | – | **3.95** (0.92) |
| reward model, (o, a), k-means | no | – | – | 1.58 (0.40) |
| reward model, (o, a, o'), k-means K=3 | no | 8.07 (0.79) | 3.68 (0.64) | 2.37 (0.59) |
| reward model, (o, a, o'), ε-gap ε=0.5 | no | 9.49 (0.79) | 4.82 (0.71) | 3.07 (0.66) |
| **reward model, (o, a, o'), kernel ε=0.5** | **no** | **9.84** (0.80) | **5.12** (0.71) | **3.32** (0.66) |
| reward model, (o, a, o'), global + residual | no | 8.10 (0.73) | 4.35 (0.62) | 3.00 (0.58) |
| reward-model-only baseline (Romoff-style) | no | 15.46 (0.73) | 9.11 (0.62) | 7.00 (0.58) |

**Findings.**
1. **The No-Warmup failure is reproduced and explained.** Grouping by GMM clusters of
   the current noisy rewards barely reduces the noise (std 3.23 vs 3.50 at σ = 3.5), and
   its SNR is *below* no correction. The consensus only quantises the noise (Prop. 4 in
   `theory_additions.md`).
2. **MSE and correlation are the wrong yardsticks.** Global grouping has lower MSE and
   higher correlation than no correction, but within-step contrast 0: the capture reward
   is shared by every agent of the step, so credit assignment is destroyed. That is
   exactly why it lost to No-Correction in Fig. 5. Report contrast / SNR (now logged as
   `reward_robustness/event_contrast`).
3. **A reward model fitted only on noisy rewards gives useful groups.** With per-step
   kernel or ε grouping on its predictions, it reaches SNR 3.3 at σ = 3.5. That is 2.3×
   No-Correction and 76% of the oracle, and above the paper's Warmup classifier
   (2.7), without clean data. At σ = 1.5 it reaches 97% of the oracle.
4. **o' is important for every learned grouper.** Adding o' raises Warmup from 2.72 to
   3.95 and the reward model (k-means) from 1.58 to 2.37. With equal inputs, the
   warm-up-free reward model reaches ~84% of Warmup (3.32 vs 3.95). The remaining gap is
   in capture contrast (0.66 vs 0.92): noisy targets make captures slower to learn. The
   stream is only 20 iterations; full training is 250, so this gap should shrink.
5. **How predictions become groups matters.** Global k-means with K = 3 is the weakest:
   its middle cluster mixes capturers and non-capturers. Per-timestep ε/kernel grouping
   is better. ε must be small relative to the action-relevant reward gap: ε = 2 loses
   much of the contrast.
6. **In-sample vs out-of-sample.** At 5 epochs, in-sample predictions are slightly better
   (fresher model). Selection bias is catastrophic only when groups follow the noise
   itself (point 1). Keep out-of-sample as the default, since it gives the exact
   conditional guarantee, but report this trade-off honestly.
7. **The reward-model-only baseline looks strong on SNR.** Its error is *systematic*
   model error (it never averages out), whereas the consensus error is zero-mean noise,
   so SNR flatters it. It must be in the end-to-end comparison: it is exactly the
   baseline Reviewer 2 asked for.
8. **The No-Warmup classifier stays worse than no correction even after the fixes.**
   Sorting the GMM labels and grouping out-of-sample raises its SNR from 0.80 to 1.28,
   still below 1.45 for No-Correction. Clustering *noisy* rewards gives labels that are
   mostly noise; *regressing* them (the reward model) is what makes learned grouping
   work.
9. **Train the reward model for 1 epoch per iteration, not 5 or 35.** With 1 epoch the
   grouping quality is unchanged (SNR 3.31 vs 3.32 at σ = 3.5) at a fifth of the cost.
   With 5 epochs the model overfits the noisy targets: it predicts *worse* than a
   constant out of sample (skill z between −5 and −13), even though its ranking is
   still useful.
10. **The skill gate behaves as intended with 1 epoch.** It stays shut while the model
    cannot predict (z ≈ −4 to 2.5) and opens once it can (z ≈ 9–20 from about the 11th
    stream iteration). Capture contrast is 0.85 when pooled over the stream. Pooled SNR
    is lower (1.8) only because the 20-iteration stream contains the closed-gate phase,
    which falls back to No-Correction.

**End-to-end pilots (indicative only).**
- Setting: 1 seed, Pursuit with 8 pursuers, σ = 2.5, 40 iterations of 3 000 frames.
- Reduced optimisation budget (lr 3e-4, 8 epochs × minibatch 400) to fit a CPU-only
  machine.
- Metric: mean true training return per 5-iteration window.

PILOT_TABLE

- Oracle ≫ No-Correction, consistent with the paper.
- The ungated reward-model groupers did not beat No-Correction end-to-end. The kernel
  variant was clearly worse. The diagnostics show why: in noisy training, captures are
  rare at first and the model has not learned them, so it predicts ~−0.1 for everyone.
  Kernel/ε grouping then merges everyone, i.e. becomes global grouping. Capture
  contrast was 0.0–0.1 for the first ~13 iterations, which erases capture credit
  exactly when the policy needs it. K-means (forced K = 3) erased less (contrast
  ~0.2–0.3) and ended level with No-Correction.
- The bootstrap-ensemble gate (`reward_model_heads=5`) did **not** catch this: the
  heads agree on "−0.1 everywhere", so the confident fraction was 1.0.
- The out-of-sample **skill gate** (`reward_model_min_skill_z=3`, 1 epoch per
  iteration) is the fix that the offline data supports. Its end-to-end pilots are in
  the table above.
- **Before claiming end-to-end gains, run the multi-seed sweep**
  (`examples/reward_robustness/sweep.sh`) on the cluster with the paper's budget.


---

## 4. Revision plan (mapped to reviewer comments)

### 4.1 Method and experiments (highest impact)

1. **Replace "No-Warmup" with the reward-model grouper** (R1, R2, R3). Keep the GMM
   variant as an ablation that shows why noise-dependent grouping fails (Prop. 4), and
   add the in-sample vs out-of-sample ablation. Make this the headline contribution
   only if the multi-seed sweep shows end-to-end gains over No-Correction. Offline it
   recovers 76–97% of the oracle's SNR without clean data, but the single-seed pilots
   were not yet better than No-Correction (§3.2). If it does not win end-to-end, the
   honest framing is "noise-independent grouping is necessary, and here is how far
   learned groupings get".
2. **Feed o' to every learned grouper** (`grouping_use_next_obs=True`). Rewards are
   R(s, a, s'); with o', Warmup's SNR rises from 2.72 to 3.95 at σ = 3.5. This cheap
   change should strengthen the existing Warmup results.
3. **Add baselines** (R2, R3):
   - reward-estimation baseline (`reward_model_only`, Romoff et al.);
   - global consensus with a robust statistic, i.e. the "Byzantine" aggregation R3
     mentions (`grouping_method=global consensus=median`);
   - the existing No-Correction and global-mean baselines.
4. **Independent seeds and statistics** (R2, R3): ≥5 training seeds per cell. Report
   mean ± 95% bootstrap CI, or IQM with stratified bootstrap (rliable), on final return
   and AUC. Add Welch tests (or bootstrap CIs of the difference) against No-Correction.
   State clearly that shaded regions are CIs over training seeds.
5. **Heterogeneous / role-differentiated rewards** (R3). Battle and Adversarial Pursuit
   are already supported by the code (`TASK_SPECS`). A cheap controlled option is
   Pursuit with agent-specific capture rewards, with the agent id added to the reward
   model input; residual consensus is designed for this case.
6. **Quantify degradation and cost** (R1):
   - a table of final return vs σ for each grouping method;
   - a scatter of final return vs grouping quality (signal / pairwise precision from
     the diagnostics), which directly validates Prop. 3;
   - the warm-up cost in frames and the grouping wall-clock overhead
     (`timers/grouping_time`).
7. **Fig. 3:** recompute with pairwise precision/recall or ARI per timestep (not F1 of
   cluster ids).

### 4.2 Theory (R1, R3)

- **Prop. 1:**
  - State it for the mean (that is what is proved). Give the median a separate result.
  - Add the assumption that G is independent of the current noise realisation given
    the transition. Explain that oracle, frozen warm-up and out-of-sample reward-model
    groupings satisfy it, and that noise-dependent groupings (GMM on r~) do not.
  - Fix the proof: the ε comes from Assumption 1, not Assumption 2 (which gives
    equality).
  - Retitle it: the result is a bias bound, not unbiasedness.
- **Selection-bias example** (new, short): if groups are formed by thresholding noisy
  rewards, E[r^_i | G] ≠ r_i even when all true rewards are equal. This formalises why
  No-Warmup fails.
- **Reward-model identifiability** (new, short): E[r~ | o,a,o'] = E[r | o,a,o'] under
  Assumption 2, so regressing on corrupted rewards is consistent. With L1 loss, the
  conditional median is recovered for Bernoulli corruption with p < 1/2.
- **Median under Bernoulli corruption** (new, R3): for a group of identical true
  rewards, the median is exact unless at least half the group is corrupted.
  P(failure) = P(Bin(|G|, p) ≥ |G|/2), which decays exponentially in |G| (Hoeffding).
  This also explains why the Bernoulli kernel does not need to be mean-preserving.
- **Prop. 2:** it is just the identity E[r^_i] − r_i = (1/|G|) Σ_j (r_j − r_i) for the
  mean. Make it a one-line lemma with proof, and drop the "impossibility" reading (R1).
- **Prop. 3:** restate as Reviewer 1 suggests: split G into ε-similar (C_t) and
  misgrouped (W_t) agents; the bias is then bounded by a weighted combination. Mention
  that it holds for the mean with an exogenous G.
- **Bias–variance trade-off for group size** (R3). For the mean,
  MSE_i(G) = ((1/|G|) Σ_j (r_j − r_i))² + σ²/|G|. Merging two agents whose rewards
  differ by δ lowers the per-sample MSE iff δ < √2σ. However, bias does not average
  out over samples while variance does, so the useful tolerance is set by the reward
  scale, not by σ. This guides the choice of ε / K.
- **Residual consensus** (new): restate Assumption 1 and Prop. 3 for residuals r − f.
  Setting f = 0 recovers the current results.
- **Definitions:**
  - Def. 3: the signature `F: O×A×R → 2^I\{∅}` should map a joint transition to a
    *partition* of I (R1).
  - Remove the leaked `def:oracle` label.
  - Assumption 1 should read "for every agent i there is a group G ∋ i" (R1).
  - Assumption 2 uses a_t^i but R^i depends on the joint action a_t.
  - The Gaussian example in Def. 2 is missing the mean (should be N(r_t, Σ)) and
    conditions on (s, a, s') instead of r_t.
  - Lemma 1's proof uses an undefined m; it should be |G|.
- **Consensus computation / communication** (R1): state that the correction runs in
  the centralised trainer (CTDE). Agents do not communicate at execution, and group
  membership is decided by the trainer. Discuss faulty or adversarial reward reports
  as the median / trimmed-mean case.
- **State the assumptions up front** (R1): independent per-agent noise from a known
  family, zero-mean (or a matching robust C), centralised training access to all
  rewards, and ε-similar subgroups.

### 4.3 Related work and positioning (R3)

- Soften "first study". Discuss reward poisoning in (offline) MARL (Wu et al., 2022,
  arXiv:2206.01888), distributional reward estimation in MARL (Hu et al., NeurIPS
  2022), and Romoff et al. (2018). Also cover Byzantine-robust aggregation and robust
  statistics, and learning with noisy labels (for the reward-model argument).
- Clarify what is new relative to these: per-agent corruption in *cooperative* MARL;
  grouping by learned reward similarity (rather than global aggregation or trust
  weights); and noise-independent grouping without clean data.

### 4.4 Writing / presentation (R1, R2)

- Fix the `??` references (p. 9 "Oracle grouping from ??", p. 14 "(??)") and the broken
  sentence "or [18, 22]" (p. 3). Define γ before use and define "MER".
- Typos: "similary", "which a correction" → "with a correction", Corollary 2 title
  ("Variance under independent heteroscedastic Gaussian noise"), and "assign" (p. 6).
- Call the configurations *instances* of the meta-algorithm, not variants (R1).
- Make the orchard example concrete (R1): who is grouped, what is ε, what is K.
- Fig. 5: plot the global-grouping ablation on the same axes as No-Correction.
- Release the code with the exact hydra launch configs. The refactor in this branch is
  a good basis (see `BenchMARL/benchmarl/reward_robustness/README.md`).
