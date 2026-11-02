# Theory additions for the revision (LaTeX-ready)

Notation follows the paper. At step t, the true joint reward is r_t ∈ ℝⁿ and the observed
reward is r̃_t. In the Gaussian case r̃_t = r_t + ξ_t. Write x_t = (s_t, a_t, s_{t+1}) for
the transition and τ_tⁱ = (o_tⁱ, a_tⁱ, o_{t+1}ⁱ) for agent i's local part of it. Groupings
are G_t = F(·), and C is the consensus function.

The statements below replace or extend Prop. 1–3. Items marked *(new)* are not in the
submission.

---

## A0. Restated assumptions

**Assumption 1′ (ε-similar groups).** For every agent i and timestep t there is a group
G ∋ i, |G| ≥ 2, with |r_tʲ − r_tⁱ| ≤ ε for all j ∈ G.
*(Fixes the quantifier order noted by Reviewer 1.)*

**Assumption 2′ (unbiased, transition-independent noise).**
E[r̃_tʲ | x_t] = r_tʲ, and the noise ξ_t = r̃_t − r_t is independent across timesteps.
*(Use the joint action a_t, not a_tⁱ, because Rⁱ depends on the joint action.)*

**Assumption 3 *(new)* (noise-independent grouping).** G_t is a function of x_t (or of
o_{t+1}, a_t, o_t) and of data from earlier iterations only. In particular
E[ξ_tʲ | G_t, x_t] = 0 for all j.

Assumption 3 holds for:
- the oracle (a function of r_t);
- a grouping model trained on a clean warm-up and then frozen;
- any grouping model used **out of sample**, i.e. fitted on iterations < k and applied
  to iteration k (`grouping_out_of_sample=True`).

It fails for groupings computed from the current noisy rewards: GMM on r̃_t, or a
classifier fitted on the current batch and then applied to it.

---

## A1. Bias of mean consensus (replaces Prop. 1 and Prop. 2)

**Proposition 1′.** Let C be the mean. Under Assumptions 2′ and 3, for i ∈ G:

$$
\mathbb E[\hat r^i_t \mid G, x_t] - r^i_t \;=\; \frac{1}{|G|}\sum_{j\in G}\big(r^j_t - r^i_t\big).
$$

If G is ε-similar for i (Assumption 1′), then
$|\mathbb E[\hat r^i_t\mid G,x_t]-r^i_t| \le \frac{|G|-1}{|G|}\,\varepsilon \le \varepsilon$.

*Proof.* $\hat r^i_t = \frac{1}{|G|}\sum_{j\in G}(r^j_t+\xi^j_t)$. Conditioning on (G, x_t)
fixes r_t. Assumption 3 gives $\mathbb E[\xi^j_t\mid G,x_t]=0$. Rearranging gives the
identity. The bound follows from $|r^j_t-r^i_t|\le\varepsilon$ for j ≠ i, and 0 for j = i. ∎

The identity is the current Prop. 2; the bound is the current Prop. 1. In the current
proof, the ε step is attributed to Assumption 2, but it comes from Assumption 1.
Assumption 3 is the missing condition that makes "conditioning on G" legitimate.

**Proposition 3′ (misgrouping).** With C_t the ε-similar members of G, W_t = G \ C_t,
η_t = |W_t|/|G| and Γ_t = max_{j∈W_t} |r_tʲ − r_tⁱ|:
$|\text{bias}| \le \eta_t\Gamma_t + (1-\eta_t)\varepsilon$.
This follows directly from Proposition 1′ by splitting the sum over C_t and W_t
(Reviewer 1's reading).

---

## A2. Why grouping on the current noisy rewards fails *(new)*

**Proposition 4 (selection bias).** Let all agents have the same true reward r
(the most favourable case for consensus). Suppose agents are grouped by thresholding
their *noisy* rewards at r, i.e. G_high = {j : r̃ʲ > r} and G_low = {j : r̃ʲ ≤ r}, with
Gaussian noise N(0, σ²). As n → ∞:

$$
\hat r^i - r \;\to\; \sigma\sqrt{2/\pi}\;\operatorname{sign}(\xi^i).
$$

So the "corrected" reward of agent i is biased by ±0.80σ, and the sign is agent i's own
noise. The consensus does not remove the noise; it only quantises it.

*Proof.* Given i ∈ G_high, the other members are i.i.d. draws of ξ conditioned on
ξ > 0. Their mean converges to E[ξ | ξ > 0] = σ√(2/π). The same holds symmetrically
for G_low. ∎

A GMM fitted on r̃ behaves the same way once σ exceeds the gaps between reward levels
(e.g. σ ≥ 1.5 vs a 0.01 tag reward in Pursuit). This is the formal reason No-Warmup
underperforms No-Correction, and it answers Reviewer 3's point that "conditioning on
data-dependent groupings introduces selection effects".

The offline benchmark makes this measurable: compare `rm_kmeans_next_insample` with
`rm_kmeans_next`, and `gmm_paper` with `gmm_fixed`.

---

## A3. Groupings can be learned from corrupted rewards *(new: identifiability)*

**Proposition 5.** Let f be any function of the local transition τⁱ.

(a) Under Assumption 2′,
$\arg\min_f \mathbb E[(f(\tau^i) - \tilde r^i)^2] = \mathbb E[\tilde r^i\mid\tau^i] = \mathbb E[r^i\mid\tau^i]$.

(b) For the Bernoulli kernel (r̃ = r w.p. 1 − p, ±M w.p. p/2 each, |r| < M), if r is
determined by τ and p < 1/2, then
$\arg\min_f \mathbb E|f(\tau^i)-\tilde r^i| = \operatorname{median}(\tilde r^i\mid\tau^i) = r^i$.

*Proof.* (a) The MSE minimiser is the conditional mean, and E[ξ | τ] = 0.
(b) The L1 minimiser is the conditional median, and
P(r̃ ≤ r) = P(r̃ ≥ r) = 1 − p/2 ≥ 1/2. ∎

So a reward model trained on corrupted rewards targets the true expected reward,
with no clean data needed. This is the identifiability argument Reviewer 3 asks for.
Grouping agents by f(τⁱ) and using f out of sample satisfies Assumption 3. The
consensus then averages the *observed* rewards, so model error enters only through
misgrouping (Prop. 3′), not directly. Estimation error decays at the usual
regression rate, roughly σ²/N_eff; the noise slows learning but does not bias it.

*Practical caveats (from the offline benchmark and pilots).* Identifiability is
asymptotic.
- **Before the model has learned the rare reward events**, it predicts nearly the same
  value for everyone. Similarity grouping on its predictions then degenerates into
  global grouping, which erases exactly the rare-event credit (within-step contrast
  ≈ 0). The code therefore only groups once f beats a constant predictor out of sample
  (paired z-test on the previous batch). This is also a clean condition to state in
  the paper: "group only when f is significantly informative".
- **Noisy targets are easy to overfit.** With many epochs per batch, f is worse than a
  constant out of sample, even though its ranking is still useful. One epoch per
  iteration was enough.

---

## A4. Median consensus under Bernoulli corruption *(new)*

**Proposition 6.** Let G be ε-similar, with each member corrupted independently with
probability p to ±M. Let K ~ Bin(|G|, p) be the number of corrupted members. If
K ≤ ⌊(|G| − 1)/2⌋, the (lower) group median lies in [min_{j uncorrupted} rʲ,
max_{j uncorrupted} rʲ], so |r̂ⁱ − rⁱ| ≤ ε. Hence

$$
\Pr\big(|\hat r^i - r^i| > \varepsilon\big) \le \Pr\big(K \ge |G|/2\big) \le \exp\!\big(-2|G|(\tfrac12 - p)^2\big)
$$

for p < 1/2 (Hoeffding). Note that K > ⌊(|G|−1)/2⌋ implies K ≥ |G|/2.

*Proof sketch.* All uncorrupted rewards lie in [rⁱ − ε, rⁱ + ε]. With K⁻ outliers below
and K⁺ above, K⁻ + K⁺ = K ≤ m = ⌊(|G|−1)/2⌋, the sorted element at index m (the lower
median) has at most m elements below it and at most |G|−1−m ≥ m above it. So it cannot
be an outlier outside that interval. ∎ The failure probability decays exponentially in the group
size. By contrast, the mean is biased by −p·r (the kernel is not mean-preserving) and
its variance grows with M².

This covers the median and the non-zero-mean Bernoulli kernel, which the current
analysis does not (Reviewer 3).

---

## A5. Residual consensus relaxes ε-similarity *(new)*

**Proposition 7.** Let f be a reward model satisfying Assumption 3 (fitted out of
sample), and define $\hat r^i = f^i + \frac{1}{|G|}\sum_{j\in G}(\tilde r^j - f^j)$.
With residuals eʲ = rʲ − fʲ:

$$
\mathbb E[\hat r^i\mid G,x] - r^i = \frac{1}{|G|}\sum_{j\in G}(e^j - e^i),
\qquad \operatorname{Var}(\hat r^i\mid G, x) = \frac{1}{|G|^2}\sum_{j,k\in G}\Sigma_{jk}.
$$

So only the **residuals** need to be ε-similar.
- f ≡ 0 recovers Propositions 1′ and 3′.
- A perfect model makes any grouping unbiased, including G = I with variance σ²/n.

This addresses heterogeneous, role-differentiated rewards (Reviewer 3): the model
absorbs the systematic differences, and the consensus removes the noise.

---

## A6. Choosing the group size / ε *(new; Reviewer 3)*

For mean consensus with independent homoscedastic noise:

$$
\mathrm{MSE}_i(G) = \Big(\tfrac{1}{|G|}\textstyle\sum_{j\in G}(r^j-r^i)\Big)^2 + \sigma^2/|G| .
$$

Adding agent j with reward gap δ to i's singleton lowers the per-sample MSE iff
δ < √2·σ. However, policy-gradient learning averages over many samples. Variance
averages out, but bias that is correlated with the action does not. The effective
tolerance should therefore be set relative to the **reward differences that matter for
the policy** (here, the capture reward), not relative to σ.

In practice:
1. Prefer groupings whose within-group *predicted* reward spread is small relative to
   the action-relevant reward gaps (k-means over predictions, or ε-gap with ε a fraction
   of that gap).
2. Monitor the diagnostics `pair_precision`, `event_contrast` and `level*_bias`.
