#  Unit tests for benchmarl.reward_robustness (no environments needed).

import pytest
import torch

from benchmarl.reward_robustness.clustering import (
    epsilon_labels,
    gmm_labels,
    kmeans_1d_labels,
    threshold_labels,
)
from benchmarl.reward_robustness.consensus import (
    apply_consensus,
    Assignment,
    group_mean,
    group_median,
    group_trimmed_mean,
)
from benchmarl.reward_robustness.groupers import (
    GroupData,
    GroupingConfig,
    make_grouper,
    oracle_labels,
)
from benchmarl.reward_robustness.metrics import (
    adjusted_rand_index,
    event_contrast_sums,
    pairwise_grouping_scores,
    reward_error_metrics,
)
from benchmarl.reward_robustness.perturbation import perturb_rewards
from benchmarl.reward_robustness.task_spec import get_task_spec


class TestConsensus:
    def test_group_mean(self):
        r = torch.tensor([[1.0, 3.0, 10.0, 20.0, 7.0]])
        labels = torch.tensor([[0, 0, 1, 1, -1]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        out = group_mean(r, labels, mask)
        assert torch.allclose(out, torch.tensor([[2.0, 2.0, 15.0, 15.0, 7.0]]))

    def test_mask_excludes_dead_agents(self):
        # Agent 1 is dead: it must neither contribute nor be modified.
        r = torch.tensor([[1.0, 100.0, 3.0]])
        labels = torch.tensor([[0, 0, 0]])
        mask = torch.tensor([[True, False, True]])
        out = group_mean(r, labels, mask)
        assert torch.allclose(out, torch.tensor([[2.0, 100.0, 2.0]]))

    def test_groups_are_per_timestep(self):
        r = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
        labels = torch.zeros(2, 2, dtype=torch.long)
        out = group_mean(r, labels, torch.ones(2, 2, dtype=torch.bool))
        assert torch.allclose(out, torch.tensor([[2.0, 2.0], [7.0, 7.0]]))

    def test_median_is_robust_to_outlier(self):
        r = torch.tensor([[1.0, 1.2, 0.9, 10.0, -3.0]])
        labels = torch.tensor([[0, 0, 0, 0, 1]])
        out = group_median(r, labels, torch.ones_like(labels, dtype=torch.bool))
        assert torch.allclose(out[0, :4], torch.full((4,), 1.0))
        assert out[0, 4] == -3.0

    def test_trimmed_mean(self):
        r = torch.tensor([[0.0, 1.0, 2.0, 3.0, 100.0]])
        labels = torch.zeros(1, 5, dtype=torch.long)
        out = group_trimmed_mean(r, labels, torch.ones(1, 5, dtype=torch.bool), trim=0.2)
        assert torch.allclose(out, torch.full((1, 5), 2.0))

    def test_soft_weights_and_offset(self):
        r = torch.tensor([[1.0, 3.0, 50.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        w = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        out = apply_consensus(r, Assignment(weights=w), mask)
        assert torch.allclose(out, torch.tensor([[2.0, 2.0, 50.0]]))
        # Residual consensus with a perfect baseline returns baseline + mean residual.
        offset = torch.tensor([[0.0, 2.0, 49.0]])
        labels = torch.zeros(1, 3, dtype=torch.long)
        out = apply_consensus(r, Assignment(labels=labels, offset=offset), mask)
        assert torch.allclose(out, offset + 1.0)

    def test_replace_and_noop(self):
        r = torch.tensor([[1.0, 2.0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        assert torch.equal(apply_consensus(r, Assignment(), mask), r)
        rep = torch.tensor([[5.0, 6.0]])
        assert torch.equal(apply_consensus(r, Assignment(replace=rep), mask), rep)

    def test_inactive_agents_are_left_alone(self):
        r = torch.tensor([[1.0, 3.0, 100.0]])
        labels = torch.zeros(1, 3, dtype=torch.long)
        active = torch.tensor([[True, True, False]])
        out = apply_consensus(r, Assignment(labels=labels, active=active))
        assert torch.allclose(out, torch.tensor([[2.0, 2.0, 100.0]]))
        out = apply_consensus(r, Assignment(replace=torch.zeros(1, 3), active=active))
        assert torch.allclose(out, torch.tensor([[0.0, 0.0, 100.0]]))

    def test_oracle_mean_is_unbiased_and_reduces_variance(self):
        torch.manual_seed(0)
        n, a, sigma = 20000, 8, 2.0
        true = torch.zeros(n, a)
        true[:, :2] = 5.0  # two agents get a large reward, six get nothing
        noisy = true + sigma * torch.randn(n, a)
        labels = (true > 1).long()
        out = group_mean(noisy, labels, torch.ones(n, a, dtype=torch.bool))
        err = out - true
        assert err.mean().abs() < 0.02  # Proposition 1
        assert torch.allclose(err[:, :2].var(), torch.tensor(sigma**2 / 2), rtol=0.05)  # Cor. 1
        assert torch.allclose(err[:, 2:].var(), torch.tensor(sigma**2 / 6), rtol=0.05)


class TestClustering:
    def test_threshold_semantics(self):
        r = torch.tensor([-0.1, -0.099, 4.9, 2.0])
        # Pursuit: r > -0.1 -> level >= 1, r > 2 -> level 2
        assert threshold_labels(r, (-0.1, 2.0), right=False).tolist() == [0, 1, 2, 1]
        # >= semantics
        assert threshold_labels(torch.tensor([0.0, 0.8]), (0.0, 0.8), right=True).tolist() == [1, 2]

    def test_gmm_sorted_labels(self):
        torch.manual_seed(0)
        x = torch.cat([torch.randn(300) * 0.1 + 5, torch.randn(300) * 0.1 - 5, torch.randn(300) * 0.1])
        labels = gmm_labels(x, 3, sort_components=True)
        assert labels[:300].unique().tolist() == [2]
        assert labels[300:600].unique().tolist() == [0]
        assert labels[600:].unique().tolist() == [1]

    def test_kmeans_keeps_rare_cluster(self):
        x = torch.cat([torch.full((990,), -0.1), torch.full((10,), 4.9)])
        labels = kmeans_1d_labels(x, 3)
        assert labels[:990].unique().numel() == 1
        assert labels[990:].unique().tolist() == [int(labels.max())]

    def test_epsilon_labels(self):
        v = torch.tensor([[0.0, 0.05, 1.0, 1.02, 5.0]])
        mask = torch.tensor([[True, True, True, True, False]])
        labels = epsilon_labels(v, 0.1, mask)
        assert labels[0, 0] == labels[0, 1] and labels[0, 2] == labels[0, 3]
        assert labels[0, 0] != labels[0, 2] and labels[0, 4] == -1


class TestMetrics:
    def test_pairwise_scores_permutation_invariant(self):
        true = torch.tensor([[0, 0, 1, 1]])
        pred = torch.tensor([[2, 2, 0, 0]])  # same partition, different ids
        s = pairwise_grouping_scores(true, pred)
        assert s["pair_precision"] == 1.0 and s["pair_recall"] == 1.0
        assert adjusted_rand_index(true, pred) == pytest.approx(1.0)

    def test_pairwise_scores_global_grouping(self):
        true = torch.tensor([[0, 0, 1, 1]])
        pred = torch.zeros(1, 4, dtype=torch.long)
        s = pairwise_grouping_scores(true, pred)
        assert s["pair_recall"] == 1.0
        assert s["pair_precision"] == pytest.approx(4 / 12)
        assert s["mean_group_size"] == pytest.approx(4.0)

    def test_event_contrast(self):
        true = torch.tensor([[5.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        event = true > 1
        num, den, rows = event_contrast_sums(true, true, event)
        assert rows == 1 and num / den == pytest.approx(1.0)
        shared = group_mean(true, torch.zeros_like(true, dtype=torch.long), torch.ones_like(event))
        num, den, _ = event_contrast_sums(true, shared, event)
        assert num / den == pytest.approx(0.0)

    def test_reward_error_levels(self):
        true = torch.tensor([0.0, 0.0, 5.0])
        corrected = torch.tensor([0.5, 0.5, 4.0])
        m = reward_error_metrics(true, true, corrected, levels=torch.tensor([0, 0, 2]))
        assert m["level2_bias"] == pytest.approx(-1.0)
        assert m["level0_bias"] == pytest.approx(0.5)


def _fake_pursuit_data(n_env=4, t=50, a=6, sigma=1.0, seed=0):
    """Reward = 5 if an 'evader' cell is set in the observation, else -0.1."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(n_env, t, a, 7, 7, 3)
    event = torch.rand(n_env, t, a, generator=g) < 0.2
    obs[..., 3, 3, 2] = event.float()
    true = torch.where(event, 4.9, -0.1)
    noisy = true + sigma * torch.randn(true.shape, generator=g)
    action = torch.nn.functional.one_hot(torch.randint(0, 5, (n_env, t, a), generator=g), 5)
    return GroupData(
        obs=obs,
        next_obs=obs.clone(),
        action=action.float(),
        reward=noisy,
        mask=torch.ones_like(true, dtype=torch.bool),
        true_reward=true,
    )


class TestGroupers:
    spec = get_task_spec("pursuit")

    def test_oracle_labels(self):
        data = _fake_pursuit_data()
        labels = oracle_labels(data.true_reward, self.spec, 3, data.mask)
        assert set(labels.unique().tolist()) == {0, 2}

    def test_untrained_learned_grouper_is_noop(self):
        data = _fake_pursuit_data()
        grouper = make_grouper(GroupingConfig(method="reward_model", epochs=1), self.spec)
        assignment, _ = grouper(data)  # out-of-sample: assigns before the first update
        assert assignment.labels is None and assignment.weights is None
        assert grouper.n_updates == 1

    @pytest.mark.parametrize("assign", ["kmeans", "epsilon", "kernel", "global"])
    def test_reward_model_grouper_learns_levels(self, assign):
        torch.manual_seed(0)
        cfg = GroupingConfig(method="reward_model", epochs=5, reward_model_assign=assign, epsilon=0.5)
        grouper = make_grouper(cfg, self.spec)
        for seed in range(3):
            grouper(_fake_pursuit_data(seed=seed))
        data = _fake_pursuit_data(seed=10)
        assignment, _ = grouper(data)
        corrected = apply_consensus(data.reward, assignment, data.mask)
        noisy_mse = ((data.reward - data.true_reward) ** 2).mean()
        corr_mse = ((corrected - data.true_reward) ** 2).mean()
        if assign != "global":
            assert corr_mse < 0.6 * noisy_mse
            truth = oracle_labels(data.true_reward, self.spec, 3, data.mask)
            if assignment.labels is not None:
                assert pairwise_grouping_scores(truth, assignment.labels)["pair_precision"] > 0.9

    def test_ensemble_reward_model_gates_on_confidence(self):
        torch.manual_seed(0)
        cfg = GroupingConfig(
            method="reward_model", epochs=3, reward_model_heads=4, reward_model_assign="epsilon"
        )
        grouper = make_grouper(cfg, self.spec)
        for seed in range(3):
            _, logs = grouper(_fake_pursuit_data(seed=seed))
        assert 0.0 <= logs["grouper/confident_frac"] <= 1.0
        data = _fake_pursuit_data(seed=7)
        assignment, _ = grouper(data)
        assert assignment.active is not None and assignment.active.dtype == torch.bool
        # Agents that are not confident are never grouped
        assert (assignment.labels[~assignment.active] == -1).all()
        corrected = apply_consensus(data.reward, assignment, data.mask)
        assert torch.equal(corrected[~assignment.active], data.reward[~assignment.active])

    def test_warmup_grouper(self):
        torch.manual_seed(0)
        grouper = make_grouper(GroupingConfig(method="warmup", epochs=40), self.spec)
        assert grouper.needs_warmup
        clean = _fake_pursuit_data(sigma=0.0)
        grouper.warmup_update(clean)
        data = _fake_pursuit_data(seed=3)
        assignment, _ = grouper(data)
        truth = oracle_labels(data.true_reward, self.spec, 3, data.mask)
        assert adjusted_rand_index(truth, assignment.labels) > 0.9

    def test_state_dict_roundtrip(self):
        cfg = GroupingConfig(method="reward_model", epochs=1)
        g1 = make_grouper(cfg, self.spec)
        g1(_fake_pursuit_data())
        g2 = make_grouper(cfg, self.spec)
        g2.load_state_dict(g1.state_dict())
        data = _fake_pursuit_data(seed=5)
        assert torch.allclose(g1.predict_rewards(data), g2.predict_rewards(data))


def test_perturbation_kernels():
    torch.manual_seed(0)
    r = torch.zeros(100000)
    noisy = perturb_rewards(r, "normal", stdev=2.0)
    assert noisy.mean().abs() < 0.05 and abs(noisy.std().item() - 2.0) < 0.05
    flipped = perturb_rewards(r, "bernoulli", flip_prob=0.1, outlier_value=10.0)
    assert set(flipped.unique().tolist()) == {-10.0, 0.0, 10.0}
    assert abs((flipped != 0).float().mean().item() - 0.1) < 0.01
    with pytest.raises(ValueError):
        perturb_rewards(r, "uniform")
