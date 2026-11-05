# Consensus-Based Individual Reward Robustness in MARL (GR²-MARL)

Code for robust cooperative MARL under individually perturbed rewards, built on
[BenchMARL](https://github.com/facebookresearch/BenchMARL) (v1.4.0, vendored in `BenchMARL/`).

- **Write-up of this revision (what was tried, what worked, how to use it): [`WRITEUP.md`](WRITEUP.md)**
- Method code: [`BenchMARL/benchmarl/reward_robustness`](BenchMARL/benchmarl/reward_robustness/README.md)
- Configuration: `reward_perturbation_*`, `grouping_*`, `consensus*`, `reward_model_*` and `certify_*` in
  `BenchMARL/benchmarl/conf/experiment/base_experiment.yaml`
- Offline grouping benchmark: `BenchMARL/examples/reward_robustness/`
- Tests: `BenchMARL/test/test_reward_robustness.py`
- Review notes and revision plan: `notes/paper_review_and_revision_plan.md`

## Setup

```bash
pip install "torchrl~=0.7.0" "pettingzoo[mpe,sisl]==1.24.3" scikit-learn hydra-core
pip install -e BenchMARL
```

(`wandb` is optional and only needed when `experiment.loggers` includes `wandb`.)

## Running

Recommended warm-up-free configuration (certified consensus, noise std estimated from the data):

```bash
cd BenchMARL
python benchmarl/run.py algorithm=mappo task=pettingzoo/pursuit \
    model=layers/cnn model@critic_model=layers/cnn \
    experiment.reward_perturbation=True experiment.reward_perturbation_stdev=3.5 \
    experiment.grouping_method=certified \
    experiment.grouping_use_next_obs=True experiment.grouping_epochs=1 \
    experiment.certify_temporal_window=9 experiment.certify_replicate_consensus=True
```

`grouping_method` is one of `none | oracle | global | warmup | gmm | reward_model |
reward_model_only | certified`. See the package README for what each one does, and
`WRITEUP.md` for variants (known σ, clean warm-up, Bernoulli noise) and the full paper sweep.
Without `wandb` installed, add `experiment.loggers=[csv]`.
