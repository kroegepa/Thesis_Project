# Consensus-Based Individual Reward Robustness in MARL (GR²-MARL)

Code for robust cooperative MARL under individually perturbed rewards, built on
[BenchMARL](https://github.com/facebookresearch/BenchMARL) (v1.4.0, vendored in `BenchMARL/`).

- Method code: [`BenchMARL/benchmarl/reward_robustness`](BenchMARL/benchmarl/reward_robustness/README.md)
- Configuration: `reward_perturbation_*`, `grouping_*`, `consensus*` and `reward_model_*` in
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

```bash
cd BenchMARL
python benchmarl/run.py algorithm=mappo task=pettingzoo/pursuit \
    model=layers/cnn model@critic_model=layers/cnn \
    experiment.reward_perturbation=True experiment.reward_perturbation_stdev=2.5 \
    experiment.grouping_method=oracle
```

`grouping_method` is one of `none | oracle | global | warmup | gmm | reward_model |
reward_model_only`. See the package README for what each one does.
