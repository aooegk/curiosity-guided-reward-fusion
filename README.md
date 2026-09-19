# Curiosity-Guided Reward Fusion for Offline Reinforcement Learning

This repository contains the core implementation accompanying the manuscript **Curiosity-Guided Reward Fusion for Offline Reinforcement Learning**.

The method studies data-dependent reward modification within Implicit Q-Learning (IQL). It combines a Random Network Distillation (RND) novelty signal, quantile-based novelty selection, a heuristic value-residual gate, and a dynamic auxiliary reward coefficient.

The environment reward is retained, and eligible transitions receive an additive reward increment. The study characterizes performance across offline datasets rather than claiming consistent improvements over IQL.

## Repository Structure

```text
.
├── main_iql.py
├── datasets.py
├── evaluate.py
├── agents/
│   ├── __init__.py
│   └── iql.py
├── networks/
│   ├── __init__.py
│   ├── actor.py
│   ├── critic.py
│   ├── value.py
│   └── mlp.py
├── ours/
│   ├── __init__.py
│   ├── main_ablation_dynamic_iql.py
│   ├── dynamic_rnd.py
│   └── beta_scheduler.py
└── utils/
    ├── __init__.py
    └── replay_buffer.py
```

- `main_iql.py`: IQL baseline training.
- `ours/main_ablation_dynamic_iql.py`: reward-fusion training and configuration.
- `ours/dynamic_rnd.py`: RND predictor, target network, and novelty signal.
- `ours/beta_scheduler.py`: auxiliary reward coefficient scheduling.
- `datasets.py`: D4RL dataset loading and preprocessing.
- `evaluate.py`: policy evaluation using D4RL normalized scores.

## Recorded Local Environment

The following versions were reported by the local Python environment:

| Component | Version |
|---|---|
| Python | 3.11.7 |
| PyTorch | 2.7.1+cu118 |
| NumPy | 2.4.6 |
| Gym | 0.26.2 |
| D4RL | 1.1 |
| mujoco-py | 2.1.2.14 |
| tqdm | 4.67.3 |

The local NVIDIA driver version is 560.94. The PyTorch package has the CUDA 11.8 build tag (`cu118`).

This table records installed package versions; it is not yet a validated clean-install specification. The current evaluation code expects the legacy Gym API. Compatibility with the installed Gym version must be checked before training.

MuJoCo and its system dependencies must also be configured. See the upstream [D4RL](https://github.com/Farama-Foundation/D4RL) and [mujoco-py](https://github.com/openai/mujoco-py) installation instructions.

## Datasets

The manuscript evaluates nine D4RL MuJoCo locomotion datasets:

| Environment | Dataset variants |
|---|---|
| HalfCheetah | medium-v2, medium-replay-v2, medium-expert-v2 |
| Hopper | medium-v2, medium-replay-v2, medium-expert-v2 |
| Walker2d | medium-v2, medium-replay-v2, medium-expert-v2 |

Datasets are accessed through D4RL and are not included in this repository.

## Running the Code

Run commands from the repository root in an environment with the required dependencies installed and compatible MuJoCo environments available.

### Reward fusion

```bash
python -m ours.main_ablation_dynamic_iql
```

The current default configuration is:

| Setting | Value |
|---|---|
| Dataset | hopper-medium-replay-v2 |
| Seed | 2 |
| Experiment | full_dynamic_beta |
| Training steps | 500,000 |
| Batch size | 256 |
| Evaluation interval | 10,000 steps |
| Evaluation episodes | 10 |
| Auxiliary reward warm-up | 50,000 steps |
| RND quantiles | 0.25 and 0.90 |
| Value-residual quantile | 0.60 |
| Dynamic coefficient range | 0 to 0.06 |

Edit `DATASETS`, `SEEDS`, and `EXPERIMENTS` in `ours/main_ablation_dynamic_iql.py` to configure additional runs.

The default command runs one configuration on one dataset with one seed. It does not automatically reproduce all manuscript experiments.

### IQL baseline

```bash
python main_iql.py
```

Configure the baseline script separately and match the dataset, seeds, preprocessing, training horizon, and evaluation settings when comparing it with reward fusion.

## Reward-Fusion Configurations

The training configuration supports:

- Disabling auxiliary rewards with `use_rnd=False`.
- Choosing fixed or dynamic coefficients through `beta_mode`.
- Enabling or disabling novelty selection through `use_rnd_gate`.
- Enabling or disabling value-residual selection through `use_td_gate`.
- Adjusting value-residual quantile bounds through `potential_quantile` and `potential_high_quantile`.

The default fixed coefficient is `1e-4`. Only the full dynamic configuration is enabled in the provided experiment list.

## Outputs

Reward-fusion runs save outputs under:

```text
ours_logs/ablation_dynamic_rnd_iql/
ours_checkpoints/ablation_dynamic_rnd_iql/
ours_done/ablation_dynamic_rnd_iql/
```

The outputs include training logs, evaluation logs, checkpoints, and completion markers.

Runs with an existing completion marker or a completed evaluation log are skipped. Use a distinct `ABLATION_NAME` when creating a separate set of runs.

## Evaluation Metrics

- **LastAvgScore**: mean normalized score over the final ten evaluation checkpoints of each run.
- **LastScore**: normalized score at the final evaluation checkpoint.
- **BestScore**: highest normalized score observed across evaluation checkpoints.

Manuscript summaries aggregate per-run metrics across seeds. Peak scores are descriptive and should be interpreted alongside late-training performance.

## Scope

This repository currently provides the core IQL and reward-fusion training code. Additional baseline implementations, analysis scripts, datasets, and pretrained checkpoints are not included in this initial upload.
