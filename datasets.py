import os

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

import gym
import d4rl
import numpy as np


def compute_return_range(raw_dataset):
    rewards = raw_dataset["rewards"]
    terminals = raw_dataset["terminals"]

    if "timeouts" in raw_dataset:
        timeouts = raw_dataset["timeouts"]
    else:
        timeouts = np.zeros_like(terminals)

    returns = []
    episode_return = 0.0

    for r, terminal, timeout in zip(
        rewards,
        terminals,
        timeouts,
    ):
        episode_return += float(r)

        if terminal or timeout:
            returns.append(
                episode_return
            )

            episode_return = 0.0

    if len(returns) == 0:
        return None, None

    return (
        min(returns),
        max(returns),
    )


def load_d4rl_dataset(
    env_name: str,
    normalize_reward: bool = True,
):


    env = gym.make(
        env_name
    )

    raw_dataset = (
        env.get_dataset()
    )


    dataset = (
        d4rl.qlearning_dataset(
            env
        )
    )

    observations = (
        dataset[
            "observations"
        ].astype(
            np.float32
        )
    )

    actions = (
        dataset[
            "actions"
        ].astype(
            np.float32
        )
    )

    rewards = (
        dataset[
            "rewards"
        ].astype(
            np.float32
        )
    )

    next_observations = (
        dataset[
            "next_observations"
        ].astype(
            np.float32
        )
    )

    terminals = (
        dataset[
            "terminals"
        ].astype(
            np.float32
        )
    )


    if normalize_reward:

        if (
            "halfcheetah" in env_name
            or "hopper" in env_name
            or "walker2d" in env_name
        ):
            (
                min_return,
                max_return,
            ) = compute_return_range(
                raw_dataset
            )

            if (
                min_return is not None
                and max_return > min_return
            ):
                rewards = (
                    rewards
                    / (
                        max_return
                        - min_return
                    )
                    * 1000.0
                )

                print(
                    f"Reward normalized: "
                    f"min_return="
                    f"{min_return:.2f}, "
                    f"max_return="
                    f"{max_return:.2f}"
                )

            else:
                print(
                    "Warning: reward "
                    "normalization skipped."
                )

        else:
            print(
                "Reward normalization "
                "not applied for this "
                "environment."
            )

    else:
        print(
            "Reward normalization disabled. "
            "Using raw D4RL rewards."
        )


    actions = np.clip(
        actions,
        -1.0 + 1e-5,
        1.0 - 1e-5,
    )

    data = {
        "observations": (
            observations
        ),

        "actions": (
            actions
        ),

        "rewards": (
            rewards.reshape(
                -1,
                1,
            )
        ),

        "next_observations": (
            next_observations
        ),

        "terminals": (
            terminals.reshape(
                -1,
                1,
            )
        ),
    }


    info = {
        "env": (
            env
        ),

        "observation_dim": (
            observations.shape[1]
        ),

        "action_dim": (
            actions.shape[1]
        ),

        "max_action": float(
            env.action_space.high[0]
        ),

        "dataset_size": (
            observations.shape[0]
        ),
    }

    return (
        data,
        info,
    )