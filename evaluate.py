import numpy as np
import torch


def evaluate_policy(
    env,
    agent,
    device: torch.device,
    eval_episodes: int = 10,
    deterministic: bool = True,
):


    episode_returns = []


    with torch.no_grad():
        for episode in range(eval_episodes):

            observation = env.reset()

            done = False
            episode_return = 0.0

            while not done:

                action = agent.act(
                    observation=observation,
                    deterministic=deterministic,
                )

                next_observation, reward, done, info = env.step(action)

                episode_return += reward


                observation = next_observation

            episode_returns.append(episode_return)


    average_return = float(np.mean(episode_returns))


    normalized_score = float(env.get_normalized_score(average_return) * 100.0)

    result = {
        "average_return": average_return,
        "normalized_score": normalized_score,
        "returns": episode_returns,
    }

    return result