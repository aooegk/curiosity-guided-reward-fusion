import os
import random
import time
import csv

import numpy as np
import torch
from tqdm import trange

from datasets import load_d4rl_dataset
from utils.replay_buffer import ReplayBuffer
from agents.iql import IQLAgent
from evaluate import evaluate_policy


def set_seed(seed: int):


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(agent: IQLAgent, save_path: str, step: int):


    checkpoint = {
        "step": step,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "value": agent.value.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
    }

    torch.save(checkpoint, save_path)


def create_csv_logs(train_log_path: str, eval_log_path: str):



    with open(train_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "value_loss",
            "critic_loss",
            "actor_loss",
            "mean_advantage",
            "elapsed_time",
        ])


    with open(eval_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "average_return",
            "normalized_score",
        ])


def write_train_log(
    train_log_path: str,
    step: int,
    value_loss: float,
    critic_loss: float,
    actor_loss: float,
    mean_advantage: float,
    elapsed_time: float,
):


    with open(train_log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            value_loss,
            critic_loss,
            actor_loss,
            mean_advantage,
            elapsed_time,
        ])


def write_eval_log(
    eval_log_path: str,
    step: int,
    average_return: float,
    normalized_score: float,
):


    with open(eval_log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            average_return,
            normalized_score,
        ])


def train_one_env(
    env_name: str,
    seed: int = 42,
    use_actor_freeze: bool = False,
):




    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_train_steps = 300_000
    step_tag = f"step{max_train_steps:_}"

    batch_size = 256

    log_interval = 1000

    eval_interval = 10_000

    eval_episodes = 10


    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)


    latest_save_path = os.path.join(
        save_dir,
        f"iql_{env_name}_seed{seed}_{step_tag}_latest.pt",
    )

    best_save_path = os.path.join(
        save_dir,
        f"iql_{env_name}_seed{seed}_{step_tag}_best.pt",
    )

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    train_log_path = os.path.join(
        log_dir,
        f"train_log_{env_name}_seed{seed}_{step_tag}.csv",
    )

    eval_log_path = os.path.join(
        log_dir,
        f"eval_log_{env_name}_seed{seed}_{step_tag}.csv",
    )

    create_csv_logs(
        train_log_path=train_log_path,
        eval_log_path=eval_log_path,
    )


    set_seed(seed)

    print("=" * 80)
    print("Basic IQL Training")
    print("=" * 80)
    print("Environment:", env_name)
    print("Seed:", seed)
    print("Device:", device)
    print("Max train steps:", max_train_steps)
    print("Batch size:", batch_size)
    print("Train log path:", train_log_path)
    print("Eval log path:", eval_log_path)
    print("Latest checkpoint path:", latest_save_path)
    print("Best checkpoint path:", best_save_path)
    print("=" * 80)


    data, info = load_d4rl_dataset(env_name)

    env = info["env"]
    observation_dim = info["observation_dim"]
    action_dim = info["action_dim"]
    max_action = info["max_action"]

    print("Observation dim:", observation_dim)
    print("Action dim:", action_dim)
    print("Max action:", max_action)
    print("Dataset size:", info["dataset_size"])
    print("=" * 80)


    try:
        env.seed(seed)
        env.action_space.seed(seed)
    except Exception:
        pass


    replay_buffer = ReplayBuffer(
        data=data,
        device=device,
    )



    agent = IQLAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,

        hidden_dim=256,
        hidden_depth=2,

        discount=0.99,
        expectile=0.7,
        temperature=3.0,
        tau=0.005,

        actor_lr=3e-4,
        critic_lr=3e-4,
        value_lr=3e-4,

        max_steps=max_train_steps,
        use_actor_lr_decay=True,
        grad_clip_norm=10.0,
    )


    print("Evaluating untrained policy...")

    eval_result = evaluate_policy(
        env=env,
        agent=agent,
        device=torch.device(device),
        eval_episodes=eval_episodes,
        deterministic=True,
    )

    print(
        f"[Before Training] "
        f"Average Return: {eval_result['average_return']:.2f} | "
        f"Normalized Score: {eval_result['normalized_score']:.2f}"
    )

    write_eval_log(
        eval_log_path=eval_log_path,
        step=0,
        average_return=eval_result["average_return"],
        normalized_score=eval_result["normalized_score"],
    )

    best_normalized_score = eval_result["normalized_score"]
    best_step = 0

    save_checkpoint(
        agent=agent,
        save_path=best_save_path,
        step=0,
    )

    print(
        f"Initial best checkpoint saved to: {best_save_path} | "
        f"Best Score: {best_normalized_score:.2f}"
    )

    print("=" * 80)

    start_time = time.time()

    actor_freeze_score = 95.0
    actor_freeze_after_step = 200_000
    actor_already_frozen = False

    recent_value_losses = []
    recent_critic_losses = []
    recent_actor_losses = []
    recent_advantages = []

    for step in trange(1, max_train_steps + 1, desc="Training IQL"):

        batch = replay_buffer.sample(batch_size=batch_size)

        logs = agent.update(batch)

        recent_value_losses.append(logs["value_loss"])
        recent_critic_losses.append(logs["critic_loss"])
        recent_actor_losses.append(logs["actor_loss"])
        recent_advantages.append(logs["mean_advantage"])

        if step % log_interval == 0:
            mean_value_loss = float(np.mean(recent_value_losses))
            mean_critic_loss = float(np.mean(recent_critic_losses))
            mean_actor_loss = float(np.mean(recent_actor_losses))
            mean_advantage = float(np.mean(recent_advantages))

            elapsed_time = float(time.time() - start_time)

            print(
                f"\nStep: {step} | "
                f"Value Loss: {mean_value_loss:.4f} | "
                f"Critic Loss: {mean_critic_loss:.4f} | "
                f"Actor Loss: {mean_actor_loss:.4f} | "
                f"Mean Advantage: {mean_advantage:.4f} | "
                f"Time: {elapsed_time:.1f}s"
            )

            write_train_log(
                train_log_path=train_log_path,
                step=step,
                value_loss=mean_value_loss,
                critic_loss=mean_critic_loss,
                actor_loss=mean_actor_loss,
                mean_advantage=mean_advantage,
                elapsed_time=elapsed_time,
            )

            recent_value_losses.clear()
            recent_critic_losses.clear()
            recent_actor_losses.clear()
            recent_advantages.clear()

        if step % eval_interval == 0:
            eval_result = evaluate_policy(
                env=env,
                agent=agent,
                device=torch.device(device),
                eval_episodes=eval_episodes,
                deterministic=True,
            )

            current_score = eval_result["normalized_score"]

            if (
                    use_actor_freeze
                    and not actor_already_frozen
                    and step >= actor_freeze_after_step
                    and current_score >= actor_freeze_score
            ):
                agent.freeze_actor()
                actor_already_frozen = True
                print(
                    f"Actor frozen at step {step} "
                    f"with score {current_score:.2f}"
                )

            print(
                f"\n[Evaluation] Step: {step} | "
                f"Average Return: {eval_result['average_return']:.2f} | "
                f"Normalized Score: {current_score:.2f}"
            )

            write_eval_log(
                eval_log_path=eval_log_path,
                step=step,
                average_return=eval_result["average_return"],
                normalized_score=current_score,
            )

            save_checkpoint(
                agent=agent,
                save_path=latest_save_path,
                step=step,
            )

            print(f"Latest checkpoint saved to: {latest_save_path}")

            if current_score > best_normalized_score:
                best_normalized_score = current_score
                best_step = step

                save_checkpoint(
                    agent=agent,
                    save_path=best_save_path,
                    step=step,
                )

                print(
                    f"New best checkpoint saved to: {best_save_path} | "
                    f"Best Step: {best_step} | "
                    f"Best Score: {best_normalized_score:.2f}"
                )
            else:
                print(
                    f"No improvement | "
                    f"Best Step: {best_step} | "
                    f"Best Score: {best_normalized_score:.2f}"
                )


    save_checkpoint(
        agent=agent,
        save_path=latest_save_path,
        step=max_train_steps,
    )

    print("=" * 80)
    print("Basic IQL training finished.")
    print(f"Final latest checkpoint saved to: {latest_save_path}")
    print(f"Best checkpoint saved to: {best_save_path}")
    print(f"Best Step: {best_step}")
    print(f"Best Normalized Score: {best_normalized_score:.2f}")
    print(f"Train log saved to: {train_log_path}")
    print(f"Eval log saved to: {eval_log_path}")
    print("=" * 80)

def main():
    experiments = [
        {
            "env_name": "halfcheetah-medium-replay-v2",
            "seed": 42,
            "use_actor_freeze": False,
        },
        {
            "env_name": "walker2d-medium-replay-v2",
            "seed": 42,
            "use_actor_freeze": False,
        },
    ]

    for exp in experiments:
        train_one_env(
            env_name=exp["env_name"],
            seed=exp["seed"],
            use_actor_freeze=exp["use_actor_freeze"],
        )


if __name__ == "__main__":
    main()