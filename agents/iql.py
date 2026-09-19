import copy
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.value import ValueNetwork
from networks.critic import CriticNetwork
from networks.actor import GaussianPolicy


def expectile_loss(
    diff: torch.Tensor,
    expectile: float,
) -> torch.Tensor:


    # 根据 diff 正负选择权重
    weight = torch.where(
        diff > 0,
        expectile,
        1.0 - expectile,
    )

    # expectile regression loss
    loss = weight * diff.pow(2)

    return loss.mean()


class IQLAgent:


    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_action: float,
        device: str = "cuda",
        hidden_dim: int = 256,
        hidden_depth: int = 2,
        discount: float = 0.99,
        expectile: float = 0.7,
        temperature: float = 3.0,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        max_exp_advantage: float = 100.0,

        max_steps: int = 500_000,
        use_actor_lr_decay: bool = True,
        grad_clip_norm: float = 10.0,
    ):

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        self.discount = discount
        self.expectile = expectile
        self.temperature = temperature
        self.tau = tau
        self.max_exp_advantage = max_exp_advantage

        self.max_steps = max_steps
        self.use_actor_lr_decay = use_actor_lr_decay
        self.grad_clip_norm = grad_clip_norm

        self.actor_frozen = False
        self.last_actor_loss = 0.0
        self.last_mean_advantage = 0.0

        self.value = ValueNetwork(
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)

        self.critic = CriticNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)

        self.actor = GaussianPolicy(
            observation_dim=observation_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
        ).to(self.device)


        self.target_critic = copy.deepcopy(self.critic).to(self.device)


        for param in self.target_critic.parameters():
            param.requires_grad = False


        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=value_lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        if self.use_actor_lr_decay:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer,
                T_max=self.max_steps,
                eta_min=0.0,
            )
        else:
            self.actor_scheduler = None

    def freeze_actor(self):
        self.actor_frozen = True

        for param in self.actor.parameters():
            param.requires_grad = False

        print("Actor has been frozen.")

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:


        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        terminals = batch["terminals"]


        value_loss = self.update_value(observations, actions)


        critic_loss = self.update_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )

        if not self.actor_frozen:
            actor_loss, mean_advantage = self.update_actor(
                observations=observations,
                actions=actions,
            )

            self.last_actor_loss = actor_loss
            self.last_mean_advantage = mean_advantage
        else:
            actor_loss = self.last_actor_loss
            mean_advantage = self.last_mean_advantage

        self.soft_update_target_critic()

        logs = {
            "value_loss": value_loss,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "mean_advantage": mean_advantage,
        }

        return logs

    def update_value(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> float:


        with torch.no_grad():
            # 使用 target critic 计算稳定的 Q 值
            q = self.target_critic.q_min(observations, actions)

        # 当前 Value 网络输出 V(s)
        v = self.value(observations)

        # diff = Q(s,a) - V(s)
        diff = q - v

        loss = expectile_loss(
            diff=diff,
            expectile=self.expectile,
        )

        self.value_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.value.parameters(),
            max_norm=self.grad_clip_norm,
        )

        self.value_optimizer.step()

        return float(loss.item())

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
    ) -> float:


        with torch.no_grad():

            next_v = self.value(next_observations)

            target_q = rewards + self.discount * (1.0 - terminals) * next_v

        q1, q2 = self.critic(observations, actions)

        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.grad_clip_norm,
        )

        self.critic_optimizer.step()

        return float(loss.item())

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):


        with torch.no_grad():
            q = self.target_critic.q_min(observations, actions)

            v = self.value(observations)

            advantage = q - v

            scaled_advantage = self.temperature * advantage

            scaled_advantage = torch.clamp(
                scaled_advantage,
                max=math.log(self.max_exp_advantage),
            )

            exp_advantage = torch.exp(scaled_advantage)

        log_probs = self.actor.log_prob(observations, actions)


        loss = -(exp_advantage * log_probs).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.grad_clip_norm,
        )

        self.actor_optimizer.step()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        mean_advantage = float(advantage.mean().item())

        return float(loss.item()), mean_advantage

    def soft_update_target_critic(self):


        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(),
                self.target_critic.parameters(),
            ):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

    @torch.no_grad()
    def act(
        self,
        observation,
        deterministic: bool = True,
    ):

        return self.actor.act(
            observation=observation,
            device=self.device,
            deterministic=deterministic,
        )