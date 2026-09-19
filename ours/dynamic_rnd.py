import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon

    def update(self, x):

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.detach().float().cpu()

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta) * self.count * batch_count / total_count
        )

        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def std(self):
        return torch.sqrt(self.var + 1e-8)


class RNDNetwork(nn.Module):


    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DynamicRND:

    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        hidden_dim=256,
        output_dim=128,
        lr=3e-4,
        reward_normalize=True,
        reward_clip=5.0,
        input_normalize=False,
    ):
        self.device = device

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.state_dim + self.action_dim + self.state_dim

        self.reward_normalize = reward_normalize
        self.reward_clip = reward_clip
        self.input_normalize = input_normalize

        self.target = RNDNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)

        self.predictor = RNDNetwork(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)

        # 固定 target network，不参与训练
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=lr,
        )

        self.reward_rms = RunningMeanStd(shape=())

        if self.input_normalize:
            self.input_rms = RunningMeanStd(shape=(self.input_dim,))
        else:
            self.input_rms = None

        self.last_rnd_loss = 0.0
        self.last_error_mean = 0.0
        self.last_error_std = 0.0
        self.last_reward_mean = 0.0
        self.last_reward_std = 0.0

    def make_input(self, states, actions, next_states):

        rnd_input = torch.cat(
            [states, actions, next_states],
            dim=-1,
        )

        if self.input_normalize:
            self.input_rms.update(rnd_input)

            mean = self.input_rms.mean.to(self.device)
            std = self.input_rms.std().to(self.device)

            rnd_input = (rnd_input - mean) / (std + 1e-8)

        return rnd_input

    def compute_error(self, states, actions, next_states):

        rnd_input = self.make_input(states, actions, next_states)

        with torch.no_grad():
            target_feature = self.target(rnd_input)

        pred_feature = self.predictor(rnd_input)

        error = F.mse_loss(
            pred_feature,
            target_feature,
            reduction="none",
        )

        error = error.mean(dim=-1, keepdim=True)

        return error

    def normalize_curiosity_reward(self, error):

        reward = error.detach()

        if self.reward_normalize:
            self.reward_rms.update(reward)

            reward_std = self.reward_rms.std().to(self.device)
            reward = reward / (reward_std + 1e-8)

        if self.reward_clip is not None:
            reward = torch.clamp(
                reward,
                min=0.0,
                max=float(self.reward_clip),
            )

        return reward.detach()

    def update(self, states, actions, next_states):

        error = self.compute_error(states, actions, next_states)
        rnd_loss = error.mean()

        self.optimizer.zero_grad()
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.last_rnd_loss = float(rnd_loss.item())
        self.last_error_mean = float(error.detach().mean().item())
        self.last_error_std = float(error.detach().std().item())

        return rnd_loss

    def get_curiosity_reward(self, states, actions, next_states):

        with torch.no_grad():
            error = self.compute_error(states, actions, next_states)

        reward = self.normalize_curiosity_reward(error)

        self.last_error_mean = float(error.detach().mean().item())
        self.last_error_std = float(error.detach().std().item())
        self.last_reward_mean = float(reward.detach().mean().item())
        self.last_reward_std = float(reward.detach().std().item())

        return reward

    def update_and_get_reward(self, states, actions, next_states):

        error = self.compute_error(states, actions, next_states)

        curiosity_reward = self.normalize_curiosity_reward(error)

        rnd_loss = error.mean()

        self.optimizer.zero_grad()
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.last_rnd_loss = float(rnd_loss.item())
        self.last_error_mean = float(error.detach().mean().item())
        self.last_error_std = float(error.detach().std().item())
        self.last_reward_mean = float(curiosity_reward.detach().mean().item())
        self.last_reward_std = float(curiosity_reward.detach().std().item())

        info = self.state_dict()

        return curiosity_reward.detach(), info

    def state_dict(self):
        """
        返回当前 RND 状态，用于日志记录。
        """
        return {
            "rnd_loss": self.last_rnd_loss,
            "rnd_error_mean": self.last_error_mean,
            "rnd_error_std": self.last_error_std,
            "curiosity_reward_mean": self.last_reward_mean,
            "curiosity_reward_std": self.last_reward_std,
            "reward_rms_mean": float(self.reward_rms.mean.item()),
            "reward_rms_std": float(self.reward_rms.std().item()),
        }

    def save(self, path):
        """
        保存 RND 模型。
        """
        torch.save(
            {
                "target": self.target.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "reward_rms_mean": self.reward_rms.mean,
                "reward_rms_var": self.reward_rms.var,
                "reward_rms_count": self.reward_rms.count,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.target.load_state_dict(checkpoint["target"])
        self.predictor.load_state_dict(checkpoint["predictor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "reward_rms_mean" in checkpoint:
            self.reward_rms.mean = checkpoint["reward_rms_mean"]
            self.reward_rms.var = checkpoint["reward_rms_var"]
            self.reward_rms.count = checkpoint["reward_rms_count"]