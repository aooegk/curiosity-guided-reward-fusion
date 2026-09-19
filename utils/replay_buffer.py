import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, data: dict, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        self.observations = data["observations"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_observations = data["next_observations"]
        self.terminals = data["terminals"]

        self.size = self.observations.shape[0]

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "observations": torch.as_tensor(
                self.observations[indices], dtype=torch.float32, device=self.device
            ),
            "actions": torch.as_tensor(
                self.actions[indices], dtype=torch.float32, device=self.device
            ),
            "rewards": torch.as_tensor(
                self.rewards[indices], dtype=torch.float32, device=self.device
            ),
            "next_observations": torch.as_tensor(
                self.next_observations[indices], dtype=torch.float32, device=self.device
            ),
            "terminals": torch.as_tensor(
                self.terminals[indices], dtype=torch.float32, device=self.device
            ),
        }

        return batch

    def __len__(self):
        return self.size