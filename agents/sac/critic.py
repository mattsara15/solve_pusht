# type: ignore[all]

import torch
import torch.nn as nn

from typing import Tuple


class Critic(nn.Module):
    def __init__(self, pix_shape, state_dim, action_dim):
        super(Critic, self).__init__()
        self._action_dim = action_dim

        self.cnn = nn.Conv2d(pix_shape[0], 32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(70688, 256)

    def forward(
        self, pixels: torch.Tensor, agent_pos: torch.Tensor, action: torch.Tensor
    ):
        y = nn.functional.relu(self.cnn(pixels))
        y = y.flatten(start_dim=1)
        return self.fc1(y)
