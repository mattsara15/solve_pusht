# type: ignore[all]

import torch
import torch.nn as nn

from typing import Tuple


class Actor(nn.Module):
    def __init__(self, pix_shape, action_dim, action_range: Tuple[int, int]):
        super(Actor, self).__init__()
        self._action_dim = action_dim
        self._min_action = action_range[0]
        self._max_action = action_range[1]

        self._cnn = nn.Conv2d(pix_shape[0], 32, kernel_size=3, stride=2)
        self._fc1 = nn.Linear(70688, 256)
        self._mean_fc = nn.Linear(256, action_dim)
        self._log_std_fc = nn.Linear(256, action_dim)

    def _adjust_log_std(self, log_std):
        log_std_min, log_std_max = (-5, 2)  # From SpinUp / Denis Yarats
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    def forward(self, pixels: torch.Tensor, agent_pos: torch.Tensor):
        y = nn.functional.relu(self._cnn(pixels))
        y = y.flatten(start_dim=1)
        y = nn.functional.relu(self._fc1(y))

        mu = self._mean_fc(y)
        log_std = torch.tanh(self._log_std_fc(y))
        log_std = self._adjust_log_std(log_std)
        return mu, log_std
