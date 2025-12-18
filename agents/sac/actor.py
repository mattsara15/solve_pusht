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

        self._cnn_backbone = nn.Sequential(
            nn.Conv2d(pix_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self._state_proj_dim = 32
        self._state_proj = nn.Sequential(
            nn.Linear(2, self._state_proj_dim),
            nn.ReLU(inplace=True),
        )
        self._mlp = nn.Sequential(
            nn.Linear(400 + self._state_proj_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self._mean_head = nn.Linear(256, action_dim)
        self._log_std_head = nn.Linear(256, action_dim)

    def _adjust_log_std(self, log_std):
        log_std_min, log_std_max = (-5, 2)  # From SpinUp / Denis Yarats
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    def forward(self, pixels: torch.Tensor, agent_pos: torch.Tensor):
        z = self._cnn_backbone(pixels)
        z_flat = torch.flatten(z, start_dim=1)
        state_proj = self._state_proj(agent_pos)
        feats = torch.cat([z_flat, state_proj], dim=1)
        
        y = self._mlp(feats)

        mu = self._mean_head(y)
        log_std = torch.tanh(self._log_std_head(y))
        log_std = self._adjust_log_std(log_std)
        return mu, log_std
