# type: ignore[all]

import torch
import torch.nn as nn

from typing import Tuple


class Critic(nn.Module):
    def __init__(self, pix_shape, state_dim, action_dim):
        super(Critic, self).__init__()
        self._action_dim = action_dim

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
        self._cnn_norm = nn.LayerNorm(400)
        self._state_proj_dim = 32
        self._state_proj = nn.Sequential(
            nn.Linear(2, self._state_proj_dim),
            nn.LayerNorm(self._state_proj_dim),
            nn.ReLU(inplace=True),
        )
        self._action_proj_dim = 32
        self._action_proj = nn.Sequential(
            nn.Linear(action_dim, self._action_proj_dim),
            nn.LayerNorm(self._action_proj_dim),
            nn.ReLU(inplace=True),
        )
        self._mlp = nn.Sequential(
            nn.Linear(400 + self._state_proj_dim + self._action_proj_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self._action_dim),
        )

    def forward(self, pixels: torch.Tensor, agent_pos: torch.Tensor, action: torch.Tensor):
        z = self._cnn_backbone(pixels)
        z_flat = self._cnn_norm(z.flatten(start_dim=1))
        state_proj = self._state_proj(agent_pos)
        action_proj = self._action_proj(action)
        feats = torch.cat([z_flat, state_proj, action_proj], dim=1)
        return self._mlp(feats)
