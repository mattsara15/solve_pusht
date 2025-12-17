# type: ignore[all]


from typing import List, Tuple
import torch

from agents.sac.actor import Actor
from agents.sac.critic import Critic


class SACConfig:
    critic_lr: float = 1e-3
    actor_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2


class SAC:
    def __init__(
        self,
        pix_dim: List[int],
        state_dim: int,
        action_dim: int,
        action_range: Tuple[int, int],
        device: torch.device,
        cfg: SACConfig = SACConfig(),
    ):
        # Setup Variables
        self._pix_dim: List[int] = pix_dim
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim
        self._action_min: int = action_range[0]
        self._action_max: int = action_range[1]
        self._action_scale_factor = torch.as_tensor(self._action_max / 2).to(device)
        self._device: torch.device = device

        # Hyper-parameters
        self._cfg: SACConfig = cfg

        # Actor
        self._actor = Actor(
            pix_shape=self._pix_dim,
            action_dim=self._action_dim,
            action_range=action_range,
        ).to(device)

        # Critic
        self._critic_1 = Critic(
            pix_shape=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
        ).to(device)
        self._critic_1_target = Critic(
            pix_shape=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
        ).to(device)
        self._critic_1_target.load_state_dict(self._critic_1.state_dict())
        self._critic_1.eval()

        self._critic_2 = Critic(
            pix_shape=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
        ).to(device)
        self._critic_2_target = Critic(
            pix_shape=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
        ).to(device)
        self._critic_2_target.load_state_dict(self._critic_2.state_dict())
        self._critic_2.eval()

        # optimizers
        self._actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=self._cfg.actor_lr
        )
        self._critic_1_optimizer = torch.optim.Adam(
            self._critic_1.parameters(), lr=self._cfg.critic_lr
        )
        self._critic_2_optimizer = torch.optim.Adam(
            self._critic_2.parameters(), lr=self._cfg.critic_lr
        )

    @torch.no_grad
    def select_action(self, observation: torch.Tensor, agent_pos: torch.Tensor):
        return self.act(observation, agent_pos)[0]

    def act(self, observation: torch.Tensor, agent_pos: torch.Tensor):
        mean, log_std = self._actor(observation, agent_pos)
        std = log_std.exp()

        # action distribution
        normal = torch.distributions.Normal(mean, std)
        sample = normal.rsample()

        action_prescaling = torch.tanh(sample)
        action = (action_prescaling * self._action_scale_factor) + 1

        log_prob = normal.log_prob(sample)
        # is this correct?!?
        log_prob -= torch.log(
            self._action_scale_factor * (1 - action_prescaling.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def update(
        self, pixels, agent_pos, actions, rewards, dones, next_pixels, next_agent_pos
    ):
        

        return {"q1_loss": 0.0, "q2_loss": 0.0, "actor_loss": 0.0}
