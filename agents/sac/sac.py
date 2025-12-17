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

    def soft_update_target_networks(self):
        for target_param, param in zip(
            self._critic_1_target.parameters(), self._critic_1.parameters()
        ):
            target_param.data.copy_(
                self._cfg.tau * param.data + (1 - self._cfg.tau) * target_param.data
            )

        for target_param, param in zip(
            self._critic_2_target.parameters(), self._critic_2.parameters()
        ):
            target_param.data.copy_(
                self._cfg.tau * param.data + (1 - self._cfg.tau) * target_param.data
            )

    def update(
        self, pixels, agent_pos, actions, rewards, dones, next_pixels, next_agent_pos
    ):
        with torch.no_grad():
            next_actions, next_log_pi = self.act(next_pixels, next_agent_pos)
            Q_1_next = self._critic_1_target(next_pixels, next_agent_pos, next_actions)
            Q_2_next = self._critic_2_target(next_pixels, next_agent_pos, next_actions)
            Q_min_next = torch.min(Q_1_next, Q_2_next)

            # TODO: use alpha decay
            y = rewards + self._cfg.gamma * (1 - dones) * (
                Q_min_next - self._cfg.alpha * next_log_pi
            )
        
        new_actions, log_pi = self.act(pixels, agent_pos)
        Q_1 = self._critic_1(pixels, agent_pos, new_actions)
        Q_2 = self._critic_2(pixels, agent_pos, new_actions)
        Q_min = torch.min(Q_1, Q_2)
        actor_loss = (self._cfg.alpha * log_pi - Q_min).mean()

        # Update Actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(), max_norm=10.0
        )
        self._actor_optimizer.step()

        # Update Critic 1
        critic_1_loss = torch.nn.functional.mse_loss(
            self._critic_1(pixels, agent_pos, actions), y
        )
        self._critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_1.parameters(), max_norm=10.0
        )
        self._critic_1_optimizer.step()

        # Update Critic 2
        critic_2_loss = torch.nn.functional.mse_loss(
            self._critic_2(pixels, agent_pos, actions), y
        )
        self._critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_2.parameters(), max_norm=10.0
        )
        self._critic_2_optimizer.step()

        # Soft update target networks
        self.soft_update_target_networks()

        return {
            "q1_loss": critic_1_loss.item(),
            "q2_loss": critic_2_loss.item(),
            "q1_grad_norm": critic_1_grad_norm,
            "q2_grad_norm": critic_2_grad_norm,
            "actor_grad_norm": actor_grad_norm,
            "actor_loss": actor_loss.item(),
        }
