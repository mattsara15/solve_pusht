# type: ignore[all]


from typing import List, Tuple
import torch
import numpy as np 

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
        action_range: Tuple[object, object],
        device: torch.device,
        cfg: SACConfig = SACConfig(),
    ):
        # Setup Variables
        self._pix_dim: List[int] = pix_dim
        self._state_dim: int = state_dim
        self._action_dim: int = action_dim

        # Action bounds can be scalars or vectors (e.g. gymnasium Box.low/high)
        self._action_low = torch.as_tensor(action_range[0], dtype=torch.float32, device=device)
        self._action_high = torch.as_tensor(action_range[1], dtype=torch.float32, device=device)
        # Map tanh output in [-1, 1] to [low, high]: a_env = a_tanh * scale + bias
        self._action_scale = (self._action_high - self._action_low) / 2.0
        self._action_scale = torch.clamp(self._action_scale, min=1e-6)
        self._action_bias = (self._action_high + self._action_low) / 2.0

        self._device: torch.device = device
        self._enhanced_debug = True

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
        normal = torch.distributions.Normal(mean, std)
        sample = normal.rsample()
        action_tanh = torch.tanh(sample)

        # Scale to environment bounds
        action = action_tanh * self._action_scale + self._action_bias

        # Log-prob of the *scaled* action.
        # First compute log-prob after tanh squashing (standard SAC correction), then apply
        # the constant Jacobian correction for the affine scaling to env bounds.
        log_prob = normal.log_prob(sample) - torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        log_prob = log_prob - torch.log(self._action_scale).sum()
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

    def update_critic(self, pixels, agent_pos, actions, rewards, dones, next_pixels, next_agent_pos):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            next_action, next_log_prob = self.act(next_pixels, next_agent_pos)
            target_Q_1_next = self._critic_1_target(next_pixels, next_agent_pos, next_action)
            target_Q_2_next = self._critic_2_target(next_pixels, next_agent_pos, next_action)
            target_Q_min_next = torch.min(target_Q_1_next, target_Q_2_next)

            # TODO: use alpha decay
            target_Q = rewards + self._cfg.gamma * (1 - dones) * (
                target_Q_min_next - self._cfg.alpha * next_log_prob
            )
            target_Q = target_Q.detach()

            current_Q_1 = self._critic_1(pixels, agent_pos, actions)
            current_Q_2 = self._critic_2(pixels, agent_pos, actions)
        
            critic_1_loss = torch.nn.functional.mse_loss(
                current_Q_1, target_Q
            )

            critic_2_loss = torch.nn.functional.mse_loss(
                current_Q_2, target_Q
            )

        # Update Critic 1
        self._critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_1.parameters(), max_norm=10.0
        )
        self._critic_1_optimizer.step()

        # Update Critic 2
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
        }

    def update_actor(self, pixels, agent_pos):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            actions, log_prob = self.act(pixels, agent_pos)
            Q_1 = self._critic_1(pixels, agent_pos, actions)
            Q_2 = self._critic_2(pixels, agent_pos, actions)
            Q_min = torch.min(Q_1, Q_2)
            actor_loss = (self._cfg.alpha * log_prob - Q_min).mean()

        # Update Actor
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._actor.parameters(), max_norm=10.0
        )
        self._actor_optimizer.step()
        
        # Introspect the quality of action predictions
        predicted_actions_x = []
        predicted_actions_y = []        
        if self._enhanced_debug:
            for action in actions:
                predicted_actions_x.append(action[0].item())
                predicted_actions_y.append(action[1].item())

        return {
            "actor_grad_norm": actor_grad_norm,
            "actor_loss": actor_loss.item(),
            "predicted_action_histogram_x": np.asarray(predicted_actions_x),
            "predicted_action_histogram_y": np.asarray(predicted_actions_y),
        }


    def update(
        self, pixels, agent_pos, actions, rewards, dones, next_pixels, next_agent_pos
    ):
        critic_stats = self.update_critic(pixels, agent_pos, actions, rewards, dones, next_pixels, next_agent_pos)
        actor_stats = self.update_actor(pixels, agent_pos)

        return {**critic_stats, **actor_stats}

    def clone(self):
        """Creates a deep copy of the SAC agent."""
        import copy
        
        # Create a new SAC instance with the same configuration
        cloned_sac = SAC(
            pix_dim=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            action_range=(self._action_low, self._action_high),
            device=self._device,
            cfg=copy.deepcopy(self._cfg),
        )
        
        # Copy actor network
        cloned_sac._actor.load_state_dict(copy.deepcopy(self._actor.state_dict()))
        
        # Copy critic networks
        cloned_sac._critic_1.load_state_dict(copy.deepcopy(self._critic_1.state_dict()))
        cloned_sac._critic_1_target.load_state_dict(copy.deepcopy(self._critic_1_target.state_dict()))
        cloned_sac._critic_2.load_state_dict(copy.deepcopy(self._critic_2.state_dict()))
        cloned_sac._critic_2_target.load_state_dict(copy.deepcopy(self._critic_2_target.state_dict()))
        
        # Copy optimizer states
        cloned_sac._actor_optimizer.load_state_dict(copy.deepcopy(self._actor_optimizer.state_dict()))
        cloned_sac._critic_1_optimizer.load_state_dict(copy.deepcopy(self._critic_1_optimizer.state_dict()))
        cloned_sac._critic_2_optimizer.load_state_dict(copy.deepcopy(self._critic_2_optimizer.state_dict()))
        
        return cloned_sac
