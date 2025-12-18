# type: ignore[all]


from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import numpy as np 
from contextlib import nullcontext

from agents.sac.actor import Actor
from agents.sac.critic import Critic


class SACConfig:
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    gamma: float = 0.98
    tau: float = 0.005
    alpha: float = 0.15


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
        self._enhanced_debug = False

        # AMP config (favor bf16 when supported; fall back to fp16)
        self._use_amp: bool = self._device.type == "cuda"
        if self._use_amp and hasattr(torch.cuda, "is_bf16_supported"):
            self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._amp_dtype = torch.bfloat16

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
        self._critic_1_target.eval()
        for p in self._critic_1_target.parameters():
            p.requires_grad_(False)

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
        self._critic_2_target.eval()
        for p in self._critic_2_target.parameters():
            p.requires_grad_(False)

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

        # Precompute constant used in scaled-action log-prob correction
        # log|det(da_env/da_tanh)| = sum(log(action_scale))
        self._log_action_scale_sum = torch.log(self._action_scale).sum()

        # Try enabling fused Adam on CUDA for speed (best-effort)
        if self._device.type == "cuda":
            for opt in (self._actor_optimizer, self._critic_1_optimizer, self._critic_2_optimizer):
                try:
                    # Re-create optimizer with fused=True if supported by this torch build
                    params = [p for group in opt.param_groups for p in group["params"]]
                    lr = opt.param_groups[0].get("lr", 1e-3)
                    new_opt = torch.optim.Adam(params, lr=lr, fused=True)
                    new_opt.load_state_dict(opt.state_dict())
                    if opt is self._actor_optimizer:
                        self._actor_optimizer = new_opt
                    elif opt is self._critic_1_optimizer:
                        self._critic_1_optimizer = new_opt
                    else:
                        self._critic_2_optimizer = new_opt
                except Exception:
                    pass

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
        log_prob = log_prob - self._log_action_scale_sum
        return action, log_prob

    def _autocast(self):
        if not self._use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=True)

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
        with self._autocast():
            # Targets do not require gradients; avoid building an autograd graph.
            with torch.no_grad():
                next_action, next_log_prob = self.act(next_pixels, next_agent_pos)
                target_Q_1_next = self._critic_1_target(next_pixels, next_agent_pos, next_action)
                target_Q_2_next = self._critic_2_target(next_pixels, next_agent_pos, next_action)
                target_Q_min_next = torch.min(target_Q_1_next, target_Q_2_next)
                target_Q = rewards + self._cfg.gamma * (1 - dones) * (
                    target_Q_min_next - self._cfg.alpha * next_log_prob
                )

            current_Q_1 = self._critic_1(pixels, agent_pos, actions)
            current_Q_2 = self._critic_2(pixels, agent_pos, actions)

            critic_1_loss = torch.nn.functional.mse_loss(current_Q_1, target_Q)
            critic_2_loss = torch.nn.functional.mse_loss(current_Q_2, target_Q)

        # Update Critic 1
        self._critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_1.parameters(), max_norm=10.0
        )
        self._critic_1_optimizer.step()

        # Update Critic 2
        self._critic_2_optimizer.zero_grad(set_to_none=True)
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
            "combined_q_loss" : (critic_1_loss + critic_2_loss).item(),
            "q1_grad_norm": critic_1_grad_norm,
            "q2_grad_norm": critic_2_grad_norm,
        }

    def update_actor(self, pixels, agent_pos):
        with self._autocast():
            actions, log_prob = self.act(pixels, agent_pos)
            with torch.no_grad():
                Q_1 = self._critic_1(pixels, agent_pos, actions)
                Q_2 = self._critic_2(pixels, agent_pos, actions)
                Q_min = torch.min(Q_1, Q_2)
            actor_loss = (self._cfg.alpha * log_prob - Q_min).mean()
    
        # Update Actor
        self._actor_optimizer.zero_grad(set_to_none=True)
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

    def state_dict(self, include_optimizers: bool = True) -> Dict[str, Any]:
        cfg_state = {
            "critic_lr": float(getattr(self._cfg, "critic_lr")),
            "actor_lr": float(getattr(self._cfg, "actor_lr")),
            "gamma": float(getattr(self._cfg, "gamma")),
            "tau": float(getattr(self._cfg, "tau")),
            "alpha": float(getattr(self._cfg, "alpha")),
        }

        state: Dict[str, Any] = {
            "meta": {
                "pix_dim": list(self._pix_dim),
                "state_dim": int(self._state_dim),
                "action_dim": int(self._action_dim),
                "action_low": self._action_low.detach().cpu(),
                "action_high": self._action_high.detach().cpu(),
            },
            "cfg": cfg_state,
            "actor": self._actor.state_dict(),
            "critic_1": self._critic_1.state_dict(),
            "critic_1_target": self._critic_1_target.state_dict(),
            "critic_2": self._critic_2.state_dict(),
            "critic_2_target": self._critic_2_target.state_dict(),
        }

        if include_optimizers:
            state["actor_optimizer"] = self._actor_optimizer.state_dict()
            state["critic_1_optimizer"] = self._critic_1_optimizer.state_dict()
            state["critic_2_optimizer"] = self._critic_2_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self._actor.load_state_dict(state_dict["actor"], strict=strict)
        self._critic_1.load_state_dict(state_dict["critic_1"], strict=strict)
        self._critic_1_target.load_state_dict(state_dict["critic_1_target"], strict=strict)
        self._critic_2.load_state_dict(state_dict["critic_2"], strict=strict)
        self._critic_2_target.load_state_dict(state_dict["critic_2_target"], strict=strict)

        # Optimizers are optional (e.g. when loading for evaluation only)
        if "actor_optimizer" in state_dict:
            self._actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
            self._move_optimizer_state_to_device(self._actor_optimizer)
        if "critic_1_optimizer" in state_dict:
            self._critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
            self._move_optimizer_state_to_device(self._critic_1_optimizer)
        if "critic_2_optimizer" in state_dict:
            self._critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
            self._move_optimizer_state_to_device(self._critic_2_optimizer)

    def save(self, path: Union[str, Path], include_optimizers: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(include_optimizers=include_optimizers), path)

    def load(
        self,
        path: Union[str, Path],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        ckpt = torch.load(
            Path(path),
            map_location=map_location if map_location is not None else self._device,
        )
        self.load_state_dict(ckpt, strict=strict)
        return ckpt

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self._device)
