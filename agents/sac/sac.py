# type: ignore[all]


from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import inspect
import torch
import numpy as np 

from agents.sac.actor import Actor
from agents.sac.critic import Critic


class SACConfig:
    critic_lr: float = 1e-4
    actor_lr: float = 1e-4
    gamma: float = 0.98
    tau: float = 0.005
    alpha: float = 0.15

    # torch.compile settings (PyTorch 2.x). Off by default for compatibility.
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    compile_dynamic: Optional[bool] = None
    # Avoid paying compile cost in evaluation clones by default.
    compile_on_clone: bool = False


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
        # Constant Jacobian term for affine mapping from tanh-space to env action-space.
        self._log_action_scale_sum = torch.log(self._action_scale).sum()

        self._device: torch.device = device
        self._enhanced_debug = False
        self._compiled = False

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

        if bool(getattr(self._cfg, "compile", False)):
            self.compile()

    @torch.no_grad
    def select_action(self, observation: torch.Tensor, agent_pos: torch.Tensor):
        return self.act(observation, agent_pos)[0]

    def act(self, observation: torch.Tensor, agent_pos: torch.Tensor):
        mean, log_std = self._actor(observation, agent_pos)
        std = log_std.exp()

        # Manual reparameterized sampling is faster than constructing a Distribution object.
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action_tanh = torch.tanh(pre_tanh)

        # Scale to environment bounds
        action = action_tanh * self._action_scale + self._action_bias

        # Log-prob of the *scaled* action.
        # Base Normal log-prob in terms of eps: log N(pre_tanh | mean, std)
        # = -0.5 * (eps^2 + 2*log_std + log(2*pi))
        log_prob = -0.5 * (eps.pow(2) + 2.0 * log_std + np.log(2.0 * np.pi))
        # Tanh correction: log|det(d tanh(x)/dx)| = sum log(1 - tanh(x)^2)
        log_prob = log_prob - torch.log1p(-action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        # Affine scaling correction (constant)
        log_prob = log_prob - self._log_action_scale_sum
        return action, log_prob

    def _autocast_enabled(self) -> bool:
        return self._device.type == "cuda"

    def _set_critics_requires_grad(self, requires_grad: bool) -> None:
        for p in self._critic_1.parameters():
            p.requires_grad_(requires_grad)
        for p in self._critic_2.parameters():
            p.requires_grad_(requires_grad)

    def compile(self) -> None:
        """Optionally compile actor/critics with torch.compile for speed."""
        if self._compiled:
            return
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available (need PyTorch 2.x)")
        # torch.compile is generally beneficial on CUDA; keep CPU behavior predictable.
        if self._device.type != "cuda":
            return

        requested_kwargs: Dict[str, Any] = {
            "mode": getattr(self._cfg, "compile_mode", "reduce-overhead"),
            "fullgraph": bool(getattr(self._cfg, "compile_fullgraph", False)),
        }
        dyn = getattr(self._cfg, "compile_dynamic", None)
        if dyn is not None:
            requested_kwargs["dynamic"] = bool(dyn)

        # Filter kwargs to match the installed torch.compile signature.
        sig = inspect.signature(torch.compile)
        compile_kwargs = {k: v for k, v in requested_kwargs.items() if k in sig.parameters}

        self._actor = torch.compile(self._actor, **compile_kwargs)
        self._critic_1 = torch.compile(self._critic_1, **compile_kwargs)
        self._critic_2 = torch.compile(self._critic_2, **compile_kwargs)
        self._critic_1_target = torch.compile(self._critic_1_target, **compile_kwargs)
        self._critic_2_target = torch.compile(self._critic_2_target, **compile_kwargs)
        self._compiled = True

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
        autocast_enabled = self._autocast_enabled()

        # Target computation should not build an autograd graph.
        with torch.no_grad():
            with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                next_action, next_log_prob = self.act(next_pixels, next_agent_pos)
                target_Q_1_next = self._critic_1_target(next_pixels, next_agent_pos, next_action)
                target_Q_2_next = self._critic_2_target(next_pixels, next_agent_pos, next_action)
                target_Q_min_next = torch.min(target_Q_1_next, target_Q_2_next)

                # TODO: use alpha decay
                target_Q = rewards + self._cfg.gamma * (1 - dones) * (
                    target_Q_min_next - self._cfg.alpha * next_log_prob
                )

        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            current_Q_1 = self._critic_1(pixels, agent_pos, actions)
            current_Q_2 = self._critic_2(pixels, agent_pos, actions)

            critic_1_loss = torch.nn.functional.mse_loss(current_Q_1, target_Q)
            critic_2_loss = torch.nn.functional.mse_loss(current_Q_2, target_Q)

        # Single backward pass reduces Python/autograd overhead.
        combined_loss = critic_1_loss + critic_2_loss
        self._critic_1_optimizer.zero_grad(set_to_none=True)
        self._critic_2_optimizer.zero_grad(set_to_none=True)
        combined_loss.backward()

        critic_1_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_1.parameters(), max_norm=10.0
        )
        critic_2_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_2.parameters(), max_norm=10.0
        )

        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()

        # Soft update target networks
        self.soft_update_target_networks()

        return {
            "q1_loss": critic_1_loss.item(),
            "q2_loss": critic_2_loss.item(),
            "combined_q_loss" : combined_loss.item(),
            "q1_grad_norm": critic_1_grad_norm,
            "q2_grad_norm": critic_2_grad_norm,
        }

    def update_actor(self, pixels, agent_pos):
        autocast_enabled = self._autocast_enabled()
        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            actions, log_prob = self.act(pixels, agent_pos)
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
        self._set_critics_requires_grad(False)
        try:
            actor_stats = self.update_actor(pixels, agent_pos)
        finally:
            self._set_critics_requires_grad(True)
        return {**critic_stats, **actor_stats}

    def clone(self):
        """Creates a deep copy of the SAC agent."""
        import copy
        
        # Create a new SAC instance with the same configuration
        cfg_copy = copy.deepcopy(self._cfg)
        if not bool(getattr(cfg_copy, "compile_on_clone", False)):
            setattr(cfg_copy, "compile", False)

        cloned_sac = SAC(
            pix_dim=self._pix_dim,
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            action_range=(self._action_low, self._action_high),
            device=self._device,
            cfg=cfg_copy,
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
