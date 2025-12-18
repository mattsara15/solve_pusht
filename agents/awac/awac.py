# type: ignore[all]

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from contextlib import nullcontext

from agents.sac.sac import SAC, SACConfig

class AWACConfig(SACConfig):
    beta: float = 2.0


class AWAC(SAC):
    def __init__(
        self,
        pix_dim: List[int],
        state_dim: int,
        action_dim: int,
        action_range: Tuple[object, object],
        device: torch.device,
        cfg: AWACConfig = AWACConfig(),
    ):
        super().__init__(
            pix_dim=pix_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=action_range,
            device=device,
            cfg=cfg,
        )

    def update_actor(self, pixels, agent_pos, actions):
        with self._autocast():
            pred_actions, log_prob = self.act(pixels, agent_pos)
            with torch.no_grad():
                Q1_pi = self._critic_1(pixels, agent_pos, pred_actions)
                Q2_pi = self._critic_2(pixels, agent_pos, pred_actions)
                V_pi = torch.min(Q1_pi, Q2_pi)
            
                Q1_old_actions = self._critic_1(pixels, agent_pos, actions)
                Q2_old_actions = self._critic_2(pixels, agent_pos, actions)
                V_old_actions = torch.min(Q1_old_actions, Q2_old_actions)
            
    
            beta = self._cfg.beta
            adv_pi = V_old_actions - V_pi
            weights = F.softmax(adv_pi / beta, dim=0)
            actor_loss = (-log_prob * len(weights) * weights.detach()).mean()

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
            for action in pred_actions:
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
        actor_stats = self.update_actor(pixels, agent_pos, actions)

        return {**critic_stats, **actor_stats}

    def state_dict(self, include_optimizers: bool = True) -> Dict[str, Any]:
        state = super().state_dict(include_optimizers=include_optimizers)
        state["cfg"]["beta"] = float(getattr(self._cfg, "beta"))
        return state

