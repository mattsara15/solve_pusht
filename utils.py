# type: ignore[all]
import random
import threading
import torch

from typing import Any, Dict, List, Optional, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        expert_buffer: Optional["ReplayBuffer"] = None,
        pin_to_device: bool = False,
    ):
        self.capacity: int = capacity
        self.pin_to_device = pin_to_device
        # NOTE: We keep experiences on CPU to avoid unbounded GPU memory growth.
        # Only sampled batches are moved to `self.device`.
        # If pin_to_device is True, we move them to GPU immediately.
        self.buffer: List[Any] = []
        self.size: int = 0
        self.device: torch.device = device
        self._write_idx: int = 0
        self._lock = threading.Lock()

        # Optional standalone buffer holding expert demonstrations.
        # Kept separate from online replay; `sample()` can mix from it.
        self.expert_buffer: Optional["ReplayBuffer"] = expert_buffer

    def _move_to_device(self, data: Any) -> Any:
        return data.to(self.device)

    def set_expert_buffer(self, expert_buffer: Optional["ReplayBuffer"]) -> None:
        self.expert_buffer = expert_buffer


    def add(self, experience: Any) -> None:
        """Add a single experience to the replay buffer.

        Thread-safe: guarded by an internal lock.
        """
        if self.pin_to_device:
            experience = self._move_to_device(experience)

        # Store experiences on CPU to prevent GPU memory growth with buffer size.
        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
                self.size += 1
            else:
                idx = self._write_idx % self.capacity
                self.buffer[idx] = experience
                self._write_idx = (self._write_idx + 1) % self.capacity

    def sample(self, batch_size: int, percent_expert: float = 0.0) -> Tuple[torch.Tensor, ...]:
        expert_k = int(round(batch_size * percent_expert))
        expert_k = max(0, min(batch_size, expert_k))
        online_k = batch_size - expert_k

        # Check buffer sizes (no copying)
        with self._lock:
            online_size = len(self.buffer)
        expert_size = 0
        if self.expert_buffer is not None:
            with self.expert_buffer._lock:
                expert_size = len(self.expert_buffer.buffer)

        # If expert buffer is empty, fall back to online
        if expert_k > 0 and expert_size == 0:
            online_k = batch_size
            expert_k = 0

        # If online buffer is empty, fall back to expert
        if online_k > 0 and online_size == 0 and expert_size > 0:
            expert_k = batch_size
            online_k = 0

        batch: List[Any] = []

        if online_k > 0:
            with self._lock:
                size_now = len(self.buffer)
                target = min(online_k, size_now)
                if target > 0:
                    idxs = random.sample(range(size_now), k=target)
                    batch.extend([self.buffer[i] for i in idxs])

        if expert_k > 0 and self.expert_buffer is not None:
            with self.expert_buffer._lock:
                size_now = len(self.expert_buffer.buffer)
                target = min(expert_k, size_now)
                if target > 0:
                    idxs = random.sample(range(size_now), k=target)
                    batch.extend([self.expert_buffer.buffer[i] for i in idxs])

        if len(batch) == 0:
            return ()

        # simplify output format
        pixels = []
        agent_pos = []
        next_pixels = []
        next_agent_pos = []
        actions = []
        rewards = []
        dones = []
        for row in batch:
            pixels.append(row[0][0])
            agent_pos.append(row[0][1])
            actions.append(row[1])
            rewards.append(row[2])
            next_pixels.append(row[3][0])
            next_agent_pos.append(row[3][1])
            dones.append(row[4])
            
        # Stack on CPU then move the batch to the training device.
        return (
            torch.stack(pixels).to(self.device, non_blocking=True),
            torch.stack(agent_pos).to(self.device, non_blocking=True),
            torch.stack(actions).to(self.device, non_blocking=True),
            torch.stack(rewards).unsqueeze(-1).to(self.device, non_blocking=True),
            torch.stack(dones).unsqueeze(-1).to(self.device, non_blocking=True),
            torch.stack(next_pixels).to(self.device, non_blocking=True),
            torch.stack(next_agent_pos).to(self.device, non_blocking=True),
        )

    def __len__(self):
        return self.size


class ExpertReplayBuffer(ReplayBuffer):
    # -----------------------------
    # Expert dataset integration
    # -----------------------------
    def preload_expert_transitions(
        self,
        dataset_id: str = "lerobot/pusht_image",
    ) -> int:
        data = LeRobotDataset(dataset_id)

        # Group entries by episode_index with their frame_index for adjacency
        episodes: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}

        for entry in data:
            # Expect these keys to exist in the dataset
            ep_idx = int(entry["episode_index"])
            fr_idx = int(entry["frame_index"])

            if ep_idx not in episodes:
                episodes[ep_idx] = []
            episodes[ep_idx].append((fr_idx, entry))

        # Sort frames within each episode
        for ep in episodes:
            episodes[ep].sort(key=lambda x: x[0])

        transitions_added = 0

        # Build transitions from adjacent frames per episode
        for ep_idx, frames in episodes.items():
            for i in range(len(frames) - 1):
                curr_entry = frames[i][1]
                next_entry = frames[i + 1][1]

                image_t = curr_entry["observation.image"]
                state_t = curr_entry["observation.state"]
                action_t = curr_entry["action"]

                # Reward and done typically refer to the transition to t+1
                reward_tp1 = curr_entry["next.reward"]
                done_tp1 = curr_entry["next.done"]

                # Use observation image/state from frame_index t+1 as next_observation
                image_tp1 = next_entry["observation.image"]
                state_tp1 = next_entry["observation.state"]

                # log experience tuple
                experience = (
                    [image_t, state_t],
                    action_t,
                    reward_tp1,
                    [image_tp1, state_tp1],
                    done_tp1.to(torch.float32),
                )

                self.add(experience=experience)
                transitions_added += 1

        return transitions_added
