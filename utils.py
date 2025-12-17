# type: ignore[all]
import time
import random
import torch

from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset  # type: ignore

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ParallelReplayBuffer:
    def __init__(self, capacity: int, num_workers: int, device: torch.device):
        self.capacity: int = capacity
        self.num_workers: int = num_workers
        self.buffers: List[List[Any]] = [[] for _ in range(num_workers)]
        self.size: int = 0
        self.device: torch.device = device
        # Per-worker write index for ring-buffer behavior when capacity is reached
        self._write_idx: List[int] = [0 for _ in range(num_workers)]

    def add(self, worker_id: int, experience: Any) -> None:
        """Add a single experience to the buffer for a given worker.

        Experience can be any tuple or object; training code should know how to consume it.
        """
        if len(self.buffers[worker_id]) < self.capacity:
            self.buffers[worker_id].append(experience)
        else:
            # Overwrite oldest using a simple ring buffer per worker
            idx = self._write_idx[worker_id] % self.capacity
            self.buffers[worker_id][idx] = experience
            self._write_idx[worker_id] = (
                self._write_idx[worker_id] + 1
            ) % self.capacity
        self.size += 1

    def sample(self, batch_size: int) -> List[Any]:
        all_experiences = [exp for buffer in self.buffers for exp in buffer]
        # Safeguard: If requested batch exceeds available, sample with replacement
        if batch_size > len(all_experiences):
            return [random.choice(all_experiences) for _ in range(batch_size)]
        return random.sample(all_experiences, batch_size)

    def __len__(self):
        return self.size

    # -----------------------------
    # Expert dataset integration
    # -----------------------------
    def preload_expert_transitions(
        self,
        dataset_id: str = "lerobot/pusht_image",
        limit: Optional[int] = None,
        target_worker: int = 0,
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

                image_t = curr_entry["observation.image"].to(self.device).unsqueeze(0)  
                state_t = curr_entry["observation.state"].to(self.device).unsqueeze(0)
                action_t = curr_entry["action"].to(self.device).unsqueeze(0)

                # Reward and done typically refer to the transition to t+1
                reward_tp1 = curr_entry["next.reward"].to(self.device).unsqueeze(0)
                done_tp1 = curr_entry["next.done"].to(self.device).unsqueeze(0)

                # Use observation image/state from frame_index t+1 as next_observation
                image_tp1 = next_entry["observation.image"].to(self.device).unsqueeze(0)
                state_tp1 = next_entry["observation.state"].to(self.device).unsqueeze(0)

                # log experience tuple
                experience = (
                    [image_t, state_t],
                    action_t,
                    reward_tp1,
                    [image_tp1, state_tp1],
                    done_tp1,
                )

                self.add(worker_id=target_worker, experience=experience)
                transitions_added += 1

                if limit is not None and transitions_added >= limit:
                    return transitions_added

        return transitions_added
