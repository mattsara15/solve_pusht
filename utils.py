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
        self.num_workers: int = num_workers + 1  # reserve one for the replay
        self.buffers: List[List[Any]] = [[] for _ in range(num_workers)]
        self.size: int = 0
        self.device: torch.device = device
        # Per-worker write index for ring-buffer behavior when capacity is reached
        self._write_idx: List[int] = [1 for _ in range(num_workers)]

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
        """Sample a batch across worker buffers without flattening them.

        This avoids creating a giant intermediate list of all experiences.
        We sample without replacement up to the number of stored experiences.
        """
        # Collect per-buffer lengths and total
        lengths = [len(buf) for buf in self.buffers]
        total = sum(lengths)

        if total == 0:
            return []

        target = min(batch_size, total)

        # Build cumulative lengths for O(log N) buffer selection
        # Example: lengths=[3,5,2] -> cumsum=[3,8,10]
        cumsum = []
        running = 0
        for L in lengths:
            running += L
            cumsum.append(running)

        batch: List[Any] = []
        seen: set[tuple[int, int]] = set()  # (buffer_idx, local_idx)

        # Helper to pick a (buffer_idx, local_idx) pair by global index
        def pick_by_global_index(gidx: int) -> tuple[int, int]:
            # Binary search over cumsum to find buffer
            lo, hi = 0, len(cumsum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if gidx < cumsum[mid]:
                    hi = mid
                else:
                    lo = mid + 1
            buf_idx = lo
            prev = cumsum[buf_idx - 1] if buf_idx > 0 else 0
            local_idx = gidx - prev
            return buf_idx, local_idx

        # Sample without replacement by tracking selected (buffer, index)
        # If duplicates occur, retry until we fill the batch or exceed attempts.
        attempts = 0
        max_attempts = target * 10  # generous cap to avoid rare infinite loops
        while len(batch) < target and attempts < max_attempts:
            attempts += 1
            gidx = random.randrange(total)
            buf_idx, local_idx = pick_by_global_index(gidx)
            key = (buf_idx, local_idx)
            if key in seen:
                continue
            seen.add(key)
            batch.append(self.buffers[buf_idx][local_idx])

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
            
        return (
            torch.stack(pixels),
            torch.stack(agent_pos),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(dones),
            torch.stack(next_pixels),
            torch.stack(next_agent_pos),
        )

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
