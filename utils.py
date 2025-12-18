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
        # NOTE: We keep experiences on CPU to avoid unbounded GPU memory growth.
        # Only sampled batches are moved to `self.device`.
        self.num_workers: int = num_workers
        self.buffers: List[List[Any]] = [[] for _ in range(num_workers)]
        self.size: int = 0
        self.device: torch.device = device
        # Per-worker write index for ring-buffer behavior when capacity is reached
        self._write_idx: List[int] = [0 for _ in range(num_workers)]

    def _to_cpu_leaf(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.detach().to("cpu")
        return x

    def _to_cpu_experience(self, experience: Any) -> Any:
        # Expected format:
        # ( [pixels, agent_pos], action, reward, [next_pixels, next_agent_pos], done )
        obs, action, reward, next_obs, done = experience
        pixels, agent_pos = obs
        next_pixels, next_agent_pos = next_obs
        return (
            [self._to_cpu_leaf(pixels), self._to_cpu_leaf(agent_pos)],
            self._to_cpu_leaf(action),
            self._to_cpu_leaf(reward),
            [self._to_cpu_leaf(next_pixels), self._to_cpu_leaf(next_agent_pos)],
            self._to_cpu_leaf(done),
        )

    def add(self, worker_id: int, experience: Any) -> None:
        """Add a single experience to the buffer for a given worker.

        Experience can be any tuple or object; training code should know how to consume it.
        """
        # Store experiences on CPU to prevent GPU memory from growing with buffer size.
        experience = self._to_cpu_experience(experience)

        if len(self.buffers[worker_id]) < self.capacity:
            self.buffers[worker_id].append(experience)
            self.size += 1
        else:
            # Overwrite oldest using a simple ring buffer per worker
            idx = self._write_idx[worker_id] % self.capacity
            self.buffers[worker_id][idx] = experience
            self._write_idx[worker_id] = (
                self._write_idx[worker_id] + 1
            ) % self.capacity

    def sample(self, batch_size: int, offline_only_iterations: int, step: int) -> List[Any]:
        """Sample a batch across worker buffers without flattening them.

        This avoids creating a giant intermediate list of all experiences.
        We sample without replacement up to the number of stored experiences.
        """
        # During offline-only warmup, sample strictly from buffer[0]
        if step < offline_only_iterations:
            buf0 = self.buffers[0]
            if len(buf0) == 0:
                return []
            target = min(batch_size, len(buf0))
            # Sample without replacement
            idxs = random.sample(range(len(buf0)), k=target) if target > 0 else []
            batch: List[Any] = [buf0[i] for i in idxs]
        else:
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

            batch = []
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

                # Keep expert transitions on CPU; sampled batches move to GPU in `sample()`.
                image_t = curr_entry["observation.image"].to("cpu")
                state_t = curr_entry["observation.state"].to("cpu")
                action_t = curr_entry["action"].to("cpu")

                # Reward and done typically refer to the transition to t+1
                reward_tp1 = curr_entry["next.reward"].to("cpu")
                done_tp1 = curr_entry["next.done"].to("cpu")

                # Use observation image/state from frame_index t+1 as next_observation
                image_tp1 = next_entry["observation.image"].to("cpu")
                state_tp1 = next_entry["observation.state"].to("cpu")

                # log experience tuple
                experience = (
                    [image_t, state_t],
                    action_t,
                    reward_tp1,
                    [image_tp1, state_tp1],
                    done_tp1.to(torch.float32),
                )

                self.add(worker_id=target_worker, experience=experience)
                transitions_added += 1

                if limit is not None and transitions_added >= limit:
                    return transitions_added

        return transitions_added
