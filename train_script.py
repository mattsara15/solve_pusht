# type: ignore[all]

import argparse
import threading
import time
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import gym_pusht  # Important: This registers the namespace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import ExpertReplayBuffer, ReplayBuffer

from agents.sac.sac import SAC


def prepare_pixels_for_agent(
    image: np.ndarray, device: torch.device, unsqueeze: bool
) -> torch.Tensor:
    tensor_val = torch.from_numpy(image)
    if unsqueeze:
        tensor_val = tensor_val.unsqueeze(0)
    y = tensor_val.permute(0, 3, 1, 2).to(torch.float32).to(device) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    y = (y - mean) / std
    return y


def evaluate_policy(
    num_episodes: int,
    device: torch.device,
    agent: SAC,
    max_steps=500,
    step_idx: int = 0,
) -> float:
    """Run evaluation rollouts with the current policy and return average reward."""
    # Create a single-environment vectorized eval env for consistency with training observation format
    eval_env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        obs_type="pixels_agent_pos",
        max_episode_steps=max_steps,
    )
    eval_env = RecordVideo(
        eval_env,
        video_folder=f"videos/{step_idx}",
        name_prefix="eval",
        episode_trigger=lambda x: True,
    )
    eval_env = RecordEpisodeStatistics(eval_env, buffer_length=num_episodes)
    returns = []
    steps = []
    for _ in range(num_episodes):
        state_dict, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0
        while not done:
            observation = prepare_pixels_for_agent(
                state_dict["pixels"], device, unsqueeze=True
            )
            agent_pos = (
                torch.from_numpy(state_dict["agent_pos"]).to(torch.float32).to(device)
            ).unsqueeze(0)

            # perform action
            actions = agent.select_action(observation, agent_pos)
            next_obs, reward, terminated, truncated, _ = eval_env.step(
                actions.detach().cpu().numpy()[0]
            )
            state_dict = next_obs
            done = terminated or truncated

            # accumulate reward
            ep_ret += reward
            ep_steps += 1

        returns.append(ep_ret)
        steps.append(ep_steps)
    eval_env.close()
    return {"average_return": np.mean(returns), "average_steps": np.mean(steps)}


def main(args):
    # Create parallel envs
    env = gym.make_vec(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        num_envs=args.num_workers,
        vectorization_mode="async",
        obs_type="pixels_agent_pos",
    )

    # TensorBoard writer
    writer = SummaryWriter(
        log_dir=f"tensorboard/{args.run_name}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
    )

    # Background evaluation thread handle (avoid overlapping eval runs)
    eval_thread: threading.Thread | None = None

    def _run_eval_async(step_idx: int, checkpoint_path: str):
        """Run policy evaluation in a background thread and log results."""
        # Create a new agent instance for evaluation and load from checkpoint
        eval_agent = SAC(
            pix_dim=pix_shape,
            state_dim=st_shape[0],
            action_dim=act_dim,
            action_range=(MIN_ACT, MAX_ACT),
            device=device,
        )
        eval_agent.load(checkpoint_path)

        eval_results = evaluate_policy(
            num_episodes=10, device=device, agent=eval_agent, step_idx=step_idx
        )
        for key, value in eval_results.items():
            writer.add_scalar(f"eval/{key}", value, global_step=step_idx)
        print(
            f"eval at step={step_idx+1}: avg_return={eval_results['average_return']:.3f}"
        )

    # Action bounds
    MIN_ACT = env.single_action_space.low
    MAX_ACT = env.single_action_space.high

    pix_shape = env.single_observation_space["pixels"].shape  # (H,W,C)
    pix_shape = (pix_shape[2], pix_shape[0], pix_shape[1])  # (C,H,W)
    st_shape = env.single_observation_space["agent_pos"].shape
    act_dim = int(np.prod(env.single_action_space.shape))

    device = torch.device("cuda")

    # Standalone expert buffer + online replay buffer
    expert_buffer = ExpertReplayBuffer(capacity=300_000, device=device)
    replay_buffer = ReplayBuffer(
        capacity=300_000, device=device, expert_buffer=expert_buffer
    )

    # Optionally preload expert demonstrations into expert buffer
    if args.use_demos:
        added = expert_buffer.preload_expert_transitions(dataset_id=args.demo_dataset)
        print(f"[info] Added: {added} expert transitions")

    agent = SAC(
        pix_dim=pix_shape,
        state_dim=st_shape[0],
        action_dim=act_dim,
        action_range=(MIN_ACT, MAX_ACT),
        device=device,
    )

    state_dict, _ = env.reset()
    for step in tqdm(range(args.iterations)):
        # Prepare state
        observation = prepare_pixels_for_agent(
            state_dict["pixels"], device, unsqueeze=False
        )
        agent_pos = (
            torch.from_numpy(state_dict["agent_pos"]).to(torch.float32).to(device)
        )

        # Sample actions
        actions = agent.select_action(observation, agent_pos)

        # Env step
        action_np = actions.detach().cpu().numpy()
        next_obs, rewards, terms, truncs, info = env.step(action_np)

        next_observation = prepare_pixels_for_agent(
            next_obs["pixels"], device, unsqueeze=False
        )
        next_agent_pos = torch.from_numpy(next_obs["agent_pos"]).to(torch.float32)

        # Store transitions by worker/env id
        for wid in range(args.num_workers):
            # Flatten dict observations for critics and replay buffer
            exp = (
                [observation[wid].to("cpu"), agent_pos[wid].to("cpu")],
                actions[wid].to("cpu"),
                torch.tensor(rewards[wid]).to(torch.float32).to("cpu"),
                [next_observation[wid].to("cpu"), next_agent_pos[wid].to("cpu")],
                torch.tensor(bool(terms[wid] or truncs[wid]))
                .to(torch.float32)
                .to("cpu"),
            )
            replay_buffer.add(experience=exp)

        # advance observation
        state_dict = next_obs

        # Reset finished envs to continue collection
        # TODO: is auto-reset being called? How to check?
        if np.any(terms | truncs):
            state_dict, _ = env.reset()

        # Learn
        if len(replay_buffer) >= args.train_start:
            # Multi-Step Training Updates
            for _ in range(args.updates_per_step):
                batch = replay_buffer.sample(
                    batch_size=args.batch_size,
                    percent_expert=args.percent_expert,
                )
                if not batch:
                    continue
                (
                    pixels,
                    agent_pos,
                    actions,
                    rewards,
                    dones,
                    next_pixels,
                    next_agent_pos,
                ) = batch
                results = agent.update(
                    pixels,
                    agent_pos,
                    actions,
                    rewards,
                    dones,
                    next_pixels,
                    next_agent_pos,
                )
                # Log training losses
                for key, value in results.items():
                    if "histogram" in key:
                        if len(value) > 0:
                            writer.add_histogram(f"train/{key}", value, step)
                    else:
                        writer.add_scalar(f"train/{key}", value, step)

        # Periodic evaluation (launch in a background thread)
        if (step + 1) % args.eval_freq == 0:
            # save the checkpoint
            checkpoint_path = f"checkpoints/sac_checkpoint_{step+1}.pt"
            agent.save(checkpoint_path)

            eval_thread = threading.Thread(
                target=_run_eval_async, args=(step, checkpoint_path), daemon=True
            )
            eval_thread.start()

    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    # Training configuration args
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=100000,
        help="Total number of training iterations (steps)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=256,
        help="Batch size for learning updates",
    )
    parser.add_argument(
        "--updates_per_step",
        "-s",
        type=int,
        default=1,
        help="Number of updates per step",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel workers/environments for training",
    )
    parser.add_argument(
        "--eval_freq",
        "-e",
        type=int,
        default=2000,
        help="Evaluation frequency in training steps",
    )
    parser.add_argument(
        "--train_start",
        type=int,
        default=2500,
        help="Number of transitions to collect before starting learning",
    )
    parser.add_argument(
        "--use_demos",
        "-D",
        action="store_true",
        help="If set, add expert demonstrations to the replay buffer before training",
    )
    parser.add_argument(
        "--percent_expert",
        type=float,
        default=0.0,
        help="Percent (0-100) or fraction (0-1) of each batch sampled from expert demonstrations",
    )
    parser.add_argument(
        "--demo_dataset",
        type=str,
        default="lerobot/pusht_image",
        help="Hugging Face dataset ID for expert transitions",
    )
    parser.add_argument(
        "--run_name",
        "-n",
        type=str,
        default="",
        help="Run name for logging and checkpointing",
    )
    args = parser.parse_args()

    main(args)
