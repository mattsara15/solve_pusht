# solve_pusht

This repository contains a minimal SAC training scaffold for the PushT environment with optional expert data preloading.

## Training

Run training with parallel environments and TensorBoard logging:

```bash
python train_script.py --iterations 100000 --num_workers 1
```

### Preload expert demonstrations (optional)

You can bootstrap the replay buffer with expert transitions downloaded from a Hugging Face dataset (default: `lerobot/pusht_image`). This requires the `datasets` Python package.

```bash
# Install datasets if needed
pip install datasets

# Run training with demos
python train_script.py --use_demos \
  --demo_dataset lerobot/pusht_image \
  --demo_split train \
  --demo_limit 10000
```

Notes:
- The loader attempts to map common LeRobot field names for PushT; if certain fields are not present (e.g., agent state), placeholders are used.
- Images are converted to NumPy arrays. The SAC agent here is a stub; adapt it to your network and preprocessing.

## TensorBoard

Logs are written to `tensorboard/rl_finetune/<timestamp>`. Launch TensorBoard:

```bash
tensorboard --logdir tensorboard/rl_finetune
```

## Project structure

- `train_script.py`: Training loop and CLI.
- `eval_script.py`: Evaluation loop.
- `utils.py`: Parallel replay buffer and dataset preload.
- `agents/sac/`: SAC agent stubs.
