# Terminal Bench Trainer

Post-train open-weight models on Terminal Bench 2.0 using the [Tinker API](https://tinker-docs.thinkingmachines.ai/) and the Harbor fork at [rishidesai/harbor](https://github.com/rishidesai/harbor).

## Overview

This repository implements RL training for Terminal Bench 2.0 tasks using:
- **Tinker API** for distributed LLM training (sampling + gradient computation)
- **Harbor Framework** for trial infrastructure (Terminus2 agent, environments, verification)

The trainer uses Harbor's `Trial` to run episodes with the Terminus2 agent, collecting token-level rollout data (token IDs, logprobs) for GRPO-style policy optimization via Tinker.

## Installation

```bash
uv pip install -e .  # installs Harbor from https://github.com/rishidesai/harbor
docker login         # optional but recommended for pulling/pushing images
```

## Quick Start

### 1. Set up credentials in `.env`

```bash
TINKER_API_KEY="your-api-key"
WANDB_API_KEY="your-api-key"
TOKENIZERS_PARALLELISM=false
```

### 2. Add tasks

Tasks follow the [Harbor task format](https://harborframework.com/docs/task-format):

```
terminal-bench-2/
├── task_name/
│   ├── task.toml           # Task configuration
│   ├── instruction.md      # Task description for the agent
│   ├── environment/
│   │   └── Dockerfile      # Container setup
│   ├── solution/
│   │   └── solve.sh        # Reference solution
│   └── tests/
│       └── test.sh         # Verification script
```

### 3. Run training

```bash
python -m src.train \
  model_name=Qwen/Qwen3-235B-A22B-Instruct-2507  \
  tasks_dir=./terminal-bench-2 \
  learning_rate=3e-4 \
  batch_size=8 \
  group_size=4 \
  n_parallel_envs=1 \
  max_tokens=4096 \
  temperature=0.7 \
  n_epochs=1 \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=train-qwen \
  wandb_name=qwen-run
```

## Configuration

Key training parameters (see `src/train.py` for full list):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | required | Model to train (e.g., `Qwen/Qwen3-235B-A22B-Instruct-2507`) |
| `tasks_dir` | required | Directory containing Harbor tasks |
| `lora_rank` | 32 | LoRA rank for fine-tuning |
| `learning_rate` | 5e-5 | Learning rate |
| `batch_size` | 8 | Tasks per batch |
| `group_size` | 4 | Rollouts per task (GRPO group size) |
| `max_turns` | None | Max agent turns per episode (None = unlimited) |
| `max_tokens` | 4096 | Max tokens per generation |
| `temperature` | 0.7 | Sampling temperature |
| `environment_type` | docker | Backend: `docker`, `modal`, `daytona`, `e2b`, `runloop` |
| `n_parallel_envs` | 1 | Parallel environments (higher for cloud backends) |
| `loss_fn` | importance_sampling | Loss: `importance_sampling` or `ppo` |

## Project Structure

```
.
├── src/
│   ├── tinker_llm.py          # TinkerLLM - Tinker sampling backend for Harbor
│   ├── terminus2_trainer.py   # GRPO trainer using Harbor Trial
│   └── train.py               # CLI entry point
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Terminus2RLTrainer                       │
├─────────────────────────────────────────────────────────────┤
│  For each batch of tasks:                                   │
│    1. Run Harbor Trials with TinkerLLM backend              │
│    2. Collect rollout_details (token IDs, logprobs)         │
│    3. Compute GRPO advantages (reward centering)            │
│    4. Build Tinker Datums and train via TrainingClient      │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────┐          ┌─────────────────────┐
│  Harbor Trial   │          │   Tinker API        │
│  - Terminus2    │          │   - SamplingClient  │
│  - Environments │          │   - TrainingClient  │
│  - Verification │          │   - LoRA weights    │
└─────────────────┘          └─────────────────────┘
```

## References

- [Tinker API Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [Harbor Framework](https://harborframework.com/)

## License

MIT
