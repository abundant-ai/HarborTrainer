#!/bin/bash

# Increase file descriptor limit to prevent "Too many open files" errors
ulimit -n 65536

python -m src.train \
  model_name=deepseek-ai/DeepSeek-V3.1  \
  tasks_dir=./datasets/terminal-bench-2 \
  learning_rate=2e-4 \
  batch_size=2 \
  group_size=24 \
  eval_split=0.2 \
  eval_group_size=16 \
  n_parallel_envs=48 \
  max_tokens=1024 \
  temperature=0.7 \
  context_limit=32000 \
  proactive_summarization_threshold=2000 \
  enable_summarize=true \
  n_epochs=2 \
  num_substeps=4 \
  remove_constant_reward_groups=true \
  normalize_advantages_by_std=true \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=harbor-training \
  wandb_name=ds-run
