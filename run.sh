#!/bin/bash

python -m src.train \
  model_name=Qwen/Qwen3-235B-A22B-Instruct-2507  \
  tasks_dir=./datasets/terminal-bench-2 \
  learning_rate=5e-4 \
  batch_size=1 \
  group_size=16 \
  n_parallel_envs=16 \
  max_tokens=4096 \
  temperature=0.7 \
  n_epochs=1 \
  loss_fn=ppo \
  environment_type=docker \
  wandb_project=train-qwen \
  wandb_name=qwen-run
