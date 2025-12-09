from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tinker
import torch
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from tinker.types import LossFnType
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.train import (
    forward_backward as cookbook_forward_backward,
    optim_step as cookbook_optim_step,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import split_list

from src.tinker_llm import TinkerLLM

logger = logging.getLogger(__name__)

# Type aliases
Metrics = dict[str, float | int | str]
TrialGroup = list[TrialResult]  # Group of trials for same task (GRPO)


@dataclass
class TrainerConfig:
    """Configuration for Terminus2RLTrainer."""

    model_name: str
    tasks_dir: Path
    logs_dir: Path

    # Tinker configuration
    tinker_base_url: str | None = None
    lora_rank: int = 32

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 8  # Tasks per batch
    group_size: int = 4  # Rollouts per task (GRPO group size)
    n_epochs: int = 1

    # RL hyperparameters
    loss_fn: LossFnType = "importance_sampling"  # "importance_sampling" or "ppo"
    num_substeps: int = 1
    remove_constant_reward_groups: bool = False

    # Agent configuration
    max_turns: int | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    context_limit: int = 128000
    trial_timeout_sec: float | None = None

    # Environment configuration
    environment_type: str = "docker"
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    n_parallel_envs: int = 1

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20


def extract_reward(verifier_result) -> float:
    """Extract reward from a VerifierResult."""
    if not verifier_result or not verifier_result.rewards:
        return 0.0
    reward_value = verifier_result.rewards.get("reward", 0.0)
    if isinstance(reward_value, (int, float)):
        return float(reward_value)
    pass_value = verifier_result.rewards.get("pass", False)
    return 1.0 if pass_value else 0.0


def compute_grpo_advantages(groups: list[TrialGroup]) -> list[list[float]]:
    """Compute GRPO-style advantages: center rewards within each group."""
    all_advantages: list[list[float]] = []

    for group in groups:
        rewards = [extract_reward(r.verifier_result) for r in group]
        if not rewards:
            all_advantages.append([])
            continue
        mean_reward = sum(rewards) / len(rewards)
        advantages = [r - mean_reward for r in rewards]
        all_advantages.append(advantages)

    return all_advantages


def build_datums_from_trials(
        groups: list[TrialGroup],
        advantages_per_group: list[list[float]],
) -> list[tinker.Datum]:
    """Build tinker.Datum directly from TrialResult rollout_details."""
    datums: list[tinker.Datum] = []

    for group, group_advantages in zip(groups, advantages_per_group):
        for result, advantage in zip(group, group_advantages):
            if not result.rollout_details:
                continue

            rd = result.rollout_details[0]  # Main agent rollout
            prompt_tokens = rd.get("prompt_token_ids", [])
            completion_tokens = rd.get("completion_token_ids", [])
            logprobs = rd.get("logprobs", [])

            n_turns = len(completion_tokens)
            if n_turns == 0:
                continue

            # Normalize advantage by turn count so each episode contributes
            # equally regardless of how many turns it took
            normalized_advantage = advantage / n_turns

            # Build one datum per turn
            for turn_idx in range(n_turns):
                if turn_idx >= len(prompt_tokens):
                    continue
                if turn_idx >= len(logprobs):
                    continue

                turn_prompt = prompt_tokens[turn_idx]
                turn_completion = completion_tokens[turn_idx]
                turn_logprobs = logprobs[turn_idx]

                if not turn_completion or not turn_logprobs:
                    continue

                token_advantages = [normalized_advantage] * len(turn_completion)

                datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens=turn_prompt),
                    loss_fn_inputs={
                        "target_tokens": turn_completion,
                        "logprobs": turn_logprobs,
                        "advantages": token_advantages,
                    },
                )
                datums.append(datum)

    return datums


def compute_batch_metrics(groups: list[TrialGroup]) -> Metrics:
    """Compute aggregate metrics for a batch of trial groups."""
    if not groups:
        return {"error": "No valid groups"}

    all_results = [r for g in groups for r in g]
    if not all_results:
        return {"error": "No valid results"}

    rewards = [extract_reward(r.verifier_result) for r in all_results]
    successes = [1 if r > 0.5 else 0 for r in rewards]

    n_turns_list = []
    for r in all_results:
        if r.rollout_details:
            n_turns_list.append(len(r.rollout_details[0].get("completion_token_ids", [])))

    return {
        "n_groups": len(groups),
        "n_episodes": len(all_results),
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate": sum(successes) / len(successes),
        "mean_turns": sum(n_turns_list) / len(n_turns_list) if n_turns_list else 0,
    }


class Terminus2RLTrainer:
    """
    RL trainer using Harbor's Trial infrastructure directly.
    
    Uses TrialResult.rollout_details - no intermediate types needed.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self._service_client: tinker.ServiceClient | None = None
        self._training_client: tinker.TrainingClient | None = None
        self._sampling_client: tinker.SamplingClient | None = None
        self._tokenizer = None
        self._renderer = None
        self._batch_count = 0
        self._ml_logger: ml_log.Logger | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def setup(self) -> None:
        """Initialize Tinker clients."""
        self._service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)

        if hasattr(self._service_client, "create_lora_training_client_async"):
            self._training_client = await self._service_client.create_lora_training_client_async(
                base_model=self.config.model_name,
                rank=self.config.lora_rank,
            )
        else:
            self._training_client = await asyncio.to_thread(
                self._service_client.create_lora_training_client,
                self.config.model_name,
                self.config.lora_rank,
            )

        self._tokenizer = self._training_client.get_tokenizer()
        renderer_name = get_recommended_renderer_name(self.config.model_name)
        self._renderer = get_renderer(renderer_name, self._tokenizer)

        self._sampling_client = await self._training_client.save_weights_and_get_sampling_client_async(
            name="initial"
        )

        self._semaphore = asyncio.Semaphore(self.config.n_parallel_envs)
        self._batch_count = 0
        logger.info(f"Initialized Terminus2RLTrainer with model {self.config.model_name}")

    def _create_llm(self) -> TinkerLLM:
        """Factory function to create TinkerLLM instances."""
        if self._sampling_client is None or self._tokenizer is None or self._renderer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        return TinkerLLM(
            sampling_client=self._sampling_client,
            tokenizer=self._tokenizer,
            renderer=self._renderer,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            context_limit=self.config.context_limit,
        )

    def _create_trial_config(self, task: Task) -> TrialConfig:
        """Create TrialConfig for a task, injecting LLM."""
        env_type = EnvironmentType(self.config.environment_type)
        return TrialConfig(
            task=TaskConfig(path=task.task_dir),
            trials_dir=self.config.logs_dir,
            agent=AgentConfig(
                name="terminus-2",
                model_name=self.config.model_name,
                kwargs={
                    "llm": self._create_llm(),
                    "collect_rollout_details": True,
                    "enable_summarize": False,
                    "max_turns": self.config.max_turns,
                },
            ),
            environment=EnvironmentConfig(
                type=env_type,
                delete=True,
                kwargs=self.config.environment_kwargs,
            ),
        )

    async def _run_trial(self, task: Task) -> TrialResult | None:
        """Run a single trial with semaphore limiting."""
        if self._semaphore is None:
            raise RuntimeError("Trainer not initialized")

        async with self._semaphore:
            try:
                trial = Trial(self._create_trial_config(task))
                if self.config.trial_timeout_sec is not None:
                    return await asyncio.wait_for(
                        trial.run(), timeout=self.config.trial_timeout_sec
                    )
                return await trial.run()
            except asyncio.TimeoutError:
                logger.error(
                    f"Trial timed out for {task.task_id} after "
                    f"{self.config.trial_timeout_sec} seconds"
                )
                return None
            except Exception as e:
                logger.error(f"Trial failed for {task.task_id}: {e}", exc_info=True)
                return None

    async def _run_group(self, task: Task) -> TrialGroup:
        """Run multiple trials for same task (GRPO grouping)."""
        results = await asyncio.gather(
            *[self._run_trial(task) for _ in range(self.config.group_size)],
            return_exceptions=True,
        )

        valid: TrialGroup = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Trial exception for {task.task_id}: {r}")
            elif r is None:
                logger.warning(f"Trial returned None for {task.task_id}")
            elif not r.rollout_details:
                logger.warning(f"No rollout details for {task.task_id}")
            else:
                valid.append(r)

        return valid

    async def _run_batch(self, tasks: list[Task]) -> list[TrialGroup]:
        """Run trials for a batch of tasks."""
        groups = await asyncio.gather(
            *[self._run_group(task) for task in tasks],
            return_exceptions=True,
        )

        valid_groups: list[TrialGroup] = []
        for task, group in zip(tasks, groups):
            if isinstance(group, Exception):
                logger.error(f"Group failed for {task.task_id}: {group}")
            elif not group:
                logger.warning(f"All trials failed for {task.task_id}")
            else:
                valid_groups.append(group)

        return valid_groups

    async def _forward_backward(self, data: list[tinker.Datum]) -> list[torch.Tensor]:
        """Accumulate gradients on a minibatch."""
        if self._training_client is None:
            raise RuntimeError("Trainer not initialized")
        return await cookbook_forward_backward(
            self._training_client, data, self.config.loss_fn
        )

    async def _optim_step(self) -> None:
        """Apply accumulated gradients."""
        if self._training_client is None:
            raise RuntimeError("Trainer not initialized")
        await cookbook_optim_step(self._training_client, self.config.learning_rate)

    async def train_batch(self, tasks: list[Task]) -> dict[str, Any]:
        """Train on a batch of tasks."""
        # Step 1: Collect trials using Harbor's Trial
        groups = await self._run_batch(tasks)

        if not groups:
            return {"error": "All trial groups failed"}

        # Optionally remove constant reward groups
        if self.config.remove_constant_reward_groups:
            groups = [
                g for g in groups
                if len(set(extract_reward(r.verifier_result) for r in g)) > 1
            ]
            if not groups:
                return {"error": "All groups had constant rewards"}

        # Get metrics after filtering
        episode_metrics = compute_batch_metrics(groups)

        # Step 2: Compute GRPO advantages
        advantages = compute_grpo_advantages(groups)

        # Step 3: Build Datums from TrialResult
        datums = build_datums_from_trials(groups, advantages)

        if not datums:
            return {"error": "No training data generated", **episode_metrics}

        # Step 4: Train via Tinker
        minibatches = split_list(datums, min(self.config.num_substeps, len(datums)))

        for batch in minibatches:
            await self._forward_backward(batch)
        await self._optim_step()

        if self._training_client is None:
            raise RuntimeError("Trainer not initialized")

        checkpoint_due = (self._batch_count + 1) % self.config.save_every == 0
        checkpoint_name = (
            f"checkpoint_{self._batch_count + 1:06d}" if checkpoint_due else "latest"
        )

        self._sampling_client = (
            await self._training_client.save_weights_and_get_sampling_client_async(
                name=checkpoint_name
            )
        )
        self._batch_count += 1

        return {
            **episode_metrics,
            "n_datums": len(datums),
            "n_minibatches": len(minibatches),
            "loss_fn": self.config.loss_fn,
        }

    async def train(self) -> None:
        """Main training loop."""
        await self.setup()

        self._ml_logger = ml_log.setup_logging(
            log_dir=str(self.config.logs_dir),
            wandb_project=self.config.wandb_project,
            wandb_name=self.config.wandb_name,
            config=self.config,
        )

        tasks = self._load_tasks()
        logger.info(f"Loaded {len(tasks)} tasks from {self.config.tasks_dir}")

        if not tasks:
            logger.error("No tasks found")
            if self._ml_logger:
                self._ml_logger.close()
            return

        for epoch in range(self.config.n_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.n_epochs}")
            random.shuffle(tasks)

            for i in range(0, len(tasks), self.config.batch_size):
                batch_tasks = tasks[i: i + self.config.batch_size]
                batch_num = i // self.config.batch_size + 1
                total_batches = (len(tasks) + self.config.batch_size - 1) // self.config.batch_size

                logger.info(f"  Batch {batch_num}/{total_batches}")

                metrics = await self.train_batch(batch_tasks)
                metrics["epoch"] = epoch + 1
                metrics["batch"] = batch_num

                logger.info(f"    Metrics: {metrics}")

                if self._ml_logger:
                    self._ml_logger.log_metrics(metrics, step=self._batch_count)

        if self._ml_logger:
            self._ml_logger.close()

    def _load_tasks(self) -> list[Task]:
        """Load tasks from tasks_dir."""
        tasks = []
        for task_dir in self.config.tasks_dir.iterdir():
            if task_dir.is_dir() and (task_dir / "task.toml").exists():
                try:
                    tasks.append(Task(task_dir=task_dir))
                except Exception as e:
                    logger.warning(f"Failed to load task from {task_dir}: {e}")
        return tasks
