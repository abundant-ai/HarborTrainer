"""
Terminal Bench Trainer

Post-training LLMs on terminal benchmarks using the Tinker API.
"""

__version__ = "0.1.0"

from .terminus2_trainer import Terminus2RLTrainer, TrainerConfig
from .tinker_llm import TinkerLLM, LogprobsMissingError
