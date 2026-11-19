"""Training package — YOLO training orchestration."""

from alpr.training.config import TrainingConfig
from alpr.training.trainer import kaggle_setup, train

__all__ = ["TrainingConfig", "kaggle_setup", "train"]
