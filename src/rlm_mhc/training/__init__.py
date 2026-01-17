"""
RLM-mHC Training Module.

Provides the training pipeline including:
- RLMTrainer: Main training loop with mHC monitoring
- Data pipeline: Dataset, collator, synthetic data generation
- Callbacks: Checkpointing, logging, early stopping
"""

from rlm_mhc.training.trainer import RLMTrainer
from rlm_mhc.training.data import RLMDataset, DataCollator, SyntheticDataGenerator

__all__ = [
    "RLMTrainer",
    "RLMDataset",
    "DataCollator",
    "SyntheticDataGenerator",
]
