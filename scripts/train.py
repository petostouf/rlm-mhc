#!/usr/bin/env python3
"""
RLM-mHC Training Script.

Entry point for training RLM-mHC models.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py model=base_1b training=finetune
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from rlm_mhc.model import RLMModel
from rlm_mhc.model.config import ModelConfig
from rlm_mhc.training import RLMTrainer, RLMDataset, SyntheticDataGenerator
from rlm_mhc.training.data.synthetic import SyntheticConfig
from rlm_mhc.types import TrainingConfig


def create_model(cfg: DictConfig) -> RLMModel:
    """Create model from config."""
    model_cfg = ModelConfig(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        head_dim=cfg.model.head_dim,
        ffn_dim=cfg.model.ffn_dim,
        vocab_size=cfg.model.vocab_size,
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        mhc_enabled=cfg.model.mhc_enabled,
        mhc_flows=cfg.model.mhc_flows,
        mhc_sinkhorn_iters=cfg.model.mhc_sinkhorn_iters,
        rope_theta=cfg.model.rope_theta,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
    )

    print(f"Creating model with {model_cfg.num_parameters() / 1e9:.2f}B parameters")

    return RLMModel(model_cfg)


def create_training_config(cfg: DictConfig) -> TrainingConfig:
    """Create training config from Hydra config."""
    return TrainingConfig(
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        betas=tuple(cfg.training.optimizer.betas),
        eps=cfg.training.optimizer.eps,
        max_grad_norm=cfg.training.max_grad_norm,
        scheduler_type=cfg.training.scheduler.type,
        warmup_steps=cfg.training.scheduler.warmup_steps,
        min_lr_ratio=cfg.training.scheduler.min_lr_ratio,
        batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_steps=cfg.training.max_steps,
        num_epochs=cfg.training.num_epochs,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        mixed_precision=cfg.training.mixed_precision,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        output_dir=cfg.training.output_dir,
        logging_steps=cfg.training.logging_steps,
        wandb_project=cfg.training.wandb.project,
        wandb_entity=cfg.training.wandb.entity,
        mhc_amax_log_steps=cfg.mhc.amax_log_steps,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training entry point."""
    print("=" * 60)
    print("RLM-mHC Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create or load dataset
    data_path = cfg.data.get('train_path')
    if data_path and Path(data_path).exists():
        print(f"Loading dataset from {data_path}")
        train_dataset = RLMDataset(
            path=data_path,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )
    else:
        print("Generating synthetic dataset...")
        generator = SyntheticDataGenerator(
            config=SyntheticConfig(num_examples=1000),
            tokenizer=tokenizer,
        )
        # Generate some sample documents
        generator.add_documents([
            "This is a sample document about machine learning. " * 100,
            "The quick brown fox jumps over the lazy dog. " * 100,
            "Artificial intelligence is transforming the world. " * 100,
        ])

        # Save synthetic data
        synthetic_path = Path(cfg.output_dir) / "synthetic_data.jsonl"
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_dataset(synthetic_path)

        train_dataset = RLMDataset(
            path=synthetic_path,
            tokenizer=tokenizer,
            max_length=cfg.data.max_length,
        )

    print(f"Dataset size: {len(train_dataset)} examples")

    # Create model
    model = create_model(cfg)

    # Create training config
    training_config = create_training_config(cfg)

    # Create trainer
    trainer = RLMTrainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting training...")
    result = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Total steps: {result.total_steps}")
    print(f"Checkpoints: {result.checkpoints}")


if __name__ == "__main__":
    main()
