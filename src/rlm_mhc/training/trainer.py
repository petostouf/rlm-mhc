"""
RLM Trainer.

Main training loop with:
- Mixed precision (bf16)
- Gradient checkpointing
- mHC metrics monitoring
- W&B integration
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from rlm_mhc.types import TrainingConfig, TrainingResult
from rlm_mhc.model.mhc import mHCLayer

# Try importing wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class RLMTrainer:
    """
    Trainer for RLM-mHC models.

    Features:
    - Mixed precision training (bf16)
    - Gradient checkpointing for memory efficiency
    - mHC-specific metrics (Amax Gain)
    - W&B logging integration
    - Gradient clipping and accumulation

    Args:
        model: RLM model to train
        train_dataset: Training dataset
        config: Training configuration
        eval_dataset: Optional evaluation dataset
        tokenizer: Tokenizer (for collator)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        config: TrainingConfig,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.tokenizer = tokenizer

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_mixed_precision()
        self._setup_gradient_checkpointing()
        self._setup_logging()

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            # Cosine with warmup
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / max(1, self.config.warmup_steps)
                progress = (step - self.config.warmup_steps) / max(
                    1, self.config.max_steps - self.config.warmup_steps
                )
                return max(
                    self.config.min_lr_ratio,
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            # Linear with warmup
            def lr_lambda(step):
                if step < self.config.warmup_steps:
                    return step / max(1, self.config.warmup_steps)
                return max(
                    self.config.min_lr_ratio,
                    1.0 - (step - self.config.warmup_steps) / (
                        self.config.max_steps - self.config.warmup_steps
                    )
                )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
            self.use_amp = True
        elif self.config.mixed_precision == "fp16":
            self.dtype = torch.float16
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.dtype = torch.float32
            self.use_amp = False

    def _setup_gradient_checkpointing(self):
        """Enable gradient checkpointing if configured."""
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

    def _setup_logging(self):
        """Setup W&B logging."""
        self.use_wandb = WANDB_AVAILABLE and self.config.wandb_project

        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config={
                    'learning_rate': self.config.learning_rate,
                    'batch_size': self.config.batch_size,
                    'max_steps': self.config.max_steps,
                    'gradient_accumulation': self.config.gradient_accumulation_steps,
                    'mixed_precision': self.config.mixed_precision,
                },
            )

    def _create_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        from rlm_mhc.training.data import DataCollator

        collator = DataCollator(
            tokenizer=self.tokenizer,
            max_length=4096,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collator,
            pin_memory=True,
        )

    def train(self) -> TrainingResult:
        """
        Main training loop.

        Returns:
            TrainingResult with final metrics
        """
        self.model.train()
        dataloader = self._create_dataloader()

        # Metrics
        total_loss = 0.0
        num_steps = 0
        checkpoints = []

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                loss = self._training_step(batch)
                total_loss += loss
                num_steps += 1

                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(loss)

                # mHC monitoring
                if self.global_step % self.config.mhc_amax_log_steps == 0:
                    self._log_mhc_metrics()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    ckpt_path = self._save_checkpoint()
                    checkpoints.append(ckpt_path)

                self.global_step += 1

                if self.global_step >= self.config.max_steps:
                    break

            if self.global_step >= self.config.max_steps:
                break

        # Final checkpoint
        final_ckpt = self._save_checkpoint()
        checkpoints.append(final_ckpt)

        # Training time
        elapsed = time.time() - start_time

        if self.use_wandb:
            wandb.finish()

        return TrainingResult(
            final_loss=total_loss / max(1, num_steps),
            total_steps=self.global_step,
            checkpoints=checkpoints,
            metrics={
                'elapsed_time': elapsed,
                'tokens_per_second': (
                    self.global_step * self.config.batch_size * 4096 / elapsed
                ),
            },
        )

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        if self.use_amp:
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
        else:
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps

        # Backward
        if self.config.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.config.mixed_precision == "fp16":
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        if self.config.mixed_precision == "fp16":
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    def _log_metrics(self, loss: float):
        """Log training metrics."""
        lr = self.scheduler.get_last_lr()[0]

        metrics = {
            'train/loss': loss,
            'train/lr': lr,
            'train/step': self.global_step,
            'train/epoch': self.epoch,
        }

        # Gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        metrics['train/grad_norm'] = total_norm

        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        # Console logging
        print(
            f"Step {self.global_step}: loss={loss:.4f}, lr={lr:.2e}, "
            f"grad_norm={total_norm:.4f}"
        )

    def _log_mhc_metrics(self):
        """Log mHC-specific metrics (Amax Gain)."""
        amax_values = []

        for name, module in self.model.named_modules():
            if isinstance(module, mHCLayer):
                amax = module.amax_gain
                amax_values.append(amax)

                if self.use_wandb:
                    wandb.log({
                        f'mhc/{name}/amax': amax,
                    }, step=self.global_step)

        if amax_values:
            mean_amax = sum(amax_values) / len(amax_values)
            max_amax = max(amax_values)

            if self.use_wandb:
                wandb.log({
                    'mhc/amax_mean': mean_amax,
                    'mhc/amax_max': max_amax,
                }, step=self.global_step)

            print(f"  mHC Amax: mean={mean_amax:.4f}, max={max_amax:.4f}")

    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = output_dir / f"checkpoint-{self.global_step}"
        ckpt_dir.mkdir(exist_ok=True)

        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(ckpt_dir)
        else:
            torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")

        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
        }, ckpt_dir / "training_state.bin")

        # Manage checkpoint limit
        self._cleanup_checkpoints(output_dir)

        print(f"Saved checkpoint to {ckpt_dir}")
        return str(ckpt_dir)

    def _cleanup_checkpoints(self, output_dir: Path):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1]),
        )

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on eval dataset."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()

        from rlm_mhc.training.data import DataCollator
        collator = DataCollator(tokenizer=self.tokenizer)

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type='cuda', dtype=self.dtype):
                    outputs = self.model(**batch)

                total_loss += outputs.loss.item()
                num_batches += 1

        self.model.train()

        metrics = {
            'eval/loss': total_loss / max(1, num_batches),
            'eval/perplexity': math.exp(total_loss / max(1, num_batches)),
        }

        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        return metrics
