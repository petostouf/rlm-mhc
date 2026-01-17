# RLM-mHC Training Guide

Complete guide for training the RLM-mHC (Recursive Language Model with Manifold-Constrained Hyper-Connections) model.

## Table of Contents

1. [Requirements](#requirements)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Monitoring](#monitoring)
7. [Checkpointing](#checkpointing)
8. [Multi-GPU Training](#multi-gpu-training)
9. [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware

| Model Size | Min VRAM | Recommended VRAM | Notes |
|------------|----------|------------------|-------|
| Small (125M) | 8 GB | 16 GB | Testing/development |
| Base (1.3B) | 24 GB | 40 GB | A100-40GB or similar |
| Large (7B) | 80 GB | 2x80 GB | Multi-GPU required |

### Software

```bash
# Python 3.10+
pip install -e .[dev]

# Required packages
pip install torch>=2.0
pip install transformers
pip install safetensors
pip install wandb  # For logging
pip install hydra-core  # For configuration
```

### Optional (Recommended)

```bash
# Flash Attention 2 (significant speedup)
pip install flash-attn --no-build-isolation

# DeepSpeed (for multi-GPU)
pip install deepspeed
```

---

## Quick Start

### 1. Validate Installation

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Run model validation
python scripts/validate_model.py
```

### 2. Train with Synthetic Data (Test Run)

```bash
# Quick test with small model
python scripts/train.py \
    model.hidden_dim=256 \
    model.num_layers=4 \
    model.num_heads=4 \
    training.max_steps=100 \
    training.batch_size=4
```

### 3. Train with Real Data

```bash
# Full training with 1.3B model
python scripts/train.py \
    data.train_path=/path/to/your/data.jsonl \
    training.max_steps=50000
```

---

## Configuration

RLM-mHC uses [Hydra](https://hydra.cc/) for configuration management.

### Config Structure

```
configs/
├── config.yaml           # Main config (defaults + runtime)
├── model/
│   └── base_1b.yaml      # 1.3B model architecture
└── training/
    └── finetune.yaml     # Training hyperparameters
```

### Model Configuration (`configs/model/base_1b.yaml`)

```yaml
model:
  # Core dimensions
  hidden_dim: 2048        # Model width
  num_layers: 24          # Number of transformer blocks
  num_heads: 16           # Attention heads
  head_dim: 128           # Per-head dimension
  ffn_dim: 5461           # FFN intermediate dim (8/3 * hidden_dim)
  vocab_size: 32000       # Vocabulary size
  max_seq_len: 8192       # Maximum sequence length

  # mHC Configuration
  mhc_enabled: true       # Enable mHC layers
  mhc_flows: 4            # Number of parallel flows
  mhc_sinkhorn_iters: 20  # Sinkhorn iterations for doubly stochastic matrix

  # Training optimizations
  gradient_checkpointing: true  # Reduce memory usage
```

### Training Configuration (`configs/training/finetune.yaml`)

```yaml
training:
  # Optimizer
  optimizer:
    type: adamw
    lr: 1e-4              # Learning rate
    weight_decay: 0.1     # L2 regularization
    betas: [0.9, 0.95]    # Adam betas

  # Scheduler
  scheduler:
    type: cosine          # Cosine annealing
    warmup_steps: 1000    # Linear warmup
    min_lr_ratio: 0.1     # Min LR = lr * min_lr_ratio

  # Batch
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
  max_steps: 50000

  # Memory optimization
  mixed_precision: bf16   # Use bfloat16
  gradient_checkpointing: true
  max_grad_norm: 1.0      # Gradient clipping
```

### Override from Command Line

```bash
# Change learning rate
python scripts/train.py training.optimizer.lr=5e-5

# Disable mHC for comparison
python scripts/train.py model.mhc_enabled=false

# Use smaller model for testing
python scripts/train.py \
    model.hidden_dim=512 \
    model.num_layers=8 \
    model.num_heads=8
```

---

## Data Preparation

### Expected Format

Data should be in JSONL format with one example per line:

```jsonl
{"text": "Your training text here..."}
{"text": "Another training example..."}
```

### From HuggingFace Datasets

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Save to JSONL
with open("train_data.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps({"text": example["text"]}) + "\n")
```

### Custom Tokenizer

By default, the training script uses Llama-2 tokenizer. To use a custom tokenizer:

```python
from transformers import AutoTokenizer

# Load your tokenizer
tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Data Preprocessing Tips

1. **Filter short sequences**: Remove examples < 128 tokens
2. **Deduplicate**: Use MinHash or exact match deduplication
3. **Shuffle**: Randomize order before training
4. **Chunk long documents**: Split into max_seq_len chunks with overlap

---

## Training

### Basic Training

```bash
python scripts/train.py data.train_path=/path/to/train.jsonl
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    data.train_path=/path/to/train.jsonl \
    +resume_from_checkpoint=/path/to/checkpoint
```

### Training Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `training.optimizer.lr` | 1e-4 | Learning rate |
| `training.batch_size` | 8 | Per-device batch size |
| `training.gradient_accumulation_steps` | 4 | Gradient accumulation |
| `training.max_steps` | 50000 | Total training steps |
| `training.mixed_precision` | bf16 | Precision (bf16, fp16, fp32) |
| `training.save_steps` | 1000 | Save checkpoint every N steps |
| `training.logging_steps` | 10 | Log metrics every N steps |

---

## Monitoring

### Weights & Biases Integration

```bash
# Login to wandb
wandb login

# Train with logging
python scripts/train.py \
    training.wandb.project=rlm-mhc \
    training.wandb.entity=your-username
```

### Key Metrics to Monitor

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| `train/loss` | Training loss | Decreasing |
| `train/lr` | Learning rate | Per scheduler |
| `mhc/amax_pre` | Pre-attention Amax | < 2.0 |
| `mhc/amax_post` | Post-attention Amax | < 2.0 |
| `grad_norm` | Gradient norm | < 10.0 |

### mHC Stability Monitoring

The Amax metric measures the maximum value in the mHC connection matrices. High values (> 2.0) indicate potential instability:

```python
# Get mHC metrics programmatically
metrics = model.get_mhc_metrics()
for layer_idx, (pre_amax, post_amax) in enumerate(metrics):
    print(f"Layer {layer_idx}: pre={pre_amax:.3f}, post={post_amax:.3f}")
```

---

## Checkpointing

### Checkpoint Structure

```
checkpoints/
├── checkpoint-1000/
│   ├── config.yaml         # Model configuration
│   ├── model.safetensors   # Model weights
│   └── optimizer.pt        # Optimizer state (optional)
├── checkpoint-2000/
└── ...
```

### Load Checkpoint

```python
from rlm_mhc.model import RLMModel

# Load model from checkpoint
model = RLMModel.from_pretrained("checkpoints/checkpoint-1000")

# Continue training or inference
model.eval()
```

### Save Checkpoint Manually

```python
# Save model
model.save_pretrained("my_checkpoint")

# Files created:
# - my_checkpoint/config.yaml
# - my_checkpoint/model.safetensors
```

---

## Multi-GPU Training

### DataParallel (Simple)

```python
import torch.nn as nn

model = RLMModel(config)
model = nn.DataParallel(model)
model = model.cuda()
```

### DistributedDataParallel (Recommended)

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 scripts/train.py \
    training.batch_size=4 \
    data.train_path=/path/to/data.jsonl
```

### DeepSpeed Integration (Coming Soon)

```bash
deepspeed scripts/train.py \
    --deepspeed configs/deepspeed/zero2.json \
    data.train_path=/path/to/data.jsonl
```

---

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size**:
   ```bash
   python scripts/train.py training.batch_size=4
   ```

2. **Enable gradient checkpointing**:
   ```bash
   python scripts/train.py training.gradient_checkpointing=true
   ```

3. **Reduce sequence length**:
   ```bash
   python scripts/train.py data.max_length=2048
   ```

4. **Use smaller model**:
   ```bash
   python scripts/train.py model.hidden_dim=1024 model.num_layers=16
   ```

### NaN Loss

1. **Lower learning rate**:
   ```bash
   python scripts/train.py training.optimizer.lr=5e-5
   ```

2. **Enable gradient clipping** (default is 1.0):
   ```bash
   python scripts/train.py training.max_grad_norm=0.5
   ```

3. **Check mHC stability**: If Amax > 2.0, the model may be unstable.

### Slow Training

1. **Install Flash Attention**:
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Use bf16 instead of fp16**:
   ```bash
   python scripts/train.py training.mixed_precision=bf16
   ```

3. **Increase batch size** (if memory allows):
   ```bash
   python scripts/train.py training.batch_size=16
   ```

### Connection Matrix Issues

If you see warnings about connection matrices:

```python
# Check connection matrix is doubly stochastic
from rlm_mhc.model.mhc import mHCLayer

layer = mHCLayer(hidden_dim=2048, num_flows=4)
P = layer.get_connection_matrix()
print(f"Row sums: {P.sum(dim=1)}")  # Should be ~1.0
print(f"Col sums: {P.sum(dim=0)}")  # Should be ~1.0
```

---

## Example Training Scripts

### Small Model for Testing

```bash
#!/bin/bash
python scripts/train.py \
    model.hidden_dim=256 \
    model.num_layers=4 \
    model.num_heads=4 \
    model.ffn_dim=683 \
    training.batch_size=8 \
    training.max_steps=1000 \
    training.logging_steps=10 \
    training.save_steps=500
```

### Full 1.3B Training

```bash
#!/bin/bash
python scripts/train.py \
    data.train_path=/data/train.jsonl \
    training.batch_size=8 \
    training.gradient_accumulation_steps=8 \
    training.max_steps=100000 \
    training.optimizer.lr=1e-4 \
    training.scheduler.warmup_steps=2000 \
    training.wandb.project=rlm-mhc-1b
```

### Fine-tuning from Checkpoint

```bash
#!/bin/bash
python scripts/train.py \
    data.train_path=/data/finetune.jsonl \
    +resume_from_checkpoint=checkpoints/pretrained \
    training.optimizer.lr=1e-5 \
    training.max_steps=10000
```

---

## Performance Benchmarks

| Model | Hardware | Batch Size | Tokens/sec | Memory |
|-------|----------|------------|------------|--------|
| 125M | RTX 3090 | 32 | ~50k | 8 GB |
| 1.3B | A100-40GB | 8 | ~15k | 35 GB |
| 1.3B | A100-80GB | 16 | ~25k | 65 GB |

*With gradient checkpointing and bf16 enabled*

---

## FAQ

**Q: Can I train without mHC layers?**

A: Yes, set `model.mhc_enabled=false`. This gives a standard transformer.

**Q: What's the recommended learning rate?**

A: Start with 1e-4 for pretraining, 1e-5 for fine-tuning.

**Q: How do I know if training is going well?**

A: Monitor:
- Loss should decrease steadily
- Amax should stay < 2.0
- Gradient norm should be stable (< 10.0)

**Q: Can I use this with my own tokenizer?**

A: Yes, modify the training script to load your tokenizer instead of Llama-2.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/petostouf/rlm-mhc/issues)
- **Documentation**: This guide + inline code comments
