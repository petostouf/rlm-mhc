# RLM-mHC-1B

**Recursive Language Model with Manifold-Constrained Hyper-Connections**

A research implementation of a transformer-based language model with manifold-constrained hyper-connections (mHC) for improved representation learning and training stability.

## Features

- **mHC Layers**: Novel hyper-connection mechanism with doubly stochastic connection matrices
- **Sinkhorn-Knopp Algorithm**: Numerically stable projection onto the Birkhoff polytope
- **RoPE**: Rotary Position Embeddings for efficient position encoding
- **Flash Attention 2**: Optional integration for faster attention computation
- **Mixed Precision**: Support for bf16/fp16 training
- **Gradient Checkpointing**: Memory-efficient training for large models

## Architecture

```
Input → Embedding → [TransformerBlock × N] → LayerNorm → LM Head → Output

TransformerBlock (with mHC):
  x → RMSNorm → mHC_pre → (+x) → RMSNorm → Attention → (+x)
    → RMSNorm → mHC_post → (+x) → RMSNorm → FFN → (+x) → output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/petostouf/rlm-mhc.git
cd rlm-mhc

# Install in development mode
pip install -e .[dev]

# Optional: Install Flash Attention for faster training
pip install flash-attn --no-build-isolation
```

## Quick Start

### Validate Installation

```bash
# Run tests
pytest tests/ -v

# Validate model functionality
python scripts/validate_model.py
```

### Basic Training

```bash
# Train with synthetic data (test run)
python scripts/train.py \
    model.hidden_dim=256 \
    model.num_layers=4 \
    training.max_steps=100

# Train with real data
python scripts/train.py data.train_path=/path/to/train.jsonl
```

### Model Usage

```python
from rlm_mhc.model import RLMModel
from rlm_mhc.model.config import ModelConfig

# Create model
config = ModelConfig(
    hidden_dim=2048,
    num_layers=24,
    num_heads=16,
    mhc_enabled=True,
)
model = RLMModel(config)

# Forward pass
import torch
input_ids = torch.randint(0, 32000, (1, 128))
output = model(input_ids)
print(output.logits.shape)  # [1, 128, 32000]

# Generate text
generated = model.generate(input_ids, max_new_tokens=50)
```

## Model Configurations

| Model | Params | Layers | Hidden | Heads | mHC Flows |
|-------|--------|--------|--------|-------|-----------|
| Small | 125M | 12 | 768 | 12 | 4 |
| Base | 1.3B | 24 | 2048 | 16 | 4 |
| Large | 7B | 32 | 4096 | 32 | 4 |

## Documentation

- **[Training Guide](docs/TRAINING.md)**: Complete guide for training the model
- **[API Reference](docs/API.md)**: Detailed API documentation (coming soon)

## Project Structure

```
rlm-mhc/
├── configs/                 # Hydra configuration files
│   ├── config.yaml         # Main config
│   ├── model/              # Model architectures
│   └── training/           # Training hyperparameters
├── docs/                    # Documentation
│   └── TRAINING.md         # Training guide
├── scripts/
│   ├── train.py            # Training entry point
│   └── validate_model.py   # Model validation
├── src/rlm_mhc/
│   ├── model/
│   │   ├── transformer.py  # Main model
│   │   ├── attention.py    # Multi-head attention
│   │   ├── components.py   # RMSNorm, SwiGLU, RoPE
│   │   └── mhc/            # mHC layers
│   │       ├── layers.py   # mHCLayer, mHCBlock
│   │       └── sinkhorn.py # Sinkhorn-Knopp algorithm
│   ├── training/           # Training utilities
│   └── scaffold/           # REPL scaffold for RLM
└── tests/                   # Unit and integration tests
```

## Key Components

### mHC Layer

The mHC layer extends the residual stream with multiple parallel flows connected by a learnable doubly stochastic matrix:

```python
from rlm_mhc.model.mhc import mHCLayer

layer = mHCLayer(
    hidden_dim=2048,
    num_flows=4,
    sinkhorn_iters=20,
)

# Get connection matrix (doubly stochastic)
P = layer.get_connection_matrix()
print(P.sum(dim=0))  # [1, 1, 1, 1]
print(P.sum(dim=1))  # [1, 1, 1, 1]
```

### Sinkhorn-Knopp Algorithm

Projects arbitrary matrices onto the set of doubly stochastic matrices:

```python
from rlm_mhc.model.mhc.sinkhorn import sinkhorn_knopp

W = torch.randn(4, 4)
P = sinkhorn_knopp(W, n_iters=20)
# P is now doubly stochastic
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_mhc.py -v

# Run with coverage
pytest tests/ --cov=src/rlm_mhc
```

## Performance

| Hardware | Model | Batch Size | Tokens/sec |
|----------|-------|------------|------------|
| RTX 3090 | 125M | 32 | ~50k |
| A100-40GB | 1.3B | 8 | ~15k |
| A100-80GB | 1.3B | 16 | ~25k |

*With gradient checkpointing and bf16*

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rlm_mhc,
  title = {RLM-mHC: Recursive Language Model with Manifold-Constrained Hyper-Connections},
  author = {petostouf},
  year = {2024},
  url = {https://github.com/petostouf/rlm-mhc}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Architecture inspired by LLaMA and Hyena
- mHC concept based on manifold learning principles
- Built with PyTorch and Hugging Face Transformers
