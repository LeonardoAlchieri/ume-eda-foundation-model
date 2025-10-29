# Foundation Models for Electrodermal Activity (EDA) Signals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Foundation models for electrodermal activity analysis using self-supervised learning**

This repository contains the official implementation of self-supervised learning methods for EDA signal analysis, supporting both contrastive learning and reconstruction-based approaches.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference API](#inference-api)
- [Configuration](#configuration)
- [Data](#data)
- [Models](#models)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This codebase provides a modular framework for self-supervised learning on EDA signals with:

- **Multiple Learning Paradigms**: Contrastive learning (SimCLR-style) and reconstruction-based learning
- **Extensive Augmentations**: 16 EDA-specific data augmentation strategies
- **Flexible Architecture**: Modular encoder-decoder architectures with configurable components
- **Production-Ready**: Built on PyTorch Lightning with Hydra configuration management
- **Reproducible**: Comprehensive testing, logging, and seed management

### Key Features

✅ **Self-Supervised Learning Engines**
- Contrastive learning with multiple views
- Reconstruction-based learning with augmentation
- Extensible base class for custom methods

✅ **EDA-Specific Augmentations**
- Temporal transformations (warping, shifting, cutout)
- Frequency-domain modifications (filtering, noise injection)
- Amplitude manipulations (scaling, warping)
- Artifact simulation (sensor artifacts, jumps)

✅ **Flexible Data Loading**
- Multi-dataset support from `.npy` files
- Channel selection and preprocessing
- Train/validation/test splitting

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (optional, for GPU support)
- 8GB+ RAM (16GB+ recommended for large datasets)

### Environment Setup

We recommend using `mamba` or `conda` for environment management:

```bash
# Clone the repository
git clone https://github.com/yourusername/eda-foundation-models.git
cd eda-foundation-models

# Create and activate environment
mamba create -n eda-foundation-models python=3.8
mamba activate eda-foundation-models

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core dependencies include:
```
pytorch>=2.0.0
pytorch-lightning>=2.0.0
hydra-core>=1.3.0
torchmetrics>=0.11.0
numpy>=1.23.0
scipy>=1.10.0
scikit-learn>=1.2.0
```

For development and testing:
```bash
pip install -r requirements-dev.txt  # includes pytest, black, etc.
```

### Verification

Verify your installation:

```bash
python verify_setup.py
```

This will check all dependencies and attempt to instantiate the models and engines.

---

## 🚀 Quick Start

### 1. Prepare Your Data

Organize your EDA data as `.npy` files:

```
data/
├── USILaughs/
│   ├── data_001.npy  # Shape: (T, C) or (T,)
│   ├── data_002.npy
│   └── ...
├── WESAD/
│   ├── data_001.npy
│   └── ...
└── ...
```

### 2. Train a Contrastive Model

```bash
python train.py --config-name=example_contrastive
```

### 3. Train a Reconstruction Model

```bash
python train.py --config-name=example_reconstruction
```

### 4. Override Configuration

```bash
# Change batch size and learning rate
python train.py --config-name=example_contrastive \
    dataloader.batch_size=128 \
    engine.optimizer_kwargs.lr=0.0001

# Use specific datasets
python train.py --config-name=example_contrastive \
    dataset.dataset_names='[USILaughs,WESAD]'
```

---

## 📁 Repository Structure

```
eda-foundation-models/
├── configs/                    # Hydra configuration files
│   └── train/
│       ├── config.yaml        # Base configuration
│       ├── example_contrastive.yaml
│       ├── example_reconstruction.yaml
│       ├── callbacks/         # Training callbacks
│       ├── data_augmentation/ # Augmentation configs
│       ├── dataloader/        # DataLoader configs
│       ├── dataset/           # Dataset configs
│       ├── engine/            # Training engine configs
│       ├── logger/            # Logger configs
│       └── model/             # Model architecture configs
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset implementation
│   │   └── __init__.py
│   ├── data_augmentation/
│   │   ├── matton_augment.py # EDA augmentations
│   │   └── __init__.py
│   ├── engine/
│   │   └── __init__.py       # Training engines
│   ├── model/
│   │   ├── encoder_decoder.py
│   │   ├── matton_constrastive.py
│   │   ├── loss/             # Loss functions
│   │   └── nn/               # Neural network modules
│   └── utils/
│       └── data.py           # Utility functions
├── tests/                     # Comprehensive test suite
│   ├── test_engine.py
│   ├── test_dataset.py
│   ├── test_data_augmentation.py
│   └── README.md
├── train.py                   # Main training script
├── verify_setup.py           # Installation verification
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                 # This file
```

---

## 🎓 Training

### Training Engines

The codebase provides two main training engines:

#### 1. **ContrastiveEngine**

Implements contrastive learning by creating two augmented views of each sample:

```python
# Input: batch['x'] with shape (N, T) or (N, T, C)
# Output: batch['x_1'], batch['x_2'] (augmented views)
```

**Configuration**: [`configs/train/engine/contrastive.yaml`](configs/train/engine/contrastive.yaml)

Key parameters:
- `optimizer`: Adam, SGD, AdamW, etc.
- `optimizer_kwargs`: Learning rate, weight decay, etc.
- `scheduler`: Optional LR scheduler
- `data_augmentation`: Compose of augmentation strategies

#### 2. **ReconstructionEngine**

Implements reconstruction-based learning with optional augmentation:

```python
# Input: batch['x'] with shape (N, T) or (N, T, C)
# Output: batch['x_aug'] (augmented version for reconstruction)
```

**Configuration**: [`configs/train/engine/reconstruction.yaml`](configs/train/engine/reconstruction.yaml)

### Training Configuration

All training is configured via Hydra. Key configuration groups:

| Group | Purpose | Example Files |
|-------|---------|---------------|
| `model` | Network architecture | `contrastive_mutton.yaml`, `encoder_decoder.yaml` |
| `engine` | Training logic | `contrastive.yaml`, `reconstruction.yaml` |
| `dataset` | Data loading | `eda_default.yaml` |
| `dataloader` | DataLoader settings | `default.yaml` |
| `logger` | Logging backend | `tensorboard.yaml`, `wandb.yaml` |
| `callbacks` | Training callbacks | `default.yaml`, `minimal.yaml` |

### Example Training Commands

**Basic contrastive learning:**
```bash
python train.py --config-name=example_contrastive
```

**Custom configuration:**
```bash
python train.py --config-name=example_contrastive \
    model=contrastive_mutton \
    engine=contrastive \
    dataset=eda_default \
    logger=wandb \
    trainer.max_epochs=100 \
    dataloader.batch_size=256
```

**Multi-GPU training:**
```bash
python train.py --config-name=example_contrastive \
    trainer.devices=4 \
    trainer.strategy=ddp
```

**Mixed precision training:**
```bash
python train.py --config-name=example_contrastive \
    trainer.precision=16-mixed
```

### Monitoring Training

Logs are saved to `outputs/YYYY-MM-DD/HH-MM-SS/`:
```
outputs/2025-10-22/11-30-17/
├── .hydra/           # Hydra configuration snapshots
├── checkpoints/      # Model checkpoints
├── train.log        # Training logs
└── tensorboard/     # TensorBoard logs (if enabled)
```

**View with TensorBoard:**
```bash
tensorboard --logdir=outputs/
```

**View with Weights & Biases:**
Configure in [`configs/train/logger/wandb.yaml`](configs/train/logger/wandb.yaml) and logs will automatically sync.

---

## 📊 Evaluation

### Loading Trained Models

```python
from src.engine import ContrastiveEngine
import torch

# Load checkpoint
engine = ContrastiveEngine.load_from_checkpoint(
    "checkpoints/epoch=99-val_loss=0.1234.ckpt"
)
engine.eval()

# Extract embeddings
with torch.no_grad():
    embeddings = engine.model.encode(data)
```

### Testing

Run tests on the test set:

```bash
python train.py --config-name=example_contrastive \
    trainer.fast_dev_run=false
```

The trainer will automatically evaluate on the test set after training.

---

## 🚀 Inference API

### EdamamePipeline: Easy Model Inference

The `EdamamePipeline` provides a simple and intuitive API for loading pretrained models and extracting embeddings from time series data.

#### Quick Start

```python
from src.pipeline import EdamamePipeline
import torch

# Load a pretrained model
pipeline = EdamamePipeline.from_pretrained(
    model_name="mae",
    weights_path="checkpoints/best.ckpt",
    model_config="configs/train/model/mae.yaml",
    device_map="auto",  # Automatically use GPU if available
)

# Prepare your data (batch_size, seq_len, in_channels)
batch_data = torch.randn(8, 240, 1)

# Extract embeddings
embeddings = pipeline.embed(batch_data)
print(f"Embeddings shape: {embeddings.shape}")
# Output: torch.Size([8, 24, 512])

# Access single sample
single_embedding = pipeline.embed(batch_data)[0]
print(f"Single embedding: {single_embedding.shape}")
```

#### Loading Models

The API supports multiple ways to load models:

```python
# Method 1: With YAML config file
pipeline = EdamamePipeline.from_pretrained(
    model_name="mae",
    weights_path="checkpoints/best.ckpt",
    model_config="configs/train/model/mae.yaml",
    device_map="auto",
)

# Method 2: With config dictionary
config = {
    "seq_len": 240,
    "in_channels": 1,
    "patch_size": 10,
    "embed_dim": 512,
    # ... other parameters
}
pipeline = EdamamePipeline.from_pretrained(
    model_name="mae",
    weights_path="checkpoints/best.ckpt",
    model_config=config,
)

# Method 3: Auto-infer from checkpoint
pipeline = EdamamePipeline.from_pretrained(
    model_name="mae",
    weights_path="checkpoints/best.ckpt",
    model_config=None,  # Will infer from checkpoint
)
```

#### Embedding Extraction

```python
# Extract embeddings without masking (full signal)
embeddings = pipeline.embed(batch_data, mask_ratio=0.0)

# Extract embeddings with masking (like during training)
embeddings_masked = pipeline.embed(batch_data, mask_ratio=0.75)

# Get embeddings with mask information
embeddings, mask, ids_restore = pipeline.embed(
    batch_data,
    mask_ratio=0.75,
    return_mask=True
)
```

#### Signal Reconstruction

```python
# Reconstruct signal with masking
reconstructed = pipeline.reconstruct(batch_data, mask_ratio=0.75)

# Compute reconstruction error
mse = torch.mean((batch_data - reconstructed) ** 2)
print(f"MSE: {mse.item():.6f}")
```

#### Batch Processing

```python
from torch.utils.data import DataLoader

all_embeddings = []
for batch in dataloader:
    embeddings = pipeline.embed(batch)
    all_embeddings.append(embeddings.cpu())

# Concatenate all embeddings
all_embeddings = torch.cat(all_embeddings, dim=0)
```

#### Device Management

```python
# CPU only
pipeline = EdamamePipeline.from_pretrained(..., device_map="cpu")

# GPU (automatic selection)
pipeline = EdamamePipeline.from_pretrained(..., device_map="auto")

# Specific GPU
pipeline = EdamamePipeline.from_pretrained(..., device_map="cuda:0")

# Half precision for memory efficiency
pipeline = EdamamePipeline.from_pretrained(
    ...,
    device_map="auto",
    torch_dtype=torch.float16
)
```

#### Complete Examples

See the comprehensive examples:
- **API Documentation**: [`src/pipeline/README.md`](src/pipeline/README.md)
- **Example Scripts**: [`examples/pipeline_usage.py`](examples/pipeline_usage.py)
- **Quick Test**: `python test_pipeline_quick.py`

#### Supported Models

Currently supported:
- ✅ **MAE (Masked Autoencoder)**: Self-supervised reconstruction model

Coming soon:
- 🔜 VAE (Variational Autoencoder)
- 🔜 Contrastive models

---

## ⚙️ Configuration

### Hydra Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management, enabling:
- **Composition**: Mix and match configuration groups
- **Override**: Command-line parameter overrides
- **Sweep**: Hyperparameter sweeps for experiments
- **Reproducibility**: Automatic configuration logging

### Configuration Structure

```yaml
# configs/train/example_contrastive.yaml
defaults:
  - model: contrastive_mutton
  - engine: contrastive
  - dataset: eda_default
  - dataloader: default
  - logger: tensorboard
  - callbacks: default
  - _self_

experiment_name: contrastive_eda_experiment
seed: 42
save_final_checkpoint: true

trainer:
  max_epochs: 400
  accelerator: auto
  devices: auto
  precision: "32"
```

### Key Configuration Files

#### Model Configuration
[`configs/train/model/contrastive_mutton.yaml`](configs/train/model/contrastive_mutton.yaml):
```yaml
_target_: src.model.matton_constrastive.ContrastiveModel

encoder_net:
  _target_: src.model.nn.cnn_encoder.CNNEncoder
  input_dim: 120
  output_dim: 64

transform_head:
  _target_: src.model.nn.linear_net.LinearNN
  input_dim: 64
  output_dim: 32

loss_fn:
  _target_: src.model.loss.contrastive.NCELoss
  temperature: 0.1
```

#### Engine Configuration
[`configs/train/engine/contrastive.yaml`](configs/train/engine/contrastive.yaml):
```yaml
_target_: src.engine.ContrastiveEngine

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
optimizer_kwargs:
  lr: 0.001
  weight_decay: 0.01

data_augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
    - ${data_augmentation.amplitude_warping}
    - ${data_augmentation.gaussian_noise}
    # ... (16 augmentations)
```

#### Dataset Configuration
[`configs/train/dataset/eda_default.yaml`](configs/train/dataset/eda_default.yaml):
```yaml
_target_: src.data.dataset.EDAtaNoLabel
base_path: ./data.nosync
dataset_names: all  # or ['USILaughs', 'WESAD']
shuffle: true
train_val_test_size: [0.8, 0.1, 0.1]
seed: 42
channels: 0  # Single channel, or [0,1,2] for multiple
```

### Hyperparameter Sweeps

Use Hydra's multirun feature:

```bash
python train.py --multirun \
    engine.optimizer_kwargs.lr=0.001,0.0001,0.00001 \
    dataloader.batch_size=64,128,256
```

---

## 💾 Data

### Data Format

The dataset loader expects `.npy` files with the following format:

**Single-channel data:**
```python
# Shape: (num_samples, time_steps)
data = np.load('data_001.npy')  # e.g., (1000, 240)
```

**Multi-channel data:**
```python
# Shape: (num_samples, time_steps, channels)
data = np.load('data_001.npy')  # e.g., (1000, 240, 3)
```

### Dataset Organization

```
data/
├── USILaughs/
│   ├── data_001.npy
│   ├── data_002.npy
│   └── ...
├── WESAD/
│   ├── data_001.npy
│   └── ...
└── CustomDataset/
    └── data_001.npy
```

### Loading Multiple Datasets

Configure in [`configs/train/dataset/eda_default.yaml`](configs/train/dataset/eda_default.yaml):

```yaml
_target_: src.data.dataset.EDAtaNoLabel
base_path: ./data
dataset_names: 
  - USILaughs
  - WESAD
  - CustomDataset
# Or use 'all' to load all subdirectories
```

### Data Preprocessing

The [`EDAtaNoLabel`](src/data/dataset.py) class handles:
- ✅ Loading and concatenation of multiple `.npy` files
- ✅ Channel selection (single or multiple channels)
- ✅ Train/validation/test splitting
- ✅ Shuffling with reproducible seeds
- ✅ Automatic shape handling

### Channel Selection

```yaml
# Single channel (output shape: N, T)
channels: 0

# Multiple channels (output shape: N, T, C)
channels: [0, 1, 2]

# All channels (output shape: N, T, C if multi-channel)
channels: null
```

---

## 🏗️ Models

### Contrastive Learning Model

**Architecture**: [`src/model/matton_constrastive.py`](src/model/matton_constrastive.py)

```
Input (N, T, C)
    ↓
Encoder Network (CNN/LSTM/Transformer)
    ↓
Latent Representation (N, D)
    ↓
Projection Head (MLP)
    ↓
Projected Embedding (N, D')
    ↓
NCE Loss (between x_1 and x_2)
```

**Components**:
- `encoder_net`: Feature extraction (e.g., CNN, LSTM)
- `transform_head`: Projection head for contrastive learning
- `loss_fn`: Contrastive loss (NCE, InfoNCE, etc.)

### Encoder-Decoder Model

**Architecture**: [`src/model/encoder_decoder.py`](src/model/encoder_decoder.py)

```
Input (N, T, C)
    ↓
Encoder Network
    ↓
Latent Representation (N, D)
    ↓
Bottleneck (optional)
    ↓
Decoder Network
    ↓
Reconstruction (N, T, C)
    ↓
Reconstruction Loss
```

**Components**:
- `encoder_net`: Encoder architecture
- `decoder_net`: Decoder architecture
- `bottleneck`: Optional transformation (e.g., VAE reparameterization)
- `loss_fn`: Reconstruction loss (MSE, MAE, etc.)

### Custom Models

To implement a custom model, extend the base classes:

```python
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define your architecture
        
    def forward(self, batch):
        # Forward pass
        x = batch['x']
        output = self.network(x)
        return output
    
    def loss(self, output, batch, step='train'):
        # Compute loss
        loss = self.criterion(output, batch['target'])
        return loss
```

Then create a configuration file in [`configs/train/model/`](configs/train/model/).

---

## 🔬 Data Augmentation

The codebase includes 16 EDA-specific augmentation strategies from [Matton et al.](https://arxiv.org/abs/2404.08783):

### Temporal Augmentations
1. **TimeWarpingStochastic**: Non-linear time warping using cubic splines
2. **TimeShiftStochastic**: Temporal shifting with wrapping
3. **TemporalCutoutStochastic**: Random temporal masking
4. **PermuteStochastic**: Chunk-based permutation

### Frequency-Domain Augmentations
5. **LowPassFilterStochastic**: Remove high frequencies
6. **HighPassFilterStochastic**: Remove low frequencies
7. **BandpassFilterStochastic**: Keep specific frequency band
8. **BandstopFilterStochastic**: Remove specific frequency band

### Amplitude Augmentations
9. **AmplitudeWarpingStochastic**: Smooth amplitude scaling
10. **ConstantAmplitudeScalingStochastic**: Uniform amplitude scaling
11. **TonicAmplitudeWarpingStochastic**: Tonic component warping
12. **TonicConstantAmplitudeScalingStochastic**: Tonic component scaling

### Noise and Artifacts
13. **GaussianNoiseStochastic**: Additive Gaussian noise
14. **HighFrequencyNoiseStochastic**: High-frequency noise injection
15. **JumpArtifactStochastic**: Simulate sudden jumps
16. **LooseSensorArtifactStochastic**: Simulate sensor disconnection

### Configuration

Augmentations are configured as a `torchvision.transforms.Compose` pipeline:

```yaml
data_augmentation:
  _target_: torchvision.transforms.Compose
  transforms:
    - ${data_augmentation.amplitude_warping}
    - ${data_augmentation.gaussian_noise}
    - ${data_augmentation.time_warping}
    # ... add more augmentations
```

Individual augmentation configs are in [`configs/train/data_augmentation/`](configs/train/data_augmentation/).

### Custom Augmentations

Implement your own augmentation:

```python
class MyCustomAugmentation:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (T,) or (T, C)
        # Apply augmentation
        return augmented_x
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_engine.py -v

# Run specific test
pytest tests/test_engine.py::TestContrastiveEngineAugmentation::test_augmentation_2d_training -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

The test suite includes:

**Engine Tests** ([`tests/test_engine.py`](tests/test_engine.py)):
- ✅ Initialization and configuration
- ✅ Forward pass (training/validation/test)
- ✅ Loss computation
- ✅ Optimizer and scheduler setup
- ✅ Metrics tracking
- ✅ Error handling

**Dataset Tests** ([`tests/test_dataset.py`](tests/test_dataset.py)):
- ✅ Data loading from `.npy` files
- ✅ Multi-dataset concatenation
- ✅ Train/val/test splitting
- ✅ Channel selection
- ✅ Shuffling and reproducibility

**Data Augmentation Tests** ([`tests/test_data_augmentation.py`](tests/test_data_augmentation.py)):
- ✅ 2D input handling (N, T)
- ✅ 3D input handling (N, T, C)
- ✅ Training vs evaluation mode
- ✅ Device preservation
- ✅ Batch independence

### Continuous Integration

Tests are automatically run on push/PR via GitHub Actions (configure in `.github/workflows/tests.yml`).

---

## 🔁 Reproducibility

### Reproducibility Checklist

✅ **Environment Management**
- Conda/Mamba environment specification
- Pinned dependency versions in `requirements.txt`
- CUDA version specification

✅ **Seed Management**
```python
seed_everything(42, workers=True)  # PyTorch Lightning
```

✅ **Configuration Tracking**
- Hydra automatically logs all configurations to `.hydra/` directory
- Model checkpoints include hyperparameters

✅ **Deterministic Operations**
```yaml
trainer:
  deterministic: true  # Enable deterministic algorithms
```

✅ **Version Control**
- Git commit hash logged in experiment outputs
- Configuration snapshots saved with checkpoints

✅ **Hardware Specifications**
```bash
# Log hardware info
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

### Reproducing Results

To reproduce experiments:

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourusername/eda-foundation-models.git
   cd eda-foundation-models
   ```

2. **Set up environment**:
   ```bash
   mamba env create -f environment.yml
   mamba activate eda-foundation-models
   ```

3. **Download data** (if publicly available):
   ```bash
   # Add data download instructions
   ```

4. **Run experiment**:
   ```bash
   python train.py --config-name=example_contrastive seed=42
   ```

5. **Check outputs**:
   ```bash
   # Outputs are in outputs/YYYY-MM-DD/HH-MM-SS/
   tensorboard --logdir=outputs/
   ```

### Expected Results

Training metrics for contrastive learning on EDA data:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Training Loss | 0.5 - 2.0 | Depends on temperature |
| Validation Loss | 0.5 - 2.0 | Should track training |
| Training Time | 2-4 hours | On NVIDIA V100, 400 epochs |

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourpaper2025,
  title={Self-Supervised Learning for Electrodermal Activity Analysis},
  author={Your Name and Collaborators},
  booktitle={Conference Name},
  year={2025}
}
```

### Related Work

This implementation is inspired by and builds upon:

```bibtex
@article{matton2024contrastive,
  title={Contrastive Learning of Electrodermal Activity Representations for Stress Detection},
  author={Matton, Katie and Lewis, Robert and Guttag, John and Picard, Rosalind},
  journal={arXiv preprint arXiv:2404.08783},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/

# Format code
black src/ tests/
```

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints for function arguments and returns
- Add docstrings for all public functions and classes
- Keep functions focused and modular

---

## 🐛 Known Issues and Limitations

- **Memory Usage**: Large datasets may require >16GB RAM
- **GPU Memory**: Batch size may need adjustment for different GPUs
- **Data Format**: Currently only supports `.npy` files
- **Multi-GPU**: DDP strategy requires careful batch size tuning

See [Issues](https://github.com/yourusername/eda-foundation-models/issues) for active bugs and feature requests.

---

## 📞 Contact

For questions or collaboration:

- **Email**: your.email@institution.edu
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/eda-foundation-models/issues)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

## 🙏 Acknowledgments

- [PyTorch Lightning](https://lightning.ai/) for the training framework
- [Hydra](https://hydra.cc/) for configuration management
- [Katie Matton et al.](https://arxiv.org/abs/2404.08783) for EDA augmentation strategies
- All contributors and users of this repository

---

## 📊 Project Status

- ✅ Core functionality complete
- ✅ Comprehensive test suite
- ✅ Documentation complete
- 🚧 Additional model architectures (in progress)
- 🚧 Pre-trained weights (coming soon)
- 📋 Benchmark results (planned)

**Last Updated**: October 22, 2025

---

<p align="center">
  <i>Built with ❤️ for the EDA research community</i>
</p>