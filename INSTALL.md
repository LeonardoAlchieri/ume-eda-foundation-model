# Installation Guide

## Install from Source

### Option 1: Development Installation (Recommended for Contributors)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/LeonardoAlchieri/eda-foundation-models.git
cd eda-foundation-models
pip install -e .
```

This allows you to modify the code and see changes immediately without reinstalling.

### Option 2: Development with All Dependencies

Install with all optional dependencies (development tools, signal processing, notebooks):

```bash
pip install -e ".[all]"
```

### Option 3: Standard Installation

Install the package directly from source:

```bash
pip install .
```

## Optional Dependencies

The package has several optional dependency groups:

### Development Tools
```bash
pip install "edamame[dev]"
```
Includes: pytest, pytest-cov, black, flake8, isort

### Signal Processing
```bash
pip install "edamame[signal]"
```
Includes: neurokit2 for advanced EDA signal processing

### Jupyter Notebooks
```bash
pip install "edamame[notebook]"
```
Includes: jupyter, matplotlib, ipykernel

### All Optional Dependencies
```bash
pip install "edamame[all]"
```

## Verify Installation

After installation, verify it works:

```python
from edamame.pipeline import EdamamePipeline
print("✓ Installation successful!")
```

Or run the quick test:

```bash
python test_pipeline_quick.py
```

## Using the Inference API

Once installed, you can use the EdamamePipeline API:

```python
from edamame.pipeline import EdamamePipeline
import torch

# Load a pretrained model
pipeline = EdamamePipeline.from_pretrained(
    model_name="mae",
    weights_path="path/to/checkpoint.ckpt",
    model_config="path/to/config.yaml",
    device_map="auto",
)

# Extract embeddings
data = torch.randn(8, 240, 1)
embeddings = pipeline.embed(data)
```

## Training Models

To train models, you'll also need the training scripts and config files:

```bash
# Make sure you have the full repository
git clone https://github.com/LeonardoAlchieri/eda-foundation-models.git
cd eda-foundation-models

# Install with dependencies
pip install -e ".[all]"

# Run training
python train.py --config-name=example_reconstruction
```

## Troubleshooting

### Import Error
If you get import errors, make sure you installed the package:
```bash
pip install -e .
```

### PyTorch/CUDA Issues
Install PyTorch separately with the correct CUDA version:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Missing Dependencies
Install all optional dependencies:
```bash
pip install -e ".[all]"
```

## Uninstall

```bash
pip uninstall edamame
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- See `pyproject.toml` for complete dependency list
