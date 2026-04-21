# 🌸 UME: Foundation Model for Electrodermal Activity (EDA) Signals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

See also [🫛 EDAMAME](), the large-scale pretrain EDA dataset we have curated to train UME.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Status](#model-status)
- [Installation](#installation)
- [Quick Start (UME)](#quick-start-ume)
- [UME Architecture (Base)](#ume-architecture-base)
- [Released UME Configurations](#released-ume-configurations)
- [EDAMAME Pretrain-Size Ablations](#edamame-pretrain-size-ablations)
- [Checkpoints and Weights](#checkpoints-and-weights)
- [Inference API (UME)](#inference-api-ume)
- [Repository Structure](#repository-structure)
- [Configuration Notes](#configuration-notes)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This repository contains the training and inference code for self-supervised EDA foundation models.

The **final and primary model is UME**, an **EfficientNet-inspired 1D contrastive model**:

- Model implementation: [`edamame/model/efficientnet_contrastive.py`](edamame/model/efficientnet_contrastive.py)
- Default model config: [`configs/pretrain/model/efficientnet_contrastive.yaml`](configs/pretrain/model/efficientnet_contrastive.yaml)
- Pretraining recipe: [`configs/pretrain/contrastive.yaml`](configs/pretrain/contrastive.yaml)

Key points:

- ✅ UME is the reference model for this repository
- ✅ Training uses contrastive learning with two augmented views
- ✅ We release multiple UME configurations (not only the base one)
- ✅ We release EDAMAME pretrain-size ablation weights for open science

---

## 🧭 Model Status

### Primary model (recommended)

- **UME (EfficientNet Contrastive)**
- Training engine: [`edamame/engine/contrastive.py`](edamame/engine/contrastive.py)
- Loss: NCE / contrastive objective in [`edamame/model/loss/contrastive.py`](edamame/model/loss/contrastive.py)

### Also present in the codebase

Other architectures remain available for research and legacy experiments (MAE, Mantis, ResNet, etc.), but the repository focus and recommended path are now UME-centered.

---

## 🔧 Installation

### Quick Install

```bash
# Clone and enter repo
cd eda-foundation-models

# Install package (editable)
pip install -e .

# Optional: install all extras
pip install -e ".[all]"
```

### Environment File

You can also use the provided environment specification:

- [`env.yaml`](env.yaml)
- Additional setup notes: [`INSTALL.md`](INSTALL.md)

### Verification

```bash
python verify_setup.py
```

---

## 🚀 Quick Start (UME)

### 1. Data Layout

Default pretraining dataset loader uses online segmented HDF5 data:

- Loader: [`edamame/data/dataset_online.py`](edamame/data/dataset_online.py)
- Default dataset config: [`configs/pretrain/dataset/eda_default.yaml`](configs/pretrain/dataset/eda_default.yaml)

### 2. Train UME Baseline

```bash
python train.py --config-name=conf-contrastive
```

Main entrypoint: [`train.py`](train.py)

### 3. Run Other Released UME Config Variants

```bash
# Alternative contrastive config variants
python train.py --config-name=conf-contrastive-2
python train.py --config-name=conf-contrastive-small-kernel
python train.py --config-name=conf-contrastive-large-kernel
```

### 4. Explicitly Select UME Size Preset

```bash
# Base (recommended)
python train.py --config-name=conf-ume-ablation-size model_size=base

# Other released presets
python train.py --config-name=conf-ume-ablation-size model_size=tiny
python train.py --config-name=conf-ume-ablation-size model_size=small
python train.py --config-name=conf-ume-ablation-size model_size=large
```

---

## 🏗️ UME Architecture (Base)

UME is an EfficientNet-inspired 1D encoder for contrastive learning:

```
Input (N, T, C)
  -> Stem Conv1D + BN + Swish
  -> MBConv1D blocks (repeated)
  -> Head Conv1D + Global Pool
  -> Linear embedding
  -> Contrastive loss on two augmented views
```

Base preset (default) from [`configs/pretrain/model/efficientnet_contrastive.yaml`](configs/pretrain/model/efficientnet_contrastive.yaml):

| Parameter | Base Value |
|-----------|------------|
| `stem_channels` | `64` |
| `mbconv_channels` | `64` |
| `num_mbconv_blocks` | `16` |
| `head_channels` | `248` |
| `embedding_dim` | `64` |
| `kernel_size` | `9` |
| `dropout` | `0.5` |
| `loss temperature` | `0.1` |

Contrastive training setup (augmentations, optimizer, scheduler):

- [`configs/pretrain/contrastive.yaml`](configs/pretrain/contrastive.yaml)
- [`configs/pretrain/engine/contrastive.yaml`](configs/pretrain/engine/contrastive.yaml)

---

## 🧪 Released UME Configurations

We focus on the **base UME** preset in practice, but we release additional UME configurations for transparency and reproducibility.

### UME model-size presets

Defined in [`configs/conf-ume-ablation-size.yaml`](configs/conf-ume-ablation-size.yaml):

| Preset | Stem | MBConv | Blocks | Head | Embedding | Kernel | Dropout |
|--------|------|--------|--------|------|-----------|--------|---------|
| `tiny` | `32` | `32` | `4`  | `96`  | `32` | `5`  | `0.3` |
| `small`| `48` | `48` | `8`  | `160` | `48` | `7`  | `0.4` |
| `base` | `64` | `64` | `16` | `248` | `64` | `9`  | `0.5` |
| `large`| `96` | `96` | `24` | `384` | `96` | `11` | `0.5` |

### Additional released contrastive variants

- [`configs/conf-contrastive.yaml`](configs/conf-contrastive.yaml)
- [`configs/conf-contrastive-2.yaml`](configs/conf-contrastive-2.yaml)
- [`configs/conf-contrastive-small-kernel.yaml`](configs/conf-contrastive-small-kernel.yaml)
- [`configs/conf-contrastive-large-kernel.yaml`](configs/conf-contrastive-large-kernel.yaml)

---

## 📏 EDAMAME Pretrain-Size Ablations

We also ran UME pretraining with different fractions of the EDAMAME pretraining pool.

Configuration:

- [`configs/conf-ume-ablation-subsample.yaml`](configs/conf-ume-ablation-subsample.yaml)

Subsample values released in that config:

- `0.05`, `0.1`, `0.2`, `0.4`, `0.6`, `0.8`, `1.0`

Example command:

```bash
python train.py --config-name=conf-ume-ablation-subsample pretrain.dataset.subsample=0.2
```

> **Open science note**
> We release ablation checkpoints for transparency and reproducibility. These subsample/ablation weights are **not** our recommended default for downstream use. Prefer the base UME checkpoint unless you are explicitly reproducing ablation analyses.

---

## 💾 Checkpoints and Weights

Examples of released weights in this repository:

- Main artifact: [`artifacts/ume.ckpt`](artifacts/ume.ckpt)
- Additional contrastive checkpoint: [`checkpoints.nosync/efficientnet-best.ckpt`](checkpoints.nosync/efficientnet-best.ckpt)
- Additional runs/checkpoints: [`checkpoints.nosync`](checkpoints.nosync)
- Extra exported models: [`checkpoints.nosync/final_models`](checkpoints.nosync/final_models)

Some checkpoints correspond to exploratory or ablation runs; use them accordingly.

---

## 🚀 Inference API (UME)

Use the packaged pipeline to load UME and extract embeddings:

```python
import torch
from edamame import EdamamePipeline

pipeline = EdamamePipeline.from_pretrained(
    model_name="efficientnet",
    weights_path="artifacts/ume.ckpt",
    model_config="configs/pretrain/model/efficientnet_contrastive.yaml",
    device_map="auto",
)

x = torch.randn(8, 240, 1)  # (batch, seq_len, channels)
embeddings = pipeline.embed(x)
print(embeddings.shape)  # (8, embedding_dim)
```

Pipeline implementation:

- [`edamame/pipeline/edamame_pipeline.py`](edamame/pipeline/edamame_pipeline.py)

---

## 📁 Repository Structure

```text
eda-foundation-models/
├── configs/
│   ├── conf-contrastive.yaml
│   ├── conf-contrastive-2.yaml
│   ├── conf-contrastive-small-kernel.yaml
│   ├── conf-contrastive-large-kernel.yaml
│   ├── conf-ume-ablation-size.yaml
│   ├── conf-ume-ablation-subsample.yaml
│   ├── pretrain/
│   │   ├── contrastive.yaml
│   │   ├── engine/contrastive.yaml
│   │   ├── model/efficientnet_contrastive.yaml
│   │   └── dataset/eda_default.yaml
│   └── downstream/
├── edamame/
│   ├── data/
│   ├── data_augmentation/
│   ├── engine/
│   ├── model/
│   │   └── efficientnet_contrastive.py
│   └── pipeline/
├── train.py
├── artifacts/
└── checkpoints.nosync/
```

---

## ⚙️ Configuration Notes

Config files currently include historical and exploratory variants. A cleanup pass is planned.

For now, if you only need the recommended UME path, start from:

1. [`configs/conf-contrastive.yaml`](configs/conf-contrastive.yaml)
2. [`configs/pretrain/model/efficientnet_contrastive.yaml`](configs/pretrain/model/efficientnet_contrastive.yaml)
3. [`configs/pretrain/engine/contrastive.yaml`](configs/pretrain/engine/contrastive.yaml)

---

<!-- ## 📚 Citation

If you use this code in your research, please cite your corresponding EDAMAME/UME paper and repository.

You may also cite related augmentation work used in the training setup:

```bibtex
@article{matton2024contrastive,
  title={Contrastive Learning of Electrodermal Activity Representations for Stress Detection},
  author={Matton, Katie and Lewis, Robert and Guttag, John and Picard, Rosalind},
  journal={arXiv preprint arXiv:2404.08783},
  year={2024}
}
``` -->

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).

---

<p align="center">
  <i>Built for open, reproducible EDA foundation modeling.</i>
</p>
