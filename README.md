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
- [Fine-Tune (UME)](#-fine-tune-ume)
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

This repo contains the code and weights for **UME**, the first foundation model for wearable EDA data. **UME** is based on EfficientNet architecture and is trained using sample-wise contrastive learning, using the set of data augmentations from [Katie Matton](https://github.com/kmatton/contrastive-learning-for-eda).

We release the weights for base UME, the default and suggested configuration, as well as other configurations (tiny, small and large). See below for more information on repo usage.

---

## 🔧 Installation

### Pixi Setup (Recommended)

This repository is configured with Pixi using [`pixi.toml`](pixi.toml)

```bash
# Install Pixi first (https://pixi.sh), then clone and enter repo
cd eda-foundation-models

# Resolve and install environment from pixi.toml
pixi install
```

---

## 🚀 Quick Use (UME)

We provide a simple API to use the pre-trained model for inference. The weights are provided inside `artifacts`.

```python
import torch
from edamame import EdamamePipeline

pipeline = EdamamePipeline.from_pretrained(
    model_name="efficientnet",
    weights_path="artifacts/ume.ckpt",
)

# NOTE: default UME requires 60s windows (at 4Hz sampling rate) and 3 channels (mixed, phasic and tonic)
x = torch.randn(8, 240, 3)  # (batch, seq_len, channels)
embeddings = pipeline.embed(x)
print(embeddings.shape)  # (8, embedding_dim)
```

---

## 🎓 Fine-Tune (UME)

Beyond frozen-feature extraction, the pipeline can be **supervised fine-tuned** for classification through a scikit-learn-style API (`fit` / `predict` / `predict_proba`). A classification head (`LayerNorm -> Linear` by default, matching Mantis and Pulse-PPG) is attached on top of the encoder embedding and trained with cross-entropy.

> **Note:** fine-tuning is currently supported **only for the EfficientNet (UME) encoder** (`model_name="efficientnet"`). MAE returns per-patch tokens rather than a pooled embedding, so it (and Mantis) raise `NotImplementedError`.

```python
import torch
from edamame import EdamamePipeline

pipeline = EdamamePipeline.from_pretrained(
    model_name="efficientnet",
    weights_path="artifacts/ume.ckpt",
)

X = torch.randn(256, 240, 3)              # (n_samples, seq_len, channels)
y = torch.randint(0, 2, (256,)).numpy()   # labels may be ints or strings

pipeline.fit(X, y, fine_tuning_type="full", num_epochs=20)

probs = pipeline.predict_proba(X)  # (n_samples, num_classes); column j -> pipeline.classes_[j]
preds = pipeline.predict(X)        # labels in the original space passed to `fit`
```

Labels are encoded internally to contiguous integer ids (sorted unique values, exposed as `pipeline.classes_`) and mapped back by `predict`.

### Fine-tuning strategies

| `fine_tuning_type` | Behaviour |
|--------------------|-----------|
| `"full"` (default) | Fine-tune the whole encoder + head. Pass `backbone_lr` for a smaller encoder LR than the head's `learning_rate`. |
| `"head"` | Freeze the encoder (linear probe) and train the head only; the backbone stays in eval mode so BatchNorm statistics are not updated. |
| `"scratch"` | Re-initialise the encoder and train everything from random weights, ignoring the pretrained checkpoint — the "no pretraining" baseline. |

### Validation, early stopping and best-checkpoint selection

Supply a validation set either explicitly with `validation_data=(X_val, y_val)` or by holding out a stratified fraction of the training data with `validation_split` (e.g. `0.2`). `validation_data` takes precedence when both are given.

```python
pipeline.fit(
    X, y,
    fine_tuning_type="full",
    learning_rate=1e-3,
    backbone_lr=1e-4,
    num_epochs=100,
    validation_split=0.2,
    class_weight="balanced",      # inverse-frequency CE weights (or pass a tensor)
    early_stopping=True,
    monitor="val_macro_f1",       # or "val_acc" (maximized) / "val_loss" (minimized)
    patience=10,
    restore_best_weights=True,    # roll back to the best epoch when training ends
)
```

Without `early_stopping`, all `num_epochs` are run and no checkpoint selection happens. With it, training stops after `patience` epochs without a `min_delta` improvement in `monitor`, and `restore_best_weights=True` (the default) restores the encoder + head from the best epoch.

### Refit on the full dataset

On small labelled datasets, permanently holding out labels for validation hurts the final model. `refit_on_full_data=True` (which requires `early_stopping=True`) instead runs training in two phases:

1. **Phase 1** trains with the validation hold-out purely to find the optimal training budget — the *optimizer-step* count of the best epoch (steps, not epochs, because the full dataset yields more batches per epoch).
2. **Phase 2** resets the encoder + head to their initial weights, rebuilds the optimizer, and retrains on 100% of the data (train + validation combined) for exactly that step budget, with no validation set.

```python
pipeline.fit(
    X, y,
    validation_split=0.2,
    early_stopping=True,
    refit_on_full_data=True,
)
```

The refit model cannot be validated internally, so use an outer evaluation loop (e.g. leave-one-person-out) for unbiased performance estimates. `restore_best_weights` is ignored in this mode, since the refit replaces the phase-1 model entirely.

### Key `fit` arguments

| Argument | Default | Purpose |
|----------|---------|---------|
| `fine_tuning_type` | `"full"` | `"full"` / `"head"` / `"scratch"` (see above). |
| `learning_rate` | `1e-3` | LR for the head (and the backbone when `backbone_lr` is None). |
| `backbone_lr` | `None` | Optional separate, typically smaller, encoder LR. |
| `batch_size` / `num_epochs` | `64` / `100` | Mini-batch size and epoch count. |
| `weight_decay` | `0.0` | AdamW weight decay. |
| `validation_data` / `validation_split` | `None` / `0.0` | Explicit validation set, or stratified hold-out fraction. |
| `class_weight` | `None` | `"balanced"` for inverse-frequency CE weights, or a `(num_classes,)` tensor. |
| `early_stopping` / `patience` / `monitor` / `min_delta` | `False` / `10` / `"val_macro_f1"` / `0.0` | Early-stopping controls. |
| `restore_best_weights` | `True` | Roll back to the best monitored epoch at the end of training. |
| `refit_on_full_data` | `False` | Two-phase budget-then-refit training (see above). |
| `use_layer_norm` | `True` | `LayerNorm -> Linear` head; set `False` for a bare linear head. |
| `verbose` | `True` | Print per-epoch loss / macro-F1. |

---

## 🏋️ Retrain (UME)

We provide config files with Hydra to reproduce the experiments from our work. Modify the config files to try different configurations.

### 1. Train base UME

```bash
pixi run python train.py --config-name conf-ume
```

### 2. Train other model and dataset sizes

```bash
# Base (default)
python train.py --config-name conf-ume model_size=base

# Other released presets
python train.py --config-name conf-ume model_size=tiny
python train.py --config-name conf-ume model_size=small
python train.py --config-name conf-ume model_size=large
```

```bash
# Base (default)
python train.py --config-name conf-ume pretrain.dataset.subsample=1.0

# Other released presets
python train.py --config-name conf-ume pretrain.dataset.subsample=0.05
python train.py --config-name conf-ume pretrain.dataset.subsample=0.5
python train.py --config-name conf-ume pretrain.dataset.subsample=0.75
```

We also provide templates for sweeps for the model and dataset size (see [sweep-ume-ablation-size.yaml](configs/sweep-ume-ablation-size.yaml) and [sweep-ume-ablation-subsample.yaml](configs/sweep-ume-ablation-subsample.yaml)).

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

Defined in [`configs/conf-ume.yaml`](configs/conf-ume.yaml):

| Preset | Stem | MBConv | Blocks | Head | Embedding | Kernel | Dropout |
|--------|------|--------|--------|------|-----------|--------|---------|
| `tiny` | `32` | `32` | `4`  | `96`  | `32` | `5`  | `0.3` |
| `small`| `48` | `48` | `8`  | `160` | `48` | `7`  | `0.4` |
| `base` | `64` | `64` | `16` | `248` | `64` | `9`  | `0.5` |
| `large`| `96` | `96` | `24` | `384` | `96` | `11` | `0.5` |

---

> **Open science note**
> We release ablation checkpoints for transparency and reproducibility. These subsample/ablation weights are **not** our recommended default for downstream use. Prefer the base UME checkpoint unless you are explicitly reproducing ablation analyses.

---

## 💾 Checkpoints and Weights

Examples of released weights in this repository:

- Main artifact: [`artifacts/ume.ckpt`](artifacts/ume.ckpt)

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

<!-- <p align="center">
  <i>People Centered Computing Lab @ USI, Switzerland.</i>
</p> -->