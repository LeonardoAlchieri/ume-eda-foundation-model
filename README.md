---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - electrodermal-activity
  - eda
  - wearables
  - physiological-signals
  - time-series
  - contrastive-learning
  - foundation-model
---

# 🌸 UME: Foundation Model for Electrodermal Activity (EDA) Signals

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3857981-blue.svg)](https://doi.org/10.1145/3857981)
[![arXiv](https://img.shields.io/badge/arXiv-2603.16878-b31b1b.svg)](https://arxiv.org/abs/2603.16878)

See also 🫛 **EDAMAME**, the large-scale EDA pretraining collection we curated to train UME: [repository](https://github.com/LeonardoAlchieri/EDAMAME) · [website](https://pc.inf.usi.ch/datasets/edamame/) (see [Pretraining data](#pretraining-data)).

---

<a id="quick-use"></a>

## 🚀 Quick Use (UME)

```bash
pip install ume-eda
```

The package installs as `ume-eda` and imports as `ume`. For the latest development version: `pip install git+https://github.com/LeonardoAlchieri/eda-foundation-models.git`.

```python
import torch
from ume import UMEPipeline

# Downloads the base UME weights from the Hugging Face Hub
pipeline = UMEPipeline.from_pretrained()

# NOTE: default UME requires 60s windows (at 4Hz sampling rate) and 3 channels (mixed, phasic and tonic)
x = torch.randn(8, 240, 3)  # (batch, seq_len, channels)
embeddings = pipeline.embed(x)
print(embeddings.shape)  # (8, 64)
```

To load from a local clone instead, pass the weights file: `UMEPipeline.from_pretrained(weights_path="model.safetensors")` (see [Checkpoints and Weights](#checkpoints)). A runnable walkthrough is in [`notebook/pipeline_demo.ipynb`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/notebook/pipeline_demo.ipynb); for other ways to install, see [Installation](#installation).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Card](#model-card)
- [Installation](#installation)
- [Fine-Tune](#fine-tune)
- [Retrain](#retrain)
- [Pretraining data](#pretraining-data)
- [UME Architecture (Base)](#architecture)
- [UME Configurations](#configurations)
- [Checkpoints and Weights](#checkpoints)
- [Citation](#citation)
- [License](#license)

---

<a id="overview"></a>

## 🎯 Overview

This repo contains the code and weights for **UME**, the first foundation model for wearable EDA data. **UME** is based on EfficientNet architecture and is trained using sample-wise contrastive learning, using the set of data augmentations from [Katie Matton](https://github.com/kmatton/contrastive-learning-for-eda).

We release the weights for base UME, the default and suggested configuration, as well as other configurations (tiny, small and large). See below for more information on repo usage.

---

<a id="model-card"></a>

## 🪪 Model Card

- **What it is:** an encoder that turns 60-second windows of wrist electrodermal activity (EDA) into 64-dimensional embeddings.
- **Architecture:** EfficientNet-style 1D CNN (stem convolution, 16 MBConv1D blocks with squeeze-and-excitation, head convolution, global average pooling, linear projection). See [`ume/model/efficientnet_contrastive.py`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/ume/model/efficientnet_contrastive.py) and [UME Architecture](#architecture).
- **Training objective:** sample-wise contrastive learning (InfoNCE, temperature 0.1) between two augmented views of the same window, using the EDA augmentations of [Matton et al. (2024)](https://github.com/kmatton/contrastive-learning-for-eda).
- **Files:** see [Checkpoints and Weights](#checkpoints).
- **Developed by:** Leonardo Alchieri, Università della Svizzera italiana (USI).
- **Paper:** [doi.org/10.1145/3857981](https://doi.org/10.1145/3857981) · [arXiv:2603.16878](https://arxiv.org/abs/2603.16878)

### Input

- Shape `(batch, 240, 3)`: 240 samples = 60 s at 4 Hz.
- Channels, in order: mixed EDA, phasic component, tonic component.
- Signals at other sampling rates or window lengths should be resampled and cut to match.

### Intended use

- Extracting features from wearable EDA for research on downstream tasks such as stress, cognitive load, or sleep/wake classification.
- Fine-tuning on a labelled research dataset with `UMEPipeline.fit` (see [Fine-Tune](#fine-tune)).

### Out-of-scope use

- Clinical diagnosis, or any medical or safety-critical decision about an individual.
- Inferring sensitive traits or emotional states of people without their informed consent, including workplace or surveillance monitoring.
- Signals that are not wrist EDA (e.g. PPG, ECG) or that are sampled very differently, without first validating performance.

### Training data

Pretrained without labels on 17 public and research EDA datasets from the EDAMAME collection (see [Pretraining data](#pretraining-data)). The data is not distributed with the model. Recordings come mostly from research studies with wrist-worn devices such as the Empatica E4, which shapes the populations, devices and settings the model has seen.

### Evaluation

Downstream evaluation (frozen features and a classical classifier, validated leave-one-person-out) is run by `train.py` through the optional `edamame_downstream` package. Results are reported in the [paper](https://doi.org/10.1145/3857981).

### Limitations and risks

- **Population and device bias.** Performance can drop on populations, devices, body locations or recording conditions that are underrepresented in the pretraining data.
- **Artifacts.** EDA is sensitive to motion, temperature and sensor contact. The model does not detect or flag poor-quality signal.
- **Individual differences.** EDA varies a lot between people. Validate with person-independent splits (e.g. leave-one-person-out) rather than random splits, which overestimate performance.
- **Privacy.** Physiological signals are personal data. Handle any data you use with the model according to its consent terms and applicable law.

---

<a id="installation"></a>

## 🔧 Installation

### Pixi (recommended)

The environment is defined in [`pixi.toml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/pixi.toml). Install [Pixi](https://pixi.sh), then:

```bash
git clone https://github.com/LeonardoAlchieri/eda-foundation-models.git
cd eda-foundation-models
pixi install --frozen
```

`--frozen` installs the default environment straight from `pixi.lock`. Without it, Pixi also checks the optional `downstream` environment, which needs the sibling repository described below.

### pip

```bash
pip install -e .            # inference and fine-tuning (UMEPipeline)
pip install -e ".[train]"   # + everything train.py needs for pretraining
```

Other extras: `mantis` (Mantis-8M backbone), `dev` (pytest, black, flake8, isort), `notebook`. See [INSTALL.md](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/INSTALL.md).

### Optional: downstream evaluation

After pretraining, `train.py` can evaluate the new model on labelled downstream tasks. This needs the separate [`edamame_downstream`](https://github.com/LeonardoAlchieri/pretrained_foundation_models_physiological_data) package, cloned next to this repository:

```bash
git clone https://github.com/LeonardoAlchieri/pretrained_foundation_models_physiological_data.git ../pretrained_foundation_models_physiological_data
pixi install -e downstream
pixi run -e downstream python train.py --config-name conf-ume
```

Without it, `train.py` pretrains as usual and skips the downstream step with a warning. Set `downstream.enabled=false` to skip it explicitly.

---

<a id="fine-tune"></a>

## 🎓 Fine-Tune (UME)

Beyond frozen-feature extraction, the pipeline can be **supervised fine-tuned** for classification through a scikit-learn-style API (`fit` / `predict` / `predict_proba`). A classification head (`LayerNorm -> Linear` by default, matching Mantis and Pulse-PPG) is attached on top of the encoder embedding and trained with cross-entropy.

> **Note:** fine-tuning is currently supported **only for the EfficientNet (UME) encoder** (`model_name="efficientnet"`). MAE returns per-patch tokens rather than a pooled embedding, so it (and Mantis) raise `NotImplementedError`.

```python
import torch
from ume import UMEPipeline

pipeline = UMEPipeline.from_pretrained()

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

<a id="retrain"></a>

## 🏋️ Retrain (UME)

We provide Hydra config files to reproduce the experiments from our work. Modify them to try different configurations. Retraining needs the pretraining data, which is not included in this repository (see [Pretraining data](#pretraining-data)).

Notes on the defaults:

- **Logging.** Runs log to [Weights & Biases](https://wandb.ai) under your default entity. Add `pretrain/logger=csv` (or `tensorboard`, `none`) to train without a W&B account.
- **Epoch length.** An "epoch" is 3000 randomly sampled batches of 256 windows (`pretrain.trainer.limit_train_batches`), as for the released model. Set it to `null` to use the full training set per epoch.
- **Quick check.** Add `pretrain.trainer.fast_dev_run=5` to run a few batches end to end.

### 1. Train base UME

```bash
pixi run python train.py --config-name conf-ume
```

### 2. Train other model and dataset sizes

```bash
# Base (default)
python train.py --config-name conf-ume model_size=base

# Other presets
python train.py --config-name conf-ume model_size=tiny
python train.py --config-name conf-ume model_size=small
python train.py --config-name conf-ume model_size=large
```

```bash
# Base (default)
python train.py --config-name conf-ume pretrain.dataset.subsample=1.0

# Other presets
python train.py --config-name conf-ume pretrain.dataset.subsample=0.05
python train.py --config-name conf-ume pretrain.dataset.subsample=0.5
python train.py --config-name conf-ume pretrain.dataset.subsample=0.75
```

We also provide templates for sweeps for the model and dataset size (see [sweep-ume-ablation-size.yaml](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/sweep-ume-ablation-size.yaml) and [sweep-ume-ablation-subsample.yaml](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/sweep-ume-ablation-subsample.yaml)).

---

<a id="pretraining-data"></a>

## 🫛 Pretraining data

UME was pretrained on 🫛 **EDAMAME**, which we release in two versions:

- **Open**: freely downloadable.
- **Restricted**: the full collection, available after signing a data sharing agreement.

See the [EDAMAME repository](https://github.com/LeonardoAlchieri/EDAMAME) for a description of the collection and the [EDAMAME website](https://pc.inf.usi.ch/datasets/edamame/) for access to both versions. The data itself is **not** included in this repository.

The base UME was pretrained on these datasets from the collection (the list in [`configs/conf-ume.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/conf-ume.yaml)):

BiHeartS, big-ideas-glycemic-wearable, Dynamics, EmpaticaE4Stress, EPM, LAUREATE, MEFAR, M2Sleep, PPG_FieldStudy, SEED, SEED-II-Lab, SEED-II-Wild, Stress Predict Dataset, ToadStool, WEEE, WESD, Workplaces.

`train.py` reads the data from `pretrain.dataset.base_path` (default `data_online.nosync/`), with one folder per dataset:

```
data_online.nosync/
└── <dataset_name>/
    └── *.h5          # one or more pandas HDFStore files
```

Each key in an HDF5 store is one recording, stored as a pandas DataFrame with the EDA channels as columns (mixed, phasic, tonic, sampled at 4 Hz). An optional `Timestamp` column is dropped. Each key's storer has an `attrs.metadata` dict with `user`, `session`, `dataset_name` and `len_data` (number of samples). The loader cuts 60 s windows (240 samples) on the fly with a stride of `shift` samples, and splits train and validation by recording, so windows from one recording never end up in both.

---

<a id="architecture"></a>

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

Base preset (default), set through the `model_size` presets in [`configs/conf-ume.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/conf-ume.yaml):

| Parameter | Base Value |
|-----------|------------|
| `stem_channels` | `64` |
| `mbconv_channels` | `64` |
| `num_mbconv_blocks` | `16` |
| `head_channels` | `248` |
| `embedding_dim` | `64` |
| `kernel_size` | `3` |
| `dropout` | `0.1` |
| `loss temperature` | `0.1` |
| optimizer | Adam, lr `5e-4`, weight decay `1e-3` |

Contrastive training setup (augmentations, optimizer, scheduler):

- [`configs/pretrain/contrastive.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/pretrain/contrastive.yaml)
- [`configs/pretrain/engine/contrastive.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/pretrain/engine/contrastive.yaml)

---

<a id="configurations"></a>

## 🧪 UME Configurations

We recommend the **base UME** preset, and only its weights are released. The other presets and the pretraining-data subsamples used in our ablations can be retrained with the configs below.

### UME model-size presets

Defined in [`configs/conf-ume.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/conf-ume.yaml):

| Preset | Stem | MBConv | Blocks | Head | Embedding | Kernel | Dropout |
|--------|------|--------|--------|------|-----------|--------|---------|
| `tiny` | `32` | `32` | `4`  | `96`  | `32` | `5`  | `0.1` |
| `small`| `48` | `48` | `8`  | `160` | `48` | `7`  | `0.1` |
| `base` | `64` | `64` | `16` | `248` | `64` | `3`  | `0.1` |
| `large`| `96` | `96` | `24` | `384` | `96` | `11` | `0.1` |

Sweeps over model size and pretraining-data size: [`sweep-ume-ablation-size.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/sweep-ume-ablation-size.yaml) and [`sweep-ume-ablation-subsample.yaml`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/configs/sweep-ume-ablation-subsample.yaml).

---

<a id="checkpoints"></a>

## 💾 Checkpoints and Weights

Three files hold the same base UME weights:

| File | Contents | Use it for |
|------|----------|------------|
| [`model.safetensors`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/model.safetensors) + [`config.json`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/config.json) | Encoder weights in [safetensors](https://huggingface.co/docs/safetensors) format, with the model config. Holds tensors only, so loading never runs code. This is what `from_pretrained()` downloads from the Hugging Face Hub. | Inference and fine-tuning (recommended). |
| [`artifacts/ume_base_inference.pt`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/artifacts/ume_base_inference.pt) | Encoder `state_dict` and model config in one PyTorch file. Loads with `torch.load(..., weights_only=True)`. | Same as above, without the safetensors dependency. |
| [`artifacts/ume_base_training.ckpt`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/artifacts/ume_base_training.ckpt) | Full PyTorch Lightning checkpoint, including optimizer and scheduler state. Needs `weights_only=False`. | Resuming or inspecting training. |

**Are they the same model?** Yes. All three hold the same 366 encoder tensors, bit-for-bit, and give identical embeddings. They differ only in what else they contain:

- `ume_base_training.ckpt` is about 3× larger because it also stores the training state (mostly the Adam optimizer state, plus epoch, step, callback and hyperparameter records). Its weight keys carry a `model.` prefix, which `from_pretrained` strips.
- `ume_base_training.ckpt` has no model config: `from_pretrained` infers the architecture from the tensor shapes and gets the same model. The other two files include the config.
- Only `ume_base_training.ckpt` needs `weights_only=False`, which can run code stored in the file, so load it only from a source you trust.

Use `ume_base_training.ckpt` only to resume pretraining. For inference and fine-tuning, use `model.safetensors` or `ume_base_inference.pt`.

`UMEPipeline.from_pretrained` accepts any of them through `weights_path`. With no `weights_path`, it downloads `model.safetensors` and `config.json` from the Hub repo given by `repo_id` (default: the released base UME); `revision` and `cache_dir` are passed to the download.

---

<a id="citation"></a>

## 📚 Citation

If you use UME, please cite the paper ([doi.org/10.1145/3857981](https://doi.org/10.1145/3857981)):

```bibtex
@article{alchieri2026foundation,
  author     = {Alchieri, Leonardo and Garzon, Matteo and Alecci, Lidia and Bombassei De Bona, Francesco and Gjoreski, Martin and De Felice, Giovanni and Santini, Silvia},
  title      = {A Foundation Model for Electrodermal Activity Data},
  journal    = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  volume     = {10},
  number     = {4},
  articleno  = {202},
  numpages   = {40},
  year       = {2026},
  month      = dec,
  publisher  = {Association for Computing Machinery},
  address    = {New York, NY, USA},
  doi        = {10.1145/3857981},
  url        = {https://doi.org/10.1145/3857981}
}
```

---

<a id="license"></a>

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/LICENSE).

---

<p align="center">
  <i>People Centered Computing Lab @ USI, Switzerland.</i>
</p>