# Installation Guide

Requires **Python 3.12 or newer**. Dependencies are declared in [`setup.py`](setup.py) (pip) and [`pixi.toml`](pixi.toml) (Pixi).

## Option 1: Pixi (recommended)

```bash
git clone https://github.com/LeonardoAlchieri/eda-foundation-models.git
cd eda-foundation-models
pixi install --frozen
```

This installs everything needed for inference, fine-tuning and pretraining, with `ume` itself in editable mode. `--frozen` installs from `pixi.lock` without re-checking the optional `downstream` environment, which needs a sibling repository (see [Downstream evaluation](#downstream-evaluation-optional)).

Run commands inside the environment with `pixi run`, e.g. `pixi run python train.py --config-name conf-ume`.

## Option 2: pip

```bash
git clone https://github.com/LeonardoAlchieri/eda-foundation-models.git
cd eda-foundation-models
pip install -e .
```

The base install covers inference and fine-tuning with `UMEPipeline`. Add extras as needed:

| Extra | Adds | Needed for |
|-------|------|------------|
| `train` | neurokit2, umap-learn, hydra-list-sweeper | Pretraining with `train.py` |
| `mantis` | mantis-tsfm | The Mantis-8M backbone (`conf-mantis`) |
| `dev` | pytest, pytest-cov, black, flake8, isort | Running the tests, contributing |
| `notebook` | jupyter, ipykernel | The notebook in `notebook/` |
| `signal`, `viz` | neurokit2 / umap-learn | Subsets of `train` |
| `all` | All of the above | |

For example: `pip install -e ".[train,dev]"`.

### PyTorch and CUDA

If pip picks the wrong PyTorch build for your GPU, install PyTorch first from the [official instructions](https://pytorch.org/get-started/locally/), e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU only
```

## Downstream evaluation (optional)

After pretraining, `train.py` evaluates the model on labelled downstream tasks using the separate [`edamame_downstream`](https://github.com/LeonardoAlchieri/pretrained_foundation_models_physiological_data) package. Clone it next to this repository:

```bash
git clone https://github.com/LeonardoAlchieri/pretrained_foundation_models_physiological_data.git ../pretrained_foundation_models_physiological_data
```

Then either use the Pixi `downstream` environment:

```bash
pixi install -e downstream
pixi run -e downstream python train.py --config-name conf-ume
```

or install it with pip: `pip install -e ../pretrained_foundation_models_physiological_data`.

If the package is missing, `train.py` skips the downstream step with a warning. Set `downstream.enabled=false` to skip it explicitly.

## Verify the installation

```bash
python examples/pipeline_usage.py   # loads artifacts/ume_base_inference.pt and embeds a random batch
python -m pytest tests/             # needs the `dev` extra with pip
```

## Uninstall

```bash
pip uninstall ume
```
