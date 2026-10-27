"""Setup script for the ``ume`` package (UME foundation model for EDA)."""

import re
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent


def read_version() -> str:
    """Read ``__version__`` from ``ume/__init__.py`` (single source of truth)."""
    init_py = (HERE / "ume" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find __version__ in ume/__init__.py")
    return match.group(1)


def read_long_description() -> str:
    readme = HERE / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


# Core runtime dependencies: enough for inference and fine-tuning with
# UMEPipeline. Pretraining with train.py also needs the `train` extra.
# The optional downstream evaluation in train.py additionally needs the
# `edamame_downstream` package, installed separately (see README).
#
# Floors are the first releases of each compiled package that work with both
# NumPy 1.x and 2.x, so any combination pip picks is binary-compatible; upper
# bounds are only set on major versions, so the package installs alongside the
# user's existing environment.
INSTALL_REQUIRES = [
    # Intel Macs top out at torch 2.2.2, which is built against NumPy 1.x and
    # fails at runtime ("Numpy is not available") with NumPy 2.
    'torch>=2.3; platform_system != "Darwin" or platform_machine != "x86_64"',
    'torch>=2.0; platform_system == "Darwin" and platform_machine == "x86_64"',
    "numpy>=1.24",
    'numpy<2; platform_system == "Darwin" and platform_machine == "x86_64"',
    "scipy>=1.13",
    # scipy 1.15 macOS wheels fail to load on macOS 27; Python 3.11+ resolves
    # to scipy >= 1.16 anyway.
    'scipy!=1.15.*; platform_system == "Darwin" and python_version < "3.11"',
    "pandas>=2.2.2",
    "scikit-learn>=1.4.2",
    "matplotlib>=3.8.4",  # imported by ume.model (plots)
    "tqdm>=4.60",
    "pyyaml>=6.0",
    "omegaconf>=2.3,<3",
    "safetensors>=0.4",  # model.safetensors weights
    "huggingface_hub>=0.20",  # UMEPipeline.from_pretrained() Hub downloads
]

EXTRAS_REQUIRE = {
    # Everything train.py needs for pretraining (and the test suite).
    "train": [
        "pytorch-lightning>=2.6,<3",  # train.py passes weights_only to Trainer.fit
        "torchmetrics>=1.0",
        "torchvision>=0.15",  # augmentation pipeline in configs/pretrain
        "hydra-core>=1.3.2,<2",
        "hydra-list-sweeper>=1.1.0,<2",
        "wandb>=0.19.9",  # configs use reinit="finish_previous"
        "tables>=3.10",  # HDF5 (.h5) dataset loading
        "neurokit2>=0.2.10",
        "umap-learn>=0.5.4",
    ],
    "mantis": [
        "mantis-tsfm>=1.0.0,<2",
    ],
    "dev": [
        "pytest>=8.0",
        "pytest-cov",
        "black",
        "flake8",
        "isort",
    ],
    "signal": [
        "neurokit2>=0.2.10",
    ],
    "notebook": [
        "jupyter",
        "notebook>=7.0",
        "ipykernel",
    ],
    "viz": [
        "umap-learn>=0.5.4",
    ],
}
# `all` = union of every optional group.
EXTRAS_REQUIRE["all"] = sorted(
    {dep for deps in EXTRAS_REQUIRE.values() for dep in deps}
)

setup(
    name="ume-eda",
    version=read_version(),
    description="UME: a foundation model for wearable electrodermal-activity (EDA) signals.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="Leonardo Alchieri",
    author_email="leonardo.alchieri@usi.ch",
    url="https://github.com/LeonardoAlchieri/eda-foundation-models",
    license="MIT",
    packages=find_packages(include=["ume", "ume.*"]),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
