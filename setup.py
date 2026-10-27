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
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "torchvision>=0.21.0",
    "pytorch-lightning>=2.6.1,<3",
    "hydra-core>=1.3.2,<2",
    "omegaconf>=2.3,<3",
    "wandb>=0.25.1,<0.26",
    "numpy>=2.0,<3",
    "pandas<3.0",
    "scipy>=1.10",
    "tqdm>=4.60",
    "pyyaml",
    "scikit-learn",
    "torchmetrics",
    "packaging",
    "tables>=3.9",  # HDF5 (.h5) dataset loading
    "safetensors>=0.4",  # model.safetensors weights
    "huggingface_hub>=0.20",  # UMEPipeline.from_pretrained() Hub downloads
    "imbalanced-learn>=0.14,<0.15",
    "matplotlib>=3.7",  # imported by ume.model / ume.engine (plots)
]

EXTRAS_REQUIRE = {
    # Everything train.py needs for pretraining.
    "train": [
        "neurokit2>=0.2.13,<0.3",
        "umap-learn>=0.5.11,<0.6",
        "hydra-list-sweeper>=1.1.0,<2",
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
        "neurokit2>=0.2.13,<0.3",
    ],
    "notebook": [
        "jupyter",
        "notebook>=7.0",
        "ipykernel",
    ],
    "viz": [
        "umap-learn>=0.5.11,<0.6",
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
    python_requires=">=3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
