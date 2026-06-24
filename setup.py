from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")

INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "pytorch-lightning>=2.0.0",
    "torchmetrics>=0.11.0",
    "numpy>=1.23.0",
    "scipy>=1.10.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "scikit-learn>=1.2.0",
    "pyyaml>=6.0",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "isort>=5.12.0",
    ],
    "notebook": [
        "jupyter>=1.0.0",
        "matplotlib>=3.5.0",
        "ipykernel>=6.0.0",
    ],
    "signal": [
        "neurokit2>=0.2.0",
    ],
}
EXTRAS_REQUIRE["all"] = list(
    dict.fromkeys(
        EXTRAS_REQUIRE["dev"]
        + EXTRAS_REQUIRE["notebook"]
        + EXTRAS_REQUIRE["signal"]
    )
)


setup(
    name="edamame",
    version="0.1.0",
    description=(
        "Foundation models for electrodermal activity (EDA) signals using "
        "self-supervised learning"
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Leonardo Alchieri",
    author_email="leonardo.alchieri@example.com",
    url="https://github.com/LeonardoAlchieri/eda-foundation-models",
    project_urls={
        "Homepage": "https://github.com/LeonardoAlchieri/eda-foundation-models",
        "Repository": "https://github.com/LeonardoAlchieri/eda-foundation-models",
        "Documentation": "https://github.com/LeonardoAlchieri/eda-foundation-models/blob/main/README.md",
    },
    packages=find_packages(exclude=("tests", "tests.*", "examples", "notebook")),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.8",
    license="MIT",
    license_files=("LICENSE",),
    keywords=[
        "deep-learning",
        "self-supervised-learning",
        "time-series",
        "eda",
        "electrodermal-activity",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)