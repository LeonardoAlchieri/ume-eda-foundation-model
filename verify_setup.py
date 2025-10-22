#!/usr/bin/env python
"""
Verify training setup and configuration.

This script checks that all components can be instantiated correctly
without actually running training.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch


def verify_imports():
    """Verify all required packages are installed."""
    print("🔍 Verifying imports...")
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning {pl.__version__}")
    except ImportError:
        print("❌ PyTorch Lightning not installed")
        return False
    
    try:
        import hydra
        print(f"✅ Hydra installed")
    except ImportError:
        print("❌ Hydra not installed")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True


def verify_model_instantiation(cfg: DictConfig):
    """Verify model can be instantiated."""
    print("\n🔍 Verifying model instantiation...")
    
    try:
        model = hydra.utils.instantiate(cfg.model)
        print(f"✅ Model instantiated: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate model: {e}")
        return False


def verify_engine_instantiation(cfg: DictConfig):
    """Verify engine can be instantiated."""
    print("\n🔍 Verifying engine instantiation...")
    
    try:
        engine = hydra.utils.instantiate(cfg.engine)
        print(f"✅ Engine instantiated: {type(engine).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate engine: {e}")
        return False


def verify_logger_instantiation(cfg: DictConfig):
    """Verify logger can be instantiated."""
    print("\n🔍 Verifying logger instantiation...")
    
    if not cfg.get("logger"):
        print("⚠️  No logger configured")
        return True
    
    try:
        logger = hydra.utils.instantiate(cfg.logger)
        print(f"✅ Logger instantiated: {type(logger).__name__}")
        return True
    except Exception as e:
        print(f"❌ Failed to instantiate logger: {e}")
        return False


@hydra.main(version_base=None, config_path="configs/train", config_name="config")
def main(cfg: DictConfig):
    """Main verification function."""
    print("=" * 60)
    print("Training Setup Verification")
    print("=" * 60)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Verify imports
    if not verify_imports():
        print("\n❌ Import verification failed!")
        print("Install dependencies: pip install -r requirements_training.txt")
        return
    
    # Verify model
    if not verify_model_instantiation(cfg):
        print("\n❌ Model verification failed!")
        return
    
    # Verify engine
    if not verify_engine_instantiation(cfg):
        print("\n❌ Engine verification failed!")
        return
    
    # Verify logger
    if not verify_logger_instantiation(cfg):
        print("\n❌ Logger verification failed!")
        return
    
    print("\n" + "=" * 60)
    print("✅ All verifications passed!")
    print("=" * 60)
    print("\n🚀 You can now run training with: python train.py")


if __name__ == "__main__":
    main()
