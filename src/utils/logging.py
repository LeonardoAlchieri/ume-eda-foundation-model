"""Utilities for logging configuration and metrics."""

import json
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def generate_experiment_name(cfg: DictConfig) -> str:
    """
    Generate an experiment name automatically from configuration components.
    
    Format: {engine}_{model}_{dataset}_{loss}
    
    Args:
        cfg: Hydra/OmegaConf configuration object
        
    Returns:
        Automatically generated experiment name string
        
    Example:
        "ReconstructionEngine_MaskedAutoencoder_all_mse"
    """
    # Extract engine name from _target_
    engine_target = cfg.get('engine', {}).get('_target_', 'engine')
    engine_name = engine_target.split('.')[-1] if engine_target else 'engine'
    
    # Extract model name from _target_
    model_target = cfg.get('model', {}).get('_target_', 'model')
    model_name = model_target.split('.')[-1] if model_target else 'model'
    
    # Extract dataset info
    dataset_names = cfg.get('dataset', {}).get('dataset_names', 'data')
    if isinstance(dataset_names, list):
        dataset_str = '_'.join(dataset_names)
    else:
        dataset_str = str(dataset_names)
    
    # Loss is typically hardcoded in models (e.g., MSE for reconstruction)
    # We can infer it from the engine type
    loss_name = 'mse'  # Default for reconstruction tasks
    
    # Construct experiment name
    exp_name = f"{engine_name}_{model_name}_{dataset_str}_{loss_name}"
    
    return exp_name


def log_hydra_config_to_lightning(trainer: Trainer, cfg: DictConfig) -> None:
    """
    Log the complete Hydra configuration to the Lightning logger.
    
    This function handles different logger types (CSVLogger, TensorBoard, WandB)
    and logs the configuration appropriately for each.
    
    Args:
        trainer: PyTorch Lightning Trainer instance with a logger
        cfg: Hydra/OmegaConf configuration object
        
    Returns:
        None
        
    Note:
        - CSVLogger: Config saved in hparams.yaml
        - TensorBoard: Config visible in Text tab as formatted JSON
        - WandB: Config logged as hyperparameters in W&B dashboard
    """
    if trainer.logger is None:
        return
    
    # Convert OmegaConf to a regular dict for compatibility
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Ensure config_dict is a dictionary (handle edge cases)
    if not isinstance(config_dict, dict):
        config_dict = {"config": config_dict}
    
    # Log as hyperparameters based on logger type
    if hasattr(trainer.logger, 'log_hyperparams'):
        # Standard Lightning logger method (CSVLogger, etc.)
        trainer.logger.log_hyperparams(config_dict)  # type: ignore
    elif hasattr(trainer.logger, 'experiment'):
        experiment = trainer.logger.experiment  # type: ignore
        
        # WandB logger
        if hasattr(experiment, 'config'):
            experiment.config.update(config_dict)
        # TensorBoard logger
        elif hasattr(experiment, 'add_text'):
            config_str = json.dumps(config_dict, indent=2)
            experiment.add_text('config', f'```\n{config_str}\n```', 0)
