"""Utilities for logging configuration and metrics."""

import json
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


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
