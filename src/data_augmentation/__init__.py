"""
Data augmentation wrapper for PyTorch tensors.

This module provides a wrapper around the Matton augmentation functions
to make them compatible with PyTorch tensors and batch processing.
"""

import torch
import numpy as np
from typing import Callable, Dict, Set, Any


class TorchAugmentationWrapper:
    """
    Wrapper for augmentation functions to work with PyTorch tensors.
    
    Applies augmentation function to each sample in a batch, handling
    conversion between PyTorch tensors and numpy arrays.
    """
    
    def __init__(self, augmentation_fn: Callable):
        """
        Args:
            augmentation_fn: Augmentation function that takes a dict with 'x' key
                           and returns augmented numpy array
        """
        self.augmentation_fn = augmentation_fn
    
    def __call__(self, batch: Dict[str, torch.Tensor], prep_keys: Set[str] = None) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to batch.
        
        Args:
            batch: Dictionary containing data tensors
            prep_keys: Set of keys to apply augmentation to (default: {'x'})
            
        Returns:
            Dictionary with augmented tensors
        """
        if prep_keys is None:
            prep_keys = {'x'}
        
        augmented_batch = {}
        
        for key, value in batch.items():
            if key in prep_keys and isinstance(value, torch.Tensor):
                # Apply augmentation to each sample in the batch
                device = value.device
                augmented_samples = []
                
                for i in range(value.shape[0]):
                    sample = value[i].cpu().numpy()
                    sample_dict = {'x': sample}
                    augmented = self.augmentation_fn(sample_dict)
                    augmented_samples.append(torch.from_numpy(augmented).float())
                
                # Stack augmented samples and move to original device
                augmented_batch[key] = torch.stack(augmented_samples).to(device)
            else:
                # Keep non-augmented keys as is
                augmented_batch[key] = value
        
        return augmented_batch


class ComposeAugmentations:
    """
    Compose multiple augmentation functions.
    
    Applies a sequence of augmentations in order.
    """
    
    def __init__(self, augmentations: list):
        """
        Args:
            augmentations: List of augmentation functions or TorchAugmentationWrapper objects
        """
        self.augmentations = augmentations
    
    def __call__(self, sample_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply all augmentations in sequence.
        
        Args:
            sample_dict: Dictionary with 'x' key containing data
            
        Returns:
            Augmented data
        """
        x = sample_dict['x']
        
        for aug in self.augmentations:
            x = aug({'x': x})
        
        return x


class RandomChoice:
    """
    Randomly choose one augmentation from a list to apply.
    """
    
    def __init__(self, augmentations: list, p: list = None):
        """
        Args:
            augmentations: List of augmentation functions
            p: Optional list of probabilities for each augmentation
        """
        self.augmentations = augmentations
        self.p = p
    
    def __call__(self, sample_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply randomly selected augmentation.
        
        Args:
            sample_dict: Dictionary with 'x' key containing data
            
        Returns:
            Augmented data
        """
        aug = np.random.choice(self.augmentations, p=self.p)
        return aug(sample_dict)


class RandomApply:
    """
    Apply augmentation with probability p.
    """
    
    def __init__(self, augmentation: Callable, p: float = 0.5):
        """
        Args:
            augmentation: Augmentation function to apply
            p: Probability of applying the augmentation
        """
        self.augmentation = augmentation
        self.p = p
    
    def __call__(self, sample_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Apply augmentation with probability p.
        
        Args:
            sample_dict: Dictionary with 'x' key containing data
            
        Returns:
            Augmented or original data
        """
        if np.random.random() < self.p:
            return self.augmentation(sample_dict)
        return sample_dict['x']
