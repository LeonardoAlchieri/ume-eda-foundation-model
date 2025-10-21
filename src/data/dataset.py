"""
Generic dataset class for loading multiple .npy files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EDAtaNoLabel(Dataset):
    """
    Generic dataset for loading and combining multiple .npy files.
    
    This dataset is agnostic to the model/engine and simply loads data from
    multiple .npy files, concatenates them, and optionally shuffles.
    
    Args:
        base_path: Base directory path containing the dataset folders
        dataset_names: List of dataset folder names to load
        file_pattern: Pattern for .npy files to load (e.g., 'data.npy', '*.npy')
        shuffle: Whether to shuffle the data after loading
        seed: Random seed for shuffling (for reproducibility)
        
    Example:
        >>> dataset = EDAtaNoLabel(
        ...     base_path='/path/to/data',
        ...     dataset_names=['dataset1', 'dataset2', 'dataset3'],
        ...     file_pattern='data.npy',
        ...     shuffle=True
        ... )
    """
    
    def __init__(
        self,
        base_path: Union[str, Path],
        dataset_names: List[str],
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the dataset."""
        self.base_path = Path(base_path)
        self.dataset_names = dataset_names
        self.shuffle = shuffle
        self.seed = seed
        
        # Load all data
        self.data = self._load_data()
        
        # Shuffle if requested
        if self.shuffle:
            self._shuffle_data()
        
        logger.info(f"Loaded dataset with {len(self.data)} samples from {len(dataset_names)} dataset(s)")
    
    def _load_data(self) -> np.ndarray:
        """
        Load data from all specified datasets.
        
        Returns:
            Concatenated numpy array of all loaded data
        """
        all_data = []
        
        for dataset_name in self.dataset_names:
            dataset_path = self.base_path / dataset_name
            
            if not dataset_path.exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                continue
            
            # Check if file_pattern is a specific file or a glob pattern
            file_paths = dataset_path.glob("*.npz")
            
            # Load each file
            for file_path in file_paths:
                if not file_path.exists():
                    logger.warning(f"File does not exist: {file_path}")
                    continue
                try:
                    data = np.load(file_path)
                    all_data.append(data)
                    logger.info(f"Loaded {data.shape} from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    raise e
        
        if len(all_data) == 0:
            raise ValueError(f"No data loaded from {self.base_path} with datasets {self.dataset_names}")
        
        # Concatenate all data along first axis
        concatenated_data = np.concatenate(all_data, axis=0)
        logger.info(f"Concatenated data shape: {concatenated_data.shape}")
        
        return concatenated_data
    
    def _shuffle_data(self):
        """Shuffle the data in-place."""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate random permutation
        indices = np.random.permutation(len(self.data))
        self.data = self.data[indices]
        logger.info(f"Shuffled data with seed={self.seed}")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'x' key containing the data sample as a torch tensor
        """
        # Load sample and convert to tensor
        sample = self.data[idx]
        
        # Convert to torch tensor
        x = torch.from_numpy(sample).float()
        
        # Return as dictionary (generic format)
        return {'x': x}
    
    def reshuffle(self):
        """Reshuffle the dataset (useful between epochs)."""
        self._shuffle_data()
    
    def get_data_shape(self) -> tuple:
        """Return the shape of a single data sample."""
        return self.data[0].shape
    
    def get_dataset_info(self) -> dict:
        """Return information about the dataset."""
        return {
            'num_samples': len(self.data),
            'data_shape': self.data.shape,
            'sample_shape': self.get_data_shape(),
            'dataset_names': self.dataset_names,
            'base_path': str(self.base_path),
            'shuffled': self.shuffle,
        }
