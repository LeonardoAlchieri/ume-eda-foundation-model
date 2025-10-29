


import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

from edamame.engine.base import BaseSelfSupervisedEngine



class ContrastiveEngine(BaseSelfSupervisedEngine):
    """
    Engine for contrastive learning.
    
    Creates two augmented views (x_1 and x_2) from batch['x'] using
    the data augmentation pipeline.
    """
    
    def forward(self, batch):
        """Forward pass through the model."""
        x_1 = batch['x_1']
        x_2 = batch['x_2']
        return self.model(x_1, x_2)
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Creates x_1 and x_2 from x using data augmentation for contrastive learning.
        Data augmentation is only applied during training.
        """
        if isinstance(batch, dict) and 'x' in batch:
            # Apply data augmentation to create two views only during training
            if self.data_augmentation is not None and self.training:
                if self.scaling_params is not None:
                    batch['x'] = self.apply_scaling(batch['x'])
                # Convert to numpy for augmentation
                x = batch['x']
                x_np = x.cpu().numpy()
                device = x.device

                # Vectorized augmentation function
                augment_fn = np.vectorize(self.data_augmentation, signature='(t)->(t)' if len(x.shape) == 2 else '(t,c)->(t,c)')

                # Apply augmentation in a vectorized way
                batch_aug_1_np = augment_fn(x_np)
                batch_aug_2_np = augment_fn(x_np)

                # Convert back to tensor
                batch_aug_1 = torch.from_numpy(batch_aug_1_np).float().to(device)
                batch_aug_2 = torch.from_numpy(batch_aug_2_np).float().to(device)
                
                batch['x_1'] = batch_aug_1
                batch['x_2'] = batch_aug_2
            else:
                # If no augmentation or not training, just copy x to x_1 and x_2
                batch['x_1'] = batch['x']
                batch['x_2'] = batch['x']
            
            # Add batch size
            batch['batch_size'] = batch['x'].shape[0]
        
        return batch