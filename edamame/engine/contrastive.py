import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from umap import UMAP

from edamame.engine.base import BaseSelfSupervisedEngine


class ContrastiveEngine(BaseSelfSupervisedEngine):
    """
    Engine for contrastive learning.

    Creates two augmented views (x_1 and x_2) from batch['x'] using
    the data augmentation pipeline.
    """

    def __init__(
        self,
        *args,
        plotting_kwargs: dict = {"max_users": 10, "max_segments_per_user": 20},
        **kwargs,
    ):
        """
        Initialize contrastive engine with embedding collection for visualization.

        Args:
            plotting_kwargs: Dictionary with 'max_users' and 'max_segments_per_user' keys
        """
        super().__init__(*args, **kwargs)
        # Storage for train embeddings and user labels
        self.train_embeddings = []
        self.train_users = []
        # Storage for validation embeddings and user labels
        self.validation_embeddings = []
        self.validation_users = []
        
        # Tracking for early stopping during collection
        self.train_user_counts = {}  # {user_id: count}
        self.val_user_counts = {}  # {user_id: count}
        
        # Maximum number of users to plot
        if "max_users" not in plotting_kwargs.keys() or "max_segments_per_user" not in plotting_kwargs.keys():
            raise ValueError(
                "plotting_kwargs must contain 'max_users' and 'max_segments_per_user' keys."
            )
        self.plotting_kwargs = plotting_kwargs

    def forward(self, batch):
        """Forward pass through the model."""
        x_1 = batch["x_1"]
        x_2 = batch["x_2"]
        return self.model(x_1, x_2)

    def get_representation(self, x):
        """
        Extract representation/embedding from input data.
        
        This method extracts feature representations from the model's encoder.
        Useful for downstream tasks, inference, or feature extraction.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
               Can also be a batch dict with 'x' key
        
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        
        Example:
            >>> engine = ContrastiveEngine(model, ...)
            >>> embeddings = engine.get_representation(batch_data)
        """
        # Handle batch dict format
        if isinstance(x, dict):
            x = x["x"]
        
        # Ensure model is in eval mode
        self.model.eval()
        
        with torch.no_grad():
            # Call the model's get_representation method
            return self.model.get_representation(x)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Hook called after batch is transferred to device.
        Creates x_1 and x_2 from x using data augmentation for contrastive learning.
        Data augmentation is only applied during training.
        """
        if isinstance(batch, dict) and "x" in batch:
            # Apply data augmentation to create two views only during training
            if self.data_augmentation is not None and self.training:
                if self.scaling_params is not None:
                    batch["x"] = self.apply_scaling(batch["x"])
                # Convert to numpy for augmentation
                x = batch["x"]
                x_np = x.cpu().numpy()
                device = x.device

                # Vectorized augmentation function
                augment_fn = np.vectorize(
                    self.data_augmentation,
                    signature="(t)->(t)" if len(x.shape) == 2 else "(t,c)->(t,c)",
                )

                # Apply augmentation in a vectorized way
                batch_aug_1_np = augment_fn(x_np)
                batch_aug_2_np = augment_fn(x_np)

                # Convert back to tensor
                batch_aug_1 = torch.from_numpy(batch_aug_1_np).float().to(device)
                batch_aug_2 = torch.from_numpy(batch_aug_2_np).float().to(device)

                batch["x_1"] = batch_aug_1
                batch["x_2"] = batch_aug_2
            else:
                # If no augmentation or not training, just copy x to x_1 and x_2
                batch["x_1"] = batch["x"]
                batch["x_2"] = batch["x"]

            # Add batch size
            batch["batch_size"] = batch["x"].shape[0]

        return batch

    def _collect_embeddings_for_visualization(self, batch, user_counts: dict, embeddings_list: list, users_list: list):
        """
        Collect embeddings and user labels for visualization if requirements not met.
        
        Args:
            batch: Current batch containing 'x_1', 'x', and 'user'
            user_counts: Dictionary tracking samples per user
            embeddings_list: List to append collected embeddings to
            users_list: List to append collected user labels to
        """
        # Only collect embeddings if we haven't reached our limits
        if "user" in batch and not self._collection_complete(user_counts):
            with torch.no_grad():
                # Get embeddings from the model
                x = batch.get("x_1", batch["x"])
                # Extract embeddings
                embeddings = self.model.get_representation(x)
                embeddings_cpu = embeddings.detach().cpu()
                
                # Get user labels
                users = batch["user"]
                if torch.is_tensor(users):
                    users = users.cpu().numpy()
                
                # Selectively collect embeddings based on requirements
                selected_indices = self._select_samples_to_collect(users, user_counts)
                
                if len(selected_indices) > 0:
                    embeddings_list.append(embeddings_cpu[selected_indices])
                    users_list.append(users[selected_indices])

    def training_step(self, batch, batch_idx):
        """
        Training step with embedding collection for visualization.
        Collects embeddings and user labels for UMAP plotting.
        """
        # Call parent's training step
        train_loss = super().training_step(batch, batch_idx)

        # Collect embeddings for visualization
        self._collect_embeddings_for_visualization(
            batch, self.train_user_counts, self.train_embeddings, self.train_users
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with embedding collection for visualization.
        Collects embeddings and user labels for UMAP plotting.
        """
        # Call parent's validation step
        val_loss = super().validation_step(batch, batch_idx)

        # Collect embeddings for visualization
        self._collect_embeddings_for_visualization(
            batch, self.val_user_counts, self.validation_embeddings, self.validation_users
        )

        return val_loss
    
    def _collection_complete(self, user_counts: dict) -> bool:
        """
        Check if we've collected enough samples for visualization.
        
        Args:
            user_counts: Dictionary tracking samples per user
            
        Returns:
            True if collection requirements are met
        """
        max_users = self.plotting_kwargs['max_users']
        max_segments = self.plotting_kwargs['max_segments_per_user']
        
        # If max_users is None, never stop (collect everything)
        if max_users is None:
            return False
            
        # Check if we have enough users
        if len(user_counts) < max_users:
            return False
            
        # Check if all users have enough segments (if max_segments is set)
        if max_segments is not None:
            return all(count >= max_segments for count in user_counts.values())
        
        return True
    
    def _select_samples_to_collect(self, users: np.ndarray, user_counts: dict) -> list:
        """
        Select which samples from the batch to collect based on current counts.
        
        Args:
            users: Array of user labels from current batch
            user_counts: Dictionary tracking samples per user
            
        Returns:
            List of indices to collect from the batch
        """
        max_users = self.plotting_kwargs['max_users']
        max_segments = self.plotting_kwargs['max_segments_per_user']
        
        selected_indices = []
        
        for idx, user in enumerate(users):
            user = int(user)  # Ensure user is hashable
            
            # Check if we already have enough users
            if max_users is not None and user not in user_counts:
                if len(user_counts) >= max_users:
                    continue  # Skip new users if we have enough
            
            # Check if we have enough segments for this user
            current_count = user_counts.get(user, 0)
            if max_segments is not None and current_count >= max_segments:
                continue  # Skip if this user has enough segments
            
            # Add this sample
            selected_indices.append(idx)
            user_counts[user] = current_count + 1
        
        return selected_indices

    def _plot_embedding_visualization(self, embeddings_list, users_list, split="val"):
        """
        Create UMAP visualization of embeddings.
        Projects embeddings to 2D and colors by user assignment.

        Args:
            embeddings_list: List of embedding tensors collected during epoch
            users_list: List of user label arrays collected during epoch
            split: Split name ("train" or "val") for logging
        """
        # Only create visualization if we have collected embeddings and users
        if len(embeddings_list) == 0 or len(users_list) == 0:
            print(
                f"No embeddings or user labels collected for {split} UMAP visualization."
            )
            return

        # Concatenate all embeddings and users
        all_embeddings = torch.cat(embeddings_list, dim=0)
        all_users = np.concatenate(users_list, axis=0)

        # Convert embeddings to numpy
        embeddings_np = all_embeddings.numpy()

        # Flatten embeddings if needed (in case they're not already 2D)
        if len(embeddings_np.shape) > 2:
            embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)

        # Apply UMAP to reduce to 2D
        print(f"Computing UMAP for {embeddings_np.shape[0]} {split} embeddings...")
        umap = UMAP(
            n_components=2,
            random_state=self.seed,
            n_neighbors=min(5, embeddings_np.shape[0] - 1),
            min_dist=0.1,
            spread=1.5,
            metric="euclidean",
        )
        embeddings_2d = umap.fit_transform(embeddings_np)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get unique users (already filtered during collection)
        unique_users = np.unique(all_users)

        # Assign colors to users
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_users)))

        # Plot each user with a different color
        for idx, user in enumerate(unique_users):
            mask = all_users == user
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[idx]],
                label=f"User {user}",
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.set_title(
            f"{split.capitalize()} Embedding Visualization (Epoch {self.current_epoch})",
            fontsize=14,
            fontweight="bold",
        )
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Log to wandb or other loggers using base class method
        self.log_figure(fig, f"{split}/embedding_umap")

        plt.close(fig)
        print(f"✓ UMAP visualization created for {split} epoch {self.current_epoch}")

    # def on_train_epoch_end(self):
    #     """
    #     Create UMAP visualization of embeddings at the end of training epoch.
    #     """
    #     self._plot_embedding_visualization(
    #         self.train_embeddings, self.train_users, split="train"
    #     )

    #     # Clear storage and counters for next epoch
    #     self.train_embeddings = []
    #     self.train_users = []
    #     self.train_user_counts = {}

    # def on_validation_epoch_end(self):
    #     """
    #     Create UMAP visualization of embeddings at the end of validation epoch.
    #     """
    #     self._plot_embedding_visualization(
    #         self.validation_embeddings, self.validation_users, split="val"
    #     )

    #     # Clear storage and counters for next epoch
    #     self.validation_embeddings = []
    #     self.validation_users = []
    #     self.val_user_counts = {}
        
    
