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
            max_users_to_plot: Maximum number of users to plot in UMAP visualization (default: 20)
        """
        super().__init__(*args, **kwargs)
        # Storage for train embeddings and user labels
        self.train_embeddings = []
        self.train_users = []
        # Storage for validation embeddings and user labels
        self.validation_embeddings = []
        self.validation_users = []
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

    def training_step(self, batch, batch_idx):
        """
        Training step with embedding collection for visualization.
        Collects embeddings and user labels for UMAP plotting.
        """
        # Call parent's training step
        train_loss = super().training_step(batch, batch_idx)

        # Collect embeddings and user labels for visualization
        with torch.no_grad():
            # Get embeddings from the model (using x_1, which is the augmented view)
            x = batch.get("x_1", batch["x"])
            # Extract embeddings
            embeddings = self.model.get_representation(x)

            # Store embeddings and user labels
            self.train_embeddings.append(embeddings.detach().cpu())

            # Get user labels if available
            if "user" in batch:
                users = batch["user"]
                if torch.is_tensor(users):
                    users = users.cpu().numpy()
                self.train_users.append(users)

        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with embedding collection for visualization.
        Collects embeddings and user labels for t-SNE plotting.
        """
        # Call parent's validation step
        val_loss = super().validation_step(batch, batch_idx)

        # Collect embeddings and user labels for visualization
        with torch.no_grad():
            # Get embeddings from the model (using x_1, which is the augmented view)
            x = batch.get("x_1", batch["x"])
            # Extract embeddings
            embeddings = self.model.get_representation(x)

            # Store embeddings and user labels
            self.validation_embeddings.append(embeddings.detach().cpu())

            # Get user labels if available
            if "user" in batch:
                users = batch["user"]
                if torch.is_tensor(users):
                    users = users.cpu().numpy()
                self.validation_users.append(users)

        return val_loss

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

        # Get unique users
        unique_users = np.unique(all_users)

        if self.plotting_kwargs['max_users'] is None:
            self.plotting_kwargs['max_users'] = len(unique_users)
            
        # Sample users if there are more than max_users_to_plot
        if len(unique_users) > self.plotting_kwargs['max_users']:
            print(
                f"Sampling {self.plotting_kwargs['max_users']} out of {len(unique_users)} users for visualization..."
            )
            rng = np.random.RandomState(self.seed)
            selected_users = rng.choice(
                unique_users, size=self.plotting_kwargs['max_users'], replace=False
            )
        else:
            selected_users = unique_users
            
        # Filter embeddings and users to only include selected users
        mask_selected = np.isin(all_users, selected_users)
        embeddings_2d_filtered = embeddings_2d[mask_selected]
        all_users_filtered = all_users[mask_selected]
        
        if self.plotting_kwargs['max_segments_per_user'] is not None:
            # Further limit number of segments per user
            filtered_indices = []
            for user in selected_users:
                user_indices = np.where(all_users_filtered == user)[0]
                if len(user_indices) > self.plotting_kwargs['max_segments_per_user']:
                    rng = np.random.RandomState(self.seed)
                    sampled_indices = rng.choice(
                        user_indices,
                        size=self.plotting_kwargs['max_segments_per_user'],
                        replace=False,
                    )
                    filtered_indices.extend(sampled_indices)
                else:
                    filtered_indices.extend(user_indices)
            embeddings_2d_filtered = embeddings_2d_filtered[filtered_indices]
            all_users_filtered = all_users_filtered[filtered_indices]

        # Assign colors to selected users
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_users)))

        # Plot each selected user with a different color
        for idx, user in enumerate(selected_users):
            mask = all_users_filtered == user
            ax.scatter(
                embeddings_2d_filtered[mask, 0],
                embeddings_2d_filtered[mask, 1],
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

    def on_train_epoch_end(self):
        """
        Create UMAP visualization of embeddings at the end of training epoch.
        """
        self._plot_embedding_visualization(
            self.train_embeddings, self.train_users, split="train"
        )

        # Clear storage for next epoch
        self.train_embeddings = []
        self.train_users = []

    def on_validation_epoch_end(self):
        """
        Create UMAP visualization of embeddings at the end of validation epoch.
        """
        self._plot_embedding_visualization(
            self.validation_embeddings, self.validation_users, split="val"
        )

        # Clear storage for next epoch
        self.validation_embeddings = []
        self.validation_users = []
