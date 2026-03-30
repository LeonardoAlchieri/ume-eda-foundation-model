"""
Example demonstrating EdamamePipeline with different normalization functions.

This example shows how to use the pipeline with:
1. No normalization (default)
2. Z-score normalization
3. Min-max normalization
4. Custom normalization function
"""

import torch
from edamame.pipeline import EdamamePipeline


def z_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Apply z-score normalization per sample.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, in_channels)
        
    Returns:
        Normalized tensor
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / (std + 1e-8)


def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Apply min-max normalization per sample to [0, 1] range.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, in_channels)
        
    Returns:
        Normalized tensor
    """
    min_val = x.min(dim=1, keepdim=True)[0]
    max_val = x.max(dim=1, keepdim=True)[0]
    return (x - min_val) / (max_val - min_val + 1e-8)


def robust_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Apply robust normalization using median and IQR.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, in_channels)
        
    Returns:
        Normalized tensor
    """
    median = x.median(dim=1, keepdim=True)[0]
    q75 = x.quantile(0.75, dim=1, keepdim=True)
    q25 = x.quantile(0.25, dim=1, keepdim=True)
    iqr = q75 - q25
    return (x - median) / (iqr + 1e-8)


def main():
    # Example 1: No normalization (default)
    print("Example 1: Pipeline without normalization")
    pipeline_no_norm = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
    )
    
    # Generate dummy data
    batch_data = torch.randn(32, 512, 4)  # (batch, seq_len, channels)
    
    embeddings = pipeline_no_norm.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 2: Z-score normalization
    print("Example 2: Pipeline with z-score normalization")
    pipeline_z_norm = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
        normalization_fn=z_normalize,
    )
    
    embeddings = pipeline_z_norm.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 3: Min-max normalization
    print("Example 3: Pipeline with min-max normalization")
    pipeline_minmax = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
        normalization_fn=min_max_normalize,
    )
    
    embeddings = pipeline_minmax.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 4: Robust normalization
    print("Example 4: Pipeline with robust normalization")
    pipeline_robust = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
        normalization_fn=robust_normalize,
    )
    
    embeddings = pipeline_robust.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 5: Lambda function for inline normalization
    print("Example 5: Pipeline with lambda normalization")
    pipeline_lambda = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
        normalization_fn=lambda x: (x - x.mean()) / (x.std() + 1e-8),
    )
    
    embeddings = pipeline_lambda.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 6: Using the same normalization for dataset and pipeline
    print("Example 6: Consistent normalization between dataset and pipeline")
    
    # Define normalization function
    def my_normalization(x):
        """Custom normalization to be used consistently."""
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
    
    # Use in dataset
    from edamame.data.dataset import EDAtaNoLabel
    dataset = EDAtaNoLabel(
        base_path="data/60s",
        dataset_names=["APSync", "BiHeartS"],
        sample_scaling_method=my_normalization,  # Same normalization
    )
    
    # Use in pipeline
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        normalization_fn=my_normalization,  # Same normalization
        device_map="auto",
    )
    
    # Get sample from dataset (already normalized)
    sample = dataset[0]['x'].unsqueeze(0)  # Add batch dimension
    
    # Pipeline will apply normalization again (be careful!)
    # If data is already normalized, you might want to pass normalization_fn=None
    embeddings = pipeline.embed(sample)
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    
    # Example 7: Reconstruction with normalization
    print("Example 7: Reconstruction with normalization")
    pipeline_recon = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints/best.ckpt",
        device_map="auto",
        normalization_fn=z_normalize,
    )
    
    # Note: Reconstruction will be in normalized space
    reconstructed = pipeline_recon.reconstruct(batch_data, mask_ratio=0.75)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print("Note: Reconstruction is in normalized space. Apply inverse transform if needed.")
    
    # If you need original scale, compute stats before normalization
    mean = batch_data.mean(dim=1, keepdim=True)
    std = batch_data.std(dim=1, unbiased=False, keepdim=True)
    
    # Inverse z-score normalization
    reconstructed_original = reconstructed * (std + 1e-8) + mean
    print(f"Reconstructed (original scale) shape: {reconstructed_original.shape}")


if __name__ == "__main__":
    main()
