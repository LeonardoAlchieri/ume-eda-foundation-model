"""
Example usage of the EdamamePipeline API.

This script demonstrates how to load a pretrained MAE model and use it for
embedding extraction and reconstruction.
"""

import torch
import numpy as np
from pathlib import Path

from edamame.pipeline import EdamamePipeline


def example_basic_usage():
    """Basic example of loading a model and extracting embeddings."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Load pretrained model
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="auto",
        torch_dtype=torch.float32,
    )
    
    print(f"\nLoaded pipeline:\n{pipeline}")
    
    # Create dummy input data
    # Shape: (batch_size, seq_len, in_channels)
    batch_size = 4
    seq_len = 240  # 60 seconds at 4 Hz
    in_channels = 1
    
    batch_data = torch.randn(batch_size, seq_len, in_channels)
    print(f"\nInput shape: {batch_data.shape}")
    
    # Extract embeddings (no masking)
    embeddings = pipeline.embed(batch_data)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    # You can also get the first sample's embedding
    single_embedding = pipeline.embed(batch_data)[0]
    print(f"Single sample embedding shape: {single_embedding.shape}")
    
    print("\n✓ Basic usage successful!\n")
    return pipeline, batch_data


def example_with_masking():
    """Example showing embedding extraction with masking."""
    print("=" * 80)
    print("Example 2: Embedding with Masking")
    print("=" * 80)
    
    # Load model
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="auto",
    )
    
    # Create input
    batch_data = torch.randn(2, 240, 1)
    
    # Extract embeddings without masking (default)
    embeddings_full = pipeline.embed(batch_data, mask_ratio=0.0)
    print(f"Full embeddings (mask_ratio=0.0): {embeddings_full.shape}")
    
    # Extract embeddings with masking (like during training)
    embeddings_masked = pipeline.embed(batch_data, mask_ratio=0.75)
    print(f"Masked embeddings (mask_ratio=0.75): {embeddings_masked.shape}")
    
    # Get embeddings with mask information
    embeddings, mask, ids_restore = pipeline.embed(
        batch_data, mask_ratio=0.75, return_mask=True
    )
    print(f"\nWith mask info:")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  IDs restore: {ids_restore.shape}")
    
    print("\n✓ Masking example successful!\n")


def example_reconstruction():
    """Example showing signal reconstruction."""
    print("=" * 80)
    print("Example 3: Signal Reconstruction")
    print("=" * 80)
    
    # Load model
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="auto",
    )
    
    # Create input
    batch_data = torch.randn(2, 240, 1)
    print(f"Input shape: {batch_data.shape}")
    
    # Reconstruct with default masking
    reconstructed = pipeline.reconstruct(batch_data)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Reconstruct with custom mask ratio
    reconstructed_75 = pipeline.reconstruct(batch_data, mask_ratio=0.75)
    print(f"Reconstructed (75% masked) shape: {reconstructed_75.shape}")
    
    # Compute reconstruction error
    mse = torch.mean((batch_data.to(reconstructed.device) - reconstructed) ** 2)
    print(f"\nReconstruction MSE: {mse.item():.6f}")
    
    print("\n✓ Reconstruction example successful!\n")


def example_different_loading_methods():
    """Example showing different ways to load the model."""
    print("=" * 80)
    print("Example 4: Different Loading Methods")
    print("=" * 80)
    
    # Method 1: With config file path
    print("\nMethod 1: Load with config file path")
    pipeline1 = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="cpu",
    )
    print(f"✓ Loaded: {pipeline1.model.__class__.__name__}")
    
    # Method 2: With config dict
    print("\nMethod 2: Load with config dictionary")
    config_dict = {
        "seq_len": 240,
        "in_channels": 1,
        "patch_size": 10,
        "embed_dim": 512,
        "encoder_depth": 4,
        "encoder_num_heads": 4,
        "decoder_embed_dim": 512,
        "decoder_depth": 4,
        "decoder_num_heads": 4,
        "mlp_ratio": 4.0,
        "mask_ratio": 0.75,
        "dropout": 0.1,
    }
    pipeline2 = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config=config_dict,
        device_map="cpu",
    )
    print(f"✓ Loaded: {pipeline2.model.__class__.__name__}")
    
    # Method 3: Without config (auto-infer from checkpoint)
    print("\nMethod 3: Load without config (auto-infer)")
    pipeline3 = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config=None,
        device_map="cpu",
    )
    print(f"✓ Loaded: {pipeline3.model.__class__.__name__}")
    print(f"  Inferred config: {pipeline3.model_config}")
    
    print("\n✓ All loading methods successful!\n")


def example_batch_processing():
    """Example showing batch processing of multiple samples."""
    print("=" * 80)
    print("Example 5: Batch Processing")
    print("=" * 80)
    
    # Load model
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="auto",
    )
    
    # Process multiple batches
    num_batches = 3
    batch_size = 8
    all_embeddings = []
    
    print(f"\nProcessing {num_batches} batches of {batch_size} samples each...")
    
    for i in range(num_batches):
        # Create batch
        batch_data = torch.randn(batch_size, 240, 1)
        
        # Extract embeddings
        embeddings = pipeline.embed(batch_data)
        all_embeddings.append(embeddings.cpu())
        
        print(f"  Batch {i+1}: {embeddings.shape}")
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"\nTotal embeddings: {all_embeddings.shape}")
    print(f"Total samples processed: {all_embeddings.shape[0]}")
    
    print("\n✓ Batch processing successful!\n")


def example_callable_interface():
    """Example showing the callable interface."""
    print("=" * 80)
    print("Example 6: Callable Interface")
    print("=" * 80)
    
    # Load model
    pipeline = EdamamePipeline.from_pretrained(
        model_name="mae",
        weights_path="checkpoints.nosync/last.ckpt",
        model_config="configs/train/model/mae.yaml",
        device_map="auto",
    )
    
    batch_data = torch.randn(4, 240, 1)
    
    # Use pipeline as a callable
    print("\nUsing pipeline as callable:")
    
    # For embedding
    embeddings = pipeline(batch_data, task="embed")
    print(f"  Embeddings via __call__: {embeddings.shape}")
    
    # For reconstruction
    reconstructed = pipeline(batch_data, task="reconstruct", mask_ratio=0.75)
    print(f"  Reconstructed via __call__: {reconstructed.shape}")
    
    print("\n✓ Callable interface successful!\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" EdamamePipeline API Examples")
    print("=" * 80 + "\n")
    
    # Check if checkpoint exists
    checkpoint_path = Path("checkpoints.nosync/last.ckpt")
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have a trained model checkpoint.")
        return
    
    try:
        # Run examples
        example_basic_usage()
        example_with_masking()
        example_reconstruction()
        example_different_loading_methods()
        example_batch_processing()
        example_callable_interface()
        
        print("\n" + "=" * 80)
        print(" All examples completed successfully! ✓")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
