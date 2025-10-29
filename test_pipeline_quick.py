"""
Quick test to verify the EdamamePipeline API works correctly.
Run this script to test the basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import EdamamePipeline


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("=" * 80)
    print("Testing EdamamePipeline API")
    print("=" * 80)
    
    # Check for checkpoint
    checkpoint_path = "checkpoints.nosync/last.ckpt"
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("Please ensure you have a trained checkpoint.")
        return False
    
    try:
        print("\n1. Loading model...")
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_path,
            model_config="configs/train/model/mae.yaml",
            device_map="cpu",  # Use CPU for testing
        )
        print("   ✓ Model loaded successfully")
        print(f"   {pipeline}")
        
        print("\n2. Creating test data...")
        batch_size = 4
        seq_len = 240
        in_channels = 1
        batch_data = torch.randn(batch_size, seq_len, in_channels)
        print(f"   ✓ Created data with shape: {batch_data.shape}")
        
        print("\n3. Testing embed() method...")
        embeddings = pipeline.embed(batch_data)
        print(f"   ✓ Embeddings shape: {embeddings.shape}")
        print(f"   ✓ Embeddings dtype: {embeddings.dtype}")
        
        print("\n4. Testing embed() with masking...")
        embeddings_masked = pipeline.embed(batch_data, mask_ratio=0.75)
        print(f"   ✓ Masked embeddings shape: {embeddings_masked.shape}")
        
        print("\n5. Testing reconstruct() method...")
        reconstructed = pipeline.reconstruct(batch_data)
        print(f"   ✓ Reconstructed shape: {reconstructed.shape}")
        assert reconstructed.shape == batch_data.shape, "Shape mismatch!"
        
        print("\n6. Testing single sample access...")
        single_embedding = pipeline.embed(batch_data)[0]
        print(f"   ✓ Single embedding shape: {single_embedding.shape}")
        
        print("\n7. Testing callable interface...")
        embeddings_call = pipeline(batch_data, task="embed")
        reconstructed_call = pipeline(batch_data, task="reconstruct")
        print(f"   ✓ Callable embed shape: {embeddings_call.shape}")
        print(f"   ✓ Callable reconstruct shape: {reconstructed_call.shape}")
        
        print("\n" + "=" * 80)
        print("✓ All tests passed successfully!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
