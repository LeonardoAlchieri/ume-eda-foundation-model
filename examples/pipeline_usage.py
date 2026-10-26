"""Minimal UME pipeline usage example.

This script shows only the base UME inference path:
- load UME from artifacts/ume_base_inference.pt
- extract embeddings from a dummy EDA batch
"""

from pathlib import Path
from sys import path
path.append("./")
import torch

from ume.pipeline import UMEPipeline


def main() -> None:
    """Run a single base UME embedding example."""
    print("\n" + "=" * 80)
    print(" UME Pipeline: Base Model Example")
    print("=" * 80 + "\n")

    weights_path = Path("artifacts/ume_base_inference.pt")
    if not weights_path.exists():
        print(f"Error: checkpoint not found at {weights_path}")
        print("Please place the base UME weights at artifacts/ume_base_inference.pt.")
        return

    pipeline = UMEPipeline.from_pretrained(
        model_name="efficientnet",
        weights_path=str(weights_path)
    )

    print(f"Loaded pipeline:\n{pipeline}\n")

    # Shape: (batch_size, seq_len, channels)
    x = torch.randn(8, 240, 3)
    embeddings = pipeline.embed(x)

    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print("\n✓ Base UME example completed successfully.\n")


if __name__ == "__main__":
    main()
