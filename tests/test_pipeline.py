"""
Unit tests for EdamamePipeline.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.pipeline import EdamamePipeline
from src.model.mae import MaskedAutoencoder


class TestEdamamePipeline:
    """Test suite for EdamamePipeline."""
    
    @pytest.fixture
    def mae_config(self):
        """MAE model configuration."""
        return {
            "seq_len": 240,
            "in_channels": 1,
            "patch_size": 10,
            "embed_dim": 128,
            "encoder_depth": 2,
            "encoder_num_heads": 4,
            "decoder_embed_dim": 64,
            "decoder_depth": 2,
            "decoder_num_heads": 4,
            "mlp_ratio": 2.0,
            "mask_ratio": 0.75,
            "dropout": 0.1,
        }
    
    @pytest.fixture
    def mae_model(self, mae_config):
        """Create a MAE model."""
        return MaskedAutoencoder(**mae_config)
    
    @pytest.fixture
    def checkpoint_file(self, mae_model):
        """Create a temporary checkpoint file."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            checkpoint_path = Path(f.name)
        
        # Save model checkpoint
        checkpoint = {
            "state_dict": mae_model.state_dict(),
            "epoch": 10,
        }
        torch.save(checkpoint, checkpoint_path)
        
        yield checkpoint_path
        
        # Cleanup
        if checkpoint_path.exists():
            checkpoint_path.unlink()
    
    def test_from_pretrained_with_config_dict(self, checkpoint_file, mae_config):
        """Test loading model with config dictionary."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        assert pipeline is not None
        assert pipeline.model_name == "mae"
        assert isinstance(pipeline.model, MaskedAutoencoder)
        assert pipeline.device.type == "cpu"
    
    def test_from_pretrained_without_config(self, checkpoint_file):
        """Test loading model without config (auto-infer)."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=None,
            device_map="cpu",
        )
        
        assert pipeline is not None
        assert pipeline.model_config is not None
        assert "seq_len" in pipeline.model_config
    
    def test_embed_basic(self, checkpoint_file, mae_config):
        """Test basic embedding extraction."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        # Create input
        batch_size = 2
        seq_len = mae_config["seq_len"]
        in_channels = mae_config["in_channels"]
        x = torch.randn(batch_size, seq_len, in_channels)
        
        # Extract embeddings
        embeddings = pipeline.embed(x)
        
        assert embeddings is not None
        assert embeddings.shape[0] == batch_size
        assert len(embeddings.shape) == 3
    
    def test_embed_no_masking(self, checkpoint_file, mae_config):
        """Test embedding extraction without masking."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # No masking
        embeddings = pipeline.embed(x, mask_ratio=0.0)
        
        # Should have all patches
        expected_num_patches = mae_config["seq_len"] // mae_config["patch_size"]
        assert embeddings.shape[1] == expected_num_patches
    
    def test_embed_with_masking(self, checkpoint_file, mae_config):
        """Test embedding extraction with masking."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # With 75% masking
        embeddings = pipeline.embed(x, mask_ratio=0.75)
        
        # Should have 25% of patches
        expected_num_patches = mae_config["seq_len"] // mae_config["patch_size"]
        expected_visible = int(expected_num_patches * 0.25)
        assert embeddings.shape[1] == expected_visible
    
    def test_embed_return_mask(self, checkpoint_file, mae_config):
        """Test embedding extraction with mask information."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # Get embeddings with mask
        result = pipeline.embed(x, mask_ratio=0.5, return_mask=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        embeddings, mask, ids_restore = result
        assert embeddings.shape[0] == 2
        assert mask.shape[0] == 2
        assert ids_restore.shape[0] == 2
    
    def test_reconstruct(self, checkpoint_file, mae_config):
        """Test signal reconstruction."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        batch_size = 2
        seq_len = mae_config["seq_len"]
        in_channels = mae_config["in_channels"]
        x = torch.randn(batch_size, seq_len, in_channels)
        
        # Reconstruct
        reconstructed = pipeline.reconstruct(x)
        
        assert reconstructed is not None
        assert reconstructed.shape == x.shape
    
    def test_reconstruct_with_mask_ratio(self, checkpoint_file, mae_config):
        """Test reconstruction with custom mask ratio."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # Reconstruct with different mask ratios
        recon_50 = pipeline.reconstruct(x, mask_ratio=0.5)
        recon_75 = pipeline.reconstruct(x, mask_ratio=0.75)
        
        assert recon_50.shape == x.shape
        assert recon_75.shape == x.shape
        # Results should be different due to different masking
        assert not torch.allclose(recon_50, recon_75)
    
    def test_callable_interface_embed(self, checkpoint_file, mae_config):
        """Test using pipeline as callable for embedding."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # Use callable interface
        embeddings = pipeline(x, task="embed")
        
        assert embeddings is not None
        assert embeddings.shape[0] == 2
    
    def test_callable_interface_reconstruct(self, checkpoint_file, mae_config):
        """Test using pipeline as callable for reconstruction."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        x = torch.randn(2, mae_config["seq_len"], mae_config["in_channels"])
        
        # Use callable interface
        reconstructed = pipeline(x, task="reconstruct")
        
        assert reconstructed is not None
        assert reconstructed.shape == x.shape
    
    def test_device_mapping_auto(self, checkpoint_file, mae_config):
        """Test automatic device mapping."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="auto",
        )
        
        # Should use GPU if available, otherwise CPU
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert pipeline.device.type == expected_device
    
    def test_device_mapping_cpu(self, checkpoint_file, mae_config):
        """Test CPU device mapping."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        assert pipeline.device.type == "cpu"
    
    def test_invalid_model_name(self, checkpoint_file, mae_config):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="not supported"):
            EdamamePipeline.from_pretrained(
                model_name="invalid_model",
                weights_path=checkpoint_file,
                model_config=mae_config,
                device_map="cpu",
            )
    
    def test_invalid_checkpoint_path(self, mae_config):
        """Test that invalid checkpoint path raises error."""
        with pytest.raises(FileNotFoundError):
            EdamamePipeline.from_pretrained(
                model_name="mae",
                weights_path="nonexistent_checkpoint.ckpt",
                model_config=mae_config,
                device_map="cpu",
            )
    
    def test_repr(self, checkpoint_file, mae_config):
        """Test string representation."""
        pipeline = EdamamePipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        repr_str = repr(pipeline)
        assert "EdamamePipeline" in repr_str
        assert "mae" in repr_str
        assert "MaskedAutoencoder" in repr_str
