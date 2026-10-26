"""
Unit tests for UMEPipeline.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from ume.pipeline import UMEPipeline
from ume.model.mae import MaskedAutoencoder


class TestUMEPipeline:
    """Test suite for UMEPipeline."""
    
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
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
        pipeline = UMEPipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="auto",
        )
        
        # Priority: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            expected_device = "mps"
        elif torch.cuda.is_available():
            expected_device = "cuda"
        else:
            expected_device = "cpu"
        assert pipeline.device.type == expected_device
    
    def test_device_mapping_cpu(self, checkpoint_file, mae_config):
        """Test CPU device mapping."""
        pipeline = UMEPipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        assert pipeline.device.type == "cpu"
    
    def test_invalid_model_name(self, checkpoint_file, mae_config):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="not supported"):
            UMEPipeline.from_pretrained(
                model_name="invalid_model",
                weights_path=checkpoint_file,
                model_config=mae_config,
                device_map="cpu",
            )
    
    def test_invalid_checkpoint_path(self, mae_config):
        """Test that invalid checkpoint path raises error."""
        with pytest.raises(FileNotFoundError):
            UMEPipeline.from_pretrained(
                model_name="mae",
                weights_path="nonexistent_checkpoint.ckpt",
                model_config=mae_config,
                device_map="cpu",
            )
    
    def test_repr(self, checkpoint_file, mae_config):
        """Test string representation."""
        pipeline = UMEPipeline.from_pretrained(
            model_name="mae",
            weights_path=checkpoint_file,
            model_config=mae_config,
            device_map="cpu",
        )
        
        repr_str = repr(pipeline)
        assert "UMEPipeline" in repr_str
        assert "mae" in repr_str
        assert "MaskedAutoencoder" in repr_str


class TestFitRefitOnFullData:
    """Test suite for `fit(..., refit_on_full_data=True)`."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline wrapping a tiny EfficientNet (UME) encoder."""
        from ume.model.efficientnet_contrastive import EfficientNetContrastive

        torch.manual_seed(0)
        model = EfficientNetContrastive(
            in_channels=1,
            seq_len=60,
            stem_channels=4,
            mbconv_channels=8,
            num_mbconv_blocks=1,
            head_channels=8,
            embedding_dim=16,
        )
        return UMEPipeline(
            model=model,
            model_name="efficientnet",
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def data(self):
        """Small, linearly separable two-class dataset."""
        torch.manual_seed(0)
        n = 40
        X = torch.randn(n, 60, 1)
        y = torch.randint(0, 2, (n,))
        X[y == 1] += 2.0
        return X, y.numpy()

    def test_requires_early_stopping(self, pipeline, data):
        X, y = data
        with pytest.raises(ValueError, match="early_stopping"):
            pipeline.fit(X, y, refit_on_full_data=True, num_epochs=1)

    def test_refit_with_validation_split(self, pipeline, data, capsys):
        X, y = data
        pipeline.fit(
            X,
            y,
            num_epochs=2,
            batch_size=8,
            validation_split=0.25,
            early_stopping=True,
            patience=5,
            refit_on_full_data=True,
        )
        out = capsys.readouterr().out
        assert "Refitting on the full dataset (40 samples)" in out
        preds = pipeline.predict(X)
        assert preds.shape == (40,)
        assert set(preds.tolist()) <= {0, 1}

    def test_refit_with_explicit_validation_data(self, pipeline, data, capsys):
        X, y = data
        pipeline.fit(
            X[:30],
            y[:30],
            num_epochs=2,
            batch_size=8,
            validation_data=(X[30:], y[30:]),
            early_stopping=True,
            refit_on_full_data=True,
        )
        out = capsys.readouterr().out
        # train (30) + validation (10) are concatenated for the refit
        assert "Refitting on the full dataset (40 samples)" in out

    def test_refit_step_budget_matches_best_epoch(self, pipeline, data, capsys):
        X, y = data
        pipeline.fit(
            X,
            y,
            num_epochs=3,
            batch_size=8,
            validation_split=0.25,
            early_stopping=True,
            patience=10,
            refit_on_full_data=True,
        )
        out = capsys.readouterr().out
        # 30 training samples / batch_size 8 -> 4 steps per phase-1 epoch
        import re

        match = re.search(r"for (\d+) optimizer steps \(best epoch (\d+)", out)
        assert match is not None
        steps, best_epoch = int(match.group(1)), int(match.group(2))
        assert steps == best_epoch * 4

    def test_refit_rejects_unknown_validation_labels(self, pipeline, data):
        X, y = data
        y_val = y[30:].copy()
        y_val[0] = 99  # label absent from the training set
        with pytest.raises(ValueError, match="labels not present"):
            pipeline.fit(
                X[:30],
                y[:30],
                num_epochs=1,
                validation_data=(X[30:], y_val),
                early_stopping=True,
                refit_on_full_data=True,
            )
