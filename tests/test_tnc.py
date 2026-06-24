"""Tests for the Temporal Neighborhood Coding (TNC) objective.

Covers the PU discriminator loss, the TNC model forward/loss/representation
contract, and the same-recording neighbor / non-neighbor sampling of
``EDAtaTNCOnlineSegmentation``. All tests are self-contained (no HDF5 data).
"""

import numpy as np
import torch

from sys import path

path.append("./")
from edamame.model.loss.tnc import TNCLoss
from edamame.model.tnc import TNCModel
from edamame.model.efficientnet_contrastive import EfficientNetContrastive
from edamame.data.dataset_online import EDAtaTNCOnlineSegmentation


# --------------------------------------------------------------------------- #
# TNCLoss
# --------------------------------------------------------------------------- #
class TestTNCLoss:
    def _embeds(self, b=6, m=4, d=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        z_ref = torch.randn(b, d, generator=g)
        z_neigh = torch.randn(b, m, d, generator=g)
        z_non = torch.randn(b, m, d, generator=g)
        mask = torch.ones(b, m)
        return z_ref, z_neigh, z_non, mask

    def test_forward_is_finite_scalar(self):
        loss_fn = TNCLoss(embedding_dim=8)
        z_ref, z_neigh, z_non, mask = self._embeds()
        loss = loss_fn(z_ref, z_neigh, z_non, mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_gradients_reach_discriminator(self):
        loss_fn = TNCLoss(embedding_dim=8)
        z_ref, z_neigh, z_non, mask = self._embeds()
        loss = loss_fn(z_ref, z_neigh, z_non, mask)
        loss.backward()
        grads = [p.grad for p in loss_fn.discriminator.parameters()]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_zero_mask_is_finite(self):
        """A recording with no valid non-neighbors must not produce NaN/inf."""
        loss_fn = TNCLoss(embedding_dim=8)
        z_ref, z_neigh, z_non, _ = self._embeds()
        mask = torch.zeros_like(z_neigh[..., 0])  # (B, M) all invalid
        loss = loss_fn(z_ref, z_neigh, z_non, mask)
        assert torch.isfinite(loss)

    def test_cross_negatives_toggle_changes_loss(self):
        z_ref, z_neigh, z_non, mask = self._embeds()
        with_cross = TNCLoss(embedding_dim=8, cross_recording_negatives=True)
        no_cross = TNCLoss(embedding_dim=8, cross_recording_negatives=False)
        # Share discriminator weights so the only difference is the cross term.
        no_cross.discriminator.load_state_dict(with_cross.discriminator.state_dict())
        l1 = with_cross(z_ref, z_neigh, z_non, mask)
        l2 = no_cross(z_ref, z_neigh, z_non, mask)
        assert not torch.isclose(l1, l2)

    def test_invalid_w_raises(self):
        for bad in (-0.1, 1.0, 1.5):
            try:
                TNCLoss(embedding_dim=8, w=bad)
                assert False, f"w={bad} should raise"
            except ValueError:
                pass


# --------------------------------------------------------------------------- #
# TNCModel
# --------------------------------------------------------------------------- #
def _tiny_model(in_channels=2, embedding_dim=8):
    encoder = EfficientNetContrastive(
        in_channels=in_channels,
        seq_len=32,
        stem_channels=8,
        mbconv_channels=8,
        num_mbconv_blocks=2,
        head_channels=16,
        embedding_dim=embedding_dim,
        kernel_size=3,
        dropout=0.0,
        loss_fn=None,
    )
    loss_fn = TNCLoss(embedding_dim=embedding_dim)
    return TNCModel(encoder=encoder, loss_fn=loss_fn)


class TestTNCModel:
    def _batch(self, b=4, m=3, t=32, c=2):
        return {
            "x": torch.randn(b, t, c),
            "x_neighbors": torch.randn(b, m, t, c),
            "x_non_neighbors": torch.randn(b, m, t, c),
            "non_neighbor_mask": torch.ones(b, m),
            "user": torch.zeros(b, dtype=torch.long),
        }

    def test_forward_shapes(self):
        model = _tiny_model()
        batch = self._batch()
        out = model(batch)
        b, m, d = 4, 3, 8
        assert out["z_ref"].shape == (b, d)
        assert out["z_neighbors"].shape == (b, m, d)
        assert out["z_non_neighbors"].shape == (b, m, d)

    def test_loss_and_backward(self):
        model = _tiny_model()
        batch = self._batch()
        out = model(batch)
        loss = model.loss(out, batch)
        assert torch.isfinite(loss)
        loss.backward()
        # Encoder must receive gradient (it is being trained, not just the head).
        enc_grad = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in enc_grad)

    def test_get_representation(self):
        model = _tiny_model()
        z = model.get_representation(torch.randn(5, 32, 2))
        assert z.shape == (5, 8)

    def test_predictions_for_metrics_none(self):
        model = _tiny_model()
        preds, targets = model.get_predictions_for_metrics({}, {})
        assert preds is None and targets is None


# --------------------------------------------------------------------------- #
# EDAtaTNCOnlineSegmentation sampling
# --------------------------------------------------------------------------- #
class TestTNCSampling:
    def test_sample_within_respects_radius(self):
        offs = EDAtaTNCOnlineSegmentation._sample_within(
            center=50, radius=5, max_start=90, n=200
        )
        assert offs.min() >= 45 and offs.max() <= 55

    def test_sample_within_clamped_at_edges(self):
        offs = EDAtaTNCOnlineSegmentation._sample_within(
            center=0, radius=5, max_start=90, n=200
        )
        assert offs.min() >= 0 and offs.max() <= 5

    def test_sample_outside_respects_gap(self):
        offs, valid = EDAtaTNCOnlineSegmentation._sample_outside(
            center=50, gap=20, max_start=90, n=500
        )
        assert valid
        assert (offs - 50).abs().min() >= 20

    def test_sample_outside_invalid_for_short_recording(self):
        # max_start=0 (recording == segment_length): no far windows exist.
        offs, valid = EDAtaTNCOnlineSegmentation._sample_outside(
            center=0, gap=20, max_start=0, n=5
        )
        assert not valid
        assert torch.equal(offs, torch.zeros(5, dtype=torch.long))

    def _fake_dataset(self, length=100, segment_length=10, mc=4):
        """Build a TNC dataset around a single synthetic recording (no I/O)."""
        ds = EDAtaTNCOnlineSegmentation.__new__(EDAtaTNCOnlineSegmentation)
        rec = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
        ds.data = [{"user": 0, "data": rec, "len_data": length}]
        ds.segment_length = segment_length
        ds.shift = 1
        ds.channels = [0, 1, 2]
        ds.sample_scaling_method = None
        ds.mc_sample_size = mc
        ds.neighbor_range = 5
        ds.non_neighbor_gap = 20
        n_windows = (length - segment_length) // 1 + 1
        ds.idx_to_series = torch.zeros(n_windows, dtype=torch.long)
        ds.idx_to_position = torch.arange(n_windows, dtype=torch.long)
        return ds

    def test_getitem_shapes_and_mask(self):
        ds = self._fake_dataset()
        sample = ds[0]
        assert sample["x"].shape == (10, 3)
        assert sample["x_neighbors"].shape == (4, 10, 3)
        assert sample["x_non_neighbors"].shape == (4, 10, 3)
        assert sample["non_neighbor_mask"].shape == (4,)
        # ref at offset 0 has valid far windows in a length-100 recording.
        assert sample["non_neighbor_mask"].sum() == 4

    def test_getitem_short_recording_masks_non_neighbors(self):
        ds = self._fake_dataset(length=10, segment_length=10)
        sample = ds[0]
        assert sample["non_neighbor_mask"].sum() == 0
        # Window content is still a valid segment (placeholder, but well-formed).
        assert sample["x_non_neighbors"].shape == (4, 10, 3)
