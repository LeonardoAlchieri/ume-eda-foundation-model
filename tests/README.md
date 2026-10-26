# Tests

Pytest suite for the `ume` package. All tests run on synthetic data built
in tmp dirs: no private data (`*.nosync`), no network, no GPU required.

## Running

```bash
# with the pixi environment
pixi run python -m pytest tests/

# or directly with the env interpreter
.pixi/envs/default/bin/python -m pytest tests -q -p no:cacheprovider

# one file / class / test
python -m pytest tests/test_engine.py
python -m pytest tests/test_engine.py::TestContrastiveEngine
python -m pytest tests/test_engine.py::TestContrastiveEngine::test_contrastive_training_step
```

Pytest options live in `pyproject.toml` (`[tool.pytest.ini_options]`: `-v`,
`--strict-markers`, `--tb=short`).

## Files

| File | Covers |
|------|--------|
| `test_engine.py` | `BaseSelfSupervisedEngine` (via a minimal `SimpleEngine` subclass, since `forward` is abstract): init/`seed`, optimizer and scheduler config, `monitor`, metrics, train/val/test steps, NaN warning, `on_after_batch_transfer` (`batch_size`, scaling, errors). `ContrastiveEngine` (`x_1`/`x_2` views) and `ReconstructionEngine` (`x_aug`) hooks and steps. |
| `test_data_augmentation.py` | Train-only augmentation in `ContrastiveEngine` and `ReconstructionEngine` for 2D `(N,T)` and 3D `(N,T,C)` batches. Augmentations are NumPy callables applied per sample through `np.vectorize`. |
| `test_contrastive_embedding_collection.py` | `ContrastiveEngine` embedding collection for UMAP plots: per-user limits, sample selection, train/val storage, `plotting_kwargs` validation. |
| `test_dataset.py` | Offline `EDAtaNoLabel`: loads `.npz` files with `values` `(N,T,C)` and `groups` `(N,)`, concatenation, channel selection and squeezing, per-dataset user encoding, shuffling, `DataLoader` use, fractional `get_splits`. |
| `test_dataset_online_mapping.py` | `EDAtaNoLabelOnlineSegmentation` on synthetic HDF5 stores (one key per recording, a `Timestamp` column, and `metadata` attrs): window counts, `idx_to_series`/`idx_to_position`/`series_to_idx`, segment extraction, skipping short recordings, `subsample`, `user_session_splitting` without recording leakage. |
| `test_mae_loss.py` | `DefaultMAELoss`, which returns `(total, masked, visible)`: MAE/MSE values, `masked_weight` mixing, edge cases, gradients. Also checks integration with `MaskedAutoencoder`. |
| `test_matton_augment_torch.py` | The torch Matton augmentations: output contract, plus statistical and spectral agreement with the NumPy versions. |
| `test_pipeline.py` | `UMEPipeline`: `from_pretrained` (MAE, with and without config), device mapping (`auto` picks MPS, then CUDA, then CPU), `embed`/`reconstruct`, errors, and `fit(refit_on_full_data=True)`. |

## Skipped tests

- `test_contrastive_embedding_collection.py::TestEpochEndHooks` is **skipped**
  because the UMAP epoch-end hooks in `ume/engine/contrastive.py` are
  commented out. Re-enable the tests together with the hooks.
