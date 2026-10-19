# Test Suite Summary

Last full run: `231 passed, 2 skipped` (about 16 s on CPU/MPS).

| File | Tests | Status |
|------|------:|--------|
| `test_engine.py` | 32 | pass |
| `test_data_augmentation.py` | 10 | pass |
| `test_contrastive_embedding_collection.py` | 28 | 26 pass, 2 skipped (epoch-end plotting hooks disabled) |
| `test_dataset.py` | 29 | pass |
| `test_dataset_online_mapping.py` | 63 | pass |
| `test_mae_loss.py` | 22 | pass |
| `test_matton_augment_torch.py` | 29 | pass |
| `test_pipeline.py` | 20 | pass |

## Conventions

- All data is synthetic, built in pytest tmp dirs. No `*.nosync` data and no network.
- Engines require a `seed` argument. `BaseSelfSupervisedEngine.forward` is
  abstract, so the tests use a small concrete subclass.
- When a test calls a step method directly, its batch must already contain
  `batch_size`. In training this key comes from `on_after_batch_transfer`.
- See `README.md` for per-file coverage and how to run.
