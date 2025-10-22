# Test Suite for SelfSupervisedEngine

This directory contains comprehensive tests for the `SelfSupervisedEngine` PyTorch Lightning module.

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run with verbose output:
```bash
python -m pytest tests/ -v
```

Run specific test file:
```bash
python -m pytest tests/test_engine.py
```

Run specific test class:
```bash
python -m pytest tests/test_engine.py::TestSelfSupervisedEngineInitialization
```

Run specific test:
```bash
python -m pytest tests/test_engine.py::TestSelfSupervisedEngineInitialization::test_basic_initialization
```

## Test Coverage

### `test_engine.py`

Comprehensive test suite for the `SelfSupervisedEngine` class with **22 tests** covering:

#### 1. Initialization Tests (4 tests)
- ✅ Basic initialization with minimal arguments
- ✅ Initialization with custom optimizer
- ✅ Initialization with scheduler
- ✅ Initialization with metrics

#### 2. Forward Pass Tests (2 tests)
- ✅ Forward method
- ✅ Predict method

#### 3. Training Tests (4 tests)
- ✅ Training step with `training_step` method
- ✅ Training step with `loss` method
- ✅ Validation step
- ✅ Test step

#### 4. Metrics Tests (2 tests)
- ✅ Metrics update and logging
- ✅ Engine works without metrics

#### 5. Batch Processing Tests (2 tests)
- ✅ Batch transfer hook with 'x' key
- ✅ Batch transfer hook with other keys

#### 6. Optimizer Tests (4 tests)
- ✅ Default optimizer configuration
- ✅ Custom optimizer configuration
- ✅ Optimizer with scheduler
- ✅ Optimizer with monitor metric

#### 7. Error Handling Tests (2 tests)
- ✅ Model without required methods raises error
- ✅ NaN loss warning

#### 8. Integration Tests (2 tests)
- ✅ Full training workflow
- ✅ Train/val/test workflow

## Test Fixtures

The test suite includes several mock models:

- **MockModelWithTrainingStep**: Model implementing `training_step` method
- **MockModelWithLossMethod**: Model implementing `loss` method
- **MockModelWithMetrics**: Model with custom metric extraction via `get_predictions_for_metrics`

## Test Data

- **simple_batch**: Basic batch with random tensors for regression tasks
- **classification_batch**: Batch with class labels for classification tasks

## Dependencies

Tests require:
- `pytest`
- `torch`
- `torchmetrics`
- `pytorch-lightning`

Install with:
```bash
pip install pytest torch torchmetrics pytorch-lightning
```

## Test Results

All 22 tests pass successfully ✅

```
====================================== 22 passed in 1.61s ======================================
```
