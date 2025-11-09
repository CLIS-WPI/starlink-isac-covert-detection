# Tests Directory

This directory contains comprehensive tests for the covert leakage detection pipeline, organized according to CI/CD best practices.

## Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── test_csi_estimation.py
│   └── test_pattern_selection.py
├── integration/       # Integration tests (medium speed, multiple components)
│   ├── test_dataset_generation.py
│   ├── test_eq_pipeline.py
│   └── test_detection_sanity.py
├── e2e/              # End-to-end tests (slow, full pipeline)
│   ├── test_scenario_a.py
│   ├── test_scenario_b.py
│   ├── test_complete_pipeline_legacy.py
│   └── test_end_to_end_legacy.py
├── conftest.py       # Shared pytest fixtures
├── pytest.ini       # Pytest configuration
└── README.md         # This file
```

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run by category
```bash
# Unit tests only (fast)
pytest tests/unit/ -v -m unit

# Integration tests
pytest tests/integration/ -v -m integration

# End-to-end tests (slow)
pytest tests/e2e/ -v -m e2e
```

### Run specific test file
```bash
pytest tests/unit/test_pattern_selection.py -v
```

### Run with markers
```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only fast tests
pytest tests/ -v -m "unit"
```

## Test Categories

### Unit Tests (`tests/unit/`)
Fast, isolated tests for individual functions and classes:
- `test_csi_estimation.py`: CSI estimation and MMSE equalization
- `test_pattern_selection.py`: Pattern selection and injection logic

**Markers:** `@pytest.mark.unit`

### Integration Tests (`tests/integration/`)
Tests that verify multiple components working together:
- `test_dataset_generation.py`: Dataset generation pipeline
- `test_eq_pipeline.py`: Complete EQ pipeline (CSI + MMSE)
- `test_detection_sanity.py`: Detection sanity checks

**Markers:** `@pytest.mark.integration`

### End-to-End Tests (`tests/e2e/`)
Full pipeline tests that generate datasets and verify complete workflows:
- `test_scenario_a.py`: Scenario A (single-hop) end-to-end
- `test_scenario_b.py`: Scenario B (dual-hop) with all patterns

**Markers:** `@pytest.mark.e2e`, `@pytest.mark.slow`

## CI/CD Integration

The `.gitlab-ci.yml` file automatically runs tests on merge requests:

```yaml
test:
  script:
    - python3 -m pytest tests/ -v --tb=short
```

## Test Fixtures

Shared fixtures are defined in `conftest.py`:
- `workspace_root`: Workspace root directory
- `test_data_dir`: Temporary directory for test data
- `clean_test_env`: Clean test environment (auto-cleanup)
- `mock_resource_grid`: Mock resource grid for testing
- `sample_ofdm_grid`: Sample OFDM grid
- `sample_injection_info`: Sample injection info
- `pattern_configs`: All pattern configurations
- `scenario_configs`: Scenario configurations

## Expected Metrics

### Scenario A (Single-hop)
- Pattern Preservation: ≈1.0
- AUC: ≈1.0

### Scenario B (Dual-hop)
- Pattern Preservation: median ≈0.48-0.50
- SNR Improvement: mean ≈30-40 dB
- Alpha Ratio: 100% in range 0.1x-3x
- injection_info: stored in metadata

## Pattern Types Supported

1. **contiguous** (mid): Fixed subcarriers 24-39
2. **random** (random16): Random 16 contiguous subcarriers
3. **hopping**: Frequency hopping (different subcarriers per symbol)
4. **sparse**: Random non-contiguous subcarriers

## Adding New Tests

1. **Unit tests**: Add to `tests/unit/` for isolated function/class tests
2. **Integration tests**: Add to `tests/integration/` for multi-component tests
3. **E2E tests**: Add to `tests/e2e/` for full pipeline tests

Use appropriate markers:
```python
@pytest.mark.unit
def test_something():
    ...

@pytest.mark.integration
@pytest.mark.slow
def test_integration():
    ...
```

## Troubleshooting

### Tests not discovered
- Ensure test files start with `test_`
- Ensure test functions start with `test_`
- Check `pytest.ini` configuration

### Import errors
- Ensure `sys.path` includes workspace root
- Check that `conftest.py` is in `tests/` directory

### Slow tests
- Use `-m "not slow"` to skip slow tests during development
- Run specific test categories: `pytest tests/unit/ -v`
