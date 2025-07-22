# Esper Test Suite

Comprehensive test coverage for the Esper morphogenetic training platform.

## Test Organization

```
tests/
├── unit/          # Unit tests for individual components
├── integration/   # Integration tests for component interactions
├── performance/   # Performance benchmarks and tests
├── contracts/     # Data contract validation tests
├── core/          # Core functionality tests
├── execution/     # Execution layer tests
├── services/      # Service-specific tests
└── utils/         # Utility function tests
```

## Running Tests

### All Tests
```bash
pytest
```

### Specific Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/

# With coverage
pytest --cov=src/esper --cov-report=html
```

### Specific Test Files
```bash
# Test morphogenetic wrapping
pytest tests/core/test_model_wrapper.py

# Test GNN policy
pytest tests/services/tamiyo/test_enhanced_policy_safety.py
```

## Test Requirements

- All new features must have tests
- Target: >90% code coverage
- Tests must be independent and deterministic
- Use appropriate fixtures from conftest.py
- Mock external dependencies

## Key Test Files

- `conftest.py` - Shared fixtures and test configuration
- `production_scenarios.py` - Real-world scenario tests
- `integration/test_phase*` - Phase-specific integration tests
- `performance/test_gnn_acceleration.py` - GNN performance validation

## Writing Tests

### Example Unit Test
```python
def test_kernel_execution(mock_kernel):
    executor = KernelExecutor()
    result = executor.execute(mock_kernel, input_data)
    assert result.success
    assert result.output.shape == expected_shape
```

### Example Integration Test
```python
@pytest.mark.integration
async def test_morphogenetic_adaptation(morphable_model):
    # Test full adaptation cycle
    initial_params = count_parameters(morphable_model)
    await trigger_adaptation(morphable_model)
    final_params = count_parameters(morphable_model)
    assert final_params > initial_params
```

## Test Infrastructure

Tests can run with or without external services:
- Mocked mode: No external dependencies required
- Integration mode: Requires Docker services running

See [Testing Documentation](../docs/testing/) for detailed strategies and guidelines.