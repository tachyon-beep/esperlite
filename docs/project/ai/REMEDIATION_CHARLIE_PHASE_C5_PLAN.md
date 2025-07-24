# Remediation Plan Charlie - Phase C5: Test Suite Modernization

**Created**: 2025-07-23
**Status**: ACTIVE

## Principles

1. **Minimal Mocks**: Only mock external services (HTTP, databases, message queues)
2. **Production Code Integrity**: Only change production code when it deviates from specification
3. **Meaningful Tests Only**: Remove tests that exist only for coverage, keep tests that verify useful functionality

## Current State Analysis

### Global Mocks (Anti-Pattern)
- `conftest.py` has 2 autouse fixtures that mock everything:
  - `prevent_real_network_calls` (line 219) - mocks all HTTP
  - `mock_oona_client_creation` (line 373) - mocks all OonaClient
- 354 mock usages across 26 test files
- Most tests mock internal components, not just external services

### Available Real Components
- `tests/fixtures/real_components.py` provides:
  - `real_kernel_cache`
  - `real_kernel_executor`
  - `real_kasmina_layer`
  - `TestKernelFactory` for creating real kernels
  - `RealComponentTestBase` with utilities

## Refactoring Strategy

### Step 1: Remove Global Mocks
1. Change autouse fixtures to opt-in:
   ```python
   # Change from:
   @pytest.fixture(autouse=True)
   def prevent_real_network_calls(request):
   
   # To:
   @pytest.fixture
   def mock_http_client():
       """Opt-in HTTP mocking for tests that need it."""
   ```

2. Add markers for tests that need real components:
   ```python
   @pytest.mark.real_components
   def test_something():
       # Uses real components
   ```

### Step 2: Identify Low-Value Tests

Tests to remove or refactor:
1. **Getter/Setter Tests**: Testing simple property access
2. **Constructor Tests**: Just verifying initialization
3. **Mock-Only Tests**: Tests that only verify mock calls
4. **Redundant Tests**: Multiple tests for same functionality

### Step 3: Refactor High-Value Tests

For each test file:
1. Identify what functionality is being tested
2. Determine if it provides value (finds bugs, ensures correctness)
3. Replace mocks with real components where possible
4. Keep mocks only for external services

## Specific File Refactoring Plan

### 1. test_seed_orchestrator.py
**Current**: All dependencies mocked
**Refactor**:
- Use real PerformanceTracker (in-memory)
- Use real BlueprintRegistry
- Mock only OonaClient and HTTP calls to Urza
- Remove simple getter tests

### 2. test_kernel_executor.py
**Current**: Likely mocking kernel loading
**Refactor**:
- Use real kernels from TestKernelFactory
- Test actual execution, not mock behavior
- Keep performance benchmarks

### 3. test_kasmina_layer.py
**Current**: Mocking kernel cache
**Refactor**:
- Use real_kasmina_layer fixture
- Test actual kernel loading and execution
- Verify blend factors affect output

### 4. Integration Tests
**Current**: Heavy mocking defeats purpose
**Refactor**:
- Use all real components
- Mock only external HTTP/Redis
- Test full workflows

## Implementation Steps

### Phase 1: Infrastructure Changes
1. Modify conftest.py to remove autouse mocks
2. Create opt-in mock fixtures
3. Add test markers for categorization

### Phase 2: Core Module Tests
1. Refactor test_seed_orchestrator.py
2. Refactor test_kernel_executor.py
3. Refactor test_kasmina_layer.py

### Phase 3: Service Tests
1. Identify which services need real testing
2. Create in-memory implementations where needed
3. Remove redundant service tests

### Phase 4: Integration Tests
1. Ensure integration tests use real components
2. Add end-to-end workflow tests
3. Performance benchmarks with real components

## Success Metrics

1. **Mock Reduction**: <30% of tests use mocks (from current ~80%)
2. **Test Value**: Every test catches real bugs or ensures correctness
3. **Performance**: Test suite runs in <5 minutes
4. **Coverage**: Maintain critical path coverage, remove trivial coverage

## Anti-Patterns to Avoid

1. **Testing Mocks**: `assert mock.called_with(...)`
2. **Testing Implementation**: Testing private methods
3. **Testing Framework**: Testing PyTorch/Pydantic behavior
4. **Coverage Theater**: Tests that just exercise code without assertions

## Example Refactoring

### Before (Too Many Mocks):
```python
def test_seed_orchestrator_apply_modification():
    mock_tracker = Mock()
    mock_registry = Mock()
    mock_layer = Mock()
    
    orchestrator = SeedOrchestrator(mock_tracker, mock_registry)
    orchestrator.apply_modification(mock_layer)
    
    assert mock_tracker.called
```

### After (Real Components):
```python
def test_seed_orchestrator_modifies_blend_factors():
    tracker = PerformanceTracker()  # Real, in-memory
    registry = BlueprintRegistry()   # Real
    layer = create_test_kasmina_layer()  # Real
    
    orchestrator = SeedOrchestrator(tracker, registry)
    
    # Set up real performance data
    tracker.record_metrics(layer_name="test", metrics={...})
    
    # Apply real modification
    decision = create_valid_adaptation_decision(
        layer_name="test",
        adaptation_type="add_seed"
    )
    success, details = orchestrator.apply_modification(layer, decision)
    
    # Verify actual behavior changed
    assert success
    assert layer.get_blend_factor(0) > 0  # Real change happened
```

## Next Steps

1. Start with conftest.py modifications
2. Pick one test file as proof of concept
3. Measure improvement in test quality
4. Roll out to other test files