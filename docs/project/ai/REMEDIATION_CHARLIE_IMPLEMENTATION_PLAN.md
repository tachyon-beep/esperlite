# Remediation Plan Charlie - Detailed Implementation Plan

**Created**: 2025-07-23
**Status**: ACTIVE
**Start Date**: 2025-07-23

## Overview

This document provides the detailed implementation plan for Remediation Plan Charlie, focusing on aligning the test suite with the actual implementation design rather than modifying code to satisfy tests.

## Phase C1: Contract-Implementation Alignment

### Objective
Fix seed_orchestrator.py to handle AdaptationDecision types according to the operational contract, not test expectations.

### Pre-Implementation Analysis

1. **Contract Definition** (operational.py:148-152)
   ```python
   adaptation_type: str = Field(
       pattern=r"^(add_seed|remove_seed|modify_architecture|optimize_parameters)$",
   )
   ```
   Valid types: `add_seed`, `remove_seed`, `modify_architecture`, `optimize_parameters`

2. **Current Implementation Issues** (seed_orchestrator.py:207-227)
   - Checks for `"add_neurons"` → Should map to `"add_seed"`
   - Checks for `"remove_neurons"` → Should map to `"remove_seed"`  
   - Checks for `"add_layer"` → Should map to `"modify_architecture"`
   - Default case maps to `REPLACE` → Should map `"optimize_parameters"`

3. **Test Expectations** (test_seed_orchestrator.py)
   - Tests use invalid adaptation types
   - Tests include forbidden fields in AdaptationDecision
   - Tests missing required `urgency` field

### Implementation Steps

#### Step 1: Fix seed_orchestrator.py adaptation type handling
```python
# File: src/esper/core/seed_orchestrator.py
# Lines: 206-228

# CURRENT (INCORRECT):
if decision.adaptation_type == "add_neurons":
    strategy = SeedStrategy.DIVERSIFY
elif decision.adaptation_type == "remove_neurons":
    strategy = SeedStrategy.SPECIALIZE
elif decision.adaptation_type == "add_layer":
    strategy = SeedStrategy.ENSEMBLE
else:
    strategy = SeedStrategy.REPLACE

# NEW (CORRECT):
if decision.adaptation_type == "add_seed":
    strategy = SeedStrategy.DIVERSIFY
elif decision.adaptation_type == "remove_seed":
    strategy = SeedStrategy.SPECIALIZE
elif decision.adaptation_type == "modify_architecture":
    strategy = SeedStrategy.ENSEMBLE
elif decision.adaptation_type == "optimize_parameters":
    strategy = SeedStrategy.REPLACE
else:
    # This should never happen due to Pydantic validation
    raise ValueError(f"Invalid adaptation type: {decision.adaptation_type}")
```

#### Step 2: Update decision creation in _execute_modification_plan
```python
# File: src/esper/core/seed_orchestrator.py
# Lines: 414-426

# Remove forbidden fields and use correct schema:
decision = AdaptationDecision(
    layer_name=plan.layer_name,
    adaptation_type="optimize_parameters",  # Valid type for kernel selection
    confidence=0.8,
    urgency=0.5,  # Add required field
    metadata={
        "category_preference": modification.get("category_preference", 0),
        "seed_idx": seed_idx,
        "reasoning": modification["reasoning"]
    }
)
```

#### Step 3: Add contract validation test
```python
# New test to ensure only valid adaptation types accepted
def test_adaptation_type_validation():
    """Test that only contract-valid adaptation types are accepted."""
    valid_types = ["add_seed", "remove_seed", "modify_architecture", "optimize_parameters"]
    invalid_types = ["add_neurons", "remove_neurons", "add_layer", "unknown"]
    
    for valid_type in valid_types:
        # Should not raise
        decision = AdaptationDecision(
            layer_name="test",
            adaptation_type=valid_type,
            confidence=0.8,
            urgency=0.5
        )
    
    for invalid_type in invalid_types:
        with pytest.raises(ValidationError):
            decision = AdaptationDecision(
                layer_name="test",
                adaptation_type=invalid_type,
                confidence=0.8,
                urgency=0.5
            )
```

### Rollback Plan
If issues arise:
1. Git revert the changes
2. Document any unexpected dependencies
3. Create compatibility layer if needed

## Phase C2: Test Schema Updates

### Objective
Update all tests to use the current AdaptationDecision schema without forbidden fields.

### Pre-Implementation Analysis

1. **Forbidden Fields in Tests**
   - `decision_id` - Not in current schema
   - `parameters` - Should use `metadata` dict
   - `reasoning` - Should be in `metadata`
   - `model_graph_state` - Not in schema
   - `epoch` - Not in schema

2. **Required Fields Missing in Tests**
   - `urgency` - Required field (0.0-1.0)
   - `timestamp` - Auto-generated if not provided

3. **Files to Update**
   - `tests/core/test_seed_orchestrator.py`
   - `tests/services/tamiyo/test_analyzer.py`
   - `tests/services/tamiyo/test_performance_tracker.py`
   - Any other files using AdaptationDecision

### Implementation Steps

#### Step 1: Create migration helper function
```python
# File: tests/helpers/adaptation_helpers.py

def create_valid_adaptation_decision(
    layer_name: str,
    adaptation_type: str,
    confidence: float = 0.8,
    urgency: float = 0.5,
    parameters: Optional[Dict] = None,
    reasoning: Optional[str] = None,
    **kwargs
) -> AdaptationDecision:
    """Create a valid AdaptationDecision with proper schema."""
    metadata = {}
    if parameters:
        metadata["parameters"] = parameters
    if reasoning:
        metadata["reasoning"] = reasoning
    
    # Map old adaptation types to new ones
    type_mapping = {
        "add_neurons": "add_seed",
        "remove_neurons": "remove_seed",
        "add_layer": "modify_architecture",
    }
    
    adaptation_type = type_mapping.get(adaptation_type, adaptation_type)
    
    return AdaptationDecision(
        layer_name=layer_name,
        adaptation_type=adaptation_type,
        confidence=confidence,
        urgency=urgency,
        metadata=metadata if metadata else {}
    )
```

#### Step 2: Update test_seed_orchestrator.py
```python
# Replace all AdaptationDecision instantiations

# OLD:
decision = AdaptationDecision(
    decision_id="test_001",
    layer_name="layer1",
    adaptation_type="add_neurons",
    confidence=0.85,
    parameters={"num_seeds": 2},
    reasoning="Layer showing high activation variance",
    model_graph_state=ModelGraphState(...),
    epoch=10
)

# NEW:
decision = create_valid_adaptation_decision(
    layer_name="layer1",
    adaptation_type="add_seed",
    confidence=0.85,
    urgency=0.7,
    parameters={"num_seeds": 2},
    reasoning="Layer showing high activation variance"
)
```

#### Step 3: Update test fixtures
```python
@pytest.fixture
def adaptation_decision():
    """Create test adaptation decision with valid schema."""
    return create_valid_adaptation_decision(
        layer_name="layer1",
        adaptation_type="add_seed",
        confidence=0.85,
        urgency=0.7,
        parameters={"num_seeds": 2},
        reasoning="Layer showing high activation variance"
    )
```

#### Step 4: Search and replace patterns
```bash
# Find all AdaptationDecision usages
grep -r "AdaptationDecision(" tests/ --include="*.py"

# Common patterns to fix:
# decision_id= → remove
# parameters= → move to metadata
# reasoning= → move to metadata
# adaptation_type="add_neurons" → "add_seed"
# adaptation_type="remove_neurons" → "remove_seed"
# adaptation_type="add_layer" → "modify_architecture"
```

### Validation Steps
1. Run each updated test individually
2. Verify no Pydantic validation errors
3. Ensure test logic still valid with new schema

## Phase C5: Test Suite Modernization

### Objective
Reduce mock usage and increase integration testing coverage.

### Pre-Implementation Analysis

1. **Current Mock Usage**
   - Global aiohttp mock in conftest.py
   - Extensive mocking of internal components
   - Tests don't reflect real usage patterns

2. **Integration Points Needing Tests**
   - Kernel compilation → execution pipeline
   - Seed selection → blueprint loading
   - Architecture modification → training loop
   - Cache → kernel loading

### Implementation Steps

#### Step 1: Create test utilities for real components
```python
# File: tests/helpers/test_infrastructure.py

class TestInfrastructure:
    """Provides real components for integration testing."""
    
    @staticmethod
    async def create_test_redis():
        """Create in-memory Redis for testing."""
        import fakeredis.aioredis
        return await fakeredis.aioredis.create_redis_pool()
    
    @staticmethod
    async def create_test_postgres():
        """Create in-memory PostgreSQL for testing."""
        # Use testing.postgresql or similar
        pass
    
    @staticmethod
    def create_test_kernel_cache():
        """Create cache with test backends."""
        return PersistentKernelCache(
            config=CacheConfig(
                redis_enabled=False,  # Use memory only for tests
                postgres_enabled=False,
                memory_cache_size=100
            )
        )
```

#### Step 2: Create integration test suites
```python
# File: tests/integration/test_compilation_execution_pipeline.py

class TestCompilationExecutionPipeline:
    """Test real compilation to execution flow."""
    
    async def test_blueprint_to_execution(self):
        """Test full pipeline without mocks."""
        # 1. Create real components
        compiler = BlueprintCompiler()
        executor = KernelExecutor()
        
        # 2. Compile blueprint
        blueprint = create_test_blueprint()
        kernel = await compiler.compile(blueprint)
        
        # 3. Execute kernel
        input_data = torch.randn(1, 128)
        output = await executor.execute(kernel, input_data)
        
        # 4. Verify output
        assert output.shape == (1, 256)
        assert not torch.isnan(output).any()
```

#### Step 3: Remove unnecessary mocks
```python
# File: tests/conftest.py

# REMOVE:
@pytest.fixture(autouse=True)
def mock_aiohttp_globally():
    """Remove this global mock."""
    pass

# ADD:
@pytest.fixture
def real_http_client():
    """Provide real HTTP client for integration tests."""
    return AsyncHttpClient(timeout=5.0)
```

### Test Categories

1. **Unit Tests** (with minimal mocks)
   - Single component functionality
   - Edge cases and error handling
   - Contract validation

2. **Integration Tests** (no mocks)
   - Component interactions
   - Data flow validation
   - Performance characteristics

3. **End-to-End Tests** (no mocks)
   - Full training scenarios
   - Architecture evolution
   - Recovery procedures

## Phase C6: Documentation and Standards

### Objective
Create comprehensive documentation to prevent future test-implementation drift.

### Implementation Steps

#### Step 1: API-First Development Guide
```markdown
# File: docs/development/API_FIRST_GUIDE.md

## API-First Development Process

1. **Define Contracts First**
   - Create Pydantic models in contracts/
   - Document all fields and constraints
   - Generate OpenAPI spec if applicable

2. **Generate Test Stubs**
   - Create tests from contract definitions
   - Test contract validation
   - Test edge cases

3. **Implement Against Contracts**
   - Use contracts as source of truth
   - Never modify contracts for implementation
   - Raise issues if contracts need change

4. **Validate Implementation**
   - Run contract validation tests
   - Check API compatibility
   - Document any deviations
```

#### Step 2: Test Development Standards
```markdown
# File: docs/development/TEST_STANDARDS.md

## Test Development Standards

### When to Use Mocks
- External services (HTTP, databases in unit tests)
- Time-sensitive operations
- Non-deterministic behavior

### When NOT to Use Mocks
- Internal component interactions
- Data transformations
- Business logic

### Integration Test Requirements
- Use real components where possible
- Test actual data flow
- Verify performance characteristics
- No global mocks
```

#### Step 3: CI/CD Enhancements
```yaml
# File: .github/workflows/contract-validation.yml

name: Contract Validation
on: [push, pull_request]

jobs:
  validate-contracts:
    steps:
      - name: Check contract compatibility
        run: |
          python scripts/validate_contracts.py
          
      - name: Check API surface
        run: |
          python scripts/check_api_exports.py
          
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v --no-mocks
```

## Execution Timeline

### Day 1 (Today)
- [x] Create detailed implementation plan
- [ ] Execute Phase C1 Step 1: Fix seed_orchestrator.py
- [ ] Execute Phase C1 Step 2: Update _execute_modification_plan
- [ ] Execute Phase C1 Step 3: Add validation tests
- [ ] Verify all tests pass with changes

### Day 2
- [ ] Execute Phase C2 Step 1: Create migration helper
- [ ] Execute Phase C2 Step 2: Update test_seed_orchestrator.py
- [ ] Execute Phase C2 Step 3: Update remaining test files
- [ ] Run full test suite and fix issues

### Day 3-4
- [ ] Execute Phase C5: Test modernization
- [ ] Create test infrastructure utilities
- [ ] Write integration test suites
- [ ] Remove unnecessary mocks

### Day 5
- [ ] Execute Phase C6: Documentation
- [ ] Create development guides
- [ ] Set up CI/CD enhancements
- [ ] Final validation and cleanup

## Success Criteria

1. **No Contract Violations**
   - seed_orchestrator only accepts valid adaptation types
   - All tests use correct AdaptationDecision schema
   - Pydantic validation passes everywhere

2. **Improved Test Quality**
   - <30% of tests use mocks
   - Full integration test coverage
   - Tests reflect real usage

3. **Clear Documentation**
   - API-first development guide
   - Test standards document
   - CI/CD automation

## Risk Mitigation

1. **Breaking Changes**
   - Run tests after each change
   - Keep changes atomic and revertable
   - Document all modifications

2. **Test Failures**
   - Fix root cause, not symptoms
   - Update tests to match design
   - Never modify implementation for tests

3. **Performance Impact**
   - Monitor test execution time
   - Use appropriate test backends
   - Balance coverage vs speed

## Next Steps

With this detailed plan complete, I will now begin implementation starting with Phase C1.