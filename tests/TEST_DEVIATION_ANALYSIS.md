# Test Suite Deviation Analysis

## Overview

This document analyzes deviations between the test suite expectations and the main implementation, identifying where code was modified to work with tests rather than maintaining the original design.

## Identified Deviations

### 1. Missing API Function: `get_logger`

**Issue:** Phase B5 modules expect `esper.utils.logging.get_logger()` function
**Location:** `src/esper/storage/cache_backends.py`, `src/esper/storage/kernel_cache.py`, etc.
**Impact:** Import errors preventing test execution

**Analysis:**
- The logging module was implemented with `setup_logging()` and `setup_high_performance_logging()` functions
- Phase B5 code expects a simple `get_logger()` function that doesn't exist
- This indicates the Phase B5 implementation was developed against a different API

**Fix Applied:** Added `get_logger()` function to maintain compatibility
```python
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)
```

### 2. Missing Module Exports

**Issue:** Test imports expect certain classes to be exported from module `__init__.py` files
**Locations:** 
- `AssetMetadata` missing from `esper.storage.__init__.py`
- `RedisConfig`, `PostgreSQLConfig` missing from `esper.storage.cache_backends`
- `ComponentType` missing from `esper.recovery.__init__.py`

**Analysis:**
- Tests were written expecting certain APIs to be publicly exported
- Implementation modules didn't export all expected classes
- Indicates incomplete API surface definition

**Fix Applied:** Updated `__init__.py` files to export expected classes

### 3. AsyncHttpClient Event Loop Initialization

**Issue:** `AsyncHttpClient.__init__()` creates `aiohttp.TCPConnector` which requires running event loop
**Location:** `src/esper/utils/http_client.py` line 82
**Impact:** RuntimeError when instantiating outside async context

**Analysis:**
- The `TCPConnector` is created eagerly in `__init__` instead of lazily when needed
- This prevents synchronous instantiation of the client
- Tests expect to be able to create clients in fixtures without async context
- This is a fundamental design deviation - async resources should be created lazily

**Current State:** Tests use global mock in `conftest.py` to work around this

**Recommended Fix:**
```python
class AsyncHttpClient:
    def __init__(self, ...):
        # Store config but don't create connector
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_connections = max_connections
        self.connector = None  # Create lazily
        
    async def _ensure_session(self):
        if self.session is None:
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(...)
            self.session = aiohttp.ClientSession(...)
```

### 4. Missing Dependencies

**Issue:** Required packages not in requirements
**Missing:**
- `asyncpg` - Required by PostgreSQL cache backend
- `prometheus-client` - Required by Nissa service

**Analysis:**
- Phase B5 implementation added dependencies without updating requirements
- Indicates development was done in isolation without full integration testing

**Fix Applied:** Installed missing packages

## Design Principle Violations

### 1. Synchronous vs Asynchronous Initialization

The original design appears to support both sync and async contexts, but Phase B5 implementations force async-only initialization patterns. This breaks the principle of gradual adoption where components should work in both contexts.

### 2. API Surface Inconsistency

The test suite expects certain APIs that weren't properly defined in the implementation. This suggests:
- Lack of API-first design
- Tests written against assumed rather than actual interfaces
- Missing integration between phases

### 3. Dependency Management

External dependencies were added without proper tracking, suggesting:
- Incomplete build system integration
- Development done in isolated environments
- Missing CI/CD validation

## Impact Assessment

1. **Test Reliability:** Tests require extensive mocking to work around design issues
2. **Integration Difficulty:** Components can't be easily instantiated in different contexts
3. **Maintenance Burden:** Divergent APIs between test expectations and implementation

## Recommendations

1. **Lazy Initialization:** Refactor async components to support sync instantiation with lazy resource creation
2. **API Documentation:** Create explicit API contracts before implementation
3. **Dependency Tracking:** Update requirements.txt with all Phase B5 dependencies
4. **Integration Testing:** Add tests that verify components work together without mocks

### 5. Contract API Changes

**Issue:** Tests expect different AdaptationDecision schema than implemented
**Location:** `tests/core/test_seed_orchestrator.py`
**Impact:** Pydantic validation errors

**Analysis:**
- Tests use fields: `decision_id`, `parameters`, `reasoning` - all forbidden by `extra="forbid"`
- Tests use adaptation types: `"add_neurons"`, `"remove_neurons"` - not in allowed pattern
- Current allowed types: `"add_seed|remove_seed|modify_architecture|optimize_parameters"`
- Missing required field: `urgency`

**Test Expectations:**
```python
AdaptationDecision(
    decision_id="test_001",  # Not in schema
    layer_name="layer1", 
    adaptation_type="add_neurons",  # Invalid pattern
    confidence=0.85,
    parameters={"num_seeds": 2},  # Not in schema
    reasoning="Layer showing high activation variance",  # Not in schema
    model_graph_state=ModelGraphState(...),  # Not in schema
    epoch=10  # Not in schema
)
```

**Actual Schema:**
```python
AdaptationDecision(
    layer_name="layer1",
    adaptation_type="add_seed",  # Must match pattern
    confidence=0.85,
    urgency=0.5,  # Required
    metadata={},  # Optional dict
    timestamp=time.time()  # Auto-generated
)
```

This indicates significant API drift between when tests were written and current implementation.

## Pattern Analysis

The deviations show a consistent pattern:
1. **API Evolution:** Contracts have evolved but tests weren't updated
2. **Missing Exports:** Implementation modules don't properly export their public APIs
3. **Async Initialization:** Components force async patterns even in sync contexts
4. **Test-First Development:** Tests written against assumed APIs that differ from implementation

## Next Steps

Continue running test suite to identify more deviations and document patterns of test-driven modifications.