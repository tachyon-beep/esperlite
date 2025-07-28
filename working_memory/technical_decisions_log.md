# Technical Decisions Log

## Major Decisions Made During Implementation

### 1. Redis Migration (Python 3.12 Compatibility)
**Problem**: aioredis incompatible with Python 3.12 (duplicate base class TimeoutError)
**Decision**: Migrate to redis.asyncio
**Impact**: Entire codebase updated, all Redis operations now use redis.asyncio
**Files Changed**: 
- src/esper/morphogenetic_v2/message_bus/clients.py
- All Redis-dependent services

### 2. Test Infrastructure Design
**Problem**: Tests failing due to missing services and fixtures
**Decision**: Create real service test infrastructure with graceful fallback
**Implementation**:
- TestRedisServer class that tries Docker → redis-server → mock
- Session-scoped fixtures for service reuse
- Real components (InMemoryPerformanceTracker, InMemoryBlueprintRegistry)
**Benefits**: More reliable integration testing, catches real issues

### 3. Sync Kernel Execution Limitation
**Problem**: Async kernel execution doesn't work in sync context
**Decision**: Implement fallback behavior that skips kernel execution in sync mode
**Workaround**: ThreadPoolExecutor for tests that need kernel execution
**Future**: TODO added to implement proper sync kernel execution

### 4. LifecycleManager Import Resolution
**Problem**: Two classes with same name causing import conflicts
**Decision**: Import from lifecycle_manager.py instead of extended_lifecycle.py
**Impact**: Fixed all lifecycle-related test failures

### 5. Service Orchestration Architecture
**Decision**: Loosely coupled services communicating via message bus
**Services**:
- Tolaria: Training orchestration
- Tamiyo: Intelligent adaptation
- Nissa: Observability and metrics
- Urza: Kernel compilation (mocked in tests)
**Benefits**: Scalable, maintainable, testable

### 6. Kernel Caching Strategy
**Decision**: Multi-level caching with metadata
**Implementation**:
- In-memory tensor cache
- Metadata cache
- Optional Redis backing
- HTTP fetch fallback
**Benefits**: Fast kernel access, reduced network calls

### 7. Test Philosophy Change
**From**: Heavy mocking of all dependencies
**To**: Real components with minimal mocking
**Principles**:
- Use real services when available
- Mock only external dependencies
- Test meaningful behavior, not implementation
- Integration over unit tests for complex systems

### 8. Message Bus Design
**Technology**: Redis Streams
**Features**:
- Topic-based routing
- Reliable delivery
- Consumer groups
- Telemetry batching
**Benefits**: Scalable, reliable, observable

### 9. Error Handling Strategy
**Approach**: Graceful degradation
**Examples**:
- Redis unavailable → use mock message bus
- Kernel load fails → continue with defaults
- Service down → queue messages
**Benefits**: System remains operational under failure

### 10. Performance Tracking Design
**Decision**: Centralized performance tracker with pluggable storage
**Features**:
- In-memory for tests
- Redis for production
- Seed-level granularity
- Historical tracking
**Benefits**: Data-driven adaptation decisions

## Lessons Learned

1. **Real > Mock**: Real service testing catches more issues
2. **Async Complexity**: Sync/async boundary requires careful handling  
3. **Import Management**: Clear module organization prevents conflicts
4. **Test Isolation**: Proper cleanup prevents test interference
5. **Graceful Fallback**: Systems should degrade gracefully
6. **Documentation**: Inline TODOs help track technical debt
7. **Peer Review**: Even tests benefit from code review
8. **Integration Testing**: Complex systems need end-to-end validation

## Future Considerations

1. **Sync Kernel Execution**: Implement proper sync support
2. **GPU Optimization**: Kernel execution on GPU
3. **Distributed Training**: Multi-node support
4. **Performance Profiling**: Built-in profiling tools
5. **Blueprint Library**: Expanded kernel templates
6. **Monitoring**: Production-ready observability