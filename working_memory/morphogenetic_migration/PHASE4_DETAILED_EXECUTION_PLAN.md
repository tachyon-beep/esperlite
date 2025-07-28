# Phase 4: Message Bus Integration - Detailed Execution Plan

*Date: 2025-01-25*
*Duration: 6 weeks*
*Status: Ready for Implementation*

## Executive Summary

Phase 4 transforms the morphogenetic system from a monolithic architecture to a distributed, event-driven system through message bus integration. This phase enables asynchronous telemetry, remote control, distributed coordination, and real-time monitoring while maintaining backward compatibility and graceful degradation.

## Technical Architecture Deep Dive

### 1. Message Bus Technology Decision

**Selected Technology: Redis Streams**

**Rationale:**
- Lower latency (<5ms p99) critical for real-time control
- Simpler operational overhead
- Built-in claim-check pattern support
- Already deployed in infrastructure
- Excellent Python async support
- Native support for consumer groups

**Fallback Option: Kafka** (if scale exceeds Redis capacity)

### 2. Detailed Message Schema Design

#### 2.1 Base Message Format

```python
@dataclass
class BaseMessage:
    """Base class for all morphogenetic messages."""
    message_id: str  # UUID v4
    timestamp: float  # Unix timestamp with microsecond precision
    source: str      # Component identifier (e.g., "kasmina_layer_12")
    version: str     # Message schema version
    correlation_id: Optional[str] = None  # For request/response patterns
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 2.2 Telemetry Messages

```python
@dataclass
class LayerHealthReport(BaseMessage):
    """Aggregated health report for all seeds in a layer."""
    layer_id: str
    total_seeds: int
    active_seeds: int
    health_metrics: Dict[int, Dict[str, float]]  # seed_id -> metrics
    performance_summary: Dict[str, float]
    telemetry_window: Tuple[float, float]  # (start_time, end_time)
    
@dataclass
class SeedMetricsSnapshot(BaseMessage):
    """Detailed metrics for a single seed."""
    layer_id: str
    seed_id: int
    lifecycle_state: str
    blueprint_id: Optional[int]
    metrics: Dict[str, float]
    error_count: int
    checkpoint_id: Optional[str]
```

#### 2.3 Control Commands

```python
@dataclass
class LifecycleTransitionCommand(BaseMessage):
    """Command to transition a seed's lifecycle state."""
    layer_id: str
    seed_id: int
    target_state: str
    parameters: Dict[str, Any]  # State-specific parameters
    priority: str = "normal"    # low, normal, high, critical
    timeout_ms: int = 5000      # Command timeout
    
@dataclass
class BlueprintUpdateCommand(BaseMessage):
    """Command to update a seed's blueprint."""
    layer_id: str
    seed_id: int
    blueprint_id: str
    grafting_strategy: str
    configuration: Dict[str, Any]
```

#### 2.4 Event Messages

```python
@dataclass
class StateTransitionEvent(BaseMessage):
    """Event emitted when a seed transitions state."""
    layer_id: str
    seed_id: int
    from_state: str
    to_state: str
    reason: str
    metrics_snapshot: Dict[str, float]
    
@dataclass
class PerformanceAlert(BaseMessage):
    """Alert for performance anomalies."""
    layer_id: str
    alert_type: str  # degradation, improvement, anomaly
    severity: str    # info, warning, critical
    details: Dict[str, Any]
```

### 3. Component Implementation Architecture

#### 3.1 Message Bus Client Implementation

```python
# src/esper/morphogenetic_v2/message_bus/client.py

class RedisStreamClient(MessageBusClient):
    """Production Redis Streams client with resilience features."""
    
    def __init__(self, config: MessageBusConfig):
        self.config = config
        self.redis = None
        self.consumer_group = f"morphogenetic_{config.instance_id}"
        self._connected = False
        self._local_buffer = asyncio.Queue(maxsize=10000)
        self._retry_queue = asyncio.Queue(maxsize=1000)
        
    async def connect(self):
        """Establish connection with retry logic."""
        retry_count = 0
        while retry_count < self.config.max_retries:
            try:
                self.redis = await aioredis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_keepalive=True
                )
                await self.redis.ping()
                self._connected = True
                # Start background tasks
                asyncio.create_task(self._flush_local_buffer())
                asyncio.create_task(self._process_retries())
                return
            except Exception as e:
                retry_count += 1
                await asyncio.sleep(2 ** retry_count)
        raise ConnectionError("Failed to connect to Redis")
```

#### 3.2 Telemetry Publisher Architecture

```python
# src/esper/morphogenetic_v2/message_bus/telemetry_publisher.py

class TelemetryPublisher:
    """High-performance telemetry publisher with batching and compression."""
    
    def __init__(self, client: MessageBusClient, config: TelemetryConfig):
        self.client = client
        self.config = config
        self.batch_queue = asyncio.Queue()
        self.claim_check_cache = None
        self._running = False
        
    async def publish_layer_health(self, layer_id: str, health_data: torch.Tensor):
        """Publish layer health report with automatic batching."""
        # Convert GPU tensor to CPU efficiently
        health_dict = self._tensor_to_dict(health_data)
        
        report = LayerHealthReport(
            message_id=str(uuid.uuid4()),
            timestamp=time.time(),
            source=f"kasmina_layer_{layer_id}",
            version="1.0",
            layer_id=layer_id,
            total_seeds=len(health_dict),
            active_seeds=sum(1 for m in health_dict.values() if m.get('active')),
            health_metrics=health_dict,
            performance_summary=self._calculate_summary(health_dict),
            telemetry_window=(self.window_start, time.time())
        )
        
        # Add to batch queue
        await self.batch_queue.put(report)
        
    async def _batch_publisher(self):
        """Background task for batched publishing."""
        batch = []
        while self._running:
            try:
                # Collect messages for batch
                deadline = asyncio.create_task(asyncio.sleep(self.config.batch_window_ms / 1000))
                
                while len(batch) < self.config.batch_size:
                    try:
                        msg = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=0.1
                        )
                        batch.append(msg)
                    except asyncio.TimeoutError:
                        break
                        
                await deadline
                
                if batch:
                    # Compress and publish
                    await self._publish_batch(batch)
                    batch = []
                    
            except Exception as e:
                logger.error(f"Batch publisher error: {e}")
```

#### 3.3 Command Handler Architecture

```python
# src/esper/morphogenetic_v2/message_bus/command_handler.py

class CommandHandler:
    """Handles incoming control commands with validation and acknowledgment."""
    
    def __init__(self, layer_registry: Dict[str, TritonChunkedKasminaLayer]):
        self.layer_registry = layer_registry
        self.command_processors = {
            'LifecycleTransitionCommand': self._process_lifecycle_transition,
            'BlueprintUpdateCommand': self._process_blueprint_update,
            'EmergencyStopCommand': self._process_emergency_stop
        }
        
    async def handle_command(self, command: BaseMessage) -> CommandResult:
        """Process command with timeout and error handling."""
        command_type = type(command).__name__
        processor = self.command_processors.get(command_type)
        
        if not processor:
            return CommandResult(
                success=False,
                error=f"Unknown command type: {command_type}"
            )
            
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                processor(command),
                timeout=command.timeout_ms / 1000
            )
            
            # Publish acknowledgment
            await self._publish_ack(command, result)
            
            return result
            
        except asyncio.TimeoutError:
            return CommandResult(
                success=False,
                error="Command timeout"
            )
```

### 4. Integration Points

#### 4.1 KasminaLayer Integration

```python
# Updates to TritonChunkedKasminaLayer

class TritonChunkedKasminaLayer(nn.Module):
    def __init__(self, ..., message_bus_client: Optional[MessageBusClient] = None):
        super().__init__()
        # ... existing init ...
        self.message_bus = message_bus_client
        self.telemetry_publisher = None
        self.command_handler = None
        
        if self.message_bus:
            self._init_message_bus_components()
            
    def _init_message_bus_components(self):
        """Initialize message bus components."""
        self.telemetry_publisher = TelemetryPublisher(
            self.message_bus,
            TelemetryConfig(
                batch_size=100,
                batch_window_ms=100,
                compression='zstd'
            )
        )
        
        self.command_handler = CommandHandler({self.layer_id: self})
        
        # Subscribe to control topics
        asyncio.create_task(self._subscribe_to_commands())
        
    async def _subscribe_to_commands(self):
        """Subscribe to relevant command topics."""
        await self.message_bus.subscribe(
            f"kasmina.control.layer.{self.layer_id}",
            self.command_handler.handle_command
        )
```

#### 4.2 Tamiyo Controller Integration

```python
# Updates to TamiyoController

class TamiyoController:
    def __init__(self, ..., message_bus_client: Optional[MessageBusClient] = None):
        # ... existing init ...
        self.message_bus = message_bus_client
        self.telemetry_consumer = None
        
    async def consume_telemetry(self):
        """Consume telemetry from message bus."""
        await self.message_bus.subscribe(
            "kasmina.telemetry.layer.+.health",
            self._process_health_report
        )
        
    async def issue_command(self, layer_id: str, seed_id: int, command: BaseMessage):
        """Issue command via message bus."""
        await self.message_bus.publish(
            f"kasmina.control.layer.{layer_id}",
            command
        )
```

### 5. Performance Optimizations

#### 5.1 Zero-Copy Telemetry Transfer

```python
class ZeroCopyTelemetryBuffer:
    """Shared memory buffer for zero-copy telemetry transfer."""
    
    def __init__(self, size_mb: int = 100):
        self.shm = shared_memory.SharedMemory(create=True, size=size_mb * 1024 * 1024)
        self.buffer = np.ndarray((size_mb * 1024 * 1024,), dtype=np.uint8, buffer=self.shm.buf)
        self.write_pos = 0
        self.read_pos = 0
        
    def write_tensor(self, tensor: torch.Tensor) -> int:
        """Write tensor to shared memory, return offset."""
        # DMA transfer from GPU to shared memory
        tensor_bytes = tensor.cpu().numpy().tobytes()
        offset = self.write_pos
        self.buffer[offset:offset + len(tensor_bytes)] = np.frombuffer(tensor_bytes, dtype=np.uint8)
        self.write_pos += len(tensor_bytes)
        return offset
```

#### 5.2 Adaptive Batching

```python
class AdaptiveBatcher:
    """Dynamically adjusts batch size based on load."""
    
    def __init__(self, min_batch: int = 10, max_batch: int = 1000):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.current_batch = min_batch
        self.latency_history = deque(maxlen=100)
        
    def adjust_batch_size(self, latency: float, throughput: float):
        """Adjust batch size based on performance metrics."""
        self.latency_history.append(latency)
        
        if latency > 10.0:  # >10ms latency
            self.current_batch = max(self.min_batch, int(self.current_batch * 0.8))
        elif latency < 5.0 and throughput < 0.8:  # Underutilized
            self.current_batch = min(self.max_batch, int(self.current_batch * 1.2))
```

### 6. Resilience and Error Handling

#### 6.1 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevents cascading failures in message bus operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise CircuitOpenError("Circuit breaker is open")
                
        try:
            result = await func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened: {e}")
                
            raise
```

#### 6.2 Message Deduplication

```python
class MessageDeduplicator:
    """Prevents duplicate message processing."""
    
    def __init__(self, window_size: int = 10000):
        self.seen_messages = deque(maxlen=window_size)
        self.bloom_filter = BloomFilter(capacity=window_size * 2, error_rate=0.001)
        
    def is_duplicate(self, message_id: str) -> bool:
        """Check if message has been seen recently."""
        if message_id in self.bloom_filter:
            return message_id in self.seen_messages
        return False
        
    def mark_processed(self, message_id: str):
        """Mark message as processed."""
        self.bloom_filter.add(message_id)
        self.seen_messages.append(message_id)
```

### 7. Monitoring and Observability

#### 7.1 Metrics Collection

```python
class MessageBusMetrics:
    """Collects and exposes metrics for monitoring."""
    
    def __init__(self):
        self.messages_published = Counter('morphogenetic_messages_published_total')
        self.messages_consumed = Counter('morphogenetic_messages_consumed_total')
        self.message_latency = Histogram('morphogenetic_message_latency_seconds')
        self.queue_depth = Gauge('morphogenetic_queue_depth')
        self.errors = Counter('morphogenetic_message_errors_total')
        
    @contextmanager
    def track_latency(self, operation: str):
        """Track operation latency."""
        start = time.time()
        try:
            yield
        finally:
            self.message_latency.labels(operation=operation).observe(time.time() - start)
```

#### 7.2 Distributed Tracing

```python
class TracingMessageBusClient(MessageBusClient):
    """Message bus client with distributed tracing support."""
    
    def __init__(self, base_client: MessageBusClient, tracer):
        self.base_client = base_client
        self.tracer = tracer
        
    async def publish(self, topic: str, data: Any) -> None:
        """Publish with tracing."""
        with self.tracer.start_active_span('message_bus.publish') as scope:
            scope.span.set_tag('topic', topic)
            scope.span.set_tag('message_type', type(data).__name__)
            
            # Inject trace context
            if hasattr(data, 'metadata'):
                self.tracer.inject(scope.span.context, data.metadata)
                
            await self.base_client.publish(topic, data)
```

### 8. Testing Framework

#### 8.1 Message Bus Test Fixtures

```python
# tests/morphogenetic_v2/message_bus/conftest.py

@pytest.fixture
async def mock_message_bus():
    """In-memory message bus for testing."""
    class MockMessageBus(MessageBusClient):
        def __init__(self):
            self.messages = defaultdict(list)
            self.subscribers = defaultdict(list)
            
        async def publish(self, topic: str, data: Any):
            self.messages[topic].append(data)
            # Notify subscribers
            for handler in self.subscribers[topic]:
                await handler(data)
                
        async def subscribe(self, topic: str, handler: Callable):
            self.subscribers[topic].append(handler)
            
    return MockMessageBus()

@pytest.fixture
async def redis_test_client():
    """Real Redis client for integration tests."""
    redis = await aioredis.from_url("redis://localhost:6379/15")  # Test DB
    yield redis
    await redis.flushdb()
    await redis.close()
```

#### 8.2 Integration Test Scenarios

```python
# tests/morphogenetic_v2/message_bus/test_integration.py

class TestMessageBusIntegration:
    async def test_end_to_end_telemetry_flow(self, redis_test_client):
        """Test complete telemetry flow from layer to controller."""
        # Setup
        layer = TritonChunkedKasminaLayer(...)
        controller = TamiyoController(...)
        
        # Connect components
        await layer.message_bus.connect()
        await controller.message_bus.connect()
        
        # Trigger telemetry
        x = torch.randn(32, 512)
        await layer.forward_async(x)
        
        # Verify controller received telemetry
        await asyncio.sleep(0.1)  # Allow propagation
        assert controller.last_health_report is not None
        assert controller.last_health_report.layer_id == layer.layer_id
```

### 9. Migration Strategy

#### 9.1 Feature Flag Integration

```python
class MessageBusFeatureFlag:
    """Gradual rollout of message bus features."""
    
    @staticmethod
    def is_enabled(feature: str, layer_id: str = None) -> bool:
        config = load_feature_config()
        
        # Check global flag
        if not config.get('message_bus_enabled', False):
            return False
            
        # Check specific feature
        feature_config = config.get(f'message_bus_{feature}', {})
        
        if layer_id and layer_id in feature_config.get('allowlist', []):
            return True
            
        rollout_percentage = feature_config.get('rollout_percentage', 0)
        return hash(layer_id) % 100 < rollout_percentage
```

### 10. Performance Benchmarks

#### 10.1 Benchmark Suite

```python
# benchmarks/morphogenetic_v2/message_bus_benchmarks.py

async def benchmark_telemetry_throughput():
    """Measure telemetry publishing throughput."""
    client = RedisStreamClient(config)
    publisher = TelemetryPublisher(client)
    
    # Generate test data
    num_layers = 10
    seeds_per_layer = 1000
    iterations = 1000
    
    start = time.time()
    
    for i in range(iterations):
        for layer_id in range(num_layers):
            health_data = torch.randn(seeds_per_layer, 4)
            await publisher.publish_layer_health(f"layer_{layer_id}", health_data)
            
    elapsed = time.time() - start
    
    messages_per_second = (iterations * num_layers) / elapsed
    print(f"Telemetry throughput: {messages_per_second:.0f} messages/second")
```

## Implementation Schedule

### Week 1: Foundation (Jan 27-31)
- [ ] Set up Redis Streams infrastructure
- [ ] Implement base MessageBusClient abstraction
- [ ] Create message schemas and serialization
- [ ] Build connection management with retry logic
- [ ] Write unit tests for core components

### Week 2: Telemetry System (Feb 3-7)
- [ ] Implement TelemetryPublisher with batching
- [ ] Add claim-check pattern for large payloads
- [ ] Create zero-copy telemetry buffer
- [ ] Build telemetry aggregator service
- [ ] Deploy monitoring dashboards

### Week 3: Command & Control (Feb 10-14)
- [ ] Implement CommandHandler with validation
- [ ] Add command acknowledgment system
- [ ] Create idempotent command processors
- [ ] Build command retry mechanism
- [ ] Test command timeout handling

### Week 4: Event System (Feb 17-21)
- [ ] Implement event emission framework
- [ ] Add event replay capability
- [ ] Create audit trail system
- [ ] Build event correlation service
- [ ] Test event ordering guarantees

### Week 5: Integration (Feb 24-28)
- [ ] Integrate with TritonChunkedKasminaLayer
- [ ] Update TamiyoController for async operation
- [ ] Add circuit breaker protection
- [ ] Implement graceful degradation
- [ ] Run integration test suite

### Week 6: Production Readiness (Mar 3-7)
- [ ] Performance optimization pass
- [ ] Chaos testing (network failures, high load)
- [ ] Documentation completion
- [ ] Deployment automation
- [ ] Production rollout plan

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|-----------|
| Redis performance bottleneck | Medium | High | Kafka fallback ready, horizontal sharding |
| Message ordering violations | Low | High | Partition keys, sequence numbers |
| Network partition handling | Medium | Medium | Circuit breakers, local buffering |
| Schema evolution breaks | Low | Medium | Versioning, backward compatibility |
| Monitoring blind spots | Low | Medium | Comprehensive metrics, tracing |

## Success Metrics

### Performance
- Message latency p99 < 10ms ✓
- Throughput > 10K msg/sec ✓
- Zero message loss under normal operation ✓
- < 100ms recovery time from failures ✓

### Reliability
- 99.9% message delivery guarantee
- Graceful degradation on bus failure
- No impact on GPU computation performance
- Automatic recovery from transient failures

### Operational
- Full observability via metrics/tracing
- Easy debugging with correlation IDs
- Clear error messages and recovery actions
- Comprehensive runbooks

## Deliverables

1. **Source Code**
   - Message bus client implementations
   - Telemetry publisher system
   - Command handler framework
   - Event emission system
   - Integration layers

2. **Tests**
   - Unit tests (>95% coverage)
   - Integration tests
   - Performance benchmarks
   - Chaos test scenarios

3. **Documentation**
   - API reference
   - Integration guide
   - Operations runbook
   - Troubleshooting guide

4. **Infrastructure**
   - Redis Streams setup
   - Monitoring dashboards
   - Alert configurations
   - Deployment scripts

## Conclusion

Phase 4 is ready for implementation with a clear path to transform the morphogenetic system into a distributed, event-driven architecture. The plan balances performance, reliability, and operational simplicity while maintaining backward compatibility and enabling gradual rollout.

---

*Prepared by: Morphogenetic Migration Team*
*Status: Ready for Execution*