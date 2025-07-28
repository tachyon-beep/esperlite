# Phase 4: Message Bus Implementation

## Overview

Phase 4 introduces a comprehensive asynchronous message bus system for the morphogenetic neural network architecture. This phase provides distributed control, monitoring, and telemetry capabilities, enabling scalable and observable neural network systems.

## Key Components

### 1. Message Bus Core
- **Clients**: Redis Streams and Mock implementations for flexible deployment
- **Publishers**: Telemetry and event publishing with batching and compression
- **Handlers**: Command processors for lifecycle and blueprint management
- **Schemas**: Strongly-typed message definitions using Pydantic

### 2. Advanced Features
- **Message Ordering**: Configurable ordering strategies (partition, global, causal)
- **Dead Letter Queue**: Failed message handling with retry and replay
- **Monitoring**: Multi-backend metrics export (Prometheus, CloudWatch, OpenTelemetry)
- **Resilience**: Circuit breakers, rate limiting, and deduplication

### 3. Integration Components
- **TritonMessageBusLayer**: Extended TritonChunkedKasminaLayer with message bus support
- **TamiyoController**: High-level orchestrator for seed population management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TamiyoController                         │
│  (Population Management, Optimization, Health Monitoring)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                  Message Bus System                         │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Commands    │  │  Telemetry   │  │     Events      │  │
│  │  (Control)   │  │  (Metrics)   │  │ (Notifications) │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              TritonMessageBusLayers                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Layer 1    │  │   Layer 2    │  │   Layer N    │    │
│  │ (100 seeds)  │  │ (100 seeds)  │  │ (100 seeds)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Message Types

### Commands
```python
# Lifecycle state transitions
LifecycleTransitionCommand(
    layer_id="layer1",
    seed_id=42,  # Optional, None for all seeds
    target_state="TRAINING",
    parameters={},
    force=False
)

# Blueprint updates
BlueprintUpdateCommand(
    layer_id="layer1", 
    seed_id=42,
    blueprint_id="optimized_v2",
    grafting_strategy="gradual",
    configuration={},
    validation_metrics={"accuracy": 0.9}
)

# Batch operations
BatchCommand(
    commands=[cmd1, cmd2, ...],
    stop_on_error=True,
    atomic=True
)

# Emergency control
EmergencyStopCommand(
    layer_id="layer1",  # Optional
    seed_id=None,       # Optional
    reason="System overload"
)
```

### Telemetry
```python
# Layer health reports
LayerHealthReport(
    layer_id="layer1",
    timestamp=time.time(),
    total_seeds=100,
    active_seeds=75,
    health_tensor=torch.tensor(...)  # [num_seeds, 4]
)

# Seed metrics
SeedMetricsSnapshot(
    layer_id="layer1",
    seed_id=42,
    lifecycle_state="TRAINING",
    metrics={"loss": 0.1, "accuracy": 0.95},
    blueprint_id=10
)

# Performance alerts
PerformanceAlert(
    layer_id="layer1",
    metric_name="loss",
    metric_value=0.9,
    threshold=0.5,
    severity=AlertSeverity.HIGH
)
```

### Events
```python
# State transitions
StateTransitionEvent(
    layer_id="layer1",
    seed_id=42,
    from_state="DORMANT",
    to_state="GERMINATED",
    reason="Scheduled activation"
)
```

## Usage Examples

### Basic Setup
```python
import asyncio
from src.esper.morphogenetic_v2.message_bus import (
    MessageBusConfig, MockMessageBusClient
)
from src.esper.morphogenetic_v2.integration import (
    TritonMessageBusLayer, TamiyoController
)

async def main():
    # Create message bus
    bus = MockMessageBusClient()
    await bus.connect()
    
    # Create layer with message bus
    layer = TritonMessageBusLayer(
        base_layer=nn.Linear(768, 768),
        chunks_per_layer=100,
        message_bus=bus,
        layer_id="layer1"
    )
    
    # Start integration
    await layer.start_message_bus_integration()
    
    # Create controller
    controller = TamiyoController(bus)
    controller.register_layer(layer)
    await controller.start()
    
    # Use the system...
    
    # Cleanup
    await controller.stop()
    await layer.stop_message_bus_integration()
    await bus.disconnect()
```

### Lifecycle Management
```python
# Transition seeds to training
result = await controller.transition_population(
    layer_id="layer1",
    target_state=ExtendedLifecycle.TRAINING,
    selection_criteria={
        "from_states": [ExtendedLifecycle.GERMINATED],
        "min_performance": 0.5
    },
    batch_size=50
)
```

### Monitoring Setup
```python
from src.esper.morphogenetic_v2.message_bus.monitoring import (
    MessageBusMonitor, PrometheusExporter
)

# Create monitor with Prometheus export
monitor = MessageBusMonitor(
    exporters=[
        PrometheusExporter("http://localhost:9091")
    ]
)

# Register health checks
monitor.register_health_check("redis", check_redis_health)

await monitor.start()
```

### Message Ordering
```python
from src.esper.morphogenetic_v2.message_bus.ordering import (
    OrderedMessageProcessor, OrderingStrategy
)

# Create ordered processor
processor = OrderedMessageProcessor(
    OrderedMessageConfig(
        ordering_strategy=OrderingStrategy.PARTITION,
        max_out_of_order_window=100
    )
)

# Register handlers
processor.register_handler("commands.*", command_handler)

# Process messages with ordering guarantees
await processor.process_message(topic, message)
```

## Configuration

### Message Bus Configuration
```python
MessageBusConfig(
    instance_id="prod-1",
    redis_url="redis://localhost:6379",
    max_retries=3,
    retry_backoff_base=2.0,
    connection_timeout=30,
    max_message_size=1048576,  # 1MB
    enable_compression=True,
    enable_local_buffer=True
)
```

### Telemetry Configuration
```python
TelemetryConfig(
    batch_size=100,
    batch_window_ms=1000,
    compression="zstd",
    enable_aggregation=True,
    anomaly_detection=True,
    anomaly_threshold_stddev=3.0,
    claim_check_threshold=10485760  # 10MB
)
```

### Tamiyo Controller Configuration
```python
TamiyoConfig(
    optimization_strategy=OptimizationStrategy.BALANCED,
    auto_optimize=True,
    performance_threshold=0.8,
    failure_threshold=0.2,
    dormancy_threshold=0.5,
    emergency_failure_rate=0.5
)
```

## Best Practices

1. **Message Design**
   - Keep messages small and focused
   - Use claim check pattern for large payloads
   - Include correlation IDs for request/response

2. **Error Handling**
   - Configure appropriate retry policies
   - Use DLQ for persistent failures
   - Monitor error rates and set alerts

3. **Performance**
   - Batch telemetry messages
   - Use partition-based ordering when possible
   - Enable compression for large messages

4. **Monitoring**
   - Export metrics to your monitoring system
   - Set up health checks for critical components
   - Track command execution times

5. **Scalability**
   - Use Redis Streams for production
   - Partition by layer ID for horizontal scaling
   - Consider message TTL for cleanup

## Testing

The phase includes comprehensive test coverage:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete message flows
- **Resilience Tests**: Test failure scenarios and recovery

Run tests with:
```bash
pytest tests/morphogenetic_v2/message_bus/
```

## Migration from Phase 3

Phase 4 is fully compatible with Phase 3 components. To migrate:

1. Wrap existing layers with `TritonMessageBusLayer`
2. Set up message bus client
3. Register layers with `TamiyoController`
4. Configure monitoring and alerts

## Future Enhancements

- Distributed tracing integration
- Advanced routing strategies
- Multi-region support
- GraphQL subscription interface
- WebSocket real-time updates