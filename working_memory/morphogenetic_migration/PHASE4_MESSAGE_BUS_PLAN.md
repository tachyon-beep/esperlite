# Phase 4: Message Bus Integration - Initial Planning Document

*Date: 2025-01-24*
*Duration: 6 weeks (Weeks 29-34)*

## Overview

Phase 4 will integrate the morphogenetic system with a message bus (Oona) to enable asynchronous communication, distributed coordination, and real-time telemetry. This phase transforms the system from a monolithic architecture to a distributed, event-driven system.

## Objectives

### Primary Goals
1. **Asynchronous Telemetry**: Decouple telemetry collection from computation
2. **Command & Control**: Enable remote seed lifecycle management
3. **Distributed Coordination**: Support multi-node deployments
4. **Event-Driven Architecture**: React to system events in real-time
5. **Scalability**: Handle thousands of concurrent messages

### Success Metrics
- Message latency: <10ms p99
- Throughput: >10K messages/second
- Zero message loss
- Graceful degradation on bus failure

## Technical Architecture

### 1. Message Bus Selection
**Recommendation**: Apache Kafka or Redis Streams

#### Kafka Pros:
- High throughput
- Durable message storage
- Built-in partitioning
- Proven at scale

#### Redis Streams Pros:
- Lower latency
- Simpler setup
- Good for claim-check pattern
- Already in infrastructure

### 2. Topic Design

```
kasmina/
├── telemetry/
│   ├── layer/{layer_id}/health
│   ├── layer/{layer_id}/performance
│   └── seed/{layer_id}/{seed_id}/metrics
├── control/
│   ├── lifecycle/transition
│   ├── lifecycle/ack
│   └── blueprint/update
├── events/
│   ├── seed/germinated
│   ├── seed/grafted
│   ├── seed/fossilized
│   └── seed/culled
└── coordination/
    ├── discovery/announce
    └── discovery/heartbeat
```

### 3. Component Architecture

```python
# Core message bus abstraction
class MessageBusClient(ABC):
    @abstractmethod
    async def publish(self, topic: str, data: Any) -> None:
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable) -> None:
        pass
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        pass

# Kafka implementation
class KafkaMessageBusClient(MessageBusClient):
    def __init__(self, brokers: List[str]):
        self.producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: json.dumps(v).encode()
        )
        self.consumer = aiokafka.AIOKafkaConsumer(
            bootstrap_servers=brokers,
            value_deserializer=lambda v: json.loads(v.decode())
        )

# Redis implementation  
class RedisMessageBusClient(MessageBusClient):
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
        self.pubsub = self.redis.pubsub()
```

### 4. Integration Points

#### A. Telemetry Publisher
- Integrates with Phase 3 Triton kernels
- Batches and compresses telemetry data
- Implements claim-check for large payloads
- Asynchronous to avoid blocking GPU

#### B. Control Command Handler
- Receives lifecycle transition requests
- Validates commands against current state
- Executes transitions asynchronously
- Publishes acknowledgments

#### C. Event Emitter
- Publishes state change events
- Includes relevant metadata
- Supports event replay
- Enables audit trail

#### D. Coordination Service
- Service discovery
- Health monitoring
- Load balancing
- Failure detection

## Implementation Plan

### Week 29-30: Foundation
1. Set up message bus infrastructure
2. Implement base client abstractions
3. Create serialization framework
4. Build connection management

### Week 31-32: Telemetry System
1. Implement telemetry publisher
2. Add claim-check pattern
3. Create telemetry aggregator
4. Build monitoring dashboards

### Week 33-34: Control & Events
1. Implement command handlers
2. Add event emission
3. Create coordination service
4. Integration testing

## Key Design Decisions

### 1. Async-First Design
All message bus operations will be asynchronous to prevent blocking GPU computation.

### 2. Graceful Degradation
System continues to function if message bus is unavailable, with local buffering.

### 3. Schema Evolution
Use Protocol Buffers or Avro for message schemas to support evolution.

### 4. Idempotency
All command handlers must be idempotent to handle duplicate messages.

### 5. Backpressure
Implement flow control to prevent overwhelming consumers.

## Risk Mitigation

### Technical Risks
1. **Message Loss**: Implement acknowledgments and retries
2. **Ordering**: Use partition keys for ordered delivery
3. **Latency**: Local caching and batching
4. **Scale**: Horizontal partitioning

### Operational Risks
1. **Monitoring**: Comprehensive metrics and alerting
2. **Debugging**: Distributed tracing integration
3. **Testing**: Chaos engineering practices
4. **Recovery**: Automated failover

## Testing Strategy

### Unit Tests
- Message serialization/deserialization
- Client connection handling
- Error scenarios

### Integration Tests
- End-to-end message flow
- Telemetry publishing
- Command execution
- Event propagation

### Performance Tests
- Throughput benchmarks
- Latency measurements
- Scalability testing
- Failure recovery

## Dependencies

### External
- Message bus (Kafka/Redis)
- Monitoring (Prometheus/Grafana)
- Tracing (Jaeger)

### Internal
- Phase 3 Triton kernels
- Phase 2 lifecycle management
- Phase 1 chunked architecture

## Next Steps

1. **Technology Decision**: Choose between Kafka and Redis
2. **Prototype**: Build proof-of-concept
3. **Schema Design**: Define message formats
4. **API Design**: Create client interfaces
5. **Infrastructure**: Set up development environment

## Success Criteria

Phase 4 will be considered complete when:
- [ ] Message bus integration operational
- [ ] Telemetry publishing at scale
- [ ] Command handling working
- [ ] Event system functional
- [ ] Performance targets met
- [ ] Graceful degradation verified
- [ ] Documentation complete
- [ ] Tests passing

---

*Ready to begin Phase 4 implementation upon approval*