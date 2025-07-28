"""
Complete example demonstrating Phase 4 message bus integration.

This example shows:
- Setting up the message bus with Redis
- Creating TritonChunkedKasminaLayer with message bus integration
- Using TamiyoController for orchestration
- Monitoring and metrics export
- Message ordering and DLQ handling
"""

import asyncio
import logging
from typing import Any
from typing import Dict

import torch
import torch.nn as nn

from src.esper.morphogenetic_v2.integration import MessageBusIntegrationConfig
from src.esper.morphogenetic_v2.integration import OptimizationStrategy
from src.esper.morphogenetic_v2.integration import TamiyoConfig
from src.esper.morphogenetic_v2.integration import TamiyoController
from src.esper.morphogenetic_v2.integration import TritonMessageBusLayer
from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle

# Import Phase 4 components
from src.esper.morphogenetic_v2.message_bus import MessageBusConfig
from src.esper.morphogenetic_v2.message_bus import MockMessageBusClient
from src.esper.morphogenetic_v2.message_bus import RedisStreamClient
from src.esper.morphogenetic_v2.message_bus.monitoring import MessageBusMonitor
from src.esper.morphogenetic_v2.message_bus.monitoring import MonitoringConfig
from src.esper.morphogenetic_v2.message_bus.monitoring import PrometheusExporter
from src.esper.morphogenetic_v2.message_bus.ordering import DeadLetterQueue
from src.esper.morphogenetic_v2.message_bus.ordering import OrderedMessageConfig
from src.esper.morphogenetic_v2.message_bus.ordering import OrderedMessageProcessor
from src.esper.morphogenetic_v2.message_bus.ordering import OrderingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def setup_message_bus(use_redis: bool = False) -> RedisStreamClient:
    """Set up message bus client."""
    config = MessageBusConfig(
        instance_id="phase4_example",
        redis_url="redis://localhost:6379" if use_redis else None,
        enable_local_buffer=True,
        max_retries=3
    )

    if use_redis:
        client = RedisStreamClient(config)
    else:
        # Use mock for testing
        client = MockMessageBusClient()

    await client.connect()
    logger.info("Message bus connected")

    return client


async def setup_monitoring(message_bus: RedisStreamClient) -> MessageBusMonitor:
    """Set up monitoring and metrics export."""
    # Configure monitoring
    monitor_config = MonitoringConfig(
        collection_interval_ms=5000,
        export_interval_ms=30000,
        enable_health_checks=True
    )

    # Create exporters
    exporters = []

    # Prometheus exporter
    prometheus = PrometheusExporter(
        pushgateway_url="http://localhost:9091",
        job_name="morphogenetic_phase4"
    )
    exporters.append(prometheus)

    # CloudWatch exporter (if on AWS)
    # cloudwatch = CloudWatchExporter(
    #     namespace="Morphogenetic/Phase4"
    # )
    # exporters.append(cloudwatch)

    # Create monitor
    monitor = MessageBusMonitor(monitor_config, exporters)

    # Register health checks
    async def message_bus_health():
        return await message_bus.is_connected()

    monitor.register_health_check("message_bus", message_bus_health)

    await monitor.start()
    logger.info("Monitoring started with %d exporters", len(exporters))

    return monitor


async def create_morphogenetic_system(
    message_bus: RedisStreamClient,
    monitor: MessageBusMonitor
) -> Dict[str, Any]:
    """Create the morphogenetic system with message bus integration."""

    # Create base neural network layers
    base_layer1 = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.LayerNorm(768)
    )

    base_layer2 = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.LayerNorm(768)
    )

    # Create TritonMessageBusLayers
    integration_config = MessageBusIntegrationConfig(
        enable_telemetry=True,
        telemetry_interval_ms=1000,
        enable_commands=True,
        enable_events=True,
        batch_telemetry=True
    )

    layer1 = TritonMessageBusLayer(
        base_layer=base_layer1,
        chunks_per_layer=100,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        enable_triton=torch.cuda.is_available(),
        message_bus=message_bus,
        integration_config=integration_config,
        layer_id="layer1"
    )

    layer2 = TritonMessageBusLayer(
        base_layer=base_layer2,
        chunks_per_layer=100,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        enable_triton=torch.cuda.is_available(),
        message_bus=message_bus,
        integration_config=integration_config,
        layer_id="layer2"
    )

    # Start message bus integration
    await layer1.start_message_bus_integration()
    await layer2.start_message_bus_integration()

    # Create Tamiyo controller
    tamiyo_config = TamiyoConfig(
        optimization_strategy=OptimizationStrategy.BALANCED,
        auto_optimize=True,
        performance_threshold=0.8,
        failure_threshold=0.2
    )

    controller = TamiyoController(message_bus, tamiyo_config)

    # Register layers with controller
    controller.register_layer(layer1)
    controller.register_layer(layer2)

    await controller.start()

    # Set up ordered message processing
    ordering_config = OrderedMessageConfig(
        ordering_strategy=OrderingStrategy.PARTITION,
        enable_dlq=True,
        max_retries=3
    )

    ordered_processor = OrderedMessageProcessor(ordering_config)
    await ordered_processor.start()

    # Set up Dead Letter Queue
    dlq = DeadLetterQueue(message_bus, ordering_config)
    await dlq.start()

    logger.info("Morphogenetic system created with %d layers", 2)

    return {
        "layers": {"layer1": layer1, "layer2": layer2},
        "controller": controller,
        "ordered_processor": ordered_processor,
        "dlq": dlq,
        "monitor": monitor
    }


async def demonstrate_lifecycle_management(system: Dict[str, Any]):
    """Demonstrate lifecycle state management."""
    logger.info("\n=== Lifecycle Management Demo ===")

    controller = system["controller"]

    # Transition some seeds to GERMINATED
    result = await controller.transition_population(
        layer_id="layer1",
        target_state=ExtendedLifecycle.GERMINATED,
        batch_size=20
    )

    logger.info("Transitioned %d seeds to GERMINATED", result["transitioned"])

    # Wait for telemetry
    await asyncio.sleep(2)

    # Transition to TRAINING
    result = await controller.transition_population(
        layer_id="layer1",
        target_state=ExtendedLifecycle.TRAINING,
        selection_criteria={"from_states": [ExtendedLifecycle.GERMINATED]},
        batch_size=10
    )

    logger.info("Transitioned %d seeds to TRAINING", result["transitioned"])

    # Get population report
    report = controller.get_population_report()
    layer1_status = report["layers"]["layer1"]

    logger.info("Layer 1 status:")
    logger.info("  Active seeds: %d", layer1_status["metrics"]["active_seeds"])
    logger.info("  Training rate: %.2f", layer1_status["metrics"]["training_rate"])
    logger.info("  Health: %s", layer1_status["health_status"])


async def demonstrate_monitoring(system: Dict[str, Any]):
    """Demonstrate monitoring and metrics."""
    logger.info("\n=== Monitoring Demo ===")

    monitor = system["monitor"]

    # Record some custom metrics
    await monitor.record_metric(
        "custom_metric",
        42.0,
        {"layer": "layer1", "type": "example"}
    )

    # Get metrics summary
    summary = await monitor.get_metrics_summary()
    logger.info("Metrics summary: %s", summary)

    # Check health
    health = monitor.get_health_status()
    logger.info("System health: %s", health)


async def demonstrate_ordered_messaging(system: Dict[str, Any]):
    """Demonstrate ordered message processing."""
    logger.info("\n=== Ordered Messaging Demo ===")

    ordered_processor = system["ordered_processor"]
    message_bus = system["layers"]["layer1"].message_bus

    # Register a test handler
    messages_received = []

    async def test_handler(message):
        messages_received.append(message)
        logger.info("Received message: %s", message.message_id)

    ordered_processor.register_handler("test.ordered.*", test_handler)

    # Send messages out of order
    from src.esper.morphogenetic_v2.message_bus.schemas import BaseMessage

    # These would normally have sequence numbers assigned
    msg1 = BaseMessage(source="test", message_id="msg1")
    msg2 = BaseMessage(source="test", message_id="msg2")
    msg3 = BaseMessage(source="test", message_id="msg3")

    # Process in wrong order
    await ordered_processor.process_message("test.ordered.topic", msg3)
    await ordered_processor.process_message("test.ordered.topic", msg1)
    await ordered_processor.process_message("test.ordered.topic", msg2)

    # Wait for reordering
    await asyncio.sleep(0.5)

    logger.info("Messages processed: %d", len(messages_received))


async def demonstrate_dlq(system: Dict[str, Any]):
    """Demonstrate Dead Letter Queue."""
    logger.info("\n=== Dead Letter Queue Demo ===")

    dlq = system["dlq"]

    # Add a failed message
    from src.esper.morphogenetic_v2.message_bus.schemas import BaseMessage

    failed_msg = BaseMessage(
        source="test",
        message_id="failed_msg_1",
        metadata={"reason": "Processing error"}
    )

    await dlq.add_failed_message(
        topic="test.failed",
        message=failed_msg,
        error="Simulated processing error"
    )

    # Get DLQ stats
    stats = await dlq.get_stats()
    logger.info("DLQ stats: %s", stats)

    # List messages
    messages = await dlq.get_messages(limit=10)
    logger.info("DLQ messages: %d", len(messages))

    # Replay message
    if messages:
        success = await dlq.replay_message(messages[0].message.message_id)
        logger.info("Message replay success: %s", success)


async def cleanup(system: Dict[str, Any]):
    """Clean up resources."""
    logger.info("\n=== Cleanup ===")

    # Stop controller
    await system["controller"].stop()

    # Stop layers
    for layer in system["layers"].values():
        await layer.stop_message_bus_integration()

    # Stop ordered processor
    await system["ordered_processor"].stop()

    # Stop DLQ
    await system["dlq"].stop()

    # Stop monitor
    await system["monitor"].stop()

    # Disconnect message bus
    if hasattr(system["layers"]["layer1"].message_bus, 'disconnect'):
        await system["layers"]["layer1"].message_bus.disconnect()

    logger.info("Cleanup complete")


async def main():
    """Main example function."""
    logger.info("Starting Phase 4 Complete Example")

    # Set up message bus
    message_bus = await setup_message_bus(use_redis=False)  # Use mock for demo

    # Set up monitoring
    monitor = await setup_monitoring(message_bus)

    # Create morphogenetic system
    system = await create_morphogenetic_system(message_bus, monitor)

    # Run demonstrations
    await demonstrate_lifecycle_management(system)
    await demonstrate_monitoring(system)
    await demonstrate_ordered_messaging(system)
    await demonstrate_dlq(system)

    # Wait a bit for final metrics
    await asyncio.sleep(2)

    # Get final report
    report = system["controller"].get_population_report()
    logger.info("\n=== Final Report ===")
    logger.info("Controller stats: %s", report["controller_stats"])

    # Cleanup
    await cleanup(system)

    logger.info("\nPhase 4 example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
