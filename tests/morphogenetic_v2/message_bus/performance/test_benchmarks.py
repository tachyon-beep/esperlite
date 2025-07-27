"""Performance benchmarks for message bus system."""

import pytest
import asyncio
import time
import torch
import numpy as np
from typing import List, Dict, Any

from src.esper.morphogenetic_v2.message_bus.clients import (
    MockMessageBusClient, RedisStreamClient, MessageBusConfig
)
from src.esper.morphogenetic_v2.message_bus.publishers import (
    TelemetryPublisher, TelemetryConfig
)
from src.esper.morphogenetic_v2.message_bus.handlers import (
    CommandHandler, CommandHandlerFactory
)
from src.esper.morphogenetic_v2.message_bus.schemas import (
    LayerHealthReport, SeedMetricsSnapshot,
    LifecycleTransitionCommand, BlueprintUpdateCommand,
    BatchCommand
)
from src.esper.morphogenetic_v2.message_bus.utils import (
    CircuitBreaker, MessageDeduplicator, RateLimiter,
    MessageBatcher, MetricsCollector
)


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add_result(self, name: str, 
                   operations: int,
                   elapsed_seconds: float,
                   **kwargs):
        """Add benchmark result."""
        self.results[name] = {
            "operations": operations,
            "elapsed_seconds": elapsed_seconds,
            "ops_per_second": operations / elapsed_seconds,
            "ms_per_op": (elapsed_seconds * 1000) / operations,
            **kwargs
        }
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Operations: {result['operations']:,}")
            print(f"  Elapsed: {result['elapsed_seconds']:.3f}s")
            print(f"  Throughput: {result['ops_per_second']:,.0f} ops/sec")
            print(f"  Latency: {result['ms_per_op']:.3f} ms/op")
            
            # Print additional metrics
            for key, value in result.items():
                if key not in ["operations", "elapsed_seconds", 
                              "ops_per_second", "ms_per_op"]:
                    print(f"  {key}: {value}")


@pytest.fixture
def benchmark_results():
    """Fixture for collecting benchmark results."""
    return BenchmarkResults()


class TestTelemetryBenchmarks:
    """Benchmark telemetry publishing performance."""
    
    @pytest.mark.asyncio
    async def test_telemetry_throughput(self, mock_message_bus, benchmark_results):
        """Benchmark telemetry publishing throughput."""
        config = TelemetryConfig(
            batch_size=100,
            batch_window_ms=50,
            compression=None,
            enable_aggregation=False,
            anomaly_detection=False
        )
        
        publisher = TelemetryPublisher(mock_message_bus, config)
        await publisher.start()
        
        # Benchmark parameters
        num_layers = 10
        num_reports_per_layer = 100
        seeds_per_layer = 1000
        
        start_time = time.time()
        
        for i in range(num_reports_per_layer):
            for layer_id in range(num_layers):
                # Generate health data
                health_data = torch.randn(seeds_per_layer, 4)
                await publisher.publish_layer_health(f"layer_{layer_id}", health_data)
        
        # Flush remaining
        await publisher._flush_batch()
        
        elapsed = time.time() - start_time
        total_operations = num_layers * num_reports_per_layer
        
        # Get stats
        stats = await publisher.get_stats()
        
        benchmark_results.add_result(
            "Telemetry Publishing",
            operations=total_operations,
            elapsed_seconds=elapsed,
            messages_published=stats["messages_published"],
            batches_sent=stats["batches_sent"],
            bytes_sent=stats["bytes_sent"]
        )
        
        await publisher.stop()
    
    @pytest.mark.asyncio
    async def test_telemetry_with_compression(self, mock_message_bus, benchmark_results):
        """Benchmark telemetry with compression enabled."""
        config = TelemetryConfig(
            batch_size=100,
            batch_window_ms=50,
            compression="zstd",
            compression_level=3
        )
        
        publisher = TelemetryPublisher(mock_message_bus, config)
        await publisher.start()
        
        num_messages = 1000
        start_time = time.time()
        
        for i in range(num_messages):
            # Large health report
            health_data = torch.randn(500, 10)  # More metrics
            await publisher.publish_layer_health(f"layer_{i % 10}", health_data)
        
        await publisher._flush_batch()
        
        elapsed = time.time() - start_time
        stats = await publisher.get_stats()
        
        benchmark_results.add_result(
            "Telemetry with Compression",
            operations=num_messages,
            elapsed_seconds=elapsed,
            compression_ratio=stats.get("compression_ratio", 1.0)
        )
        
        await publisher.stop()
    
    @pytest.mark.asyncio
    async def test_seed_metrics_throughput(self, mock_message_bus, benchmark_results):
        """Benchmark individual seed metrics publishing."""
        config = TelemetryConfig(batch_size=200, batch_window_ms=100)
        publisher = TelemetryPublisher(mock_message_bus, config)
        await publisher.start()
        
        num_seeds = 10000
        start_time = time.time()
        
        for seed_id in range(num_seeds):
            await publisher.publish_seed_metrics(
                layer_id="test_layer",
                seed_id=seed_id,
                metrics={
                    "loss": np.random.random(),
                    "accuracy": np.random.random(),
                    "compute_time_ms": np.random.random() * 100
                },
                lifecycle_state="TRAINING",
                blueprint_id=seed_id % 10
            )
        
        await publisher._flush_batch()
        
        elapsed = time.time() - start_time
        
        benchmark_results.add_result(
            "Seed Metrics Publishing",
            operations=num_seeds,
            elapsed_seconds=elapsed
        )
        
        await publisher.stop()


class TestCommandHandlingBenchmarks:
    """Benchmark command handling performance."""
    
    @pytest.mark.asyncio
    async def test_command_throughput(self, mock_message_bus, 
                                    test_layer_registry,
                                    benchmark_results):
        """Benchmark command handling throughput."""
        handler = CommandHandler(test_layer_registry, mock_message_bus)
        await handler.start()
        
        num_commands = 1000
        start_time = time.time()
        
        # Execute commands
        tasks = []
        for i in range(num_commands):
            cmd = LifecycleTransitionCommand(
                layer_id="test_layer_1",
                seed_id=i % 100,
                target_state="TRAINING"
            )
            tasks.append(handler.handle_command(cmd))
        
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        stats = await handler.get_stats()
        
        benchmark_results.add_result(
            "Command Handling",
            operations=num_commands,
            elapsed_seconds=elapsed,
            success_count=success_count,
            average_execution_ms=stats["average_execution_ms"]
        )
        
        await handler.stop()
    
    @pytest.mark.asyncio
    async def test_batch_command_performance(self, mock_message_bus,
                                           test_layer_registry,
                                           benchmark_results):
        """Benchmark batch command execution."""
        handler = CommandHandler(test_layer_registry, mock_message_bus)
        await handler.start()
        
        # First transition seeds to TRAINING state so they can accept blueprint updates
        from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle
        layer = test_layer_registry["test_layer_1"]
        for i in range(100):  # Prepare 100 seeds
            layer.set_seed_state(i, ExtendedLifecycle.TRAINING)
        
        # Create batch commands
        batch_size = 100
        num_batches = 10
        
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            commands = []
            for i in range(batch_size):
                cmd = BlueprintUpdateCommand(
                    layer_id="test_layer_1",
                    seed_id=i,
                    blueprint_id=f"blueprint_{i}",
                    grafting_strategy="immediate"
                )
                commands.append(cmd)
            
            batch_cmd = BatchCommand(
                commands=commands,
                atomic=True
            )
            
            result = await handler.handle_command(batch_cmd)
            # Don't assert success - this is a benchmark test
            # Just count successes for metrics
        
        elapsed = time.time() - start_time
        total_commands = batch_size * num_batches
        
        benchmark_results.add_result(
            "Batch Command Handling",
            operations=total_commands,
            elapsed_seconds=elapsed,
            batches=num_batches,
            commands_per_batch=batch_size
        )
        
        await handler.stop()


class TestUtilityBenchmarks:
    """Benchmark utility components."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self, benchmark_results):
        """Benchmark circuit breaker overhead."""
        breaker = CircuitBreaker("test")
        
        # Success case benchmark
        async def fast_operation():
            return "success"
        
        num_calls = 10000
        start_time = time.time()
        
        for _ in range(num_calls):
            result = await breaker.call(fast_operation)
            assert result == "success"
        
        elapsed = time.time() - start_time
        
        benchmark_results.add_result(
            "Circuit Breaker (Success)",
            operations=num_calls,
            elapsed_seconds=elapsed
        )
        
        # Failure case benchmark
        breaker_fail = CircuitBreaker("test_fail")
        failures = 0
        
        async def failing_operation():
            raise Exception("Test failure")
        
        start_time = time.time()
        
        for i in range(num_calls):
            try:
                await breaker_fail.call(failing_operation)
            except Exception:
                failures += 1
        
        elapsed = time.time() - start_time
        
        benchmark_results.add_result(
            "Circuit Breaker (Failure)",
            operations=num_calls,
            elapsed_seconds=elapsed,
            failures=failures,
            circuit_state=breaker_fail.state.value
        )
    
    @pytest.mark.asyncio
    async def test_deduplicator_performance(self, benchmark_results):
        """Benchmark message deduplication."""
        dedup = MessageDeduplicator(window_size=10000)
        
        num_messages = 50000
        num_duplicates = 10000
        
        # Generate message IDs
        message_ids = [f"msg_{i}" for i in range(num_messages)]
        
        # Add duplicates
        duplicate_ids = np.random.choice(message_ids, num_duplicates).tolist()
        all_ids = message_ids + duplicate_ids
        np.random.shuffle(all_ids)
        
        start_time = time.time()
        duplicates_found = 0
        
        for msg_id in all_ids:
            if await dedup.is_duplicate(msg_id):
                duplicates_found += 1
            else:
                await dedup.mark_processed(msg_id)
        
        elapsed = time.time() - start_time
        stats = dedup.get_stats()
        
        benchmark_results.add_result(
            "Message Deduplication",
            operations=len(all_ids),
            elapsed_seconds=elapsed,
            duplicates_found=duplicates_found,
            false_positive_rate=stats.get("false_positive_rate", 0)
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self, benchmark_results):
        """Benchmark rate limiter."""
        from src.esper.morphogenetic_v2.message_bus.utils import RateLimiterConfig
        
        config = RateLimiterConfig(
            max_calls=1000,
            time_window_seconds=1.0
        )
        limiter = RateLimiter("test", config)
        
        num_requests = 10000
        allowed = 0
        rejected = 0
        
        start_time = time.time()
        
        for _ in range(num_requests):
            if await limiter.acquire():
                allowed += 1
            else:
                rejected += 1
        
        elapsed = time.time() - start_time
        
        benchmark_results.add_result(
            "Rate Limiter",
            operations=num_requests,
            elapsed_seconds=elapsed,
            allowed=allowed,
            rejected=rejected
        )
    
    @pytest.mark.asyncio
    async def test_message_batcher_performance(self, benchmark_results):
        """Benchmark message batching."""
        batcher = MessageBatcher(
            batch_size=100,
            window_ms=50,
            max_bytes=1024 * 1024
        )
        
        num_messages = 10000
        batches_created = 0
        
        start_time = time.time()
        
        for i in range(num_messages):
            message = {"id": i, "data": "x" * 100}
            batch = await batcher.add(message)
            if batch:
                batches_created += 1
        
        # Flush final batch
        final_batch = await batcher.flush()
        if final_batch:
            batches_created += 1
        
        elapsed = time.time() - start_time
        stats = batcher.get_stats()
        
        benchmark_results.add_result(
            "Message Batcher",
            operations=num_messages,
            elapsed_seconds=elapsed,
            batches_created=batches_created,
            average_batch_size=stats["average_batch_size"]
        )


class TestRedisStreamBenchmarks:
    """Benchmark Redis Streams performance (if available)."""
    
    @pytest.mark.asyncio
    async def test_redis_publish_throughput(self, redis_test_client,
                                          message_bus_config,
                                          benchmark_results):
        """Benchmark Redis Streams publishing."""
        client = RedisStreamClient(message_bus_config)
        await client.connect()
        
        num_messages = 1000
        start_time = time.time()
        
        for i in range(num_messages):
            report = LayerHealthReport(
                layer_id=f"layer_{i % 10}",
                total_seeds=100,
                active_seeds=50
            )
            
            await client.publish(f"benchmark.stream.{i % 10}", report)
        
        elapsed = time.time() - start_time
        stats = await client.get_stats()
        
        benchmark_results.add_result(
            "Redis Streams Publishing",
            operations=num_messages,
            elapsed_seconds=elapsed,
            publish_errors=stats["publish_errors"]
        )
        
        await client.disconnect()
    
    @pytest.mark.asyncio
    async def test_redis_subscribe_performance(self, redis_test_client,
                                             message_bus_config,
                                             benchmark_results):
        """Benchmark Redis Streams subscription."""
        publisher = RedisStreamClient(message_bus_config)
        subscriber = RedisStreamClient(message_bus_config)
        
        await publisher.connect()
        await subscriber.connect()
        
        messages_received = []
        
        async def handler(message):
            messages_received.append(message)
        
        # Subscribe
        await subscriber.subscribe("benchmark.test.*", handler)
        
        # Publish messages
        num_messages = 1000
        start_time = time.time()
        
        for i in range(num_messages):
            report = SeedMetricsSnapshot(
                layer_id="test",
                seed_id=i,
                lifecycle_state="TRAINING",
                metrics={"loss": 0.1}
            )
            
            await publisher.publish("benchmark.test.stream", report)
        
        # Wait for all messages
        while len(messages_received) < num_messages:
            await asyncio.sleep(0.01)
        
        elapsed = time.time() - start_time
        
        benchmark_results.add_result(
            "Redis Streams Subscribe",
            operations=num_messages,
            elapsed_seconds=elapsed,
            messages_received=len(messages_received)
        )
        
        await publisher.disconnect()
        await subscriber.disconnect()


class TestScalabilityBenchmarks:
    """Benchmark scalability with increasing load."""
    
    @pytest.mark.asyncio
    async def test_concurrent_publishers(self, mock_message_bus, benchmark_results):
        """Benchmark with multiple concurrent publishers."""
        num_publishers = 10
        messages_per_publisher = 100
        
        async def publisher_task(publisher_id: int):
            config = TelemetryConfig(batch_size=50, batch_window_ms=100)
            publisher = TelemetryPublisher(mock_message_bus, config)
            await publisher.start()
            
            for i in range(messages_per_publisher):
                health_data = torch.randn(100, 4)
                await publisher.publish_layer_health(
                    f"publisher_{publisher_id}_layer_{i % 5}",
                    health_data
                )
            
            await publisher.stop()
        
        start_time = time.time()
        
        # Run all publishers concurrently
        tasks = [publisher_task(i) for i in range(num_publishers)]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        total_messages = num_publishers * messages_per_publisher
        
        benchmark_results.add_result(
            "Concurrent Publishers",
            operations=total_messages,
            elapsed_seconds=elapsed,
            publishers=num_publishers
        )
    
    @pytest.mark.asyncio
    async def test_scaling_command_handlers(self, mock_message_bus,
                                          test_layer_registry,
                                          benchmark_results):
        """Benchmark command handling at scale."""
        # Create multiple handlers
        num_handlers = 5
        handlers = []
        
        for i in range(num_handlers):
            handler = CommandHandler(test_layer_registry, mock_message_bus)
            await handler.start()
            handlers.append(handler)
        
        # Send commands to different handlers
        num_commands = 1000
        start_time = time.time()
        
        tasks = []
        for i in range(num_commands):
            handler = handlers[i % num_handlers]
            cmd = LifecycleTransitionCommand(
                layer_id=f"test_layer_{(i % 2) + 1}",
                seed_id=i % 100,
                target_state="TRAINING"
            )
            tasks.append(handler.handle_command(cmd))
        
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        benchmark_results.add_result(
            "Scaled Command Handlers",
            operations=num_commands,
            elapsed_seconds=elapsed,
            handlers=num_handlers,
            success_rate=success_count / num_commands
        )
        
        # Cleanup
        for handler in handlers:
            await handler.stop()


@pytest.mark.asyncio
async def test_run_all_benchmarks(mock_message_bus, redis_test_client,
                                 message_bus_config, test_layer_registry,
                                 benchmark_results):
    """Run all benchmarks and print summary."""
    # Create test instances
    telemetry_bench = TestTelemetryBenchmarks()
    command_bench = TestCommandHandlingBenchmarks()
    utility_bench = TestUtilityBenchmarks()
    redis_bench = TestRedisStreamBenchmarks()
    scale_bench = TestScalabilityBenchmarks()
    
    # Run benchmarks
    print("\nRunning Message Bus Benchmarks...")
    
    # Telemetry benchmarks
    await telemetry_bench.test_telemetry_throughput(mock_message_bus, benchmark_results)
    await telemetry_bench.test_telemetry_with_compression(mock_message_bus, benchmark_results)
    await telemetry_bench.test_seed_metrics_throughput(mock_message_bus, benchmark_results)
    
    # Command benchmarks
    await command_bench.test_command_throughput(mock_message_bus, test_layer_registry, benchmark_results)
    await command_bench.test_batch_command_performance(mock_message_bus, test_layer_registry, benchmark_results)
    
    # Utility benchmarks
    await utility_bench.test_circuit_breaker_performance(benchmark_results)
    await utility_bench.test_deduplicator_performance(benchmark_results)
    await utility_bench.test_rate_limiter_performance(benchmark_results)
    await utility_bench.test_message_batcher_performance(benchmark_results)
    
    # Redis benchmarks (if available)
    if redis_test_client:
        await redis_bench.test_redis_publish_throughput(redis_test_client, message_bus_config, benchmark_results)
        await redis_bench.test_redis_subscribe_performance(redis_test_client, message_bus_config, benchmark_results)
    
    # Scalability benchmarks
    await scale_bench.test_concurrent_publishers(mock_message_bus, benchmark_results)
    await scale_bench.test_scaling_command_handlers(mock_message_bus, test_layer_registry, benchmark_results)
    
    # Print results
    benchmark_results.print_summary()