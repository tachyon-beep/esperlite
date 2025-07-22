"""
Comprehensive logging performance test suite for Phase 1.2 optimization.

Tests logging overhead <0.1ms per call and validates memory efficiency.
"""

import gc
import logging
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.esper.utils.logging import AsyncEsperLogger
from src.esper.utils.logging import OptimizedStructuredFormatter
from src.esper.utils.logging import setup_high_performance_logging
from src.esper.utils.logging import setup_logging


class TestLoggingPerformance:
    """Comprehensive logging performance test suite."""

    def test_logging_latency_target(self):
        """Test logging overhead <0.1ms per call."""
        logger = setup_logging("performance_test")

        # Warm up JIT and caches
        for _ in range(100):
            logger.info("Warm up message")

        # Performance test with precise timing
        times = []
        for i in range(1000):
            start = time.perf_counter()
            logger.info("Performance test message %d", i)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Statistical analysis
        avg_latency_ms = statistics.mean(times) * 1000
        p95_latency_ms = statistics.quantiles(times, n=20)[18] * 1000  # 95th percentile
        p99_latency_ms = (
            statistics.quantiles(times, n=100)[98] * 1000
        )  # 99th percentile

        # Performance assertions
        assert (
            avg_latency_ms < 0.1
        ), f"Average logging overhead {avg_latency_ms:.3f}ms exceeds 0.1ms target"
        assert (
            p95_latency_ms < 0.2
        ), f"P95 logging overhead {p95_latency_ms:.3f}ms exceeds 0.2ms target"
        assert (
            p99_latency_ms < 0.5
        ), f"P99 logging overhead {p99_latency_ms:.3f}ms exceeds 0.5ms target"

        print(
            f"✅ Logging Performance: avg={avg_latency_ms:.3f}ms, p95={p95_latency_ms:.3f}ms, p99={p99_latency_ms:.3f}ms"
        )

    def test_high_frequency_logging_stability(self):
        """Test stability under high-frequency logging."""
        logger = setup_high_performance_logging("stability_test")

        start_time = time.perf_counter()
        for i in range(10000):  # 10x higher frequency
            logger.debug("High frequency message %d", i)
        elapsed = time.perf_counter() - start_time

        # Should handle 10k messages in reasonable time
        messages_per_second = 10000 / elapsed
        assert (
            elapsed < 5.0
        ), f"High frequency logging took {elapsed:.2f}s, expected <5s"
        assert (
            messages_per_second > 5000
        ), f"Throughput {messages_per_second:.0f} msg/s too low"

        print(f"✅ High Frequency Stability: {messages_per_second:.0f} messages/second")

    def test_logging_memory_efficiency(self):
        """Verify logging doesn't leak memory over extended use."""
        tracemalloc.start()
        logger = setup_logging("memory_test")

        # Initial memory measurement
        gc.collect()
        initial_memory, _ = tracemalloc.get_traced_memory()

        # Log 10,000 messages
        for i in range(10000):
            logger.info("Memory test message %d", i)

        # Force garbage collection and measure final memory
        gc.collect()
        final_memory, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory analysis
        memory_growth_mb = (final_memory - initial_memory) / 1024 / 1024
        bytes_per_message = (final_memory - initial_memory) / 10000

        assert (
            memory_growth_mb < 10.0
        ), f"Memory usage {memory_growth_mb:.1f}MB too high"
        assert (
            bytes_per_message < 1024
        ), f"Memory per message {bytes_per_message:.0f} bytes too high"

        print(
            f"✅ Memory Efficiency: {memory_growth_mb:.1f}MB growth, {bytes_per_message:.0f} bytes/message"
        )

    def test_concurrent_logging_performance(self):
        """Test logging performance under concurrent access."""
        logger = setup_logging("concurrent_test")
        results = []

        def log_worker(worker_id: int, message_count: int):
            """Worker function for concurrent logging."""
            start = time.perf_counter()
            for i in range(message_count):
                logger.info("Worker %d message %d", worker_id, i)
            return time.perf_counter() - start

        # Test with 4 concurrent workers, 250 messages each
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(log_worker, i, 250) for i in range(4)]
            results = [f.result() for f in futures]

        # Performance analysis
        max_time = max(results)
        total_messages = 4 * 250
        total_time = max(results)  # Parallel execution time
        throughput = total_messages / total_time

        assert max_time < 2.0, f"Concurrent logging too slow: {max_time:.2f}s"
        assert throughput > 500, f"Concurrent throughput {throughput:.0f} msg/s too low"

        print(
            f"✅ Concurrent Performance: {throughput:.0f} messages/second across 4 workers"
        )

    def test_cross_platform_compatibility(self):
        """Verify logging works across different environments."""
        import platform

        logger = setup_logging(f"cross_platform_test_{platform.system()}")

        # Test various message types that could cause encoding issues
        test_messages = [
            "Standard ASCII message",
            "Unicode test: 测试 🚀 Ñoël",
            "Special chars: !@#$%^&*(){}[]|\\:;\"'<>?/~`",
            "Long message: " + "x" * 1000,
            f"Platform info: {platform.system()} {platform.release()}",
        ]

        start_time = time.perf_counter()
        for i, message in enumerate(test_messages):
            logger.info("Cross-platform test %d: %s", i, message)
            logger.warning("Warning %d: %s", i, message)
            logger.error("Error %d: %s", i, message)

        elapsed = time.perf_counter() - start_time

        # Should complete without exceptions and maintain performance
        assert elapsed < 1.0, f"Cross-platform test took {elapsed:.2f}s, expected <1s"

        print(
            f"✅ Cross-Platform Compatibility: {len(test_messages) * 3} diverse messages in {elapsed:.3f}s"
        )

    def test_performance_regression_check(self):
        """Ensure optimizations don't break existing functionality."""
        logger = setup_logging("regression_test")

        # Test that all existing functionality still works
        logger.info("Standard message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")

        # Performance should still meet targets
        start_time = time.perf_counter()
        for i in range(100):
            logger.info("Regression test message %d", i)
        elapsed = time.perf_counter() - start_time

        # Should be faster than 0.1ms per call
        avg_latency = elapsed / 100 * 1000  # Convert to ms
        assert (
            avg_latency < 0.1
        ), f"Performance regression: {avg_latency:.3f}ms per call"

        print(
            f"✅ Regression Check: {avg_latency:.3f}ms average latency (no performance regression)"
        )


class TestLoggingPerformance:
    """Comprehensive logging performance test suite."""

    def test_logging_latency_target(self):
        """Test logging overhead <0.1ms per call."""
        logger = setup_logging("performance_test")

        # Warm up JIT and caches
        for _ in range(100):
            logger.info("Warm up message")

        # Performance test with precise timing
        times = []
        for i in range(1000):
            start = time.perf_counter()
            logger.info("Performance test message %d", i)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Statistical analysis
        avg_latency_ms = statistics.mean(times) * 1000
        p95_latency_ms = statistics.quantiles(times, n=20)[18] * 1000  # 95th percentile
        p99_latency_ms = (
            statistics.quantiles(times, n=100)[98] * 1000
        )  # 99th percentile

        # Performance assertions
        assert (
            avg_latency_ms < 0.1
        ), f"Average logging overhead {avg_latency_ms:.3f}ms exceeds 0.1ms target"
        assert (
            p95_latency_ms < 0.2
        ), f"P95 logging overhead {p95_latency_ms:.3f}ms exceeds 0.2ms target"
        assert (
            p99_latency_ms < 0.5
        ), f"P99 logging overhead {p99_latency_ms:.3f}ms exceeds 0.5ms target"

        print(
            f"✅ Logging Performance: avg={avg_latency_ms:.3f}ms, p95={p95_latency_ms:.3f}ms, p99={p99_latency_ms:.3f}ms"
        )

    def test_high_frequency_logging_stability(self):
        """Test stability under high-frequency logging."""
        logger = setup_high_performance_logging("stability_test")

        start_time = time.perf_counter()
        for i in range(10000):  # 10x higher frequency
            logger.debug("High frequency message %d", i)
        elapsed = time.perf_counter() - start_time

        # Should handle 10k messages in reasonable time
        messages_per_second = 10000 / elapsed
        assert (
            elapsed < 5.0
        ), f"High frequency logging took {elapsed:.2f}s, expected <5s"
        assert (
            messages_per_second > 5000
        ), f"Throughput {messages_per_second:.0f} msg/s too low"

        print(f"✅ High Frequency Stability: {messages_per_second:.0f} messages/second")

    def test_logging_memory_efficiency(self):
        """Verify logging doesn't leak memory over extended use."""
        tracemalloc.start()
        logger = setup_logging("memory_test")

        # Initial memory measurement
        gc.collect()
        initial_memory, _ = tracemalloc.get_traced_memory()

        # Log 10,000 messages
        for i in range(10000):
            logger.info("Memory test message %d", i)

        # Force garbage collection and measure final memory
        gc.collect()
        final_memory, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory analysis
        memory_growth_mb = (final_memory - initial_memory) / 1024 / 1024
        bytes_per_message = (final_memory - initial_memory) / 10000

        assert (
            memory_growth_mb < 10.0
        ), f"Memory usage {memory_growth_mb:.1f}MB too high"
        assert (
            bytes_per_message < 1024
        ), f"Memory per message {bytes_per_message:.0f} bytes too high"

        print(
            f"✅ Memory Efficiency: {memory_growth_mb:.1f}MB growth, {bytes_per_message:.0f} bytes/message"
        )

    def test_concurrent_logging_performance(self):
        """Test logging performance under concurrent access."""
        logger = setup_logging("concurrent_test")
        results = []

        def log_worker(worker_id: int, message_count: int):
            """Worker function for concurrent logging."""
            start = time.perf_counter()
            for i in range(message_count):
                logger.info("Worker %d message %d", worker_id, i)
            return time.perf_counter() - start

        # Test with 4 concurrent workers, 250 messages each
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(log_worker, i, 250) for i in range(4)]
            results = [f.result() for f in futures]

        # Performance analysis
        max_time = max(results)
        total_messages = 4 * 250
        total_time = max(results)  # Parallel execution time
        throughput = total_messages / total_time

        assert max_time < 2.0, f"Concurrent logging too slow: {max_time:.2f}s"
        assert throughput > 500, f"Concurrent throughput {throughput:.0f} msg/s too low"

        print(
            f"✅ Concurrent Performance: {throughput:.0f} messages/second across 4 workers"
        )

    def test_cross_platform_compatibility(self):
        """Verify logging works across different environments."""
        import platform

        logger = setup_logging(f"cross_platform_test_{platform.system()}")

        # Test various message types that could cause encoding issues
        test_messages = [
            "Standard ASCII message",
            "Unicode test: 测试 🚀 Ñoël",
            "Special chars: !@#$%^&*(){}[]|\\:;\"'<>?/~`",
            "Long message: " + "x" * 1000,
            f"Platform info: {platform.system()} {platform.release()}",
        ]

        start_time = time.perf_counter()
        for i, message in enumerate(test_messages):
            logger.info("Cross-platform test %d: %s", i, message)
            logger.warning("Warning %d: %s", i, message)
            logger.error("Error %d: %s", i, message)

        elapsed = time.perf_counter() - start_time

        # Should complete without exceptions and maintain performance
        assert elapsed < 1.0, f"Cross-platform test took {elapsed:.2f}s, expected <1s"

        print(
            f"✅ Cross-Platform Compatibility: {len(test_messages) * 3} diverse messages in {elapsed:.3f}s"
        )

    def test_formatter_caching_effectiveness(self):
        """Test OptimizedStructuredFormatter caching performance."""
        formatter = OptimizedStructuredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        import logging

        # Create test records
        record1 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message 1",
            args=(),
            exc_info=None,
        )
        record2 = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=1,
            msg="Test message 2",
            args=(),
            exc_info=None,
        )

        # Time first format (cache miss)
        start = time.perf_counter()
        result1 = formatter.format(record1)
        first_time = time.perf_counter() - start

        # Time second format (should benefit from caching optimizations)
        start = time.perf_counter()
        result2 = formatter.format(record2)
        second_time = time.perf_counter() - start

        # Verify both work correctly
        assert "Test message 1" in result1
        assert "Test message 2" in result2
        assert "INFO" in result1 and "INFO" in result2

        print(
            f"✅ Formatter Caching: first={first_time * 1000:.3f}ms, second={second_time * 1000:.3f}ms"
        )

    def test_async_logger_performance(self):
        """Test AsyncEsperLogger performance characteristics."""
        async_logger = AsyncEsperLogger("async_perf_test")

        try:
            # Performance test
            start_time = time.perf_counter()
            for i in range(1000):
                # Async logging should have minimal blocking time
                async_logger.queue_handler.handle(
                    logging.LogRecord(
                        name="async_test",
                        level=logging.INFO,
                        pathname="",
                        lineno=1,
                        msg=f"Async test message {i}",
                        args=(),
                        exc_info=None,
                    )
                )
            elapsed = time.perf_counter() - start_time

            # Async operations should be very fast
            avg_time_ms = elapsed / 1000 * 1000
            assert (
                avg_time_ms < 0.05
            ), f"Async logging overhead {avg_time_ms:.3f}ms too high"

            print(f"✅ Async Logger: {avg_time_ms:.3f}ms average overhead per message")

            # Allow time for background processing
            time.sleep(0.1)

        finally:
            async_logger.shutdown()

    def test_performance_regression_check(self):
        """Ensure optimizations don't break existing functionality."""
        logger = setup_logging("regression_test")

        # Test that all existing functionality still works
        logger.info("Standard message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.debug("Debug message")

        # Performance should still meet targets
        start_time = time.perf_counter()
        for i in range(100):
            logger.info("Regression test message %d", i)
        elapsed = time.perf_counter() - start_time

        # Should be faster than 0.1ms per call
        avg_latency = elapsed / 100 * 1000  # Convert to ms
        assert (
            avg_latency < 0.1
        ), f"Performance regression: {avg_latency:.3f}ms per call"

        print(
            f"✅ Regression Check: {avg_latency:.3f}ms average latency (no performance regression)"
        )

    @pytest.mark.stress
    def test_extended_operation_stability(self):
        """Test logging stability over extended operation."""
        logger = setup_logging("extended_stability_test")

        # Run extended logging operation
        start_time = time.perf_counter()
        message_count = 50000

        for i in range(message_count):
            logger.info("Extended stability message %d", i)

            # Periodic performance check
            if i % 10000 == 0 and i > 0:
                current_time = time.perf_counter()
                elapsed = current_time - start_time
                rate = i / elapsed
                assert (
                    rate > 1000
                ), f"Performance degraded: {rate:.0f} msg/s at message {i}"

        total_elapsed = time.perf_counter() - start_time
        final_rate = message_count / total_elapsed

        assert (
            final_rate > 5000
        ), f"Extended operation rate {final_rate:.0f} msg/s too low"
        assert (
            total_elapsed < 30
        ), f"Extended operation took {total_elapsed:.1f}s, expected <30s"

        print(
            f"✅ Extended Stability: {message_count} messages in {total_elapsed:.1f}s ({final_rate:.0f} msg/s)"
        )
