"""
Test service infrastructure for integration tests.

This module provides fixtures that spin up real services (Redis, etc.)
for integration testing, using Docker containers or embedded servers.
"""

import os
import time
import socket
import subprocess
import pytest
import logging
from contextlib import closing
from typing import Optional

logger = logging.getLogger(__name__)

def find_free_port(start=6379, end=7000) -> int:
    """Find a free port to use for services."""
    for port in range(start, end):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free ports found in range {start}-{end}")


class RedisTestServer:
    """Manages a Redis server instance for testing."""

    def __init__(self, port: Optional[int] = None):
        self.port = port or find_free_port()
        self.process = None
        self.redis_url = f"redis://localhost:{self.port}"

    def start(self) -> str:
        """Start Redis server and return connection URL."""
        # First try to use Docker
        if self._try_docker():
            return self.redis_url

        # Fall back to redis-server if available
        if self._try_redis_server():
            return self.redis_url

        # If neither work, we'll use mock
        logger.warning("No Redis available, tests will use mock")
        return None

    def _try_docker(self) -> bool:
        """Try to start Redis using Docker."""
        try:
            # Check if Docker is available
            subprocess.run(["docker", "--version"], check=True, capture_output=True)

            # Stop any existing container
            subprocess.run(
                ["docker", "stop", "test-redis"],
                capture_output=True
            )
            subprocess.run(
                ["docker", "rm", "test-redis"],
                capture_output=True
            )

            # Start Redis container
            cmd = [
                "docker", "run",
                "--name", "test-redis",
                "-d",  # detached
                "-p", f"{self.port}:6379",
                "redis:7-alpine"
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            # Wait for Redis to be ready
            time.sleep(1)

            # Test connection
            import redis
            client = redis.Redis(host='localhost', port=self.port)
            client.ping()

            logger.info(f"Started Redis in Docker on port {self.port}")
            return True

        except (subprocess.CalledProcessError, ImportError, Exception) as e:
            logger.debug(f"Docker Redis failed: {e}")
            return False

    def _try_redis_server(self) -> bool:
        """Try to start Redis using redis-server."""
        try:
            # Check if redis-server is available
            subprocess.run(["redis-server", "--version"], check=True, capture_output=True)

            # Start Redis server
            cmd = ["redis-server", "--port", str(self.port), "--save", ""]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Wait for Redis to be ready
            time.sleep(0.5)

            # Test connection
            import redis
            client = redis.Redis(host='localhost', port=self.port)
            client.ping()

            logger.info(f"Started redis-server on port {self.port}")
            return True

        except (subprocess.CalledProcessError, ImportError, Exception) as e:
            logger.debug(f"redis-server failed: {e}")
            if self.process:
                self.process.terminate()
                self.process = None
            return False

    def stop(self):
        """Stop the Redis server."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        else:
            # Try to stop Docker container
            try:
                subprocess.run(
                    ["docker", "stop", "test-redis"],
                    capture_output=True
                )
                subprocess.run(
                    ["docker", "rm", "test-redis"],
                    capture_output=True
                )
            except Exception:
                pass


# Global test Redis instance
_test_redis: Optional[RedisTestServer] = None


@pytest.fixture(scope="session")
def redis_server():
    """Session-scoped Redis server for all tests."""
    global _test_redis

    if _test_redis is None:
        _test_redis = RedisTestServer()
        redis_url = _test_redis.start()

        if redis_url:
            # Set environment variable for OonaClient
            os.environ['REDIS_URL'] = redis_url

    yield _test_redis

    # Cleanup happens at end of test session
    if _test_redis:
        _test_redis.stop()
        _test_redis = None


@pytest.fixture
def redis_client(redis_server):
    """Get a Redis client connected to test server."""
    if redis_server.redis_url:
        import redis
        client = redis.Redis.from_url(redis_server.redis_url)
        yield client
        client.flushdb()  # Clean up after test
    else:
        # Return None if no Redis available
        yield None


@pytest.fixture
def oona_client(redis_server, mock_oona_client):
    """Get OonaClient - real if Redis available, mock otherwise."""
    if redis_server.redis_url:
        try:
            from esper.services.oona_client import OonaClient
            client = OonaClient(redis_url=redis_server.redis_url)
            yield client
            client.close()
        except Exception as e:
            logger.warning(f"Failed to create real OonaClient: {e}")
            yield mock_oona_client
    else:
        yield mock_oona_client


@pytest.fixture(scope="session")
def redis_async_available():
    """Check if redis async is available."""
    try:
        import redis.asyncio  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def message_bus_client(redis_server, redis_async_available):
    """Get appropriate message bus client based on Redis availability."""
    from src.esper.morphogenetic_v2.message_bus.clients import (
        MockMessageBusClient, RedisStreamClient, MessageBusConfig
    )

    if redis_server.redis_url and redis_async_available:
        config = MessageBusConfig(redis_url=redis_server.redis_url)
        client = RedisStreamClient(config)
    else:
        client = MockMessageBusClient()

    yield client

    # Cleanup
    if hasattr(client, 'disconnect'):
        import asyncio
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(client.disconnect())


# Additional test services can be added here
# e.g., TestUrzaServer, TestKafkaServer, etc.