"""
Circuit breaker pattern implementation for resilient service communication.

This module provides circuit breaker functionality to prevent cascading failures
when external services become unavailable or slow to respond.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)
        self.message = message


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successes needed in half-open to close
    timeout: int = 30  # Request timeout in seconds


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.

    The circuit breaker prevents cascading failures by failing fast when
    a service is known to be unavailable, and automatically recovers
    when the service becomes available again.
    """

    def __init__(
        self,
        name: str = "default",
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker (for logging)
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_circuit_opens = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("Circuit breaker '%s' initialized", self.name)

    async def call(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a function call through the circuit breaker.

        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Any exception from the wrapped function
        """
        async with self._lock:
            self.total_requests += 1

            # Check if circuit breaker should transition states
            await self._check_state_transition()

            # Fail fast if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                logger.warning("Circuit breaker '%s' is OPEN, failing fast", self.name)
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")

        # Execute the function call with timeout
        try:
            # Add timeout to the function call
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.config.timeout
            )

            # Record success
            async with self._lock:
                await self._record_success()

            return result

        except asyncio.TimeoutError:
            async with self._lock:
                self.total_timeouts += 1
                await self._record_failure()
            logger.warning(
                f"Circuit breaker '{self.name}' timeout after {self.config.timeout}s"
            )
            raise

        except Exception as e:
            async with self._lock:
                await self._record_failure()
            logger.warning("Circuit breaker '%s' failure: %s", self.name, e)
            raise

    async def _check_state_transition(self) -> None:
        """Check if circuit breaker should transition between states."""
        current_time = time.time()

        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.config.recovery_timeout:
                await self._transition_to_half_open()

    async def _record_success(self) -> None:
        """Record a successful function call."""
        self.total_successes += 1

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            # Check if we have enough successes to close
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed function call."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()

        # Reset success count on failure
        self.success_count = 0

        # Check if we should open the circuit
        if (
            self.state != CircuitBreakerState.OPEN
            and self.failure_count >= self.config.failure_threshold
        ):
            await self._transition_to_open()

    async def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = time.time()
        self.total_circuit_opens += 1
        self.success_count = 0

        logger.warning(
            f"Circuit breaker '{self.name}' OPENED after {self.failure_count} failures"
        )

    async def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = time.time()
        self.success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' HALF-OPEN, testing service recovery"
        )

    async def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = time.time()
        self.failure_count = 0
        self.success_count = 0

        logger.info("Circuit breaker '%s' CLOSED, service recovered", self.name)

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary containing circuit breaker metrics
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "total_timeouts": self.total_timeouts,
            "total_circuit_opens": self.total_circuit_opens,
            "last_failure_time": self.last_failure_time,
            "last_state_change": self.last_state_change,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()

        logger.info("Circuit breaker '%s' reset to CLOSED state", self.name)

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is in CLOSED state."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is in OPEN state."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is in HALF_OPEN state."""
        return self.state == CircuitBreakerState.HALF_OPEN
