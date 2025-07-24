"""
TamiyoClient: Production HTTP client for Tamiyo strategic controller service.

This client provides reliable, circuit-breaker protected communication
with the Tamiyo service for strategic adaptation decisions.
"""

import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.utils.circuit_breaker import CircuitBreaker
from esper.utils.circuit_breaker import CircuitBreakerConfig
from esper.utils.config import ServiceConfig
from esper.utils.http_client import AsyncHttpClient

logger = logging.getLogger(__name__)


class TamiyoClient:
    """
    Production HTTP client for Tamiyo strategic controller service.

    Provides circuit-breaker protected, reliable communication with Tamiyo
    for strategic analysis and adaptation decision requests.
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        Initialize TamiyoClient.

        Args:
            config: Service configuration (uses default if None)
        """
        from esper.utils.config import get_service_config

        self.config = config or get_service_config()
        self.base_url = self.config.tamiyo_url.rstrip("/")
        self.timeout = self.config.http_timeout

        # Circuit breaker for reliability
        self._circuit_breaker = CircuitBreaker(
            name="tamiyo_client",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=45,
                success_threshold=2,
                timeout=self.timeout,
            ),
        )

        # Statistics tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_failures": 0,
            "last_request_time": None,
        }

        logger.info("Initialized TamiyoClient for %s", self.base_url)

    async def analyze_model_state(
        self,
        health_signals: List[HealthSignal],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AdaptationDecision]:
        """
        Request strategic analysis from Tamiyo service.

        Args:
            health_signals: List of health signals from KasminaLayers
            context: Additional context (epoch, learning rate, etc.)

        Returns:
            List of adaptation decisions from Tamiyo
        """
        if not health_signals:
            logger.debug("No health signals provided, skipping Tamiyo analysis")
            return []

        self._stats["total_requests"] += 1
        self._stats["last_request_time"] = time.time()

        payload = {
            "health_signals": [signal.model_dump() for signal in health_signals],
            "context": context or {},
            "timestamp": time.time(),
            "client_id": "tolaria_trainer",
        }

        try:
            async with AsyncHttpClient(timeout=self.timeout) as http_client:
                response = await self._circuit_breaker.call(
                    self._make_analyze_request, http_client, payload
                )

                # Parse response
                if response and "decisions" in response:
                    decisions = [
                        AdaptationDecision(**decision_data)
                        for decision_data in response["decisions"]
                    ]

                    self._stats["successful_requests"] += 1
                    logger.info("Tamiyo analysis returned %d decisions", len(decisions))
                    return decisions
                else:
                    logger.warning("Tamiyo response missing 'decisions' field")
                    return []

        except Exception as e:
            self._stats["failed_requests"] += 1
            if "CircuitBreakerOpen" in str(type(e)):
                self._stats["circuit_breaker_failures"] += 1
                logger.warning("Tamiyo circuit breaker is open, skipping analysis")
            else:
                logger.error("Tamiyo analysis failed: %s", e)
            return []

    async def _make_analyze_request(
        self, http_client: AsyncHttpClient, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Make the actual HTTP request to Tamiyo analyze endpoint.

        Args:
            http_client: Async HTTP client
            payload: Request payload

        Returns:
            Response data or None if failed
        """
        url = f"{self.base_url}/api/v1/analyze"

        response = await http_client.post(url, json=payload)
        response.raise_for_status()

        return await response.json()

    async def submit_adaptation_feedback(
        self,
        decision: AdaptationDecision,
        success: bool,
        performance_impact: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Submit feedback about adaptation results to Tamiyo.

        Args:
            decision: The adaptation decision that was applied
            success: Whether the adaptation was successful
            performance_impact: Performance metrics after adaptation

        Returns:
            True if feedback was submitted successfully
        """
        self._stats["total_requests"] += 1

        payload = {
            "decision": decision.model_dump(),
            "success": success,
            "performance_impact": performance_impact or {},
            "timestamp": time.time(),
            "client_id": "tolaria_trainer",
        }

        try:
            async with AsyncHttpClient(timeout=self.timeout) as http_client:
                await self._circuit_breaker.call(
                    self._make_feedback_request, http_client, payload
                )

                self._stats["successful_requests"] += 1
                logger.debug(
                    f"Submitted adaptation feedback for {decision.adaptation_type}"
                )
                return True

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.error("Failed to submit adaptation feedback: %s", e)
            return False

    async def _make_feedback_request(
        self, http_client: AsyncHttpClient, payload: Dict[str, Any]
    ) -> None:
        """
        Make the actual HTTP request to Tamiyo feedback endpoint.

        Args:
            http_client: Async HTTP client
            payload: Request payload
        """
        url = f"{self.base_url}/api/v1/feedback"

        response = await http_client.post(url, json=payload)
        response.raise_for_status()

    async def get_tamiyo_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current status of Tamiyo service.

        Returns:
            Status information or None if unavailable
        """
        try:
            async with AsyncHttpClient(timeout=self.timeout) as http_client:
                response = await self._circuit_breaker.call(
                    self._make_status_request, http_client
                )

                self._stats["successful_requests"] += 1
                return response

        except Exception as e:
            self._stats["failed_requests"] += 1
            logger.debug("Failed to get Tamiyo status: %s", e)
            return None

    async def _make_status_request(
        self, http_client: AsyncHttpClient
    ) -> Dict[str, Any]:
        """
        Make the actual HTTP request to Tamiyo status endpoint.

        Args:
            http_client: Async HTTP client

        Returns:
            Status response data
        """
        url = f"{self.base_url}/api/v1/status"

        response = await http_client.get(url)
        response.raise_for_status()

        return await response.json()

    async def health_check(self) -> bool:
        """
        Perform health check on Tamiyo service.

        Returns:
            True if Tamiyo service is healthy
        """
        try:
            status = await self.get_tamiyo_status()
            return status is not None and status.get("healthy", False)
        except Exception:
            return False

    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get client statistics and circuit breaker status.

        Returns:
            Dictionary containing client statistics
        """
        cb_stats = self._circuit_breaker.get_stats()

        return {
            "client_stats": self._stats.copy(),
            "circuit_breaker_stats": cb_stats,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    def reset_stats(self) -> None:
        """Reset client statistics."""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_failures": 0,
            "last_request_time": None,
        }
        logger.debug("TamiyoClient statistics reset")


class MockTamiyoClient(TamiyoClient):
    """
    Mock implementation of TamiyoClient for testing and development.

    Provides simulated Tamiyo responses without requiring a real service.
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize mock client."""
        # Initialize parent but don't use circuit breaker
        from esper.utils.config import get_service_config

        self.config = config or get_service_config()
        self.base_url = self.config.tamiyo_url.rstrip("/")
        self.timeout = self.config.http_timeout

        # Mock statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_failures": 0,
            "last_request_time": None,
        }

        logger.info("Initialized MockTamiyoClient (simulated for %s)", self.base_url)

    async def analyze_model_state(
        self,
        health_signals: List[HealthSignal],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AdaptationDecision]:
        """
        Mock strategic analysis with simulated decisions.

        Args:
            health_signals: List of health signals from KasminaLayers
            context: Additional context

        Returns:
            Simulated adaptation decisions
        """
        self._stats["total_requests"] += 1
        self._stats["last_request_time"] = time.time()

        if not health_signals:
            return []

        # Simulate analysis delay
        import asyncio

        await asyncio.sleep(0.01)

        decisions = []

        # Generate decisions based on health signals
        for signal in health_signals:
            if signal.health_score < 0.7:  # Poor health threshold
                decision = AdaptationDecision(
                    adaptation_type="add_seed",
                    layer_name=f"layer_{hash(str(signal)) % 1000}",
                    confidence=0.8,
                    urgency=1.0 - signal.health_score,
                    metadata={
                        "target_seed_index": 0,
                        "estimated_improvement": 0.15,
                        "kernel_artifact_id": f"kernel_{int(time.time() * 1000) % 10000}",
                        "reason": f"Health score {signal.health_score:.2f} below threshold",
                    },
                )
                decisions.append(decision)

        self._stats["successful_requests"] += 1
        logger.debug("Mock Tamiyo analysis generated %d decisions", len(decisions))

        return decisions

    async def submit_adaptation_feedback(
        self,
        decision: AdaptationDecision,
        success: bool,
        performance_impact: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Mock feedback submission."""
        self._stats["total_requests"] += 1
        self._stats["successful_requests"] += 1

        logger.debug(
            f"Mock feedback: {decision.adaptation_type} -> {'success' if success else 'failure'}"
        )
        return True

    async def get_tamiyo_status(self) -> Dict[str, Any]:
        """Mock status response."""
        return {
            "healthy": True,
            "version": "mock-1.0.0",
            "uptime": time.time() % 3600,
            "active_analyses": 0,
        }

    async def health_check(self) -> bool:
        """Mock health check always returns True."""
        return True

    def get_client_stats(self) -> Dict[str, Any]:
        """
        Get mock client statistics (no circuit breaker).

        Returns:
            Dictionary containing client statistics
        """
        return {
            "client_stats": self._stats.copy(),
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    def reset_stats(self) -> None:
        """Reset mock client statistics."""
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_failures": 0,
            "last_request_time": None,
        }
        logger.debug("MockTamiyoClient statistics reset")
