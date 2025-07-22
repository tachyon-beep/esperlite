"""
Async HTTP client utilities for Esper services.

This module provides high-performance async HTTP clients optimized for
service-to-service communication with retry logic and connection pooling.
"""

import asyncio
import logging
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import aiohttp

logger = logging.getLogger(__name__)


class AsyncHttpResponse:
    """
    Response wrapper for AsyncHttpClient.

    Provides a simple interface to response data without requiring
    context manager usage.
    """

    def __init__(self, status: int, data: bytes, content_type: str):
        """
        Initialize response wrapper.

        Args:
            status: HTTP status code
            data: Response body data
            content_type: Response content type
        """
        self.status = status
        self._data = data
        self.content_type = content_type

    async def json(self) -> Dict[str, Any]:
        """Parse response as JSON."""
        import json

        return json.loads(self._data.decode("utf-8"))

    async def text(self) -> str:
        """Get response as text."""
        return self._data.decode("utf-8")

    async def read(self) -> bytes:
        """Get raw response data."""
        return self._data


class AsyncHttpClient:
    """
    High-performance async HTTP client with connection pooling and retries.

    Provides optimized HTTP operations for inter-service communication
    with comprehensive error handling and performance monitoring.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_connections: int = 100,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ):
        """
        Initialize async HTTP client.

        Args:
            timeout: Request timeout in seconds
            max_connections: Maximum connection pool size
            max_retries: Number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session: Optional[aiohttp.ClientSession] = None

        # Performance tracking
        self.request_count = 0
        self.total_request_time = 0.0
        self.error_count = 0

        logger.debug(
            f"Initialized AsyncHttpClient with {max_connections} max connections"
        )

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=self.connector,
            headers={"User-Agent": "Esper-AsyncHttpClient/1.0"},
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """
        Perform async GET request with retry logic.

        Args:
            url: Request URL
            params: Query parameters
            headers: Additional headers
            **kwargs: Additional aiohttp parameters

        Returns:
            HTTP response
        """
        return await self._request("GET", url, params=params, headers=headers, **kwargs)

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """
        Perform async POST request with retry logic.

        Args:
            url: Request URL
            json: JSON payload
            data: Raw data payload
            headers: Additional headers
            **kwargs: Additional aiohttp parameters

        Returns:
            HTTP response
        """
        return await self._request(
            "POST", url, json=json, data=data, headers=headers, **kwargs
        )

    async def put(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Union[str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> aiohttp.ClientResponse:
        """
        Perform async PUT request with retry logic.

        Args:
            url: Request URL
            json: JSON payload
            data: Raw data payload
            headers: Additional headers
            **kwargs: Additional aiohttp parameters

        Returns:
            HTTP response
        """
        return await self._request(
            "PUT", url, json=json, data=data, headers=headers, **kwargs
        )

    async def _request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Execute HTTP request with retry logic and error handling.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Request parameters

        Returns:
            HTTP response

        Raises:
            aiohttp.ClientError: For HTTP errors after retries exhausted
        """
        if not self.session:
            raise RuntimeError(
                "HTTP client not initialized. Use async context manager."
            )

        start_time = time.perf_counter()
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Perform request
                async with self.session.request(method, url, **kwargs) as response:
                    # Update statistics
                    self.request_count += 1
                    self.total_request_time += time.perf_counter() - start_time

                    # Check for HTTP errors
                    response.raise_for_status()

                    logger.debug(
                        f"{method} {url} -> {response.status} (attempt {attempt + 1})"
                    )

                    # Read the response data and create a wrapper
                    response_data = await response.read()

                    # Create a response wrapper that provides the data we need
                    response_wrapper = AsyncHttpResponse(
                        status=response.status,
                        data=response_data,
                        content_type=response.content_type,
                    )

                    return response_wrapper

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                self.error_count += 1

                # Don't retry for certain errors
                if isinstance(e, aiohttp.ClientResponseError):
                    if e.status in (400, 401, 403, 404, 422):  # Client errors
                        logger.warning("%s %s failed with %s: %s", method, url, e.status, e)
                        raise

                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"{method} {url} failed (attempt {attempt + 1}), "
                        f"retrying in {wait_time:.2f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"{method} {url} failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise

        # Should not reach here, but raise last exception if we do
        if last_exception:
            raise last_exception

    def get_stats(self) -> Dict[str, Any]:
        """
        Get HTTP client performance statistics.

        Returns:
            Performance statistics dictionary
        """
        avg_request_time = 0.0
        if self.request_count > 0:
            avg_request_time = self.total_request_time / self.request_count

        error_rate = 0.0
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "total_request_time": self.total_request_time,
            "avg_request_time_ms": avg_request_time * 1000,
            "max_connections": self.connector.limit if self.connector else 0,
        }


class HttpClientError(Exception):
    """Exception raised for HTTP client operation failures."""

    pass


# Convenience function for creating client instances
def create_http_client(
    timeout: int = 30, max_connections: int = 100, max_retries: int = 3
) -> AsyncHttpClient:
    """
    Create configured async HTTP client.

    Args:
        timeout: Request timeout in seconds
        max_connections: Maximum connection pool size
        max_retries: Number of retry attempts

    Returns:
        Configured AsyncHttpClient instance
    """
    return AsyncHttpClient(
        timeout=timeout, max_connections=max_connections, max_retries=max_retries
    )
