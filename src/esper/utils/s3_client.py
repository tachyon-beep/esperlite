"""
Optimized S3 client for Esper with connection pooling and retry logic.

This module provides a high-performance S3 client implementation with:
- Connection pooling for reduced latency
- Intelligent retry mechanisms with exponential backoff
- Comprehensive error handling and recovery
- Performance monitoring and metrics
- Thread-safe operations
"""

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import boto3
import botocore.exceptions
from botocore.config import Config

logger = logging.getLogger(__name__)

# Constants
BUCKET_NAME_REQUIRED_ERROR = "Bucket name must be provided"


@dataclass
class S3ClientConfig:
    """Configuration for optimized S3 client."""

    endpoint_url: Optional[str] = None
    bucket_name: Optional[str] = None
    max_pool_connections: int = 50
    max_attempts: int = 3
    tcp_keepalive: bool = True
    read_timeout: int = 60
    connect_timeout: int = 10
    max_bandwidth: Optional[int] = None  # MB/s

    @classmethod
    def from_environment(cls) -> "S3ClientConfig":
        """Create config from environment variables."""
        return cls(
            endpoint_url=os.getenv("S3_ENDPOINT_URL"),
            bucket_name=os.getenv("S3_BUCKET_NAME"),
            max_pool_connections=int(os.getenv("S3_MAX_POOL_CONNECTIONS", "50")),
            max_attempts=int(os.getenv("S3_MAX_ATTEMPTS", "3")),
            tcp_keepalive=os.getenv("S3_TCP_KEEPALIVE", "true").lower() == "true",
            read_timeout=int(os.getenv("S3_READ_TIMEOUT", "60")),
            connect_timeout=int(os.getenv("S3_CONNECT_TIMEOUT", "10")),
        )


@dataclass
class S3Operation:
    """Metrics for S3 operations."""

    operation: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    retry_count: int = 0
    bytes_transferred: int = 0


class OptimizedS3Client:
    """High-performance S3 client with connection pooling and retry logic."""

    def __init__(self, config: Optional[S3ClientConfig] = None):
        """Initialize the optimized S3 client."""
        self.config = config or S3ClientConfig.from_environment()
        self._client: Optional[boto3.client] = None
        self._session_cache: Dict[str, boto3.Session] = {}
        self._operation_metrics: List[S3Operation] = []
        self._lock = asyncio.Lock()

        # Performance monitoring
        self._total_operations = 0
        self._successful_operations = 0
        self._total_bytes_transferred = 0

    def _create_optimized_client(self) -> boto3.client:
        """Create an optimized S3 client with connection pooling."""
        config = Config(
            max_pool_connections=self.config.max_pool_connections,
            retries={"max_attempts": self.config.max_attempts, "mode": "adaptive"},
            tcp_keepalive=self.config.tcp_keepalive,
            read_timeout=self.config.read_timeout,
            connect_timeout=self.config.connect_timeout,
            # Enable request/response compression
            parameter_validation=False,  # Skip validation for performance
        )

        return boto3.client("s3", endpoint_url=self.config.endpoint_url, config=config)

    @property
    def client(self) -> boto3.client:
        """Get or create the S3 client."""
        if self._client is None:
            self._client = self._create_optimized_client()
        return self._client

    async def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Upload a file to S3 with retry logic and performance monitoring.

        Args:
            local_path: Path to local file
            s3_key: S3 object key
            bucket: S3 bucket name (uses config default if not provided)
            extra_args: Additional S3 upload arguments

        Returns:
            True if successful, False otherwise
        """
        bucket = bucket or self.config.bucket_name
        if not bucket:
            raise ValueError(BUCKET_NAME_REQUIRED_ERROR)

        start_time = time.time()
        retry_count = 0

        # Get file size safely
        try:
            file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        except (OSError, IOError):
            file_size = 0

        for attempt in range(self.config.max_attempts):
            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        self.client.upload_file,
                        local_path,
                        bucket,
                        s3_key,
                        extra_args,
                    )

                # Record successful operation
                duration_ms = (time.time() - start_time) * 1000
                self._record_operation(
                    S3Operation(
                        operation="upload",
                        duration_ms=duration_ms,
                        success=True,
                        retry_count=retry_count,
                        bytes_transferred=file_size,
                    )
                )
                return True

            except botocore.exceptions.ClientError as e:
                retry_count += 1
                error_code = e.response.get("Error", {}).get("Code", "Unknown")

                if attempt == self.config.max_attempts - 1:
                    # Final attempt failed
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_operation(
                        S3Operation(
                            operation="upload",
                            duration_ms=duration_ms,
                            success=False,
                            error=f"{error_code}: {str(e)}",
                            retry_count=retry_count,
                        )
                    )
                    logger.error("Upload failed after %d retries: %s", retry_count, e)
                    return False

                # Exponential backoff
                wait_time = (2**attempt) * 0.1
                logger.warning(
                    "Upload attempt %d failed, retrying in %ss: %s",
                    attempt + 1,
                    wait_time,
                    e,
                )
                await asyncio.sleep(wait_time)

            except (OSError, IOError, RuntimeError) as e:
                duration_ms = (time.time() - start_time) * 1000
                self._record_operation(
                    S3Operation(
                        operation="upload",
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        retry_count=retry_count,
                    )
                )
                logger.error("Unexpected error during upload: %s", e)
                return False

        return False

    async def download_file(
        self, s3_key: str, local_path: str, bucket: Optional[str] = None
    ) -> bool:
        """
        Download a file from S3 with retry logic and performance monitoring.

        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
            bucket: S3 bucket name (uses config default if not provided)

        Returns:
            True if successful, False otherwise
        """
        bucket = bucket or self.config.bucket_name
        if not bucket:
            raise ValueError(BUCKET_NAME_REQUIRED_ERROR)

        start_time = time.time()
        retry_count = 0

        for attempt in range(self.config.max_attempts):
            try:
                async with self._lock:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self.client.download_file, bucket, s3_key, local_path
                    )

                # Get file size for metrics
                file_size = (
                    os.path.getsize(local_path) if os.path.exists(local_path) else 0
                )

                # Record successful operation
                duration_ms = (time.time() - start_time) * 1000
                self._record_operation(
                    S3Operation(
                        operation="download",
                        duration_ms=duration_ms,
                        success=True,
                        retry_count=retry_count,
                        bytes_transferred=file_size,
                    )
                )
                return True

            except botocore.exceptions.ClientError as e:
                retry_count += 1
                error_code = e.response.get("Error", {}).get("Code", "Unknown")

                if attempt == self.config.max_attempts - 1:
                    # Final attempt failed
                    duration_ms = (time.time() - start_time) * 1000
                    self._record_operation(
                        S3Operation(
                            operation="download",
                            duration_ms=duration_ms,
                            success=False,
                            error=f"{error_code}: {str(e)}",
                            retry_count=retry_count,
                        )
                    )
                    logger.error("Download failed after %d retries: %s", retry_count, e)
                    return False

                # Exponential backoff
                wait_time = (2**attempt) * 0.1
                logger.warning(
                    "Download attempt %d failed, retrying in %ss: %s",
                    attempt + 1,
                    wait_time,
                    e,
                )
                await asyncio.sleep(wait_time)

            except (OSError, IOError, RuntimeError) as e:
                duration_ms = (time.time() - start_time) * 1000
                self._record_operation(
                    S3Operation(
                        operation="download",
                        duration_ms=duration_ms,
                        success=False,
                        error=str(e),
                        retry_count=retry_count,
                    )
                )
                logger.error("Unexpected error during download: %s", e)
                return False

        return False

    def upload_bytes(
        self, data: bytes, s3_key: str, bucket: Optional[str] = None
    ) -> str:
        """
        Upload bytes data to S3 synchronously.

        Args:
            data: Bytes data to upload
            s3_key: S3 object key
            bucket: S3 bucket name (uses config default if not provided)

        Returns:
            S3 object reference string

        Raises:
            ValueError: If bucket name is not provided
            RuntimeError: If upload fails
        """
        bucket = bucket or self.config.bucket_name
        if not bucket:
            raise ValueError(BUCKET_NAME_REQUIRED_ERROR)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(data)
            tmp_path = tmp_file.name

        try:
            # Use asyncio to run the async upload
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(
                self.upload_file(tmp_path, s3_key, bucket)
            )

            if success:
                return f"s3://{bucket}/{s3_key}"

            raise RuntimeError(f"Failed to upload to s3://{bucket}/{s3_key}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def object_exists(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """
        Check if an S3 object exists.

        Args:
            s3_key: S3 object key
            bucket: S3 bucket name (uses config default if not provided)

        Returns:
            True if object exists, False otherwise
        """
        bucket = bucket or self.config.bucket_name
        if not bucket:
            raise ValueError(BUCKET_NAME_REQUIRED_ERROR)

        start_time = time.time()

        try:
            async with self._lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: self.client.head_object(Bucket=bucket, Key=s3_key)
                )

            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(
                S3Operation(
                    operation="head_object", duration_ms=duration_ms, success=True
                )
            )
            return True

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            duration_ms = (time.time() - start_time) * 1000

            if error_code == "404":
                # Object doesn't exist - this is expected, not an error
                self._record_operation(
                    S3Operation(
                        operation="head_object",
                        duration_ms=duration_ms,
                        success=True,  # Not finding is a successful check
                    )
                )
                return False

            # Actual error
            self._record_operation(
                S3Operation(
                    operation="head_object",
                    duration_ms=duration_ms,
                    success=False,
                    error=f"{error_code}: {str(e)}",
                )
            )
            logger.error("Error checking object existence: %s", e)
            return False

        except (OSError, IOError, RuntimeError) as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_operation(
                S3Operation(
                    operation="head_object",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
            )
            logger.error("Unexpected error checking object existence: %s", e)
            return False

    def _record_operation(self, operation: S3Operation) -> None:
        """Record operation metrics for monitoring."""
        self._operation_metrics.append(operation)
        self._total_operations += 1

        if operation.success:
            self._successful_operations += 1
            self._total_bytes_transferred += operation.bytes_transferred

        # Keep only last 1000 operations for memory efficiency
        if len(self._operation_metrics) > 1000:
            self._operation_metrics = self._operation_metrics[-1000:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        if self._total_operations == 0:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "average_duration_ms": 0.0,
                "total_bytes_transferred": 0,
                "operations_per_second": 0.0,
            }

        success_rate = self._successful_operations / self._total_operations
        avg_duration = sum(op.duration_ms for op in self._operation_metrics) / len(
            self._operation_metrics
        )

        # Calculate operations per second over last 100 operations
        recent_ops = (
            self._operation_metrics[-100:]
            if len(self._operation_metrics) >= 100
            else self._operation_metrics
        )
        if len(recent_ops) >= 2:
            time_span = (
                sum(op.duration_ms for op in recent_ops) / 1000
            )  # Convert to seconds
            ops_per_second = len(recent_ops) / time_span if time_span > 0 else 0
        else:
            ops_per_second = 0

        return {
            "total_operations": self._total_operations,
            "success_rate": success_rate,
            "average_duration_ms": avg_duration,
            "total_bytes_transferred": self._total_bytes_transferred,
            "operations_per_second": ops_per_second,
            "recent_operations": len(self._operation_metrics),
        }

    @asynccontextmanager
    async def batch_context(self):
        """Context manager for batch operations with optimized connection reuse."""
        try:
            # Pre-warm the connection
            if self._client is None:
                self._client = self._create_optimized_client()
            yield self
        finally:
            # Connection cleanup handled by boto3's connection pooling
            pass


# Module-level client instance for convenient access
class _ClientManager:
    """Manages the default S3 client instance."""

    def __init__(self):
        self._default_client: Optional[OptimizedS3Client] = None

    def get_client(self, config: Optional[S3ClientConfig] = None) -> OptimizedS3Client:
        """Get the default S3 client instance."""
        if config is not None:
            # If custom config is provided, always create a new client
            return OptimizedS3Client(config)

        if self._default_client is None:
            self._default_client = OptimizedS3Client(config)
        return self._default_client


_client_manager = _ClientManager()


def get_s3_client(config: Optional[S3ClientConfig] = None) -> OptimizedS3Client:
    """Get the default S3 client instance."""
    return _client_manager.get_client(config)


async def upload_file(
    local_path: str,
    s3_key: str,
    bucket: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function for uploading files."""
    client = get_s3_client()
    return await client.upload_file(local_path, s3_key, bucket, extra_args)


async def download_file(
    s3_key: str, local_path: str, bucket: Optional[str] = None
) -> bool:
    """Convenience function for downloading files."""
    client = get_s3_client()
    return await client.download_file(s3_key, local_path, bucket)


async def object_exists(s3_key: str, bucket: Optional[str] = None) -> bool:
    """Convenience function for checking object existence."""
    client = get_s3_client()
    return await client.object_exists(s3_key, bucket)


def upload_bytes(s3_client, data: bytes, bucket: str, key: str) -> str:
    """
    Upload bytes data to S3 and return object reference.

    Note: This is a compatibility function for the existing tezzeret worker.
    For new code, prefer using the async upload_file method.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name

    try:
        # Use asyncio to run the async upload
        loop = asyncio.get_event_loop()
        success = loop.run_until_complete(s3_client.upload_file(tmp_path, key, bucket))

        if success:
            return f"s3://{bucket}/{key}"

        raise RuntimeError(f"Failed to upload to s3://{bucket}/{key}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
