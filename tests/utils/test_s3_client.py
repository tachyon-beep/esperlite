"""Tests for the optimized S3 client module."""

import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import botocore.exceptions
import pytest

from esper.utils.s3_client import OptimizedS3Client
from esper.utils.s3_client import S3ClientConfig
from esper.utils.s3_client import S3Operation
from esper.utils.s3_client import download_file
from esper.utils.s3_client import get_s3_client
from esper.utils.s3_client import object_exists
from esper.utils.s3_client import upload_bytes
from esper.utils.s3_client import upload_file


class TestS3ClientConfig:
    """Test S3ClientConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = S3ClientConfig()

        assert config.endpoint_url is None
        assert config.bucket_name is None
        assert config.max_pool_connections == 50
        assert config.max_attempts == 3
        assert config.tcp_keepalive is True
        assert config.read_timeout == 60
        assert config.connect_timeout == 10
        assert config.max_bandwidth is None

    @patch.dict(
        os.environ,
        {
            "S3_ENDPOINT_URL": "http://localhost:9000",
            "S3_BUCKET_NAME": "test-bucket",
            "S3_MAX_POOL_CONNECTIONS": "100",
            "S3_MAX_ATTEMPTS": "5",
            "S3_TCP_KEEPALIVE": "false",
            "S3_READ_TIMEOUT": "120",
            "S3_CONNECT_TIMEOUT": "20",
        },
    )
    def test_from_environment(self):
        """Test configuration from environment variables."""
        config = S3ClientConfig.from_environment()

        assert config.endpoint_url == "http://localhost:9000"
        assert config.bucket_name == "test-bucket"
        assert config.max_pool_connections == 100
        assert config.max_attempts == 5
        assert config.tcp_keepalive is False
        assert config.read_timeout == 120
        assert config.connect_timeout == 20

    @patch.dict(os.environ, {}, clear=True)
    def test_from_environment_defaults(self):
        """Test configuration from environment with defaults."""
        config = S3ClientConfig.from_environment()

        assert config.endpoint_url is None
        assert config.bucket_name is None
        assert config.max_pool_connections == 50
        assert config.max_attempts == 3
        assert config.tcp_keepalive is True
        assert config.read_timeout == 60
        assert config.connect_timeout == 10


class TestS3Operation:
    """Test S3Operation dataclass."""

    def test_basic_operation(self):
        """Test basic operation creation."""
        op = S3Operation(
            operation="upload",
            duration_ms=123.45,
            success=True,
            retry_count=2,
            bytes_transferred=1024,
        )

        assert op.operation == "upload"
        assert abs(op.duration_ms - 123.45) < 0.001
        assert op.success is True
        assert op.error is None
        assert op.retry_count == 2
        assert op.bytes_transferred == 1024

    def test_operation_with_error(self):
        """Test operation with error."""
        op = S3Operation(
            operation="download",
            duration_ms=456.78,
            success=False,
            error="NoSuchKey: The specified key does not exist",
        )

        assert op.operation == "download"
        assert abs(op.duration_ms - 456.78) < 0.001
        assert op.success is False
        assert op.error == "NoSuchKey: The specified key does not exist"
        assert op.retry_count == 0
        assert op.bytes_transferred == 0


class TestOptimizedS3Client:
    """Test OptimizedS3Client class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = S3ClientConfig(
            endpoint_url="http://localhost:9000",
            bucket_name="test-bucket",
            max_attempts=2,
        )
        self.client = OptimizedS3Client(self.config)

    @patch("boto3.client")
    def test_create_optimized_client(self, mock_boto3_client):
        """Test S3 client creation with optimized configuration."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        self.client._create_optimized_client()

        mock_boto3_client.assert_called_once()
        args, kwargs = mock_boto3_client.call_args

        assert args[0] == "s3"
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert "config" in kwargs

        # Verify config settings
        config = kwargs["config"]
        assert config.max_pool_connections == 50
        assert config.retries["max_attempts"] == 2
        assert config.retries["mode"] == "adaptive"

    @patch("boto3.client")
    def test_client_property_lazy_initialization(self, mock_boto3_client):
        """Test that client property creates client on first access."""
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        # First access should create client
        client1 = self.client.client
        assert mock_boto3_client.call_count == 1

        # Second access should reuse existing client
        client2 = self.client.client
        assert mock_boto3_client.call_count == 1
        assert client1 is client2

    @pytest.mark.asyncio
    @patch("os.path.getsize")
    @patch("os.path.exists")
    async def test_upload_file_success(self, mock_exists, mock_getsize):
        """Test successful file upload."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        mock_s3_client = MagicMock()
        mock_s3_client.upload_file = MagicMock()

        # Patch the internal _client attribute directly
        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=None)

                result = await self.client.upload_file("/tmp/test.txt", "test-key")

                assert result is True
                mock_loop.run_in_executor.assert_called_once()

                # Check metrics
                assert self.client._total_operations == 1
                assert self.client._successful_operations == 1
                assert len(self.client._operation_metrics) == 1

                op = self.client._operation_metrics[0]
                assert op.operation == "upload"
                assert op.success is True
                assert op.bytes_transferred == 1024

    @pytest.mark.asyncio
    async def test_upload_file_no_bucket(self):
        """Test upload file with no bucket specified."""
        client = OptimizedS3Client()  # No bucket in config

        with pytest.raises(ValueError, match="Bucket name must be provided"):
            await client.upload_file("/tmp/test.txt", "test-key")

    @pytest.mark.asyncio
    async def test_upload_file_client_error_retry(self):
        """Test upload file with client error and retry logic."""
        mock_s3_client = MagicMock()

        # Create a ClientError
        error_response = {"Error": {"Code": "SlowDown", "Message": "Slow down"}}
        client_error = botocore.exceptions.ClientError(error_response, "upload_file")

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop

                # First call fails, second succeeds
                mock_loop.run_in_executor = AsyncMock(side_effect=[client_error, None])

                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    with patch("os.path.getsize", return_value=1024):
                        with patch("os.path.exists", return_value=True):
                            result = await self.client.upload_file(
                                "/tmp/test.txt", "test-key"
                            )

                assert result is True
                assert mock_loop.run_in_executor.call_count == 2
                mock_sleep.assert_called_once_with(0.1)  # Exponential backoff

    @pytest.mark.asyncio
    async def test_upload_file_max_retries_exceeded(self):
        """Test upload file when max retries are exceeded."""
        mock_s3_client = MagicMock()

        # Create a ClientError that will persist
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        client_error = botocore.exceptions.ClientError(error_response, "upload_file")

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=client_error)

                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with patch("os.path.getsize", return_value=1024):
                        with patch("os.path.exists", return_value=True):
                            result = await self.client.upload_file(
                                "/tmp/test.txt", "test-key"
                            )

                assert result is False
                assert mock_loop.run_in_executor.call_count == 2

                # Check error metrics
                assert self.client._total_operations == 1
                assert self.client._successful_operations == 0

                op = self.client._operation_metrics[0]
                assert op.operation == "upload"
                assert op.success is False
                assert "AccessDenied" in op.error

    @pytest.mark.asyncio
    async def test_download_file_success(self):
        """Test successful file download."""
        mock_s3_client = MagicMock()
        mock_s3_client.download_file = MagicMock()

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(return_value=None)

                with patch("os.path.getsize", return_value=2048):
                    with patch("os.path.exists", return_value=True):
                        result = await self.client.download_file(
                            "test-key", "/tmp/downloaded.txt"
                        )

                assert result is True
                mock_loop.run_in_executor.assert_called_once()

                # Check metrics
                assert self.client._total_operations == 1
                assert self.client._successful_operations == 1

                op = self.client._operation_metrics[0]
                assert op.operation == "download"
                assert op.success is True
                assert op.bytes_transferred == 2048

    @pytest.mark.asyncio
    async def test_object_exists_true(self):
        """Test object exists check when object exists."""
        mock_s3_client = MagicMock()
        mock_s3_client.head_object = MagicMock(return_value={"ContentLength": 1024})

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(
                    return_value={"ContentLength": 1024}
                )

                result = await self.client.object_exists("test-key")

                assert result is True
                mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_object_exists_false_404(self):
        """Test object exists check when object doesn't exist (404)."""
        mock_s3_client = MagicMock()

        # Create a 404 ClientError
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        client_error = botocore.exceptions.ClientError(error_response, "head_object")

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=client_error)

                result = await self.client.object_exists("missing-key")

                assert result is False

                # Check that 404 is recorded as success (expected behavior)
                op = self.client._operation_metrics[0]
                assert op.operation == "head_object"
                assert op.success is True

    @pytest.mark.asyncio
    async def test_object_exists_error(self):
        """Test object exists check with non-404 error."""
        mock_s3_client = MagicMock()

        # Create an access denied error
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        client_error = botocore.exceptions.ClientError(error_response, "head_object")

        with patch.object(self.client, "_client", mock_s3_client):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(side_effect=client_error)

                result = await self.client.object_exists("test-key")

                assert result is False

                # Check that error is recorded as failure
                op = self.client._operation_metrics[0]
                assert op.operation == "head_object"
                assert op.success is False
                assert "AccessDenied" in op.error

    def test_record_operation(self):
        """Test operation metrics recording."""
        op1 = S3Operation("upload", 100.0, True, bytes_transferred=1024)
        op2 = S3Operation("download", 200.0, False, error="Some error")

        self.client._record_operation(op1)
        self.client._record_operation(op2)

        assert len(self.client._operation_metrics) == 2
        assert self.client._total_operations == 2
        assert self.client._successful_operations == 1
        assert self.client._total_bytes_transferred == 1024

    def test_record_operation_memory_limit(self):
        """Test that operation metrics are limited to prevent memory growth."""
        # Add more than 1000 operations
        for i in range(1200):
            op = S3Operation(f"op_{i}", 100.0, True)
            self.client._record_operation(op)

        # Should keep only last 1000
        assert len(self.client._operation_metrics) == 1000
        assert self.client._total_operations == 1200

        # Check that we kept the most recent ones
        assert self.client._operation_metrics[0].operation == "op_200"
        assert self.client._operation_metrics[-1].operation == "op_1199"

    def test_get_performance_metrics_empty(self):
        """Test performance metrics with no operations."""
        metrics = self.client.get_performance_metrics()

        expected = {
            "total_operations": 0,
            "success_rate": 0.0,
            "average_duration_ms": 0.0,
            "total_bytes_transferred": 0,
            "operations_per_second": 0.0,
        }
        assert metrics == expected

    def test_get_performance_metrics_with_operations(self):
        """Test performance metrics calculation."""
        # Add some test operations
        ops = [
            S3Operation("upload", 100.0, True, bytes_transferred=1024),
            S3Operation("download", 200.0, True, bytes_transferred=2048),
            S3Operation("upload", 150.0, False, error="Error"),
        ]

        for op in ops:
            self.client._record_operation(op)

        metrics = self.client.get_performance_metrics()

        assert metrics["total_operations"] == 3
        assert metrics["success_rate"] == 2 / 3  # 2 successful out of 3
        assert abs(metrics["average_duration_ms"] - 150.0) < 0.001
        assert metrics["total_bytes_transferred"] == 3072  # 1024 + 2048
        assert metrics["recent_operations"] == 3

    @pytest.mark.asyncio
    async def test_batch_context(self):
        """Test batch context manager."""
        with patch.object(self.client, "_create_optimized_client") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            async with self.client.batch_context() as ctx:
                assert ctx is self.client
                assert self.client._client is mock_client

            # Client should remain after context exit
            assert self.client._client is mock_client


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Reset global client between tests."""
        import esper.utils.s3_client

        esper.utils.s3_client._default_client = None

    @patch("esper.utils.s3_client.OptimizedS3Client")
    def test_get_s3_client_singleton(self, mock_client_class):
        """Test that get_s3_client returns singleton instance."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        # First call creates instance
        client1 = get_s3_client()
        assert client1 is mock_instance
        mock_client_class.assert_called_once_with(None)

        # Second call returns same instance
        client2 = get_s3_client()
        assert client2 is mock_instance
        assert mock_client_class.call_count == 1

    @patch("esper.utils.s3_client.OptimizedS3Client")
    def test_get_s3_client_with_config(self, mock_client_class):
        """Test get_s3_client with custom config."""
        mock_instance = MagicMock()
        mock_client_class.return_value = mock_instance

        config = S3ClientConfig(bucket_name="custom-bucket")
        client = get_s3_client(config)

        assert client is mock_instance
        mock_client_class.assert_called_once_with(config)

    @pytest.mark.asyncio
    async def test_upload_file_convenience(self):
        """Test convenience upload_file function."""
        with patch("esper.utils.s3_client.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.upload_file = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client

            result = await upload_file("/tmp/test.txt", "test-key", "test-bucket")

            assert result is True
            mock_client.upload_file.assert_called_once_with(
                "/tmp/test.txt", "test-key", "test-bucket", None
            )

    @pytest.mark.asyncio
    async def test_download_file_convenience(self):
        """Test convenience download_file function."""
        with patch("esper.utils.s3_client.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.download_file = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client

            result = await download_file(
                "test-key", "/tmp/downloaded.txt", "test-bucket"
            )

            assert result is True
            mock_client.download_file.assert_called_once_with(
                "test-key", "/tmp/downloaded.txt", "test-bucket"
            )

    @pytest.mark.asyncio
    async def test_object_exists_convenience(self):
        """Test convenience object_exists function."""
        with patch("esper.utils.s3_client.get_s3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.object_exists = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_client

            result = await object_exists("test-key", "test-bucket")

            assert result is True
            mock_client.object_exists.assert_called_once_with("test-key", "test-bucket")


class TestUploadBytes:
    """Test upload_bytes compatibility function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.test_data = b"test file content"

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.unlink")
    @patch("asyncio.get_event_loop")
    def test_upload_bytes_success(
        self, mock_get_loop, mock_unlink, mock_exists, mock_temp_file
    ):
        """Test successful bytes upload."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/temp_file_123"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        mock_exists.return_value = True

        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = True

        # Mock the client's upload_file method
        self.mock_client.upload_file = MagicMock()

        result = upload_bytes(
            self.mock_client, self.test_data, "test-bucket", "test-key"
        )

        assert result == "s3://test-bucket/test-key"

        # Verify temp file was written
        mock_temp.write.assert_called_once_with(self.test_data)

        # Verify upload was called
        mock_loop.run_until_complete.assert_called_once()

        # Verify cleanup
        mock_unlink.assert_called_once_with("/tmp/temp_file_123")

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.unlink")
    @patch("asyncio.get_event_loop")
    def test_upload_bytes_failure(
        self, mock_get_loop, mock_unlink, mock_exists, mock_temp_file
    ):
        """Test bytes upload failure."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/temp_file_123"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        mock_exists.return_value = True

        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = False  # Upload fails

        # Mock the client's upload_file method
        self.mock_client.upload_file = MagicMock()

        with pytest.raises(
            RuntimeError, match="Failed to upload to s3://test-bucket/test-key"
        ):
            upload_bytes(self.mock_client, self.test_data, "test-bucket", "test-key")

        # Verify cleanup still happens
        mock_unlink.assert_called_once_with("/tmp/temp_file_123")

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.unlink")
    @patch("asyncio.get_event_loop")
    def test_upload_bytes_cleanup_on_exception(
        self, mock_get_loop, mock_unlink, mock_exists, mock_temp_file
    ):
        """Test that temporary file is cleaned up even on exception."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/temp_file_123"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        mock_exists.return_value = True

        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_until_complete.side_effect = Exception("Unexpected error")

        with pytest.raises(Exception, match="Unexpected error"):
            upload_bytes(self.mock_client, self.test_data, "test-bucket", "test-key")

        # Verify cleanup still happens
        mock_unlink.assert_called_once_with("/tmp/temp_file_123")

    @patch("tempfile.NamedTemporaryFile")
    @patch("os.path.exists")
    @patch("os.unlink")
    @patch("asyncio.get_event_loop")
    def test_upload_bytes_cleanup_missing_file(
        self, mock_get_loop, mock_unlink, mock_exists, mock_temp_file
    ):
        """Test cleanup when temporary file doesn't exist."""
        # Setup mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/temp_file_123"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        # File doesn't exist during cleanup
        mock_exists.return_value = False

        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_until_complete.return_value = True

        result = upload_bytes(
            self.mock_client, self.test_data, "test-bucket", "test-key"
        )

        assert result == "s3://test-bucket/test-key"

        # unlink should not be called if file doesn't exist
        mock_unlink.assert_not_called()


class TestIntegration:
    """Integration tests for S3 client functionality."""

    @pytest.mark.asyncio
    async def test_full_upload_download_cycle(self):
        """Test complete upload and download cycle with mocked S3."""
        config = S3ClientConfig(
            endpoint_url="http://localhost:9000", bucket_name="test-bucket"
        )
        client = OptimizedS3Client(config)

        # Test data for performance verification
        test_data = "Hello, World!\nThis is test data."

        with patch.object(client, "_client") as mock_s3_client:
            # Mock successful operations
            mock_s3_client.upload_file = MagicMock()
            mock_s3_client.download_file = MagicMock()
            mock_s3_client.head_object = MagicMock(
                return_value={"ContentLength": len(test_data)}
            )

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor = AsyncMock(
                    side_effect=[
                        None,  # upload_file success
                        {"ContentLength": len(test_data)},  # head_object success
                        None,  # download_file success
                    ]
                )

                with patch("os.path.getsize", return_value=len(test_data)):
                    with patch("os.path.exists", return_value=True):
                        # Test upload (mock will handle this, no real file needed)
                        upload_result = await client.upload_file(
                            "/dummy/path/test.txt", "test-key"
                        )
                        assert upload_result is True

                        # Test existence check
                        exists_result = await client.object_exists("test-key")
                        assert exists_result is True

                        # Test download
                        download_path = f"/tmp/test_download_{os.getpid()}.txt"
                        download_result = await client.download_file(
                            "test-key", download_path
                        )
                        assert download_result is True

            # Verify metrics
            metrics = client.get_performance_metrics()
            assert metrics["total_operations"] == 3
            assert abs(metrics["success_rate"] - 1.0) < 0.001
