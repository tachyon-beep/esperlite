"""Tests for the logging configuration module."""

import logging
import sys
from unittest.mock import MagicMock, patch

from esper.utils.logging import EsperStreamHandler, setup_logging


class TestSetupLogging:
    """Test the setup_logging function."""

    def setup_method(self):
        """Reset logging state between tests."""
        # Clear root logger handlers to avoid interference between tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.WARNING)  # Reset to default level

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging("test-service")

        assert logger.name == "test-service"
        assert isinstance(logger, logging.Logger)

        # Check root logger configuration
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

        # Find our EsperStreamHandler
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        assert len(esper_handlers) == 1
        assert esper_handlers[0].esper_service == "test-service"

    def test_setup_logging_custom_level(self):
        """Test setup with custom log level."""
        logger = setup_logging("debug-service", level=logging.DEBUG)

        assert logger.name == "debug-service"
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_custom_level_warning(self):
        """Test setup with warning level."""
        logger = setup_logging("warn-service", level=logging.WARNING)

        assert logger.name == "warn-service"
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_custom_level_error(self):
        """Test setup with error level."""
        logger = setup_logging("error-service", level=logging.ERROR)

        assert logger.name == "error-service"
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR

    def test_formatter_format(self):
        """Test that the formatter produces correct format."""
        setup_logging("test-service")
        root_logger = logging.getLogger()

        # Find our EsperStreamHandler
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        assert len(esper_handlers) == 1

        handler = esper_handlers[0]
        assert handler.formatter is not None

        # Test the formatter directly
        log_record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted_message = handler.formatter.format(log_record)
        assert "test-service" in formatted_message
        assert "INFO" in formatted_message
        assert "test.module" in formatted_message
        assert "42" in formatted_message
        assert "Test message" in formatted_message

    def test_no_duplicate_handlers(self):
        """Test that multiple calls don't create duplicate handlers."""
        # First call
        setup_logging("service1")
        root_logger = logging.getLogger()

        # Count our EsperStreamHandlers
        initial_esper_handlers = len(
            [h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)]
        )

        # Second call with same service
        setup_logging("service1")
        final_esper_handlers = len(
            [h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)]
        )

        # Should still have only one handler for service1
        assert initial_esper_handlers == final_esper_handlers == 1

    def test_different_service_names(self):
        """Test different service names create different loggers."""
        logger1 = setup_logging("service-a")
        logger2 = setup_logging("service-b")

        assert logger1.name == "service-a"
        assert logger2.name == "service-b"
        assert logger1 is not logger2

        # Should have two handlers now
        root_logger = logging.getLogger()
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        assert len(esper_handlers) == 2

    def test_same_service_name_returns_same_logger(self):
        """Test that same service name returns the same logger."""
        logger1 = setup_logging("same-service")
        logger2 = setup_logging("same-service")

        assert logger1 is logger2
        assert logger1.name == "same-service"

    def test_logging_output(self, caplog):
        """Test that logging actually produces output."""
        with caplog.at_level(logging.INFO):
            logger = setup_logging("output-service")
            logger.info("Test log message")

            assert "Test log message" in caplog.text

    def test_log_level_filtering(self, caplog):
        """Test that log level filtering works correctly."""
        with caplog.at_level(logging.WARNING):
            logger = setup_logging("filter-service", level=logging.WARNING)

            # This should not appear (below threshold)
            logger.info("Info message")

            # This should appear
            logger.warning("Warning message")

            # Info should not be in output
            assert "Info message" not in caplog.text
            # Warning should be in output
            assert "Warning message" in caplog.text

    def test_root_logger_configuration(self):
        """Test that root logger is properly configured."""
        setup_logging("test-service", level=logging.DEBUG)

        root_logger = logging.getLogger()

        # Check level
        assert root_logger.level == logging.DEBUG

        # Check handler configuration - find our EsperStreamHandler
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        assert len(esper_handlers) == 1

        handler = esper_handlers[0]
        assert handler.stream is sys.stdout
        assert handler.formatter is not None

    def test_service_logger_hierarchy(self):
        """Test that logger hierarchy works correctly."""
        setup_logging("parent-service")

        parent_logger = logging.getLogger("parent-service")
        child_logger = logging.getLogger("parent-service.child")

        assert parent_logger.name == "parent-service"
        assert child_logger.name == "parent-service.child"
        assert child_logger.parent is parent_logger

    def test_child_logger_output(self, caplog):
        """Test that child loggers inherit configuration."""
        with caplog.at_level(logging.INFO):
            setup_logging("parent-service")

            child_logger = logging.getLogger("parent-service.child")
            child_logger.info("Child message")

            assert "Child message" in caplog.text

    def test_multiple_services_same_root(self):
        """Test multiple services with shared root logger."""
        setup_logging("service-x")
        setup_logging("service-y")

        root_logger = logging.getLogger()

        # Should have handlers for both services
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        service_names = {h.esper_service for h in esper_handlers}

        assert len(esper_handlers) == 2
        assert service_names == {"service-x", "service-y"}

    def test_logging_with_special_characters(self, caplog):
        """Test logging with special characters in service name."""
        with caplog.at_level(logging.INFO):
            logger = setup_logging("service-with-dashes_and_underscores.dots")
            logger.info("Special character test")

            assert "Special character test" in caplog.text

    def test_formatter_includes_all_components(self, caplog):
        """Test that formatter includes all required components."""
        with caplog.at_level(logging.INFO):
            setup_logging("test-service")

            # Log from a specific module/line to test formatter
            test_logger = logging.getLogger("test.specific.module")
            test_logger.info("Formatted test message")

            # Check that the message was logged
            assert "Formatted test message" in caplog.text

            # Check that the service is configured
            root_logger = logging.getLogger()
            esper_handlers = [
                h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
            ]
            assert len(esper_handlers) == 1
            assert esper_handlers[0].esper_service == "test-service"

    def test_setup_logging_idempotent(self):
        """Test that multiple calls with same service don't add duplicate handlers."""
        setup_logging("idempotent-service")
        setup_logging("idempotent-service")
        setup_logging("idempotent-service")

        root_logger = logging.getLogger()
        esper_handlers = [
            h
            for h in root_logger.handlers
            if isinstance(h, EsperStreamHandler)
            and h.esper_service == "idempotent-service"
        ]

        # Should only have one handler for this service
        assert len(esper_handlers) == 1

    @patch("logging.StreamHandler")
    def test_handler_configuration(self, mock_stream_handler):
        """Test that StreamHandler is configured correctly."""
        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        setup_logging("test-service")

        # Our code uses EsperStreamHandler, not the mocked StreamHandler
        # So we verify the real behavior
        root_logger = logging.getLogger()
        esper_handlers = [
            h for h in root_logger.handlers if isinstance(h, EsperStreamHandler)
        ]
        assert len(esper_handlers) == 1

        handler = esper_handlers[0]
        assert handler.stream is sys.stdout
        assert handler.formatter is not None
        assert handler.esper_service == "test-service"

    def test_existing_handlers_preserved(self):
        """Test that existing handlers are preserved when setting up logging."""
        root_logger = logging.getLogger()

        # Add a dummy handler
        dummy_handler = logging.NullHandler()
        root_logger.addHandler(dummy_handler)
        initial_handler_count = len(root_logger.handlers)

        setup_logging("preserve-service")

        # Should have the dummy handler plus our new one
        final_handler_count = len(root_logger.handlers)
        assert final_handler_count == initial_handler_count + 1

        # Dummy handler should still be there
        assert dummy_handler in root_logger.handlers
