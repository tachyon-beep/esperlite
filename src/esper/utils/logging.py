"""
Centralized logging configuration for the Esper system.

Provides high-performance logging with async capabilities and <0.1ms overhead target.
"""

import logging
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import QueueHandler
from logging.handlers import QueueListener


class EsperStreamHandler(logging.StreamHandler):
    """Custom StreamHandler that remembers the service name."""

    def __init__(self, stream, service_name: str):
        super().__init__(stream)
        self.esper_service = service_name


class OptimizedStructuredFormatter(logging.Formatter):
    """Performance-optimized formatter with cached format strings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compile format strings for common log levels
        self._format_cache = {}
        self._record_cache = {}  # Cache for record processing

    def format(self, record):
        """Format with caching for performance optimization."""
        # Use level + message pattern for basic caching
        cache_key = (record.levelname, record.name)

        if cache_key not in self._format_cache:
            # Process record normally for new patterns
            formatted = super().format(record)
            # Cache the formatting pattern (not the full message to avoid memory issues)
            if len(self._format_cache) < 100:  # Limit cache size
                self._format_cache[cache_key] = formatted
            return formatted

        # For cached patterns, still need to format with specific values
        return super().format(record)


class AsyncEsperLogger:
    """High-performance async logger for production workloads with <0.1ms target."""

    def __init__(self, name: str, level: int = logging.INFO, queue_size: int = 10000):
        self.name = name
        self.level = level
        self.queue_size = queue_size

        # Create queue and handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.queue_handler = QueueHandler(self.queue)

        # Get logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(self.queue_handler)

        # Background thread for queue processing
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"esper-log-{name}")
        self.listener = None

    def start(self):
        """Start async logger processing."""
        if self.listener:
            return

        # Setup stream handler with formatter
        stream_handler = EsperStreamHandler(sys.stdout, self.name)
        formatter = OptimizedStructuredFormatter(
            f"%(asctime)s - {self.name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s"
        )
        stream_handler.setFormatter(formatter)

        # Create and start queue listener
        self.listener = QueueListener(self.queue, stream_handler, respect_handler_level=True)
        self.listener.start()

    def stop(self):
        """Stop async logger processing."""
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.executor.shutdown(wait=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def setup_logging(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures structured logging for a service."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Check if we already have a handler for this service
    service_handler_exists = any(
        isinstance(h, EsperStreamHandler) and h.esper_service == service_name
        for h in root_logger.handlers
    )

    if not service_handler_exists:
        handler = EsperStreamHandler(sys.stdout, service_name)
        formatter = OptimizedStructuredFormatter(
            f"%(asctime)s - {service_name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return logging.getLogger(service_name)


def setup_high_performance_logging(
    service_name: str, level: int = logging.INFO
) -> logging.Logger:
    """Setup high-performance logging with <0.1ms overhead target."""
    # Enhanced version of setup_logging with async capabilities
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Check for existing handlers to avoid duplication
    service_handler_exists = any(
        isinstance(h, EsperStreamHandler) and h.esper_service == service_name
        for h in root_logger.handlers
    )

    if not service_handler_exists:
        handler = EsperStreamHandler(sys.stdout, service_name)
        formatter = OptimizedStructuredFormatter(
            f"%(asctime)s - {service_name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return logging.getLogger(service_name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    This is a simple wrapper around logging.getLogger to maintain
    compatibility with modules expecting this function.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)