"""
Centralized logging configuration for the Esper system.
"""

import logging
import sys


class EsperStreamHandler(logging.StreamHandler):
    """Custom StreamHandler that remembers the service name."""

    def __init__(self, stream, service_name: str):
        super().__init__(stream)
        self.esper_service = service_name


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
        formatter = logging.Formatter(
            f"%(asctime)s - {service_name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return logging.getLogger(service_name)
