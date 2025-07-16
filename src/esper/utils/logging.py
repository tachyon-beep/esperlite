"""
Centralized logging configuration for the Esper system.
"""

import logging
import sys


def setup_logging(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures structured logging for a service."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        f"%(asctime)s - {service_name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    return logging.getLogger(service_name)
