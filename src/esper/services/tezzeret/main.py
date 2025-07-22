"""
Tezzeret main entrypoint.

This module provides the main entry point for the Tezzeret compilation worker.
"""

import asyncio
import logging
import os
import sys

from esper.services.tezzeret.worker import TezzeretWorker
from esper.utils.config import init_service_config
from esper.utils.logging import setup_logging


async def main():
    """Main entry point for Tezzeret worker."""
    try:
        # Initialize and validate configuration
        config = init_service_config()

        # Setup logging with configuration
        setup_logging("tezzeret", getattr(logging, config.log_level.upper()))
        logger = logging.getLogger(__name__)

        logger.info("Starting Tezzeret compilation worker")
        logger.info("Configuration: %s", config.to_dict())

        # Get worker ID from environment or generate one
        worker_id = os.getenv("TEZZERET_WORKER_ID", "tezzeret-worker-01")

        # Create and start worker with validated configuration
        worker = TezzeretWorker(worker_id=worker_id, config=config)
        await worker.start_polling()

    except Exception as e:
        logger.error("Failed to start Tezzeret worker: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
