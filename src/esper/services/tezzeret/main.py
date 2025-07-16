"""
Tezzeret main entrypoint.

This module provides the main entry point for the Tezzeret compilation worker.
"""

import logging
import os

from esper.services.tezzeret.worker import TezzeretWorker


def main():
    """Main entry point for Tezzeret worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get worker ID from environment or generate one
    worker_id = os.getenv("TEZZERET_WORKER_ID", "tezzeret-worker-01")

    # Create and start worker
    worker = TezzeretWorker(worker_id=worker_id)
    worker.start_polling()


if __name__ == "__main__":
    main()
