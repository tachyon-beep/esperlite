"""
Tolaria Training Service - Main orchestration service.

This module provides the main service interface for the Tolaria training
orchestrator, including health checking, configuration management, and
graceful startup/shutdown.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from esper.services.tolaria.config import TolariaConfig
from esper.services.tolaria.trainer import TolariaTrainer
from esper.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class TolariaService:
    """
    Main Tolaria training service orchestrator.

    This service manages the complete training lifecycle including:
    - Configuration loading and validation
    - Trainer initialization and management
    - Health monitoring and status reporting
    - Graceful shutdown handling
    """

    def __init__(self, config: TolariaConfig):
        """Initialize the Tolaria service."""
        self.config = config
        self.trainer: Optional[TolariaTrainer] = None
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(
            "Tolaria service initialized with config: %s", config.model.architecture
        )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame) -> None:
            logger.info("Received shutdown signal %d", signum)
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self) -> None:
        """Start the Tolaria training service."""
        logger.info("Starting Tolaria training service...")

        try:
            # Initialize trainer
            self.trainer = TolariaTrainer(self.config)
            await self.trainer.initialize()

            # Mark service as running
            self.running = True

            # Start training
            logger.info("Beginning training process...")
            metrics_history = await self.trainer.train()

            # Log training completion
            if metrics_history:
                final_metrics = metrics_history[-1]
                logger.info(
                    "Training completed successfully. Final metrics: "
                    "train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
                    final_metrics.train_loss,
                    final_metrics.train_accuracy,
                    final_metrics.val_loss,
                    final_metrics.val_accuracy,
                )

        except Exception as e:
            logger.error("Training failed: %s", e)
            raise
        finally:
            await self.shutdown()

    def health_check(self) -> dict:
        """Perform health check and return status."""
        status = {
            "service": "tolaria",
            "status": "healthy" if self.running else "stopped",
            "trainer_running": self.trainer.running if self.trainer else False,
            "config_valid": True,
        }

        if self.trainer:
            training_state = self.trainer.get_training_state()
            status.update(
                {
                    "current_epoch": training_state.epoch,
                    "total_adaptations": training_state.total_adaptations,
                    "best_val_accuracy": training_state.best_val_accuracy,
                }
            )

        return status

    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Tolaria service...")

        self.running = False

        if self.trainer:
            await self.trainer.shutdown()

        logger.info("Tolaria service shutdown complete")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


async def run_service(config_path: str) -> None:
    """Run the Tolaria service with the given configuration."""
    # Setup logging
    setup_logging("tolaria", level=logging.INFO)

    logger.info("Loading Tolaria configuration from: %s", config_path)

    try:
        # Load configuration
        config = TolariaConfig.from_yaml(Path(config_path))

        # Create and start service
        service = TolariaService(config)

        # Create task for the main service
        service_task = asyncio.create_task(service.start())

        # Create task for shutdown monitoring
        shutdown_task = asyncio.create_task(service.wait_for_shutdown())

        # Wait for either completion or shutdown
        done, pending = await asyncio.wait(
            [service_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Task was cancelled, this is expected
                logger.debug("Task cancelled during shutdown")
                raise

        # If service task completed, check for exceptions
        if service_task in done:
            try:
                await service_task
            except Exception as e:
                logger.error("Service failed: %s", e)
                sys.exit(1)

        logger.info("Tolaria service exited gracefully")

    except Exception as e:
        logger.error("Failed to start Tolaria service: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tolaria Training Service")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Tolaria configuration file"
    )

    args = parser.parse_args()

    # Run the service
    asyncio.run(run_service(args.config))
