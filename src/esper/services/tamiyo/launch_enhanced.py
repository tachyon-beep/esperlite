#!/usr/bin/env python3
"""
Launch Enhanced Tamiyo Service with REMEDIATION A1 Integration

This script provides a production-ready launcher for the enhanced Tamiyo service
that includes all components from REMEDIATION ACTIVITY A1:
- Blueprint library with 18 templates
- Multi-metric intelligent reward system
- Phase 1-2 seamless integration
- Production health signal collection
"""

import asyncio
import logging
import os
from typing import Optional

from esper.services.oona_client import OonaClient
from esper.services.tamiyo import EnhancedTamiyoService
from esper.services.tamiyo import PolicyConfig
from esper.utils.config import ServiceConfig

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TamiyoLauncher:
    """Production launcher for Enhanced Tamiyo Service."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        urza_url: Optional[str] = None,
        enable_learning: bool = True,
    ):
        """Initialize the launcher with service URLs."""
        # Service configuration
        self.service_config = ServiceConfig(
            name="tamiyo",
            port=8082,  # Tamiyo standard port
            redis_url=redis_url or os.getenv("REDIS_URL", "redis://localhost:6379"),
        )

        # Set Urza service URL
        if urza_url:
            self.service_config.service_urls["urza"] = urza_url
        else:
            self.service_config.service_urls["urza"] = os.getenv(
                "URZA_URL", "http://localhost:8080"
            )

        # Initialize Oona client
        self.oona_client = OonaClient(self.service_config.get_redis_url())

        # Policy configuration with production settings
        self.policy_config = PolicyConfig(
            # Enhanced GNN architecture
            hidden_dim=128,
            num_gnn_layers=4,
            num_attention_heads=4,
            attention_dropout=0.1,
            # Safety settings
            enable_uncertainty=True,
            uncertainty_samples=10,
            health_threshold=0.3,
            adaptation_confidence_threshold=0.7,
            uncertainty_threshold=0.3,
            max_adaptations_per_epoch=3,
            safety_margin=0.2,
        )

        # Initialize enhanced service
        self.tamiyo_service = EnhancedTamiyoService(
            service_config=self.service_config,
            oona_client=self.oona_client,
            policy_config=self.policy_config,
            enable_learning=enable_learning,
        )

        logger.info(
            "🚀 Enhanced Tamiyo Service initialized with REMEDIATION A1 components"
        )

    async def start(self):
        """Start the enhanced Tamiyo service."""
        logger.info("Starting Enhanced Tamiyo Service...")
        logger.info(f"  - Redis URL: {self.service_config.get_redis_url()}")
        logger.info(f"  - Urza URL: {self.service_config.get_service_url('urza')}")
        logger.info(f"  - Learning enabled: {self.tamiyo_service.enable_learning}")
        logger.info(
            f"  - Blueprints loaded: "
            f"{len(self.tamiyo_service.blueprint_registry.blueprints)}"
        )

        try:
            # Start the service
            await self.tamiyo_service.start()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Service error: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the service."""
        logger.info("Shutting down Enhanced Tamiyo Service...")
        try:
            await self.tamiyo_service.stop()
            logger.info("✅ Service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_status(self):
        """Get comprehensive service status."""
        return self.tamiyo_service.get_status()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch Enhanced Tamiyo Service with REMEDIATION A1"
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL for Oona message bus (default: redis://localhost:6379)"
    )
    parser.add_argument(
        "--urza-url",
        default=None,
        help="Urza asset service URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--no-learning",
        action="store_true",
        help="Disable real-time policy learning"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and start launcher
    launcher = TamiyoLauncher(
        redis_url=args.redis_url,
        urza_url=args.urza_url,
        enable_learning=not args.no_learning,
    )

    # Print startup banner
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                 Enhanced Tamiyo Strategic Controller              ║
    ║                    with REMEDIATION ACTIVITY A1                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  Components:                                                      ║
    ║  ✓ Blueprint Library: 18 production-ready templates              ║
    ║  ✓ Intelligent Reward: Multi-metric with temporal analysis       ║
    ║  ✓ Phase 1-2 Bridge: Seamless kernel integration                ║
    ║  ✓ Health Collection: Real-time signal processing               ║
    ║  ✓ GNN Policy: Advanced graph neural network decisions          ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Start the service
    await launcher.start()


if __name__ == "__main__":
    asyncio.run(main())
