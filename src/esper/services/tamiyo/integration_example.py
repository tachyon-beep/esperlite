"""
Production Integration Example for Autonomous Tamiyo Service

This module demonstrates how to integrate and deploy the complete autonomous
Tamiyo Strategic Controller in a production environment with all Phase 2
components working together.

This example shows:
- Complete service initialization with production configuration
- Integration with Oona message bus and health signal collection
- Real-time autonomous decision making with safety validation
- Continuous policy learning with multi-metric rewards
- Production monitoring and alerting systems
"""

import asyncio
import logging
from typing import Any
from typing import Dict

from esper.services.oona_client import OonaClient

from .autonomous_service import AutonomousServiceConfig
from .autonomous_service import AutonomousTamiyoService
from .policy import PolicyConfig
from .policy_trainer import ProductionTrainingConfig
from .reward_system import RewardConfig

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTamiyoDeployment:
    """
    Production deployment wrapper for the Autonomous Tamiyo Service.
    
    This class provides a complete production-ready deployment with:
    - Configuration management
    - Service lifecycle management
    - Health monitoring and alerting
    - Performance metrics collection
    - Graceful shutdown handling
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize production deployment with Redis connection."""
        # Initialize Oona client for message bus communication
        self.oona_client = OonaClient(redis_url=redis_url)

        # Production service configuration
        self.service_config = AutonomousServiceConfig(
            # Conservative production settings
            decision_interval_ms=100,  # 100ms decision cycles
            max_decisions_per_minute=3,  # Conservative adaptation rate
            min_confidence_threshold=0.75,  # High confidence requirement
            safety_cooldown_seconds=60.0,  # 1 minute cooldown between adaptations
            enable_real_time_learning=True,
            enable_safety_validation=True,
            enable_correlation_analysis=True
        )

        # Enhanced policy configuration for production
        self.policy_config = PolicyConfig(
            # Advanced GNN architecture
            hidden_dim=128,
            num_gnn_layers=4,
            num_attention_heads=4,
            attention_dropout=0.1,

            # Enhanced uncertainty quantification
            enable_uncertainty=True,
            uncertainty_samples=10,
            epistemic_weight=0.1,

            # Conservative safety thresholds
            health_threshold=0.4,
            adaptation_confidence_threshold=0.75,
            uncertainty_threshold=0.15,
            max_adaptations_per_epoch=2,
            safety_margin=0.15
        )

        # Production training configuration
        self.training_config = ProductionTrainingConfig(
            # Robust learning parameters
            learning_rate=2e-4,  # Conservative learning rate
            batch_size=64,
            num_training_steps=200000,
            gradient_clip_norm=0.5,

            # PPO parameters optimized for safety
            ppo_epochs=4,
            clip_ratio=0.15,  # Conservative clipping
            value_loss_coeff=0.5,
            entropy_bonus_coeff=0.01,
            gae_lambda=0.95,

            # Enhanced safety regularization
            safety_loss_weight=2.0,  # Strong safety penalty
            uncertainty_regularization=0.15,
            safety_penalty_weight=3.0,

            # Production stability
            min_buffer_size=1000,
            warmup_steps=10000,
            early_stopping_patience=15,

            # Checkpointing and monitoring
            checkpoint_interval=3600,  # Hourly checkpoints
            log_interval=100,
            tensorboard_logging=True
        )

        # Multi-metric reward configuration
        self.reward_config = RewardConfig(
            # Balanced component weights for production
            accuracy_weight=0.25,
            speed_weight=0.20,
            memory_weight=0.15,
            stability_weight=0.25,  # Higher stability weight
            safety_weight=0.20,     # Higher safety weight
            innovation_weight=0.05,

            # Conservative temporal discounting
            immediate_discount=1.0,
            short_term_discount=0.95,
            medium_term_discount=0.85,
            long_term_discount=0.75,

            # Strict safety thresholds
            safety_failure_penalty=-3.0,  # Harsh safety penalty
            stability_failure_penalty=-2.0,

            # Advanced correlation analysis
            correlation_window_size=200,
            enable_adaptive_weights=True,
            weight_adaptation_rate=0.005,  # Slow adaptation
            weight_momentum=0.95
        )

        # Initialize the autonomous service
        self.tamiyo_service = AutonomousTamiyoService(
            oona_client=self.oona_client,
            service_config=self.service_config,
            policy_config=self.policy_config,
            training_config=self.training_config,
            reward_config=self.reward_config
        )

        logger.info("🏭 Production Tamiyo deployment initialized")

    async def deploy(self) -> None:
        """Deploy the autonomous Tamiyo service in production mode."""
        logger.info("🚀 Deploying Autonomous Tamiyo Service in production mode...")

        try:
            # Start monitoring task
            monitoring_task = asyncio.create_task(self._production_monitoring())

            # Start the main service
            service_task = asyncio.create_task(self.tamiyo_service.start())

            # Wait for both tasks
            await asyncio.gather(monitoring_task, service_task)

        except KeyboardInterrupt:
            logger.info("🛑 Received shutdown signal, stopping service...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"❌ Production deployment error: {e}")
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Graceful shutdown of the production deployment."""
        logger.info("🔄 Initiating graceful shutdown...")

        try:
            # Stop the autonomous service
            await self.tamiyo_service.stop()

            # Close Oona client connection
            # await self.oona_client.close()

            logger.info("✅ Production deployment shutdown completed")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")

    async def _production_monitoring(self) -> None:
        """Production monitoring and alerting system."""
        logger.info("📊 Starting production monitoring system...")

        monitoring_interval = 30  # 30 seconds
        alert_thresholds = {
            'max_decision_latency_ms': 500,
            'min_health_processing_rate': 500,
            'max_safety_rejection_rate': 0.7,
            'min_success_rate': 0.6,
            'min_average_health': 0.4
        }

        while True:
            try:
                # Get comprehensive service status
                status = self.tamiyo_service.get_comprehensive_status()

                # Check alert conditions
                alerts = self._check_alert_conditions(status, alert_thresholds)

                if alerts:
                    for alert in alerts:
                        logger.warning(f"🚨 PRODUCTION ALERT: {alert}")

                # Log key metrics
                stats = status.get('statistics', {})
                decisions = stats.get('decisions', {})
                performance = status.get('performance_metrics', {})

                logger.info(
                    f"📈 Production Metrics - "
                    f"Decisions: {decisions.get('total', 0)}, "
                    f"Success Rate: {decisions.get('success_rate', 0.0):.1%}, "
                    f"Latency: {performance.get('decision_latency_ms', 0.0):.1f}ms, "
                    f"Health Rate: {performance.get('health_processing_rate', 0.0):.0f} Hz"
                )

                await asyncio.sleep(monitoring_interval)

            except Exception as e:
                logger.error(f"❌ Error in production monitoring: {e}")
                await asyncio.sleep(monitoring_interval)

    def _check_alert_conditions(
        self,
        status: Dict[str, Any],
        thresholds: Dict[str, float]
    ) -> list[str]:
        """Check for alert conditions in service status."""
        alerts = []

        try:
            performance = status.get('performance_metrics', {})
            statistics = status.get('statistics', {})
            decisions = statistics.get('decisions', {})
            health = statistics.get('health_monitoring', {})

            # Decision latency alert
            decision_latency = performance.get('decision_latency_ms', 0.0)
            if decision_latency > thresholds['max_decision_latency_ms']:
                alerts.append(f"High decision latency: {decision_latency:.1f}ms")

            # Health processing rate alert
            processing_rate = performance.get('health_processing_rate', 0.0)
            if processing_rate < thresholds['min_health_processing_rate']:
                alerts.append(f"Low health processing rate: {processing_rate:.0f} Hz")

            # Safety rejection rate alert
            total_decisions = decisions.get('total', 1)
            safety_rejections = decisions.get('safety_rejections', 0)
            safety_rejection_rate = safety_rejections / max(total_decisions, 1)

            if safety_rejection_rate > thresholds['max_safety_rejection_rate']:
                alerts.append(f"High safety rejection rate: {safety_rejection_rate:.1%}")

            # Success rate alert
            success_rate = decisions.get('success_rate', 0.0)
            if success_rate < thresholds['min_success_rate'] and total_decisions > 10:
                alerts.append(f"Low adaptation success rate: {success_rate:.1%}")

            # Average health alert
            avg_health = health.get('average_health', 1.0)
            if avg_health < thresholds['min_average_health']:
                alerts.append(f"Low system health: {avg_health:.3f}")

        except Exception as e:
            alerts.append(f"Error checking alert conditions: {e}")

        return alerts

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status for external monitoring."""
        try:
            service_status = self.tamiyo_service.get_comprehensive_status()

            return {
                'deployment': {
                    'mode': 'production',
                    'configuration': {
                        'decision_interval_ms': self.service_config.decision_interval_ms,
                        'max_adaptations_per_minute': self.service_config.max_decisions_per_minute,
                        'confidence_threshold': self.service_config.min_confidence_threshold,
                        'safety_enabled': self.service_config.enable_safety_validation,
                        'learning_enabled': self.service_config.enable_real_time_learning
                    }
                },
                'service_status': service_status,
                'health_trends': self.tamiyo_service.get_health_trends(),
                'reward_analysis': self.tamiyo_service.get_reward_analysis(),
                'training_progress': self.tamiyo_service.get_training_progress()
            }

        except Exception as e:
            return {'error': f'Failed to get deployment status: {e}'}


async def main():
    """Main entry point for production deployment."""
    logger.info("🏭 Starting Autonomous Tamiyo Production Deployment")

    # Initialize production deployment
    deployment = ProductionTamiyoDeployment(
        redis_url="redis://localhost:6379"
    )

    try:
        # Deploy and run the service
        await deployment.deploy()

    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
    finally:
        await deployment.shutdown()


# Integration examples for different deployment scenarios

async def basic_autonomous_example():
    """Basic autonomous service example with minimal configuration."""
    logger.info("🔧 Basic Autonomous Tamiyo Example")

    # Simple setup
    oona_client = OonaClient("redis://localhost:6379")

    tamiyo_service = AutonomousTamiyoService(
        oona_client=oona_client,
        service_config=AutonomousServiceConfig(
            decision_interval_ms=200,  # Slower for demo
            max_decisions_per_minute=2,
            min_confidence_threshold=0.8
        )
    )

    try:
        # Start service
        service_task = asyncio.create_task(tamiyo_service.start())

        # Run for demonstration period
        await asyncio.sleep(30)  # 30 seconds

        # Get status
        status = tamiyo_service.get_comprehensive_status()
        logger.info(f"📊 Final Status: {status}")

    finally:
        await tamiyo_service.stop()


async def monitoring_integration_example():
    """Example showing integration with external monitoring systems."""
    logger.info("📊 Monitoring Integration Example")

    oona_client = OonaClient("redis://localhost:6379")
    tamiyo_service = AutonomousTamiyoService(oona_client=oona_client)

    async def external_monitoring():
        """Simulate external monitoring system integration."""
        while True:
            try:
                # Get comprehensive status
                status = tamiyo_service.get_comprehensive_status()

                # Example: Send to Prometheus, DataDog, etc.
                logger.info(f"📈 Metrics for external system: {status}")

                # Get reward analysis
                reward_analysis = tamiyo_service.get_reward_analysis()
                logger.info(f"🎯 Reward Analysis: {reward_analysis}")

                await asyncio.sleep(10)  # Every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break

    try:
        # Run service with monitoring
        await asyncio.gather(
            tamiyo_service.start(),
            external_monitoring()
        )
    finally:
        await tamiyo_service.stop()


if __name__ == "__main__":
    # Run the production deployment
    asyncio.run(main())
