"""
TritonChunkedKasminaLayer with integrated message bus support.

This layer extends the base TritonChunkedKasminaLayer with:
- Command handling for lifecycle transitions and blueprint updates
- Telemetry publishing for monitoring
- Event notifications for state changes
- Integration with the morphogenetic message bus system
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from ..kasmina.triton_chunked_layer import TritonChunkedKasminaLayer
from ..lifecycle import ExtendedLifecycle
from ..message_bus.clients import MessageBusClient
from ..message_bus.publishers import EventPublisher
from ..message_bus.publishers import TelemetryConfig
from ..message_bus.publishers import TelemetryPublisher
from ..message_bus.schemas import BlueprintUpdateCommand
from ..message_bus.schemas import LifecycleTransitionCommand

logger = logging.getLogger(__name__)


@dataclass
class MessageBusIntegrationConfig:
    """Configuration for message bus integration."""
    enable_telemetry: bool = True
    telemetry_interval_ms: int = 1000
    enable_commands: bool = True
    enable_events: bool = True
    command_timeout_ms: int = 5000
    batch_telemetry: bool = True
    telemetry_batch_size: int = 100


class TritonMessageBusLayer(TritonChunkedKasminaLayer):
    """
    TritonChunkedKasminaLayer with full message bus integration.
    
    Provides:
    - Asynchronous command handling
    - Real-time telemetry publishing
    - Event-driven state notifications
    - Distributed monitoring and control
    """

    def __init__(
        self,
        base_layer: nn.Module,
        chunks_per_layer: int = 1000,
        device: torch.device = torch.device('cuda'),
        checkpoint_dir: Optional[str] = None,
        enable_triton: bool = True,
        message_bus: Optional[MessageBusClient] = None,
        integration_config: Optional[MessageBusIntegrationConfig] = None,
        layer_id: Optional[str] = None
    ):
        """
        Initialize layer with message bus integration.
        
        Args:
            base_layer: Base neural network layer
            chunks_per_layer: Number of morphogenetic seeds
            device: Computation device
            checkpoint_dir: Directory for checkpoints
            enable_triton: Whether to use Triton kernels
            message_bus: Message bus client (optional)
            integration_config: Integration configuration
            layer_id: Unique layer identifier
        """
        super().__init__(
            base_layer=base_layer,
            chunks_per_layer=chunks_per_layer,
            device=device,
            checkpoint_dir=checkpoint_dir,
            enable_triton=enable_triton
        )

        self.layer_id = layer_id or f"triton_layer_{id(self)}"
        self.integration_config = integration_config or MessageBusIntegrationConfig()

        # Message bus components
        self.message_bus = message_bus
        self.telemetry_publisher: Optional[TelemetryPublisher] = None
        self.event_publisher: Optional[EventPublisher] = None

        # Command subscription ID
        self._command_subscription: Optional[str] = None

        # Background tasks
        self._telemetry_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize if message bus provided
        if self.message_bus:
            self._init_message_bus()

    def _init_message_bus(self):
        """Initialize message bus components."""
        if not self.message_bus:
            return

        # Create publishers
        if self.integration_config.enable_telemetry:
            telemetry_config = TelemetryConfig(
                batch_size=self.integration_config.telemetry_batch_size,
                batch_window_ms=self.integration_config.telemetry_interval_ms,
                enable_aggregation=True
            )
            self.telemetry_publisher = TelemetryPublisher(
                self.message_bus, telemetry_config
            )

        if self.integration_config.enable_events:
            self.event_publisher = EventPublisher(self.message_bus)

    async def start_message_bus_integration(self):
        """Start message bus integration."""
        if not self.message_bus:
            logger.warning("No message bus client provided")
            return

        self._running = True

        # Start publishers
        if self.telemetry_publisher:
            await self.telemetry_publisher.start()

        # Subscribe to commands
        if self.integration_config.enable_commands:
            await self._subscribe_to_commands()

        # Start telemetry task
        if self.integration_config.enable_telemetry:
            self._telemetry_task = asyncio.create_task(self._telemetry_loop())

        logger.info("Message bus integration started for layer %s", self.layer_id)

    async def stop_message_bus_integration(self):
        """Stop message bus integration."""
        self._running = False

        # Cancel telemetry task
        if self._telemetry_task:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from commands
        if self._command_subscription and self.message_bus:
            await self.message_bus.unsubscribe(self._command_subscription)

        # Stop publishers
        if self.telemetry_publisher:
            await self.telemetry_publisher.stop()

        logger.info("Message bus integration stopped for layer %s", self.layer_id)

    async def _subscribe_to_commands(self):
        """Subscribe to command topics."""
        # Subscribe to layer-specific commands
        topic_pattern = f"morphogenetic.control.*.layer.{self.layer_id}"

        self._command_subscription = await self.message_bus.subscribe(
            topic_pattern,
            self._handle_command
        )

        logger.info("Subscribed to commands on %s", topic_pattern)

    async def _handle_command(self, message):
        """Handle incoming command."""
        try:
            if isinstance(message, LifecycleTransitionCommand):
                await self._handle_lifecycle_command(message)
            elif isinstance(message, BlueprintUpdateCommand):
                await self._handle_blueprint_command(message)
            else:
                logger.warning("Unknown command type: %s", type(message).__name__)

        except Exception as e:
            logger.error("Error handling command: %s", e)

    async def _handle_lifecycle_command(self, command: LifecycleTransitionCommand):
        """Handle lifecycle transition command."""
        seed_id = command.seed_id

        # Handle all seeds if seed_id is None
        if seed_id is None:
            success_count = 0
            for i in range(self.chunks_per_layer):
                if await self._transition_seed(i, command.target_state,
                                             command.parameters, command.force):
                    success_count += 1

            logger.info("Transitioned %d/%d seeds to %s",
                       success_count, self.chunks_per_layer, command.target_state)
        else:
            # Single seed transition
            success = await self._transition_seed(
                seed_id, command.target_state,
                command.parameters, command.force
            )

            if success:
                logger.info("Transitioned seed %d to %s", seed_id, command.target_state)
            else:
                logger.warning("Failed to transition seed %d to %s",
                             seed_id, command.target_state)

    async def _transition_seed(self, seed_id: int, target_state: str,
                              parameters: Dict[str, Any], force: bool) -> bool:
        """Transition a single seed."""
        try:
            target = ExtendedLifecycle[target_state.upper()]

            # Get current state
            current_state = self.get_seed_state(seed_id)

            # Request transition
            success = self.request_state_transition(
                seed_id, target, parameters
            )

            if success and self.event_publisher:
                # Publish state transition event
                await self.event_publisher.publish_state_transition(
                    layer_id=self.layer_id,
                    seed_id=seed_id,
                    from_state=current_state.name,
                    to_state=target.name,
                    reason="Command request",
                    metrics=self._get_seed_metrics(seed_id)
                )

            return success

        except Exception as e:
            logger.error("Error transitioning seed %d: %s", seed_id, e)
            return False

    async def _handle_blueprint_command(self, command: BlueprintUpdateCommand):
        """Handle blueprint update command."""
        seed_id = command.seed_id

        if seed_id is None:
            logger.warning("Blueprint update requires specific seed_id")
            return

        try:
            # Update blueprint in state tensor
            self.extended_state.update_state(
                seed_id,
                {
                    'blueprint': int(command.blueprint_id),
                    'grafting_strategy': self._get_strategy_id(command.grafting_strategy)
                }
            )

            # Sync Triton state
            if self.enable_triton:
                self._sync_triton_state()

            # Apply configuration if provided
            if command.configuration:
                self._apply_blueprint_config(seed_id, command.configuration)

            logger.info("Updated blueprint for seed %d to %s",
                       seed_id, command.blueprint_id)

        except Exception as e:
            logger.error("Error updating blueprint: %s", e)

    def _get_strategy_id(self, strategy_name: str) -> int:
        """Convert strategy name to ID."""
        strategies = {
            "immediate": 0,
            "gradual": 1,
            "conditional": 2,
            "adaptive": 3,
            "safe": 4
        }
        return strategies.get(strategy_name, 1)  # Default to gradual

    def _apply_blueprint_config(self, seed_id: int, config: Dict[str, Any]):
        """Apply blueprint configuration."""
        # This could update learning rates, regularization, etc.
        # For now, just log
        logger.debug("Applied config to seed %d: %s", seed_id, config)

    async def _telemetry_loop(self):
        """Background task to publish telemetry."""
        while self._running:
            try:
                # Publish layer health
                await self._publish_layer_health()

                # Publish seed metrics
                await self._publish_seed_metrics()

                # Wait for next interval
                await asyncio.sleep(
                    self.integration_config.telemetry_interval_ms / 1000
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Telemetry error: %s", e)
                await asyncio.sleep(1.0)

    async def _publish_layer_health(self):
        """Publish layer health telemetry."""
        if not self.telemetry_publisher:
            return

        # Convert telemetry buffer to health tensor
        # Format: [lifecycle_state, loss, accuracy, compute_time_ms]
        health_tensor = torch.zeros(self.chunks_per_layer, 4, device=self.device)

        for seed_id in range(self.chunks_per_layer):
            state = int(self.extended_state.state_tensor[seed_id, 0])
            metrics = self.telemetry_buffer[seed_id]

            health_tensor[seed_id, 0] = state
            health_tensor[seed_id, 1] = metrics[0]  # loss
            health_tensor[seed_id, 2] = metrics[1]  # accuracy
            health_tensor[seed_id, 3] = metrics[3]  # compute time

        await self.telemetry_publisher.publish_layer_health(
            self.layer_id, health_tensor
        )

    async def _publish_seed_metrics(self):
        """Publish individual seed metrics for active seeds."""
        if not self.telemetry_publisher:
            return

        # Only publish metrics for active seeds
        for seed_id in range(self.chunks_per_layer):
            state = ExtendedLifecycle(int(self.extended_state.state_tensor[seed_id, 0]))

            if state not in [ExtendedLifecycle.DORMANT, ExtendedLifecycle.HARVESTED]:
                metrics = self._get_seed_metrics(seed_id)

                if metrics:
                    await self.telemetry_publisher.publish_seed_metrics(
                        layer_id=self.layer_id,
                        seed_id=seed_id,
                        metrics=metrics,
                        lifecycle_state=state.name,
                        blueprint_id=int(self.extended_state.state_tensor[seed_id, 1])
                    )

    def get_seed_state(self, seed_id: int) -> ExtendedLifecycle:
        """Get current state of a seed."""
        return ExtendedLifecycle(
            int(self.extended_state.state_tensor[seed_id, 0])
        )

    def set_seed_state(self, seed_id: int, state: ExtendedLifecycle):
        """Set seed state (for emergency stop)."""
        self.extended_state.update_state(
            seed_id,
            {'lifecycle': state.value}
        )
        if self.enable_triton:
            self._sync_triton_state()

    async def emergency_stop(self):
        """Emergency stop all seeds."""
        logger.warning("Emergency stop requested for layer %s", self.layer_id)

        # Transition all seeds to DORMANT
        for seed_id in range(self.chunks_per_layer):
            self.set_seed_state(seed_id, ExtendedLifecycle.DORMANT)

        # Publish event
        if self.event_publisher:
            await self.event_publisher.publish_state_transition(
                layer_id=self.layer_id,
                seed_id=None,  # All seeds
                from_state="*",
                to_state="DORMANT",
                reason="Emergency stop"
            )

    async def stop_seed(self, seed_id: int):
        """Stop a specific seed."""
        self.set_seed_state(seed_id, ExtendedLifecycle.DORMANT)

        if self.event_publisher:
            await self.event_publisher.publish_state_transition(
                layer_id=self.layer_id,
                seed_id=seed_id,
                from_state=self.get_seed_state(seed_id).name,
                to_state="DORMANT",
                reason="Seed stop"
            )

    # Make layer compatible with command handler expectations
    @property
    def num_seeds(self) -> int:
        """Number of seeds in layer."""
        return self.chunks_per_layer

    async def transition_seed(self, seed_id: int, target_state: ExtendedLifecycle,
                            parameters: Dict[str, Any], force: bool) -> bool:
        """Async wrapper for state transition."""
        return self.request_state_transition(seed_id, target_state, parameters)

    async def update_seed_blueprint(self, seed_id: int, blueprint_id: str,
                                  strategy: str, config: Dict[str, Any]) -> bool:
        """Update seed blueprint."""
        try:
            self.extended_state.update_state(
                seed_id,
                {
                    'blueprint': int(blueprint_id),
                    'grafting_strategy': self._get_strategy_id(strategy)
                }
            )

            if self.enable_triton:
                self._sync_triton_state()

            return True

        except Exception as e:
            logger.error("Error updating blueprint: %s", e)
            return False

    async def get_seed_metrics(self, seed_id: int) -> Dict[str, float]:
        """Async wrapper for getting seed metrics."""
        return self._get_seed_metrics(seed_id)

    def __repr__(self) -> str:
        return (f"TritonMessageBusLayer(id={self.layer_id}, "
                f"seeds={self.chunks_per_layer}, "
                f"triton={self.enable_triton}, "
                f"bus={'connected' if self.message_bus else 'disconnected'})")
