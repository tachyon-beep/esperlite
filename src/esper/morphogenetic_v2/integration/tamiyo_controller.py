"""
Tamiyo, the Seed Controller - Orchestrates morphogenetic seed lifecycle.

Named after Tamiyo, the Moon Sage, this controller manages the complex
lifecycle of morphogenetic seeds across distributed layers. It provides
high-level orchestration, monitoring, and optimization of seed populations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from ..lifecycle import ExtendedLifecycle
from ..message_bus.clients import MessageBusClient
from ..message_bus.handlers import CommandHandler
from ..message_bus.handlers import CommandHandlerFactory
from ..message_bus.publishers import EventPublisher
from ..message_bus.schemas import AlertSeverity
from ..message_bus.schemas import AlertType
from ..message_bus.schemas import BatchCommand
from ..message_bus.schemas import EmergencyStopCommand
from ..message_bus.schemas import LifecycleTransitionCommand
from ..message_bus.schemas import PerformanceAlert
from .triton_message_bus_layer import TritonMessageBusLayer

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Seed optimization strategies."""
    AGGRESSIVE = "aggressive"  # Maximize performance
    BALANCED = "balanced"      # Balance performance and stability
    CONSERVATIVE = "conservative"  # Prioritize stability
    ADAPTIVE = "adaptive"      # Adapt based on metrics


@dataclass
class SeedPopulationMetrics:
    """Metrics for a population of seeds."""
    total_seeds: int = 0
    active_seeds: int = 0
    dormant_seeds: int = 0
    training_seeds: int = 0
    evaluating_seeds: int = 0
    grafting_seeds: int = 0
    average_performance: float = 0.0
    best_performance: float = 0.0
    worst_performance: float = 0.0
    failure_rate: float = 0.0
    transition_success_rate: float = 0.0


@dataclass
class TamiyoConfig:
    """Configuration for Tamiyo controller."""
    # Monitoring
    monitor_interval_ms: int = 5000
    metrics_window_size: int = 100

    # Optimization
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    auto_optimize: bool = True
    optimization_interval_ms: int = 30000

    # Thresholds
    performance_threshold: float = 0.8
    failure_threshold: float = 0.2
    dormancy_threshold: float = 0.5

    # Batch operations
    max_batch_size: int = 100
    parallel_transitions: int = 10

    # Emergency response
    emergency_failure_rate: float = 0.5
    emergency_response_time_ms: int = 1000


class TamiyoController:
    """
    High-level controller for morphogenetic seed populations.
    
    Provides:
    - Population-level lifecycle management
    - Performance monitoring and optimization
    - Automated seed scheduling
    - Emergency response coordination
    - Cross-layer synchronization
    """

    def __init__(
        self,
        message_bus: MessageBusClient,
        config: Optional[TamiyoConfig] = None
    ):
        """
        Initialize Tamiyo controller.
        
        Args:
            message_bus: Message bus client
            config: Controller configuration
        """
        self.message_bus = message_bus
        self.config = config or TamiyoConfig()

        # Layer registry
        self.layers: Dict[str, TritonMessageBusLayer] = {}

        # Command handler
        self.command_handler: Optional[CommandHandler] = None

        # Event publisher
        self.event_publisher = EventPublisher(message_bus)

        # Monitoring data
        self.population_metrics: Dict[str, SeedPopulationMetrics] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.alert_history: List[PerformanceAlert] = []

        # Optimization state
        self.optimization_queue: asyncio.Queue = asyncio.Queue()
        self.pending_optimizations: Set[str] = set()

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._optimize_task: Optional[asyncio.Task] = None
        self._alert_subscription: Optional[str] = None
        self._running = False

        # Statistics
        self.stats = {
            "layers_managed": 0,
            "total_seeds": 0,
            "transitions_requested": 0,
            "transitions_successful": 0,
            "optimizations_performed": 0,
            "emergencies_handled": 0,
            "alerts_received": 0
        }

    async def start(self):
        """Start the controller."""
        if self._running:
            return

        self._running = True

        # Create command handler
        self.command_handler = CommandHandlerFactory.create(
            self.layers, self.message_bus
        )
        await self.command_handler.start()

        # Subscribe to alerts
        await self._subscribe_to_alerts()

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        if self.config.auto_optimize:
            self._optimize_task = asyncio.create_task(self._optimize_loop())

        logger.info("Tamiyo controller started")

    async def stop(self):
        """Stop the controller."""
        self._running = False

        # Cancel tasks
        for task in [self._monitor_task, self._optimize_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop command handler
        if self.command_handler:
            await self.command_handler.stop()

        # Unsubscribe from alerts
        if self._alert_subscription:
            await self.message_bus.unsubscribe(self._alert_subscription)

        logger.info("Tamiyo controller stopped")

    def register_layer(self, layer: TritonMessageBusLayer):
        """
        Register a layer for management.
        
        Args:
            layer: Layer to manage
        """
        self.layers[layer.layer_id] = layer
        self.population_metrics[layer.layer_id] = SeedPopulationMetrics()
        self.performance_history[layer.layer_id] = []

        self.stats["layers_managed"] = len(self.layers)
        self.stats["total_seeds"] += layer.num_seeds

        logger.info("Registered layer %s with %d seeds",
                   layer.layer_id, layer.num_seeds)

    def unregister_layer(self, layer_id: str):
        """Unregister a layer."""
        if layer_id in self.layers:
            seeds = self.layers[layer_id].num_seeds
            del self.layers[layer_id]
            del self.population_metrics[layer_id]
            del self.performance_history[layer_id]

            self.stats["layers_managed"] = len(self.layers)
            self.stats["total_seeds"] -= seeds

            logger.info("Unregistered layer %s", layer_id)

    async def transition_population(
        self,
        layer_id: str,
        target_state: ExtendedLifecycle,
        selection_criteria: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Transition a population of seeds.
        
        Args:
            layer_id: Target layer
            target_state: Target lifecycle state
            selection_criteria: Criteria for selecting seeds
            batch_size: Number of seeds to transition
            
        Returns:
            Transition results
        """
        if layer_id not in self.layers:
            raise ValueError(f"Unknown layer: {layer_id}")

        layer = self.layers[layer_id]
        seeds_to_transition = self._select_seeds(
            layer, target_state, selection_criteria
        )

        # Limit batch size
        if batch_size:
            seeds_to_transition = seeds_to_transition[:batch_size]
        else:
            seeds_to_transition = seeds_to_transition[:self.config.max_batch_size]

        # Create batch command
        commands = [
            LifecycleTransitionCommand(
                layer_id=layer_id,
                seed_id=seed_id,
                target_state=target_state.name
            )
            for seed_id in seeds_to_transition
        ]

        if not commands:
            return {"success": True, "transitioned": 0}

        batch = BatchCommand(
            commands=commands,
            stop_on_error=False,
            atomic=False
        )

        # Execute
        self.stats["transitions_requested"] += len(commands)
        result = await self.command_handler.handle_command(batch)

        if result.success:
            self.stats["transitions_successful"] += result.details.get("successful", 0)

        return {
            "success": result.success,
            "transitioned": result.details.get("successful", 0),
            "failed": result.details.get("failed", 0),
            "seeds": seeds_to_transition
        }

    def _select_seeds(
        self,
        layer: TritonMessageBusLayer,
        target_state: ExtendedLifecycle,
        criteria: Optional[Dict[str, Any]]
    ) -> List[int]:
        """Select seeds based on criteria."""
        selected = []

        for seed_id in range(layer.num_seeds):
            current_state = layer.get_seed_state(seed_id)

            # Basic state transition validation
            if current_state == target_state:
                continue

            # Apply criteria
            if criteria:
                metrics = layer._get_seed_metrics(seed_id)

                # Performance threshold
                if "min_performance" in criteria:
                    if metrics.get("accuracy", 0) < criteria["min_performance"]:
                        continue

                # State filter
                if "from_states" in criteria:
                    if current_state not in criteria["from_states"]:
                        continue

                # Max failures
                if "max_failures" in criteria:
                    failures = int(layer.extended_state.state_tensor[seed_id, 6])
                    if failures > criteria["max_failures"]:
                        continue

            selected.append(seed_id)

        return selected

    async def optimize_layer(self, layer_id: str) -> Dict[str, Any]:
        """
        Optimize seed population in a layer.
        
        Args:
            layer_id: Layer to optimize
            
        Returns:
            Optimization results
        """
        if layer_id not in self.layers:
            raise ValueError(f"Unknown layer: {layer_id}")

        metrics = self.population_metrics[layer_id]
        strategy = self.config.optimization_strategy

        results = {
            "layer_id": layer_id,
            "strategy": strategy.value,
            "actions": []
        }

        # Dormancy management
        if metrics.dormant_seeds / metrics.total_seeds > self.config.dormancy_threshold:
            # Wake up dormant seeds
            action = await self._wake_dormant_seeds(layer_id, strategy)
            results["actions"].append(action)

        # Performance optimization
        if metrics.average_performance < self.config.performance_threshold:
            # Optimize underperforming seeds
            action = await self._optimize_performance(layer_id, strategy)
            results["actions"].append(action)

        # Failure management
        if metrics.failure_rate > self.config.failure_threshold:
            # Handle failing seeds
            action = await self._handle_failures(layer_id, strategy)
            results["actions"].append(action)

        self.stats["optimizations_performed"] += 1

        return results

    async def _wake_dormant_seeds(
        self,
        layer_id: str,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Wake up dormant seeds based on strategy."""
        # Number to wake based on strategy
        wake_percentages = {
            OptimizationStrategy.AGGRESSIVE: 0.8,
            OptimizationStrategy.BALANCED: 0.5,
            OptimizationStrategy.CONSERVATIVE: 0.2,
            OptimizationStrategy.ADAPTIVE: 0.4
        }

        wake_pct = wake_percentages.get(strategy, 0.5)

        result = await self.transition_population(
            layer_id,
            ExtendedLifecycle.GERMINATED,
            selection_criteria={"from_states": [ExtendedLifecycle.DORMANT]},
            batch_size=int(self.layers[layer_id].num_seeds * wake_pct)
        )

        return {
            "action": "wake_dormant",
            "percentage": wake_pct,
            "transitioned": result["transitioned"]
        }

    async def _optimize_performance(
        self,
        layer_id: str,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Optimize underperforming seeds."""
        # Transition underperformers to training
        criteria = {
            "from_states": [ExtendedLifecycle.EVALUATING, ExtendedLifecycle.STABILIZATION],
            "min_performance": 0.0,
            "max_performance": self.config.performance_threshold
        }

        result = await self.transition_population(
            layer_id,
            ExtendedLifecycle.TRAINING,
            selection_criteria=criteria
        )

        return {
            "action": "retrain_underperformers",
            "threshold": self.config.performance_threshold,
            "transitioned": result["transitioned"]
        }

    async def _handle_failures(
        self,
        layer_id: str,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Handle failing seeds."""
        # Reset high-failure seeds
        if strategy == OptimizationStrategy.AGGRESSIVE:
            max_failures = 3
        elif strategy == OptimizationStrategy.CONSERVATIVE:
            max_failures = 10
        else:
            max_failures = 5

        criteria = {
            "max_failures": max_failures
        }

        result = await self.transition_population(
            layer_id,
            ExtendedLifecycle.DORMANT,
            selection_criteria=criteria
        )

        return {
            "action": "reset_failures",
            "max_failures": max_failures,
            "transitioned": result["transitioned"]
        }

    async def _subscribe_to_alerts(self):
        """Subscribe to performance alerts."""
        self._alert_subscription = await self.message_bus.subscribe(
            "morphogenetic.telemetry.alert.*",
            self._handle_alert
        )

    async def _handle_alert(self, alert: PerformanceAlert):
        """Handle performance alert."""
        self.stats["alerts_received"] += 1
        self.alert_history.append(alert)

        # Keep history limited
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        # Handle critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            await self._handle_critical_alert(alert)

    async def _handle_critical_alert(self, alert: PerformanceAlert):
        """Handle critical performance alert."""
        logger.warning("Critical alert: %s", alert)

        # Check if emergency response needed
        if alert.alert_type == AlertType.SYSTEM_FAILURE:
            await self._emergency_response(alert.layer_id)

    async def _emergency_response(self, layer_id: Optional[str]):
        """Execute emergency response."""
        self.stats["emergencies_handled"] += 1

        if layer_id:
            # Stop specific layer
            cmd = EmergencyStopCommand(
                layer_id=layer_id,
                reason="Critical system failure"
            )
        else:
            # Stop all layers
            cmd = EmergencyStopCommand(
                reason="System-wide emergency"
            )

        await self.command_handler.handle_command(cmd)

        logger.error("Emergency stop executed for %s",
                    layer_id or "all layers")

    async def _monitor_loop(self):
        """Background monitoring task."""
        while self._running:
            try:
                # Update metrics for each layer
                for layer_id, layer in self.layers.items():
                    await self._update_layer_metrics(layer_id, layer)

                # Check for issues
                await self._check_population_health()

                # Wait for next interval
                await asyncio.sleep(self.config.monitor_interval_ms / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitor error: %s", e)
                await asyncio.sleep(1.0)

    async def _update_layer_metrics(self, layer_id: str, layer: TritonMessageBusLayer):
        """Update metrics for a layer."""
        metrics = SeedPopulationMetrics(total_seeds=layer.num_seeds)

        # Count seeds by state
        state_counts = {}
        total_performance = 0.0
        best_performance = 0.0
        worst_performance = 1.0
        failures = 0

        for seed_id in range(layer.num_seeds):
            state = layer.get_seed_state(seed_id)
            state_counts[state] = state_counts.get(state, 0) + 1

            # Get performance
            seed_metrics = layer._get_seed_metrics(seed_id)
            if seed_metrics:
                perf = seed_metrics.get("accuracy", 0.0)
                total_performance += perf
                best_performance = max(best_performance, perf)
                worst_performance = min(worst_performance, perf)

            # Count failures
            error_count = int(layer.extended_state.state_tensor[seed_id, 6])
            if error_count > 0:
                failures += 1

        # Update metrics
        metrics.dormant_seeds = state_counts.get(ExtendedLifecycle.DORMANT, 0)
        metrics.training_seeds = state_counts.get(ExtendedLifecycle.TRAINING, 0)
        metrics.evaluating_seeds = state_counts.get(ExtendedLifecycle.EVALUATING, 0)
        metrics.grafting_seeds = state_counts.get(ExtendedLifecycle.GRAFTING, 0)

        metrics.active_seeds = layer.num_seeds - metrics.dormant_seeds

        if metrics.active_seeds > 0:
            metrics.average_performance = total_performance / metrics.active_seeds
            metrics.best_performance = best_performance
            metrics.worst_performance = worst_performance

        metrics.failure_rate = failures / layer.num_seeds if layer.num_seeds > 0 else 0

        self.population_metrics[layer_id] = metrics

        # Update performance history
        history = self.performance_history[layer_id]
        history.append(metrics.average_performance)

        # Keep history limited
        if len(history) > self.config.metrics_window_size:
            self.performance_history[layer_id] = history[-self.config.metrics_window_size:]

    async def _check_population_health(self):
        """Check overall population health."""
        for layer_id, metrics in self.population_metrics.items():
            # Check failure rate
            if metrics.failure_rate > self.config.emergency_failure_rate:
                logger.error("High failure rate in layer %s: %.2f",
                           layer_id, metrics.failure_rate)
                await self._emergency_response(layer_id)

            # Queue for optimization if needed
            if (metrics.average_performance < self.config.performance_threshold or
                metrics.failure_rate > self.config.failure_threshold):

                if layer_id not in self.pending_optimizations:
                    self.pending_optimizations.add(layer_id)
                    await self.optimization_queue.put(layer_id)

    async def _optimize_loop(self):
        """Background optimization task."""
        while self._running:
            try:
                # Wait for optimization request or timeout
                layer_id = await asyncio.wait_for(
                    self.optimization_queue.get(),
                    timeout=self.config.optimization_interval_ms / 1000
                )

                # Remove from pending
                self.pending_optimizations.discard(layer_id)

                # Optimize
                await self.optimize_layer(layer_id)

            except asyncio.TimeoutError:
                # Periodic optimization check
                for layer_id in self.layers:
                    if layer_id not in self.pending_optimizations:
                        metrics = self.population_metrics.get(layer_id)
                        if metrics and metrics.average_performance < self.config.performance_threshold:
                            await self.optimize_layer(layer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Optimization error: %s", e)
                await asyncio.sleep(1.0)

    def get_population_report(self) -> Dict[str, Any]:
        """Get comprehensive population report."""
        report = {
            "timestamp": time.time(),
            "controller_stats": self.stats.copy(),
            "layers": {}
        }

        for layer_id, metrics in self.population_metrics.items():
            report["layers"][layer_id] = {
                "metrics": {
                    "total_seeds": metrics.total_seeds,
                    "active_seeds": metrics.active_seeds,
                    "dormant_rate": metrics.dormant_seeds / metrics.total_seeds,
                    "training_rate": metrics.training_seeds / metrics.total_seeds,
                    "average_performance": metrics.average_performance,
                    "performance_range": [metrics.worst_performance, metrics.best_performance],
                    "failure_rate": metrics.failure_rate
                },
                "performance_trend": self._calculate_trend(layer_id),
                "health_status": self._assess_health(metrics)
            }

        return report

    def _calculate_trend(self, layer_id: str) -> str:
        """Calculate performance trend."""
        history = self.performance_history.get(layer_id, [])

        if len(history) < 10:
            return "insufficient_data"

        # Simple trend analysis
        recent = sum(history[-5:]) / 5
        older = sum(history[-10:-5]) / 5

        if recent > older * 1.05:
            return "improving"
        elif recent < older * 0.95:
            return "degrading"
        else:
            return "stable"

    def _assess_health(self, metrics: SeedPopulationMetrics) -> str:
        """Assess overall health of population."""
        if metrics.failure_rate > self.config.emergency_failure_rate:
            return "critical"
        elif metrics.failure_rate > self.config.failure_threshold:
            return "unhealthy"
        elif metrics.average_performance < self.config.performance_threshold:
            return "underperforming"
        elif metrics.dormant_seeds / metrics.total_seeds > self.config.dormancy_threshold:
            return "dormant"
        else:
            return "healthy"
