"""Advanced grafting strategies for morphogenetic seeds.

Provides sophisticated blending strategies for integrating
blueprints into the main network with various control mechanisms.
"""

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GraftingConfig:
    """Configuration for grafting strategies."""
    ramp_duration: int = 50  # Epochs for full integration
    drift_threshold: float = 0.01  # Weight drift threshold
    pause_duration: int = 10  # Pause epochs on drift
    momentum_factor: float = 0.1  # Momentum acceleration
    stability_window: int = 5  # Epochs for stability check
    min_alpha: float = 0.0  # Minimum blending factor
    max_alpha: float = 1.0  # Maximum blending factor


@dataclass
class GraftingContext:
    """Runtime context for grafting decisions."""
    seed_id: int
    current_epoch: int
    total_epochs: int
    current_alpha: float
    metrics: Dict[str, float]
    model_weights: Optional[torch.Tensor] = None
    blueprint_weights: Optional[torch.Tensor] = None


class GraftingStrategyBase(ABC):
    """Base class for all grafting strategies."""

    def __init__(self, config: GraftingConfig):
        """Initialize grafting strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.reset()

    @abstractmethod
    def compute_alpha(
        self,
        context: GraftingContext
    ) -> float:
        """Compute blending alpha for current epoch.

        Args:
            context: Current grafting context

        Returns:
            Alpha value in [0, 1]
        """

    @abstractmethod
    def reset(self):
        """Reset strategy state."""
        pass

    def should_abort(self, context: GraftingContext) -> bool:
        """Check if grafting should be aborted.

        Args:
            context: Current context

        Returns:
            True if grafting should stop
        """
        # Default: abort on excessive errors
        error_rate = context.metrics.get('error_rate', 0.0)
        return error_rate > 0.5

    def on_epoch_end(self, context: GraftingContext):
        """Hook called at end of each epoch.

        Args:
            context: Current context
        """


class LinearGrafting(GraftingStrategyBase):
    """Simple linear ramping strategy."""

    def compute_alpha(self, context: GraftingContext) -> float:
        """Linear ramp from 0 to 1."""
        progress = context.current_epoch / self.config.ramp_duration
        return np.clip(progress, self.config.min_alpha, self.config.max_alpha)

    def reset(self):
        """No state to reset."""
        pass


class DriftControlledGrafting(GraftingStrategyBase):
    """Pauses grafting if weight drift exceeds threshold.

    Monitors the rate of weight change in the model and pauses
    integration if changes are too rapid, allowing the network
    to stabilize before continuing.
    """

    def __init__(self, config: GraftingConfig):
        """Initialize drift-controlled strategy."""
        self.pause_counter = 0
        self.weight_history = []
        self.weight_ema = None
        super().__init__(config)

    def compute_alpha(self, context: GraftingContext) -> float:
        """Compute alpha with drift control."""
        # Check if we're in pause mode
        if self.pause_counter > 0:
            self.pause_counter -= 1
            logger.debug(
                "Grafting paused for seed %d (%d epochs remaining)",
                context.seed_id, self.pause_counter
            )
            return context.current_alpha  # Hold current value

        # Calculate drift if weights provided
        if context.model_weights is not None:
            drift = self._calculate_drift(context.model_weights)

            if drift > self.config.drift_threshold:
                logger.warning(
                    "Weight drift %.4f exceeds threshold %.4f for seed %d",
                    drift, self.config.drift_threshold, context.seed_id
                )
                self.pause_counter = self.config.pause_duration
                return context.current_alpha

        # Normal linear progression
        progress = context.current_epoch / self.config.ramp_duration
        return np.clip(progress, self.config.min_alpha, self.config.max_alpha)

    def _calculate_drift(self, weights: torch.Tensor) -> float:
        """Calculate weight drift using EMA.

        Args:
            weights: Current model weights

        Returns:
            Normalized drift value
        """
        if self.weight_ema is None:
            self.weight_ema = weights.clone()
            return 0.0

        # Calculate drift
        with torch.no_grad():
            drift = (weights - self.weight_ema).norm() / self.weight_ema.norm()

            # Update EMA
            self.weight_ema = 0.9 * self.weight_ema + 0.1 * weights

        return drift.item()

    def reset(self):
        """Reset strategy state."""
        self.pause_counter = 0
        self.weight_history.clear()
        self.weight_ema = None

    def on_epoch_end(self, context: GraftingContext):
        """Track weight history."""
        if context.model_weights is not None:
            self.weight_history.append(context.model_weights.clone())

            # Keep only recent history
            if len(self.weight_history) > self.config.stability_window:
                self.weight_history.pop(0)


class MomentumGrafting(GraftingStrategyBase):
    """Accelerates grafting based on positive performance trends.

    Monitors performance metrics and speeds up integration when
    improvements are consistent, while slowing down on degradation.
    """

    def __init__(self, config: GraftingConfig):
        """Initialize momentum strategy."""
        self.velocity = 0.0
        self.performance_history = []
        super().__init__(config)

    def compute_alpha(self, context: GraftingContext) -> float:
        """Compute alpha with momentum adjustment."""
        # Get performance delta
        performance_delta = context.metrics.get('performance_delta', 0.0)

        # Update velocity based on performance
        self.velocity = (
            self.config.momentum_factor * self.velocity +
            performance_delta * 0.01  # Scale factor
        )

        # Base linear progression
        base_progress = context.current_epoch / self.config.ramp_duration

        # Apply momentum adjustment
        adjusted_progress = base_progress + self.velocity

        # Ensure we don't go backwards or exceed limits
        adjusted_alpha = np.clip(
            adjusted_progress,
            max(self.config.min_alpha, context.current_alpha),
            self.config.max_alpha
        )

        logger.debug(
            "Momentum grafting for seed %d: base=%.3f, velocity=%.3f, alpha=%.3f",
            context.seed_id, base_progress, self.velocity, adjusted_alpha
        )

        return adjusted_alpha

    def reset(self):
        """Reset strategy state."""
        self.velocity = 0.0
        self.performance_history.clear()

    def on_epoch_end(self, context: GraftingContext):
        """Track performance history."""
        performance = context.metrics.get('performance_score', 0.0)
        self.performance_history.append(performance)

        # Keep only recent history
        if len(self.performance_history) > self.config.stability_window:
            self.performance_history.pop(0)


class AdaptiveGrafting(GraftingStrategyBase):
    """Combines multiple strategies with adaptive switching.

    Monitors grafting progress and switches between strategies
    based on current conditions and performance.
    """

    def __init__(self, config: GraftingConfig):
        """Initialize adaptive strategy."""
        # Initialize sub-strategies
        self.strategies = {
            'linear': LinearGrafting(config),
            'drift': DriftControlledGrafting(config),
            'momentum': MomentumGrafting(config)
        }

        self.current_strategy = 'linear'
        self.strategy_scores = {
            'linear': 0.0,
            'drift': 0.0,
            'momentum': 0.0
        }
        super().__init__(config)

    def compute_alpha(self, context: GraftingContext) -> float:
        """Compute alpha using best strategy."""
        # Select strategy based on conditions
        self._select_strategy(context)

        # Use selected strategy
        strategy = self.strategies[self.current_strategy]
        alpha = strategy.compute_alpha(context)

        # Update strategy scores
        self._update_scores(context)

        return alpha

    def _select_strategy(self, context: GraftingContext):
        """Select best strategy for current conditions."""
        # High weight drift -> use drift control
        if context.model_weights is not None:
            drift_strategy = self.strategies['drift']
            drift = drift_strategy._calculate_drift(context.model_weights)

            if drift > self.config.drift_threshold * 0.5:
                self.current_strategy = 'drift'
                return

        # Consistent performance improvement -> use momentum
        performance_delta = context.metrics.get('performance_delta', 0.0)
        if performance_delta > 0.01:
            self.current_strategy = 'momentum'
            return

        # Default to linear
        self.current_strategy = 'linear'

    def _update_scores(self, context: GraftingContext):
        """Update strategy effectiveness scores."""
        # Simple scoring based on performance
        performance = context.metrics.get('performance_score', 0.0)

        # Reward current strategy if performance improves
        if performance > 0:
            self.strategy_scores[self.current_strategy] += 0.1
        else:
            self.strategy_scores[self.current_strategy] -= 0.05

        # Normalize scores
        total = sum(self.strategy_scores.values())
        if total > 0:
            for key in self.strategy_scores:
                self.strategy_scores[key] /= total

    def reset(self):
        """Reset all sub-strategies."""
        for strategy in self.strategies.values():
            strategy.reset()

        self.current_strategy = 'linear'
        self.strategy_scores = {
            'linear': 0.33,
            'drift': 0.33,
            'momentum': 0.34
        }

    def on_epoch_end(self, context: GraftingContext):
        """Forward to current strategy."""
        self.strategies[self.current_strategy].on_epoch_end(context)


class StabilityGrafting(GraftingStrategyBase):
    """Grafting with stability checkpoints.

    Implements periodic stability checks and can slow down
    or reverse grafting if instability is detected.
    """

    def __init__(self, config: GraftingConfig):
        """Initialize stability strategy."""
        self.stability_checkpoints = []
        self.last_stable_alpha = 0.0
        self.instability_counter = 0
        super().__init__(config)

    def compute_alpha(self, context: GraftingContext) -> float:
        """Compute alpha with stability checks."""
        # Check stability every N epochs
        if context.current_epoch % self.config.stability_window == 0:
            is_stable = self._check_stability(context)

            if is_stable:
                self.last_stable_alpha = context.current_alpha
                self.instability_counter = 0
            else:
                self.instability_counter += 1

                if self.instability_counter >= 3:
                    # Revert to last stable point
                    logger.warning(
                        "Reverting seed %d to last stable alpha %.3f",
                        context.seed_id, self.last_stable_alpha
                    )
                    return self.last_stable_alpha

        # Normal progression with potential slowdown
        progress = context.current_epoch / self.config.ramp_duration

        # Slow down if we've seen instability
        if self.instability_counter > 0:
            progress *= 0.5  # Half speed

        return np.clip(progress, self.config.min_alpha, self.config.max_alpha)

    def _check_stability(self, context: GraftingContext) -> bool:
        """Check if current state is stable.

        Args:
            context: Current context

        Returns:
            True if stable
        """
        # Multiple stability criteria
        loss_stable = context.metrics.get('loss_variance', 0.0) < 0.1
        gradient_stable = context.metrics.get('gradient_norm', 1.0) < 10.0
        accuracy_stable = context.metrics.get('accuracy_variance', 0.0) < 0.05

        return loss_stable and gradient_stable and accuracy_stable

    def reset(self):
        """Reset strategy state."""
        self.stability_checkpoints.clear()
        self.last_stable_alpha = 0.0
        self.instability_counter = 0


# Registry of available strategies
GRAFTING_STRATEGIES = {
    'linear': LinearGrafting,
    'drift_controlled': DriftControlledGrafting,
    'momentum': MomentumGrafting,
    'adaptive': AdaptiveGrafting,
    'stability': StabilityGrafting
}


class GraftingStrategyFactory:
    """Factory for creating grafting strategies."""

    @staticmethod
    def create(
        strategy_name: str,
        config: Optional[GraftingConfig] = None
    ) -> GraftingStrategyBase:
        """Create a grafting strategy.

        Args:
            strategy_name: Name of strategy to create
            config: Optional configuration

        Returns:
            Configured strategy instance
        """
        if strategy_name not in GRAFTING_STRATEGIES:
            raise ValueError(
                f"Unknown grafting strategy: {strategy_name}. "
                f"Available: {list(GRAFTING_STRATEGIES.keys())}"
            )

        config = config or GraftingConfig()
        strategy_class = GRAFTING_STRATEGIES[strategy_name]
        return strategy_class(config)


def create_grafting_strategy(
    strategy_name: str,
    config: Optional[GraftingConfig] = None
) -> GraftingStrategyBase:
    """Factory function to create grafting strategies.

    Args:
        strategy_name: Name of strategy to create
        config: Optional configuration

    Returns:
        Configured strategy instance
    """
    if strategy_name not in GRAFTING_STRATEGIES:
        raise ValueError(
            f"Unknown grafting strategy: {strategy_name}. "
            f"Available: {list(GRAFTING_STRATEGIES.keys())}"
        )

    if config is None:
        config = GraftingConfig()

    strategy_class = GRAFTING_STRATEGIES[strategy_name]
    return strategy_class(config)
