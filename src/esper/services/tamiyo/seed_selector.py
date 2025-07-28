"""
Intelligent seed selection framework for morphogenetic training.

This module provides various seed selection strategies including
UCB, Thompson Sampling, Epsilon-Greedy, and performance-weighted approaches.
"""

import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.services.tamiyo.performance_tracker import SeedPerformanceMetrics


class SelectionStrategy(str, Enum):
    """Available seed selection strategies."""

    UCB = "ucb"
    THOMPSON = "thompson"
    EPSILON_GREEDY = "epsilon_greedy"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    ROUND_ROBIN = "round_robin"  # For testing


@dataclass
class SelectionContext:
    """Context information for seed selection."""

    current_epoch: int
    total_epochs: int
    current_loss: float
    learning_rate: float
    layer_type: str  # e.g., "Linear", "Conv2d"
    available_memory_mb: float
    urgency: float = 0.5  # 0-1, higher means more urgent adaptation


@dataclass
class SelectionReason:
    """Explanation for why a seed was selected."""

    strategy: str
    seed_idx: int
    score: float
    reason: str
    alternatives_considered: int


class SeedSelectionStrategy(ABC):
    """Base class for seed selection strategies."""

    @abstractmethod
    def select(
        self,
        viable_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
        context: SelectionContext,
    ) -> int:
        """Select a seed from viable options."""

    @abstractmethod
    def get_reason(self) -> str:
        """Get human-readable selection reason."""


class UCBStrategy(SeedSelectionStrategy):
    """Upper Confidence Bound selection strategy."""

    def __init__(self, exploration_constant: float = 2.0):
        self.exploration_constant = exploration_constant
        self.last_scores = {}

    def select(
        self,
        viable_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
        context: SelectionContext,
    ) -> int:
        """Select seed using UCB algorithm."""
        if not viable_seeds:
            return 0  # Fallback to seed 0

        # Calculate UCB scores
        scores = {}
        for seed_idx in viable_seeds:
            if seed_idx in metrics:
                metric = metrics[seed_idx]
                # UCB score already computed in performance tracker
                scores[seed_idx] = metric.ucb_score
            else:
                # Unexplored seeds get infinite score
                scores[seed_idx] = float("inf")

        self.last_scores = scores

        # Select seed with highest UCB score
        best_seed = max(scores.keys(), key=lambda k: scores[k])
        return best_seed

    def get_reason(self) -> str:
        """Get selection reason."""
        if not self.last_scores:
            return "UCB: No scores available"

        sorted_seeds = sorted(
            self.last_scores.items(), key=lambda x: x[1], reverse=True
        )
        best_seed, best_score = sorted_seeds[0]

        if best_score == float("inf"):
            return f"UCB: Selected unexplored seed {best_seed}"
        else:
            return f"UCB: Selected seed {best_seed} with score {best_score:.3f}"


class ThompsonSamplingStrategy(SeedSelectionStrategy):
    """Thompson Sampling selection strategy."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.last_samples = {}

    def select(
        self,
        viable_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
        context: SelectionContext,
    ) -> int:
        """Select seed using Thompson Sampling."""
        if not viable_seeds:
            return 0

        # Sample from Beta distribution for each seed
        samples = {}
        for seed_idx in viable_seeds:
            if seed_idx in metrics:
                metric = metrics[seed_idx]
                # Thompson score already computed with Beta sampling
                samples[seed_idx] = metric.thompson_score
            else:
                # Use prior for unexplored seeds
                samples[seed_idx] = np.random.beta(self.prior_alpha, self.prior_beta)

        self.last_samples = samples

        # Select seed with highest sample
        best_seed = max(samples.keys(), key=lambda k: samples[k])
        return best_seed

    def get_reason(self) -> str:
        """Get selection reason."""
        if not self.last_samples:
            return "Thompson: No samples available"

        sorted_seeds = sorted(
            self.last_samples.items(), key=lambda x: x[1], reverse=True
        )
        best_seed, best_sample = sorted_seeds[0]

        return (
            f"Thompson: Selected seed {best_seed} "
            f"with sampled value {best_sample:.3f}"
        )


class EpsilonGreedyStrategy(SeedSelectionStrategy):
    """Epsilon-Greedy selection strategy."""

    def __init__(self, epsilon: float = 0.1, decay_rate: float = 0.995):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.last_choice = None

    def select(
        self,
        viable_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
        context: SelectionContext,
    ) -> int:
        """Select seed using epsilon-greedy approach."""
        if not viable_seeds:
            return 0

        # Decay epsilon over time
        self.epsilon = self.initial_epsilon * (
            self.decay_rate ** context.current_epoch
        )

        # Exploration vs exploitation
        if random.random() < self.epsilon:
            # Explore: random selection
            self.last_choice = "explore"
            return random.choice(viable_seeds)
        else:
            # Exploit: choose best performing
            self.last_choice = "exploit"
            best_seed = viable_seeds[0]
            best_score = -float("inf")

            for seed_idx in viable_seeds:
                if seed_idx in metrics:
                    score = metrics[seed_idx].performance_score
                    if score > best_score:
                        best_score = score
                        best_seed = seed_idx

            return best_seed

    def get_reason(self) -> str:
        """Get selection reason."""
        return (
            f"Epsilon-Greedy: {self.last_choice} "
            f"(epsilon={self.epsilon:.3f})"
        )


class PerformanceWeightedStrategy(SeedSelectionStrategy):
    """Direct performance-based selection."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.last_weights = {}

    def select(
        self,
        viable_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
        context: SelectionContext,
    ) -> int:
        """Select seed based on performance weighting."""
        if not viable_seeds:
            return 0

        # Calculate performance weights
        weights = {}
        for seed_idx in viable_seeds:
            if seed_idx in metrics:
                # Use exponentially weighted performance score
                score = metrics[seed_idx].performance_score
                # Apply urgency modifier
                score *= 1 + context.urgency
                weights[seed_idx] = score
            else:
                # Small weight for unexplored seeds
                weights[seed_idx] = 0.1

        self.last_weights = weights

        # Softmax selection with temperature
        if self.temperature > 0:
            # Convert to probabilities
            scores = np.array(list(weights.values()))
            scores = scores / self.temperature
            probs = np.exp(scores) / np.sum(np.exp(scores))

            # Sample based on probabilities
            seed_indices = list(weights.keys())
            return np.random.choice(seed_indices, p=probs)
        else:
            # Greedy selection
            return max(weights.keys(), key=lambda k: weights[k])

    def get_reason(self) -> str:
        """Get selection reason."""
        if not self.last_weights:
            return "Performance: No weights available"

        sorted_seeds = sorted(
            self.last_weights.items(), key=lambda x: x[1], reverse=True
        )
        best_seed, best_weight = sorted_seeds[0]

        return f"Performance: Selected seed {best_seed} with weight {best_weight:.3f}"


class SeedSelector:
    """
    Main seed selection framework.

    Coordinates seed selection using various strategies and
    tracks selection history.
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.UCB,
        performance_tracker: Optional[PerformanceTracker] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize seed selector.

        Args:
            strategy: Selection strategy to use
            performance_tracker: Performance tracking system
            config: Strategy-specific configuration
        """
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.config = config or {}

        # Initialize strategy
        self.strategy = self._create_strategy(strategy)

        # Track selection history
        self.selection_history = []

        # Constraints
        self.min_dormant_epochs = self.config.get("min_dormant_epochs", 5)
        self.max_concurrent_seeds = self.config.get("max_concurrent_seeds", 2)
        self.fallback_to_zero = self.config.get("fallback_to_zero", True)

    def _create_strategy(self, strategy: SelectionStrategy) -> SeedSelectionStrategy:
        """Create strategy instance based on type."""
        if strategy == SelectionStrategy.UCB:
            return UCBStrategy(
                exploration_constant=self.config.get("ucb", {}).get(
                    "exploration_constant", 2.0
                )
            )
        elif strategy == SelectionStrategy.THOMPSON:
            return ThompsonSamplingStrategy(
                prior_alpha=self.config.get("thompson", {}).get("prior_alpha", 1.0),
                prior_beta=self.config.get("thompson", {}).get("prior_beta", 1.0),
            )
        elif strategy == SelectionStrategy.EPSILON_GREEDY:
            return EpsilonGreedyStrategy(
                epsilon=self.config.get("epsilon_greedy", {}).get("epsilon", 0.1),
                decay_rate=self.config.get("epsilon_greedy", {}).get(
                    "decay_rate", 0.995
                ),
            )
        elif strategy == SelectionStrategy.PERFORMANCE_WEIGHTED:
            return PerformanceWeightedStrategy(
                temperature=self.config.get("performance", {}).get("temperature", 1.0)
            )
        else:
            # Round robin for testing
            return UCBStrategy()  # Default to UCB

    async def select_seed(
        self,
        layer_name: str,
        available_seeds: List[int],
        context: SelectionContext,
        active_seeds: Optional[List[int]] = None,
    ) -> Tuple[int, SelectionReason]:
        """
        Select optimal seed for kernel loading.

        Args:
            layer_name: Target layer name
            available_seeds: List of available seed indices
            context: Selection context
            active_seeds: Currently active seeds (to avoid)

        Returns:
            (seed_idx, reason) tuple
        """
        # Get performance metrics for this layer
        metrics = await self.performance_tracker.get_layer_metrics(layer_name)

        # Filter viable seeds
        viable_seeds = await self._filter_viable_seeds(
            available_seeds, layer_name, context, active_seeds or [], metrics
        )

        if not viable_seeds:
            # No viable seeds, fallback if enabled
            if self.fallback_to_zero and 0 in available_seeds:
                reason = SelectionReason(
                    strategy=str(self.strategy.__class__.__name__),
                    seed_idx=0,
                    score=0.0,
                    reason="No viable seeds, falling back to seed 0",
                    alternatives_considered=len(available_seeds),
                )
                return 0, reason
            else:
                # Return first available seed
                seed_idx = available_seeds[0] if available_seeds else 0
                reason = SelectionReason(
                    strategy=str(self.strategy.__class__.__name__),
                    seed_idx=seed_idx,
                    score=0.0,
                    reason="No viable seeds, using first available",
                    alternatives_considered=len(available_seeds),
                )
                return seed_idx, reason

        # Apply selection strategy
        selected_seed = self.strategy.select(viable_seeds, metrics, context)

        # Update tracking
        await self.performance_tracker.record_selection(
            layer_name, selected_seed, context.current_epoch
        )

        # Create reason
        reason = SelectionReason(
            strategy=str(self.strategy.__class__.__name__),
            seed_idx=selected_seed,
            score=metrics.get(selected_seed, SeedPerformanceMetrics(
                seed_idx=selected_seed, layer_name=layer_name
            )).performance_score,
            reason=self.strategy.get_reason(),
            alternatives_considered=len(viable_seeds),
        )

        # Record history
        self.selection_history.append(
            {
                "layer_name": layer_name,
                "seed_idx": selected_seed,
                "epoch": context.current_epoch,
                "reason": reason,
            }
        )

        return selected_seed, reason

    async def _filter_viable_seeds(
        self,
        available_seeds: List[int],
        layer_name: str,
        context: SelectionContext,
        active_seeds: List[int],
        metrics: Dict[int, SeedPerformanceMetrics],
    ) -> List[int]:
        """Filter seeds based on viability constraints."""
        viable = []

        for seed_idx in available_seeds:
            # Skip if already active
            if seed_idx in active_seeds:
                continue

            # Check concurrent limit
            if len(active_seeds) >= self.max_concurrent_seeds:
                continue

            # Check dormancy requirement
            if seed_idx in metrics:
                metric = metrics[seed_idx]
                epochs_since_activation = (
                    context.current_epoch - metric.last_activation_epoch
                )
                if epochs_since_activation < self.min_dormant_epochs:
                    continue

            # Check memory constraints
            if seed_idx in metrics:
                avg_memory = np.mean(metrics[seed_idx].memory_usage[-5:])
                if avg_memory > context.available_memory_mb * 0.8:
                    continue

            viable.append(seed_idx)

        return viable

    def get_selection_summary(self) -> Dict:
        """Get summary of selection history."""
        if not self.selection_history:
            return {"total_selections": 0}

        # Aggregate by layer
        layer_stats = {}
        for record in self.selection_history:
            layer = record["layer_name"]
            if layer not in layer_stats:
                layer_stats[layer] = {"selections": 0, "unique_seeds": set()}

            layer_stats[layer]["selections"] += 1
            layer_stats[layer]["unique_seeds"].add(record["seed_idx"])

        # Convert sets to counts
        for layer in layer_stats:
            layer_stats[layer]["unique_seeds"] = len(
                layer_stats[layer]["unique_seeds"]
            )

        return {
            "total_selections": len(self.selection_history),
            "layer_stats": layer_stats,
            "strategy": str(self.strategy.__class__.__name__),
        }
