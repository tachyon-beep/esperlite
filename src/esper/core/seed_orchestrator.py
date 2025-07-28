"""
Seed Orchestrator for Dynamic Architecture Modification via Kasmina Seeds.

This module implements Phase B4 by orchestrating dynamic kernel loading
and seed management to achieve morphogenetic behavior without traditional
model surgery.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch.nn as nn

from esper.blueprints.registry import BlueprintRegistry
from esper.contracts.operational import AdaptationDecision
from esper.execution.kasmina_layer import KasminaLayer
from esper.services.oona_client import OonaClient
from esper.services.tamiyo.blueprint_integration import Phase2IntegrationOrchestrator
from esper.services.tamiyo.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class SeedStrategy(Enum):
    """Strategy for managing seeds during architecture modification."""

    REPLACE = "replace"  # Replace underperforming kernel
    DIVERSIFY = "diversify"  # Load different kernels across seeds
    SPECIALIZE = "specialize"  # Specialize seeds for different tasks
    ENSEMBLE = "ensemble"  # Use all seeds as ensemble


@dataclass
class SeedModificationPlan:
    """Plan for modifying seeds in a layer."""

    layer_name: str
    strategy: SeedStrategy
    seed_modifications: Dict[int, Dict[str, Any]]  # seed_idx -> modification details
    expected_improvement: float
    risk_score: float
    reasoning: str


@dataclass
class SeedOrchestratorConfig:
    """Configuration for seed orchestrator."""

    max_seeds_per_layer: int = 4
    min_performance_threshold: float = 0.3
    adaptation_cooldown_epochs: int = 5
    blend_adjustment_rate: float = 0.1
    diversity_bonus: float = 0.2
    specialization_threshold: float = 0.7


class SeedOrchestrator:
    """
    Orchestrates dynamic architecture modification through Kasmina seeds.

    Instead of traditional model surgery, this achieves morphogenetic behavior by:
    1. Loading different kernels into seeds
    2. Adjusting seed blend factors
    3. Managing seed lifecycle and specialization
    """

    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        blueprint_registry: BlueprintRegistry,
        oona_client: OonaClient,
        urza_url: str,
        config: Optional[SeedOrchestratorConfig] = None,
    ):
        self.performance_tracker = performance_tracker
        self.config = config or SeedOrchestratorConfig()

        # Initialize Phase 2 integration for kernel loading
        self.integration_orchestrator = Phase2IntegrationOrchestrator(
            blueprint_registry=blueprint_registry,
            oona_client=oona_client,
            urza_url=urza_url,
        )

        # Track seed specializations
        self.seed_specializations: Dict[
            str, Dict[int, str]
        ] = {}  # layer -> seed_idx -> specialization

        # Track modification history
        self.modification_history: List[Dict[str, Any]] = []
        self.last_modification_epoch: Dict[str, int] = {}  # layer -> epoch

    async def apply_architecture_modification(
        self,
        model: nn.Module,
        decision: AdaptationDecision,
        model_state: Optional[Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply architecture modification through seed orchestration.

        Args:
            model: Model containing Kasmina layers
            decision: Adaptation decision from Tamiyo
            model_state: Optional model state for adaptation pipeline

        Returns:
            (success, details) tuple
        """
        start_time = time.time()

        try:
            # 1. Find the target Kasmina layer
            kasmina_layer = self._find_kasmina_layer(model, decision.layer_name)
            if kasmina_layer is None:
                return False, {
                    "error": f"Layer {decision.layer_name} is not a KasminaLayer"
                }

            # 2. Analyze current seed performance
            seed_analysis = await self._analyze_seed_performance(
                decision.layer_name, kasmina_layer
            )

            # 3. Create modification plan based on decision type
            plan = self._create_modification_plan(
                decision, kasmina_layer, seed_analysis
            )

            # 4. Execute the plan
            success = await self._execute_modification_plan(
                kasmina_layer, plan, model_state
            )

            # 5. Record the modification
            duration_ms = (time.time() - start_time) * 1000
            self._record_modification(decision, plan, success, duration_ms)

            if success:
                details = {
                    "strategy": plan.strategy.value,
                    "modified_seeds": len(plan.seed_modifications),
                    "expected_improvement": plan.expected_improvement,
                    "duration_ms": duration_ms,
                }
                logger.info(
                    f"Successfully applied {plan.strategy.value} modification to {decision.layer_name}: "
                    f"{details}"
                )
                return True, details
            else:
                return False, {"error": "Failed to execute modification plan"}

        except Exception as e:
            logger.error(f"Error applying architecture modification: {e}")
            return False, {"error": str(e)}

    def _find_kasmina_layer(
        self, model: nn.Module, layer_name: str
    ) -> Optional[KasminaLayer]:
        """Find Kasmina layer by name in the model."""
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, KasminaLayer):
                return module
        return None

    async def _analyze_seed_performance(
        self, layer_name: str, kasmina_layer: KasminaLayer
    ) -> Dict[int, Dict[str, float]]:
        """Analyze performance of each seed in the layer."""
        analysis = {}

        for seed_idx in range(kasmina_layer.num_seeds):
            # Get performance metrics from tracker
            metrics = await self.performance_tracker.get_seed_metrics(
                layer_name, seed_idx
            )

            # Calculate composite scores
            accuracy_score = metrics.get("accuracy_trend", 0.5)
            loss_score = 1.0 - metrics.get("loss_trend", 0.5)  # Lower loss is better
            efficiency_score = metrics.get("efficiency", 0.5)

            # Check if seed is active
            active_seeds_mask = kasmina_layer.state_layout.get_active_seeds()
            is_active = active_seeds_mask[seed_idx].item()

            analysis[seed_idx] = {
                "is_active": is_active,
                "accuracy_score": accuracy_score,
                "loss_score": loss_score,
                "efficiency_score": efficiency_score,
                "composite_score": (accuracy_score + loss_score + efficiency_score) / 3,
                "blend_factor": kasmina_layer.state_layout.alpha_blend[seed_idx].item(),
            }

        return analysis

    def _create_modification_plan(
        self,
        decision: AdaptationDecision,
        kasmina_layer: KasminaLayer,
        seed_analysis: Dict[int, Dict[str, float]],
    ) -> SeedModificationPlan:
        """Create a plan for modifying seeds based on decision and analysis."""

        # Determine strategy based on adaptation type and current state
        # Using contract-valid adaptation types from operational.py
        if decision.adaptation_type == "add_seed":
            # Add diversity by loading different kernels into new/inactive seeds
            plan = self._create_diversify_plan(kasmina_layer, seed_analysis, decision)

        elif decision.adaptation_type == "remove_seed":
            # Consolidate by reducing active seeds and specializing remaining ones
            plan = self._create_specialize_plan(kasmina_layer, seed_analysis, decision)

        elif decision.adaptation_type == "modify_architecture":
            # Create ensemble by activating all seeds with different kernels
            plan = self._create_ensemble_plan(kasmina_layer, seed_analysis, decision)

        elif decision.adaptation_type == "optimize_parameters":
            # Replace underperforming seeds with better alternatives
            plan = self._create_replace_plan(kasmina_layer, seed_analysis, decision)

        else:
            # This should never happen due to Pydantic validation
            raise ValueError(
                f"Invalid adaptation type: {decision.adaptation_type}. "
                f"Valid types are: add_seed, remove_seed, modify_architecture, optimize_parameters"
            )

        return plan

    def _create_diversify_plan(
        self,
        kasmina_layer: KasminaLayer,
        seed_analysis: Dict[int, Dict[str, float]],
        decision: AdaptationDecision,
    ) -> SeedModificationPlan:
        """Create plan to diversify seeds with different kernels."""
        modifications = {}

        # Find inactive or underperforming seeds
        candidates = []
        for seed_idx, analysis in seed_analysis.items():
            if (
                not analysis["is_active"]
                or analysis["composite_score"] < self.config.min_performance_threshold
            ):
                candidates.append((seed_idx, analysis["composite_score"]))

        # Sort by score (lowest first)
        candidates.sort(key=lambda x: x[1])

        # Plan to load diverse kernels into top candidates
        # Get parameters from metadata (not direct parameters field)
        params = decision.metadata.get("parameters", {})
        num_to_modify = min(len(candidates), params.get("num_seeds", 2))
        for i in range(num_to_modify):
            seed_idx = candidates[i][0]
            modifications[seed_idx] = {
                "action": "load_diverse_kernel",
                "category_preference": i % 3,  # Rotate through different categories
                "initial_blend": 0.3,  # Start with moderate blend
                "reasoning": f"Diversifying seed {seed_idx} with category {i % 3}",
            }

        expected_improvement = self.config.diversity_bonus * num_to_modify

        return SeedModificationPlan(
            layer_name=decision.layer_name,
            strategy=SeedStrategy.DIVERSIFY,
            seed_modifications=modifications,
            expected_improvement=expected_improvement,
            risk_score=0.3,  # Moderate risk
            reasoning=f"Diversifying {num_to_modify} seeds to increase representation capacity",
        )

    def _create_specialize_plan(
        self,
        kasmina_layer: KasminaLayer,
        seed_analysis: Dict[int, Dict[str, float]],
        decision: AdaptationDecision,
    ) -> SeedModificationPlan:
        """Create plan to specialize seeds by consolidating to best performers."""
        modifications = {}

        # Find best performing seeds
        active_seeds = [
            (idx, analysis)
            for idx, analysis in seed_analysis.items()
            if analysis["is_active"]
        ]
        active_seeds.sort(key=lambda x: x[1]["composite_score"], reverse=True)

        # Keep only top performers
        keep_count = max(1, len(active_seeds) // 2)

        for i, (seed_idx, analysis) in enumerate(active_seeds):
            if i < keep_count:
                # Increase blend factor for top performers
                modifications[seed_idx] = {
                    "action": "increase_blend",
                    "new_blend": min(0.8, analysis["blend_factor"] + 0.2),
                    "reasoning": f"Strengthening top performing seed {seed_idx}",
                }
            else:
                # Deactivate underperformers
                modifications[seed_idx] = {
                    "action": "unload_kernel",
                    "reasoning": f"Consolidating by removing underperforming seed {seed_idx}",
                }

        expected_improvement = 0.2 * keep_count  # Focused improvement

        return SeedModificationPlan(
            layer_name=decision.layer_name,
            strategy=SeedStrategy.SPECIALIZE,
            seed_modifications=modifications,
            expected_improvement=expected_improvement,
            risk_score=0.4,  # Higher risk due to consolidation
            reasoning=f"Specializing to {keep_count} best performing seeds",
        )

    def _create_ensemble_plan(
        self,
        kasmina_layer: KasminaLayer,
        seed_analysis: Dict[int, Dict[str, float]],
        decision: AdaptationDecision,
    ) -> SeedModificationPlan:
        """Create plan to use all seeds as an ensemble."""
        modifications = {}

        # Activate all seeds with balanced blending
        target_blend = 1.0 / kasmina_layer.num_seeds

        for seed_idx in range(kasmina_layer.num_seeds):
            if not seed_analysis[seed_idx]["is_active"]:
                # Load kernel for inactive seed
                modifications[seed_idx] = {
                    "action": "load_ensemble_kernel",
                    "category_preference": seed_idx,  # Different category per seed
                    "initial_blend": target_blend,
                    "reasoning": f"Activating seed {seed_idx} for ensemble",
                }
            else:
                # Adjust blend for active seed
                modifications[seed_idx] = {
                    "action": "adjust_blend",
                    "new_blend": target_blend,
                    "reasoning": f"Balancing seed {seed_idx} for ensemble",
                }

        expected_improvement = 0.15 * kasmina_layer.num_seeds

        return SeedModificationPlan(
            layer_name=decision.layer_name,
            strategy=SeedStrategy.ENSEMBLE,
            seed_modifications=modifications,
            expected_improvement=expected_improvement,
            risk_score=0.2,  # Low risk, high redundancy
            reasoning=f"Creating ensemble with all {kasmina_layer.num_seeds} seeds",
        )

    def _create_replace_plan(
        self,
        kasmina_layer: KasminaLayer,
        seed_analysis: Dict[int, Dict[str, float]],
        decision: AdaptationDecision,
    ) -> SeedModificationPlan:
        """Create plan to replace underperforming seeds."""
        modifications = {}

        # Find worst performing active seed
        worst_seed = None
        worst_score = float("inf")

        for seed_idx, analysis in seed_analysis.items():
            if analysis["is_active"] and analysis["composite_score"] < worst_score:
                worst_seed = seed_idx
                worst_score = analysis["composite_score"]

        if (
            worst_seed is not None
            and worst_score < self.config.min_performance_threshold
        ):
            modifications[worst_seed] = {
                "action": "replace_kernel",
                "reasoning": f"Replacing underperforming seed {worst_seed} (score: {worst_score:.3f})",
            }
            expected_improvement = 0.3
        else:
            # No replacement needed
            expected_improvement = 0.0

        return SeedModificationPlan(
            layer_name=decision.layer_name,
            strategy=SeedStrategy.REPLACE,
            seed_modifications=modifications,
            expected_improvement=expected_improvement,
            risk_score=0.3,
            reasoning=f"Replacing {len(modifications)} underperforming seeds",
        )

    async def _execute_modification_plan(
        self, kasmina_layer: KasminaLayer, plan: SeedModificationPlan, model_state: Any
    ) -> bool:
        """Execute the seed modification plan."""
        try:
            success_count = 0

            for seed_idx, modification in plan.seed_modifications.items():
                action = modification["action"]

                if action in [
                    "load_diverse_kernel",
                    "load_ensemble_kernel",
                    "replace_kernel",
                ]:
                    # Request blueprint compilation and load kernel
                    # Using correct schema - no decision_id, parameters/reasoning in metadata
                    decision = AdaptationDecision(
                        layer_name=plan.layer_name,
                        adaptation_type="optimize_parameters",  # Valid type for kernel optimization
                        confidence=0.8,
                        urgency=0.5,  # Required field
                        metadata={
                            "parameters": {
                                "category_preference": modification.get(
                                    "category_preference", 0
                                ),
                                "seed_idx": seed_idx,
                            },
                            "reasoning": modification["reasoning"],
                            "action": action,
                        }
                    )

                    # Execute adaptation pipeline
                    (
                        success,
                        details,
                    ) = await self.integration_orchestrator.execute_adaptation_pipeline(
                        decision=decision, model_state=model_state
                    )

                    if success:
                        # Adjust blend factor if specified
                        if "initial_blend" in modification:
                            kasmina_layer.set_seed_alpha(
                                seed_idx, modification["initial_blend"]
                            )
                        success_count += 1
                    else:
                        logger.warning(
                            f"Failed to load kernel for seed {seed_idx}: {details}"
                        )

                elif action == "unload_kernel":
                    # Unload kernel from seed
                    await kasmina_layer.unload_kernel(seed_idx)
                    success_count += 1

                elif action in ["adjust_blend", "increase_blend"]:
                    # Adjust blend factor
                    new_blend = modification.get("new_blend", 0.5)
                    kasmina_layer.set_seed_alpha(seed_idx, new_blend)
                    success_count += 1

                else:
                    logger.warning(f"Unknown modification action: {action}")

            # Consider success if at least half of modifications succeeded
            return success_count >= len(plan.seed_modifications) / 2

        except Exception as e:
            logger.error(f"Error executing modification plan: {e}")
            return False

    def _record_modification(
        self,
        decision: AdaptationDecision,
        plan: SeedModificationPlan,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Record modification in history."""
        record = {
            "timestamp": decision.timestamp,  # Use timestamp from decision
            "layer_name": decision.layer_name,
            "adaptation_type": decision.adaptation_type,
            "strategy": plan.strategy.value,
            "num_seeds_modified": len(plan.seed_modifications),
            "expected_improvement": plan.expected_improvement,
            "risk_score": plan.risk_score,
            "confidence": decision.confidence,
            "urgency": decision.urgency,
            "success": success,
            "duration_ms": duration_ms,
            "metadata": decision.metadata,
        }

        self.modification_history.append(record)

        # Note: epoch tracking moved to caller since AdaptationDecision doesn't have epoch
        # The caller should track epochs separately if needed

    def can_modify_layer(self, layer_name: str, current_epoch: int) -> bool:
        """Check if layer can be modified based on cooldown period."""
        if layer_name not in self.last_modification_epoch:
            return True

        epochs_since_last = current_epoch - self.last_modification_epoch[layer_name]
        return epochs_since_last >= self.config.adaptation_cooldown_epochs

    def get_modification_stats(self) -> Dict[str, Any]:
        """Get statistics about seed modifications."""
        if not self.modification_history:
            return {
                "total_modifications": 0,
                "success_rate": 0.0,
                "strategies_used": {},
                "avg_duration_ms": 0.0,
            }

        total = len(self.modification_history)
        successful = sum(1 for m in self.modification_history if m["success"])

        strategies_used = {}
        total_duration = 0.0

        for mod in self.modification_history:
            strategy = mod["strategy"]
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
            total_duration += mod["duration_ms"]

        return {
            "total_modifications": total,
            "success_rate": successful / total,
            "strategies_used": strategies_used,
            "avg_duration_ms": total_duration / total,
            "recent_modifications": self.modification_history[-5:],  # Last 5
        }
