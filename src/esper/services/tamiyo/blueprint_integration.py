"""
Blueprint integration layer for Phase 1-2 connection.

Provides seamless integration between Tamiyo's intelligent decisions,
blueprint selection, and Phase 1 kernel execution system.
"""

import asyncio
import logging
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from esper.blueprints.metadata import BlueprintCategory
from esper.blueprints.metadata import BlueprintMetadata
from esper.blueprints.registry import BlueprintRegistry
from esper.contracts.assets import Blueprint
from esper.contracts.enums import BlueprintState
from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import ModelGraphState
from esper.services.clients.tezzeret_client import TezzeretClient
from esper.services.oona_client import OonaClient
from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.services.tamiyo.seed_selector import SeedSelector
from esper.services.tamiyo.seed_selector import SelectionContext
from esper.services.tamiyo.seed_selector import SelectionStrategy
from esper.utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class BlueprintSelector:
    """
    Intelligent blueprint selection based on GNN policy output.
    
    Maps policy decisions to appropriate blueprint templates from
    the registry.
    """

    def __init__(self, blueprint_registry: BlueprintRegistry):
        self.registry = blueprint_registry
        self.selection_history: List[Dict] = []

        # Action type to blueprint category mapping
        self.action_mapping = {
            0: BlueprintCategory.TRANSFORMER,    # Attention/FFN
            1: BlueprintCategory.MIXTURE_OF_EXPERTS,  # MoE components
            2: BlueprintCategory.EFFICIENCY,     # LoRA/compression
            3: BlueprintCategory.ROUTING,        # Distributed
            4: BlueprintCategory.DIAGNOSTICS,    # Monitoring
        }

    def select_blueprint(
        self,
        action_type: int,
        model_state: ModelGraphState,
        layer_name: str,
        constraints: Optional[Dict] = None
    ) -> Optional[BlueprintMetadata]:
        """
        Select appropriate blueprint based on policy action and context.
        
        Args:
            action_type: Policy network action output
            model_state: Current model graph state
            layer_name: Target layer for adaptation
            constraints: Optional constraints (memory, latency, etc.)
            
        Returns:
            Selected blueprint metadata or None
        """
        # Map action to category
        category = self.action_mapping.get(action_type)
        if not category:
            logger.warning(f"Unknown action type: {action_type}")
            return None

        # Get layer info from model state
        layer_info = self._extract_layer_info(model_state, layer_name)
        layer_type = layer_info.get("type", "Linear")

        # Filter blueprints by criteria
        candidates = self.registry.list_blueprints(
            category=category,
            safe_only=True,  # Safety first
            compatible_with=layer_type
        )

        if not candidates:
            logger.info(f"No compatible blueprints for {layer_type} in {category}")
            return None

        # Apply constraints
        if constraints:
            candidates = self._apply_constraints(candidates, constraints)

        # Score and rank candidates
        scored_candidates = []
        for blueprint in candidates:
            score = self._score_blueprint(blueprint, model_state, layer_info)
            scored_candidates.append((score, blueprint))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        if scored_candidates:
            best_score, best_blueprint = scored_candidates[0]

            # Record selection
            self.selection_history.append({
                "timestamp": time.time(),
                "action_type": action_type,
                "layer_name": layer_name,
                "blueprint_id": best_blueprint.blueprint_id,
                "score": best_score
            })

            logger.info(
                f"Selected blueprint {best_blueprint.blueprint_id} "
                f"for {layer_name} with score {best_score:.3f}"
            )

            return best_blueprint

        return None

    def _extract_layer_info(
        self,
        model_state: ModelGraphState,
        layer_name: str
    ) -> Dict:
        """Extract layer information from model state."""
        # Find layer in graph
        for node in model_state.graph_data.nodes:
            if node.get("name") == layer_name:
                return {
                    "type": node.get("type", "Linear"),
                    "parameters": node.get("parameters", 0),
                    "health_score": node.get("health_score", 0.5),
                    "current_adaptations": node.get("adaptations", [])
                }

        # Default if not found
        return {"type": "Linear", "parameters": 0, "health_score": 0.5}

    def _apply_constraints(
        self,
        candidates: List[BlueprintMetadata],
        constraints: Dict
    ) -> List[BlueprintMetadata]:
        """Filter blueprints by constraints."""
        filtered = []

        max_memory = constraints.get("max_memory_mb", float('inf'))
        max_latency = constraints.get("max_latency_ms", float('inf'))
        max_params = constraints.get("max_param_increase", float('inf'))

        for blueprint in candidates:
            # Check memory constraint
            if blueprint.memory_footprint_kb / 1024 > max_memory:
                continue

            # Check latency constraint
            if blueprint.expected_latency_ms > max_latency:
                continue

            # Check parameter constraint
            if blueprint.param_delta > max_params:
                continue

            filtered.append(blueprint)

        return filtered

    def _score_blueprint(
        self,
        blueprint: BlueprintMetadata,
        model_state: ModelGraphState,
        layer_info: Dict
    ) -> float:
        """Score blueprint based on expected benefit and cost."""
        score = 0.0

        # Benefit scores
        score += blueprint.past_accuracy_gain_estimate * 10.0
        score += blueprint.stability_improvement_estimate * 5.0
        score += blueprint.speed_improvement_estimate * 3.0

        # Cost penalties
        score -= blueprint.risk_score * 5.0
        score -= (blueprint.memory_footprint_kb / 1024) * 0.01
        score -= blueprint.expected_latency_ms * 0.1

        # Context bonuses
        if layer_info["health_score"] < 0.3:
            # Prioritize stability for unhealthy layers
            if blueprint.stability_improvement_estimate > 0.01:
                score += 2.0

        # Avoid redundant adaptations
        if blueprint.blueprint_id in layer_info.get("current_adaptations", []):
            score -= 10.0

        return score


class ExecutionSystemIntegrator:
    """
    Integrates Tamiyo decisions with Phase 1 kernel execution system.
    
    Handles the complete flow from blueprint selection to kernel loading.
    """

    def __init__(
        self,
        oona_client: OonaClient,
        urza_url: str,
        tezzeret_client: Optional[TezzeretClient] = None,
        seed_selector: Optional[SeedSelector] = None,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        self.oona_client = oona_client
        self.urza_url = urza_url
        self.tezzeret_client = tezzeret_client or TezzeretClient(urza_url)

        # Initialize seed selection system
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.seed_selector = seed_selector or SeedSelector(
            strategy=SelectionStrategy.UCB,
            performance_tracker=self.performance_tracker
        )

        # Circuit breaker for reliability
        from esper.utils.circuit_breaker import CircuitBreakerConfig
        self.circuit_breaker = CircuitBreaker(
            name="execution_integrator",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60
            )
        )

        # Track active adaptations
        self.active_adaptations: Dict[str, Dict] = {}
        self.current_epoch = 0

    async def select_seed_for_layer(
        self,
        layer_name: str,
        available_seeds: List[int],
        current_loss: float = 0.0,
        learning_rate: float = 0.001,
        available_memory_mb: float = 1000.0,
        urgency: float = 0.5
    ) -> Tuple[int, str]:
        """
        Select optimal seed for a layer using intelligent selection.
        
        Args:
            layer_name: Target layer name
            available_seeds: List of available seed indices
            current_loss: Current training loss
            learning_rate: Current learning rate
            available_memory_mb: Available GPU memory
            urgency: Adaptation urgency (0-1)
            
        Returns:
            (seed_idx, reason) tuple
        """
        # Create selection context
        context = SelectionContext(
            current_epoch=self.current_epoch,
            total_epochs=100,  # Will be updated from training config
            current_loss=current_loss,
            learning_rate=learning_rate,
            layer_type="Linear",  # Will be determined from layer
            available_memory_mb=available_memory_mb,
            urgency=urgency
        )

        # Get currently active seeds for this layer
        active_seeds = [
            int(key.split(":")[1])
            for key in self.active_adaptations.keys()
            if key.startswith(f"{layer_name}:")
        ]

        # Select seed using intelligent strategy
        seed_idx, reason = await self.seed_selector.select_seed(
            layer_name=layer_name,
            available_seeds=available_seeds,
            context=context,
            active_seeds=active_seeds
        )

        logger.info(
            f"Selected seed {seed_idx} for {layer_name}: {reason.reason}"
        )

        return seed_idx, reason.reason

    async def load_kernel(
        self,
        layer_name: str,
        kernel_id: str,
        seed_idx: Optional[int] = None,
        use_intelligent_selection: bool = True
    ) -> bool:
        """
        Load kernel into KasminaLayer using Phase 1 system.
        
        Args:
            layer_name: Target layer name
            kernel_id: Compiled kernel ID from Urza
            seed_idx: Seed index in KasminaLayer (None for intelligent selection)
            use_intelligent_selection: Whether to use intelligent seed selection
            
        Returns:
            Success status
        """
        try:
            # Use intelligent seed selection if not specified
            if seed_idx is None and use_intelligent_selection:
                # Default to 4 seeds per layer (standard config)
                available_seeds = list(range(4))
                seed_idx, selection_reason = await self.select_seed_for_layer(
                    layer_name=layer_name,
                    available_seeds=available_seeds
                )
                logger.info(f"Using intelligent selection: {selection_reason}")
            elif seed_idx is None:
                # Fallback to seed 0 if not using intelligent selection
                seed_idx = 0

            # Use circuit breaker for protection
            async with self.circuit_breaker:
                # Publish kernel load command via Oona
                command = {
                    "action": "load_kernel",
                    "layer_name": layer_name,
                    "kernel_id": kernel_id,
                    "seed_idx": seed_idx,
                    "timestamp": time.time()
                }

                await self.oona_client.publish(
                    topic="control.kasmina.commands",
                    message=command
                )

                # Track active adaptation
                self.active_adaptations[f"{layer_name}:{seed_idx}"] = {
                    "kernel_id": kernel_id,
                    "loaded_at": time.time(),
                    "status": "active"
                }

                logger.info(f"Loaded kernel {kernel_id} into {layer_name}:{seed_idx}")
                return True

        except Exception as e:
            logger.error(f"Failed to load kernel: {e}")
            return False

    async def unload_kernel(
        self,
        layer_name: str,
        seed_idx: int = 0
    ) -> bool:
        """Unload kernel from KasminaLayer."""
        try:
            async with self.circuit_breaker:
                command = {
                    "action": "unload_kernel",
                    "layer_name": layer_name,
                    "seed_idx": seed_idx,
                    "timestamp": time.time()
                }

                await self.oona_client.publish(
                    topic="control.kasmina.commands",
                    message=command
                )

                # Update tracking
                key = f"{layer_name}:{seed_idx}"
                if key in self.active_adaptations:
                    self.active_adaptations[key]["status"] = "unloaded"
                    self.active_adaptations[key]["unloaded_at"] = time.time()

                logger.info(f"Unloaded kernel from {layer_name}:{seed_idx}")
                return True

        except Exception as e:
            logger.error(f"Failed to unload kernel: {e}")
            return False

    async def request_blueprint_compilation(
        self,
        blueprint: BlueprintMetadata,
        target_layer: str
    ) -> Optional[str]:
        """
        Request blueprint compilation via Tezzeret.
        
        Args:
            blueprint: Blueprint metadata to compile
            target_layer: Target layer for adaptation
            
        Returns:
            Blueprint ID if submitted successfully
        """
        try:
            # Create Blueprint contract
            blueprint_contract = Blueprint(
                name=blueprint.name,
                description=blueprint.description,
                state=BlueprintState.PROPOSED,
                architecture={
                    "blueprint_id": blueprint.blueprint_id,
                    "target_layer": target_layer,
                    "category": blueprint.category.value
                },
                hyperparameters={},
                created_by="tamiyo",
                performance_metrics={}
            )

            # Submit to Tezzeret via Urza
            blueprint_id = await self.tezzeret_client.submit_blueprint(
                blueprint_contract
            )

            if blueprint_id:
                logger.info(f"Submitted blueprint {blueprint_id} for compilation")
                return blueprint_id

        except Exception as e:
            logger.error(f"Failed to submit blueprint: {e}")

        return None

    async def wait_for_compilation(
        self,
        blueprint_id: str,
        timeout: float = 300.0  # 5 minutes
    ) -> Optional[str]:
        """
        Wait for blueprint compilation to complete.
        
        Args:
            blueprint_id: Blueprint ID to monitor
            timeout: Maximum wait time
            
        Returns:
            Kernel ID if compilation successful
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check compilation status
                status = await self.tezzeret_client.get_blueprint_status(
                    blueprint_id
                )

                if status and status.get("state") == BlueprintState.CHARACTERIZED:
                    # Compilation complete
                    kernel_id = status.get("kernel_id")
                    if kernel_id:
                        logger.info(f"Compilation complete: {kernel_id}")
                        return kernel_id

                elif status and status.get("state") == BlueprintState.INVALID:
                    logger.error(f"Blueprint compilation failed: {blueprint_id}")
                    return None

            except Exception as e:
                logger.warning(f"Error checking compilation status: {e}")

            # Wait before next check
            await asyncio.sleep(5.0)

        logger.error(f"Compilation timeout for blueprint {blueprint_id}")
        return None

    def get_active_adaptations(self) -> Dict[str, Dict]:
        """Get currently active adaptations."""
        return {
            k: v for k, v in self.active_adaptations.items()
            if v.get("status") == "active"
        }

    async def update_performance_metrics(
        self,
        layer_name: str,
        seed_idx: int,
        accuracy_delta: float,
        loss_delta: float,
        latency_ms: float,
        memory_mb: float
    ) -> None:
        """
        Update performance metrics for a seed based on execution results.
        
        Args:
            layer_name: Layer name
            seed_idx: Seed index
            accuracy_delta: Change in accuracy
            loss_delta: Change in loss
            latency_ms: Execution latency
            memory_mb: Memory usage
        """
        from esper.services.tamiyo.performance_tracker import PerformanceDelta

        delta = PerformanceDelta(
            accuracy_delta=accuracy_delta,
            loss_delta=loss_delta,
            latency_ms=latency_ms,
            memory_mb=memory_mb
        )

        # Get kernel ID if available
        kernel_id = None
        adaptation_key = f"{layer_name}:{seed_idx}"
        if adaptation_key in self.active_adaptations:
            kernel_id = self.active_adaptations[adaptation_key].get("kernel_id")

        await self.performance_tracker.update_metrics(
            layer_name=layer_name,
            seed_idx=seed_idx,
            performance_delta=delta,
            kernel_id=kernel_id
        )

        logger.info(
            f"Updated metrics for {layer_name}:{seed_idx} - "
            f"accuracy_delta={accuracy_delta:.4f}, loss_delta={loss_delta:.4f}"
        )

    def update_epoch(self, epoch: int) -> None:
        """Update current epoch for selection context."""
        self.current_epoch = epoch


class Phase2IntegrationOrchestrator:
    """
    Complete Phase 1-2 integration orchestrator.
    
    Handles the full adaptation pipeline from Tamiyo decision to
    kernel execution.
    """

    def __init__(
        self,
        blueprint_registry: BlueprintRegistry,
        oona_client: OonaClient,
        urza_url: str,
        seed_selection_strategy: SelectionStrategy = SelectionStrategy.UCB,
        seed_selection_config: Optional[Dict] = None
    ):
        self.blueprint_selector = BlueprintSelector(blueprint_registry)

        # Create shared performance tracker
        performance_tracker = PerformanceTracker()

        # Create seed selector with strategy
        seed_selector = SeedSelector(
            strategy=seed_selection_strategy,
            performance_tracker=performance_tracker,
            config=seed_selection_config or {}
        )

        # Pass to execution integrator
        self.execution_integrator = ExecutionSystemIntegrator(
            oona_client=oona_client,
            urza_url=urza_url,
            seed_selector=seed_selector,
            performance_tracker=performance_tracker
        )

        # Track adaptation pipeline state
        self.pipeline_state = {}

    async def execute_adaptation_pipeline(
        self,
        decision: AdaptationDecision,
        model_state: ModelGraphState,
        constraints: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """
        Execute complete adaptation pipeline.
        
        Args:
            decision: Tamiyo adaptation decision
            model_state: Current model state
            constraints: Optional constraints
            
        Returns:
            Success status and execution details
        """
        pipeline_id = f"{decision.layer_name}:{time.time()}"
        self.pipeline_state[pipeline_id] = {
            "start_time": time.time(),
            "decision": decision,
            "status": "started"
        }

        try:
            # 1. Select appropriate blueprint
            blueprint = self.blueprint_selector.select_blueprint(
                action_type=self._map_adaptation_to_action(decision),
                model_state=model_state,
                layer_name=decision.layer_name,
                constraints=constraints
            )

            if not blueprint:
                logger.warning("No suitable blueprint found")
                self.pipeline_state[pipeline_id]["status"] = "no_blueprint"
                return False, {"reason": "no_suitable_blueprint"}

            self.pipeline_state[pipeline_id]["blueprint_id"] = blueprint.blueprint_id

            # 2. Request compilation
            blueprint_id = await self.execution_integrator.request_blueprint_compilation(
                blueprint, decision.layer_name
            )

            if not blueprint_id:
                self.pipeline_state[pipeline_id]["status"] = "compilation_failed"
                return False, {"reason": "compilation_request_failed"}

            # 3. Wait for compilation
            kernel_id = await self.execution_integrator.wait_for_compilation(
                blueprint_id
            )

            if not kernel_id:
                self.pipeline_state[pipeline_id]["status"] = "compilation_timeout"
                return False, {"reason": "compilation_timeout"}

            self.pipeline_state[pipeline_id]["kernel_id"] = kernel_id

            # 4. Load kernel into execution layer with intelligent seed selection
            success = await self.execution_integrator.load_kernel(
                layer_name=decision.layer_name,
                kernel_id=kernel_id
                # seed_idx not specified - will use intelligent selection
            )

            if success:
                self.pipeline_state[pipeline_id]["status"] = "completed"
                self.pipeline_state[pipeline_id]["end_time"] = time.time()

                duration = (
                    self.pipeline_state[pipeline_id]["end_time"] -
                    self.pipeline_state[pipeline_id]["start_time"]
                )

                return True, {
                    "pipeline_id": pipeline_id,
                    "blueprint_id": blueprint.blueprint_id,
                    "kernel_id": kernel_id,
                    "duration_seconds": duration
                }
            else:
                self.pipeline_state[pipeline_id]["status"] = "load_failed"
                return False, {"reason": "kernel_load_failed"}

        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            self.pipeline_state[pipeline_id]["status"] = "error"
            self.pipeline_state[pipeline_id]["error"] = str(e)
            return False, {"reason": "pipeline_error", "error": str(e)}

    def _map_adaptation_to_action(self, decision: AdaptationDecision) -> int:
        """Map adaptation decision to action type."""
        # Simple mapping based on adaptation type
        mapping = {
            "add_attention": 0,
            "add_ffn": 0,
            "add_moe": 1,
            "add_efficiency": 2,
            "add_routing": 3,
            "add_diagnostic": 4,
        }

        # Default to transformer category
        return mapping.get(decision.adaptation_type, 0)
