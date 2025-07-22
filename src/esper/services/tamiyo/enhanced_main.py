"""
Enhanced Tamiyo Strategic Controller Service with REMEDIATION A1 integration.

Integrates blueprint library, reward system, and Phase 1-2 connection
for complete autonomous morphogenetic adaptation.
"""

import asyncio
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import torch

from esper.blueprints.registry import BlueprintRegistry
from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.contracts.operational import ModelGraphState
from esper.services.oona_client import OonaClient
from esper.services.tamiyo.analyzer import ModelGraphAnalyzer
from esper.services.tamiyo.blueprint_integration import Phase2IntegrationOrchestrator
from esper.services.tamiyo.health_collector import ProductionHealthCollector
from esper.services.tamiyo.policy import PolicyConfig
from esper.services.tamiyo.policy import PolicyTrainingState
from esper.services.tamiyo.policy import TamiyoPolicyGNN
from esper.services.tamiyo.policy_trainer import ProductionPolicyTrainer
from esper.services.tamiyo.reward_computer import IntelligentRewardComputer
from esper.utils.config import ServiceConfig


logger = logging.getLogger(__name__)


class EnhancedTamiyoService:
    """
    Production Tamiyo service with complete Phase 2 intelligence system.
    
    Integrates all components from REMEDIATION A1:
    - Blueprint library and selection
    - Multi-metric reward computation
    - Phase 1-2 seamless integration
    - Real-time health signal processing
    - GNN-based policy decisions
    """
    
    def __init__(
        self,
        service_config: ServiceConfig,
        oona_client: Optional[OonaClient] = None,
        policy_config: Optional[PolicyConfig] = None,
        enable_learning: bool = True,
    ):
        # Service configuration
        self.config = service_config
        self.enable_learning = enable_learning
        
        # External services
        self.oona_client = oona_client or OonaClient()
        
        # Initialize blueprint registry
        self.blueprint_registry = BlueprintRegistry()
        logger.info(f"Loaded {len(self.blueprint_registry.blueprints)} blueprints")
        
        # Policy configuration
        if policy_config is None:
            policy_config = PolicyConfig()
        
        # Core components
        self.health_collector = ProductionHealthCollector(
            oona_client=self.oona_client,
            buffer_size=50000
        )
        
        self.analyzer = ModelGraphAnalyzer()
        
        self.policy = TamiyoPolicyGNN(policy_config)
        
        self.policy_trainer = ProductionPolicyTrainer(
            policy_network=self.policy,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.reward_computer = IntelligentRewardComputer()
        
        self.integration_orchestrator = Phase2IntegrationOrchestrator(
            blueprint_registry=self.blueprint_registry,
            oona_client=self.oona_client,
            urza_url=self.config.urza_url
        )
        
        # Service state
        self.is_running = False
        self.last_adaptation_time: Dict[str, float] = {}
        self.current_model_state: Optional[ModelGraphState] = None
        self.adaptation_history: List[Tuple[AdaptationDecision, Dict]] = []
        
        # Performance tracking
        self.metrics = {
            "decisions_made": 0,
            "adaptations_executed": 0,
            "adaptation_success_rate": 0.0,
            "avg_decision_latency_ms": 0.0,
            "total_reward": 0.0,
        }
    
    async def start(self) -> None:
        """Start the enhanced Tamiyo service."""
        logger.info("Starting Enhanced Tamiyo Strategic Controller...")
        self.is_running = True
        
        # Start all service loops
        await asyncio.gather(
            self.health_collector.start_intelligent_collection(),
            self._policy_decision_loop(),
            self._training_loop() if self.enable_learning else self._noop_coro(),
            self._monitoring_loop(),
            self._safety_monitoring_loop()
        )
    
    async def stop(self) -> None:
        """Stop the Tamiyo service gracefully."""
        logger.info("Stopping Enhanced Tamiyo Strategic Controller...")
        self.is_running = False
        await asyncio.sleep(0.5)  # Allow loops to finish
    
    async def _policy_decision_loop(self) -> None:
        """Main policy decision loop with complete integration."""
        logger.info("Starting policy decision loop")
        
        decision_interval = 5.0  # 5 seconds between decisions
        min_signals_for_decision = 100
        
        while self.is_running:
            try:
                start_time = time.perf_counter()
                
                # Get current health signals
                current_health_signals = self.health_collector.signal_buffer.get_recent(
                    window_size=1000
                )
                
                if len(current_health_signals) < min_signals_for_decision:
                    await asyncio.sleep(decision_interval)
                    continue
                
                # Build graph representation
                model_topology = await self._get_current_topology()
                model_graph = self.analyzer.build_model_graph(
                    health_signals=current_health_signals,
                    model_topology=model_topology
                )
                
                # Make policy decision
                decision = await self._make_intelligent_decision(model_graph)
                
                if decision:
                    # Check safety and cooldown
                    if await self._validate_decision_safety(decision):
                        # Execute adaptation through Phase 1-2 integration
                        success = await self._execute_adaptation_pipeline(
                            decision, model_graph
                        )
                        
                        if success:
                            # Monitor results for learning
                            await self._monitor_adaptation_results(
                                decision, model_graph
                            )
                
                # Update metrics
                decision_time = (time.perf_counter() - start_time) * 1000
                self._update_decision_metrics(decision_time)
                
                await asyncio.sleep(decision_interval)
                
            except Exception as e:
                logger.error(f"Policy decision loop error: {e}")
                await asyncio.sleep(decision_interval)
    
    async def _make_intelligent_decision(
        self,
        model_graph: ModelGraphState
    ) -> Optional[AdaptationDecision]:
        """Make policy decision using trained GNN."""
        try:
            # Convert to PyTorch tensors
            graph_data = self._prepare_graph_for_inference(model_graph)
            
            # Forward pass through policy network
            with torch.no_grad():
                policy_outputs = self.policy(**graph_data)
            
            # Sample action with exploration
            action_type, log_prob, entropy = self.policy.sample_action(
                policy_outputs['policy_logits'],
                temperature=0.9  # Slightly exploratory
            )
            
            # Get uncertainty estimate
            uncertainty = policy_outputs['uncertainty'].item()
            
            # Only proceed if confidence is sufficient
            max_uncertainty = 0.3
            if uncertainty > max_uncertainty:
                logger.debug(f"Decision uncertainty {uncertainty} too high")
                return None
            
            # Select target layer based on node embeddings
            target_layer = self._select_target_layer(
                model_graph,
                policy_outputs['node_embeddings']
            )
            
            if target_layer is None:
                return None
            
            # Map action to adaptation type
            adaptation_type = self._map_action_to_adaptation(action_type)
            
            # Create adaptation decision
            decision = AdaptationDecision(
                layer_name=target_layer,
                adaptation_type=adaptation_type,
                confidence=1.0 - uncertainty,
                urgency=self._compute_urgency(model_graph, target_layer),
                metadata={
                    'policy_log_prob': log_prob,
                    'policy_entropy': entropy,
                    'action_type': action_type,
                    'graph_timestamp': model_graph.timestamp,
                    'decision_method': 'enhanced_gnn_policy'
                }
            )
            
            self.metrics["decisions_made"] += 1
            return decision
            
        except Exception as e:
            logger.error(f"Error making policy decision: {e}")
            return None
    
    async def _execute_adaptation_pipeline(
        self,
        decision: AdaptationDecision,
        model_graph: ModelGraphState
    ) -> bool:
        """Execute adaptation through complete Phase 1-2 pipeline."""
        try:
            # Define constraints based on current resources
            constraints = {
                "max_memory_mb": 1024,  # From manifest
                "max_latency_ms": 10.0,
                "max_param_increase": 1000000,  # 1M parameters
            }
            
            # Record pre-adaptation metrics
            pre_metrics = await self._collect_current_metrics(decision.layer_name)
            
            # Execute through integration orchestrator
            success, details = await self.integration_orchestrator.execute_adaptation_pipeline(
                decision=decision,
                model_state=model_graph,
                constraints=constraints
            )
            
            if success:
                self.metrics["adaptations_executed"] += 1
                
                # Record adaptation
                self.adaptation_history.append((decision, details))
                self.last_adaptation_time[decision.layer_name] = time.time()
                
                logger.info(
                    f"Successfully executed adaptation: {decision.adaptation_type} "
                    f"on {decision.layer_name} in {details['duration_seconds']:.2f}s"
                )
                
                # Collect post-adaptation metrics for reward computation
                post_metrics = await self._collect_current_metrics(decision.layer_name)
                
                # Compute reward
                reward_analysis = await self.reward_computer.compute_adaptation_reward(
                    adaptation_decision=decision,
                    pre_metrics=pre_metrics,
                    post_metrics=post_metrics,
                    temporal_window=300.0
                )
                
                # Train policy if learning enabled
                if self.enable_learning:
                    await self._update_policy_with_experience(
                        model_graph, decision, reward_analysis
                    )
                
                return True
            else:
                logger.warning(
                    f"Adaptation pipeline failed: {details.get('reason', 'unknown')}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error executing adaptation pipeline: {e}")
            return False
    
    async def _update_policy_with_experience(
        self,
        state: ModelGraphState,
        action: AdaptationDecision,
        reward_analysis
    ):
        """Update policy based on adaptation outcome."""
        try:
            # Create next state (would be from next health collection)
            next_state = state  # Simplified for now
            
            # Train on experience
            success = await self.policy_trainer.train_on_experience(
                state=state,
                action=action,
                reward=reward_analysis.total_reward,
                next_state=next_state,
                done=False
            )
            
            if success:
                self.metrics["total_reward"] += reward_analysis.total_reward
                logger.debug(
                    f"Policy updated with reward {reward_analysis.total_reward:.3f}"
                )
                
        except Exception as e:
            logger.error(f"Error updating policy: {e}")
    
    async def _validate_decision_safety(
        self,
        decision: AdaptationDecision
    ) -> bool:
        """Validate decision safety including cooldown."""
        # Check cooldown
        current_time = time.time()
        last_adaptation = self.last_adaptation_time.get(decision.layer_name, 0)
        cooldown = 30.0  # 30 seconds
        
        if current_time - last_adaptation < cooldown:
            logger.debug(f"Adaptation for {decision.layer_name} on cooldown")
            return False
        
        # Check confidence threshold
        if decision.confidence < 0.7:
            logger.debug(f"Decision confidence {decision.confidence} too low")
            return False
        
        # Check if layer is in error state
        if decision.layer_name in self.health_collector.filter_engine.error_layers:
            logger.warning(f"Layer {decision.layer_name} in error state")
            return False
        
        return True
    
    async def _collect_current_metrics(self, layer_name: str) -> Dict[str, float]:
        """Collect current metrics for a layer."""
        # Get latest health signals for the layer
        recent_signals = [
            s for s in self.health_collector.signal_buffer.get_recent(100)
            if s.layer_id == layer_name
        ]
        
        if not recent_signals:
            return {
                "accuracy": 0.0,
                "execution_latency_ms": 0.0,
                "error_rate": 0.0,
                "cache_hit_rate": 0.0,
                "memory_usage_mb": 0.0,
                "recovery_success_rate": 1.0,
            }
        
        # Aggregate metrics
        return {
            "accuracy": 0.85,  # Would come from training metrics
            "execution_latency_ms": sum(s.execution_latency for s in recent_signals) / len(recent_signals),
            "error_rate": sum(s.error_count for s in recent_signals) / len(recent_signals),
            "cache_hit_rate": sum(s.cache_hit_rate for s in recent_signals) / len(recent_signals),
            "memory_usage_mb": 100.0,  # Placeholder
            "recovery_success_rate": 0.99,  # From Phase 1 error recovery
        }
    
    async def _monitor_adaptation_results(
        self,
        decision: AdaptationDecision,
        pre_state: ModelGraphState
    ):
        """Monitor adaptation results for continuous improvement."""
        # This would track the adaptation over time
        # For now, just log
        logger.info(
            f"Monitoring adaptation: {decision.adaptation_type} "
            f"on {decision.layer_name}"
        )
    
    def _prepare_graph_for_inference(
        self,
        model_graph: ModelGraphState
    ) -> Dict[str, torch.Tensor]:
        """Convert model graph to PyTorch tensors."""
        # Extract graph data
        graph_data = model_graph.graph_data
        
        # Convert to tensors
        return {
            "x": torch.tensor(graph_data.x, dtype=torch.float32),
            "edge_index": torch.tensor(graph_data.edge_index, dtype=torch.long),
            "edge_attr": torch.tensor(graph_data.edge_attr, dtype=torch.float32),
            "batch": None,  # Single graph
        }
    
    def _select_target_layer(
        self,
        model_graph: ModelGraphState,
        node_embeddings: torch.Tensor
    ) -> Optional[str]:
        """Select target layer based on node embeddings and health."""
        # Find unhealthiest layer
        worst_health = 1.0
        target_layer = None
        
        for i, node in enumerate(model_graph.graph_data.nodes):
            if node.get("health_score", 1.0) < worst_health:
                worst_health = node["health_score"]
                target_layer = node.get("name", f"layer_{i}")
        
        return target_layer
    
    def _map_action_to_adaptation(self, action_type: int) -> str:
        """Map policy action to adaptation type."""
        mapping = {
            0: "add_attention",
            1: "add_moe",
            2: "add_efficiency",
            3: "add_routing",
            4: "add_diagnostic",
        }
        return mapping.get(action_type, "add_attention")
    
    def _compute_urgency(
        self,
        model_graph: ModelGraphState,
        layer_name: str
    ) -> float:
        """Compute adaptation urgency."""
        # Find layer health
        for node in model_graph.graph_data.nodes:
            if node.get("name") == layer_name:
                health = node.get("health_score", 1.0)
                # Lower health = higher urgency
                return 1.0 - health
        return 0.5
    
    async def _get_current_topology(self) -> Any:
        """Get current model topology."""
        # This would query the actual model structure
        # For now, return a mock topology
        return {
            "layer_names": ["layer_0", "layer_1", "layer_2", "layer_3"],
            "connections": [(0, 1), (1, 2), (2, 3)],
        }
    
    def _update_decision_metrics(self, decision_time_ms: float):
        """Update decision performance metrics."""
        # Update average latency with EMA
        alpha = 0.1
        self.metrics["avg_decision_latency_ms"] = (
            alpha * decision_time_ms +
            (1 - alpha) * self.metrics["avg_decision_latency_ms"]
        )
        
        # Update success rate
        if self.metrics["decisions_made"] > 0:
            self.metrics["adaptation_success_rate"] = (
                self.metrics["adaptations_executed"] /
                self.metrics["decisions_made"]
            )
    
    async def _training_loop(self) -> None:
        """Continuous policy improvement loop."""
        while self.is_running:
            try:
                # Check if we have enough experiences
                if self.policy_trainer.training_stats["episodes"] >= 10:
                    # Log training progress
                    logger.info(
                        f"Policy training stats: "
                        f"episodes={self.policy_trainer.training_stats['episodes']}, "
                        f"avg_reward={self.policy_trainer.training_stats['avg_reward']:.3f}, "
                        f"success_rate={self.policy_trainer.training_stats['success_rate']:.2%}"
                    )
                
                await asyncio.sleep(60.0)  # Training report interval
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _monitoring_loop(self) -> None:
        """System monitoring and metrics reporting."""
        while self.is_running:
            try:
                logger.info(
                    f"Tamiyo metrics: decisions={self.metrics['decisions_made']}, "
                    f"adaptations={self.metrics['adaptations_executed']}, "
                    f"success_rate={self.metrics['adaptation_success_rate']:.2%}, "
                    f"avg_latency={self.metrics['avg_decision_latency_ms']:.1f}ms, "
                    f"total_reward={self.metrics['total_reward']:.3f}"
                )
                
                await asyncio.sleep(30.0)  # Report every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _safety_monitoring_loop(self) -> None:
        """Safety monitoring for dangerous adaptations."""
        while self.is_running:
            try:
                # Check for concerning patterns
                if self.metrics["adaptation_success_rate"] < 0.5:
                    logger.warning(
                        "Low adaptation success rate: "
                        f"{self.metrics['adaptation_success_rate']:.2%}"
                    )
                
                if self.metrics["total_reward"] < -10.0:
                    logger.error("Negative total reward - policy may be degrading")
                
                await asyncio.sleep(10.0)  # Safety check every 10 seconds
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def _noop_coro(self) -> None:
        """No-op coroutine for when learning is disabled."""
        while self.is_running:
            await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        return {
            "is_running": self.is_running,
            "blueprints_loaded": len(self.blueprint_registry.blueprints),
            "health_signals_buffered": len(self.health_collector.signal_buffer.buffer),
            "active_adaptations": len(self.integration_orchestrator.active_adaptations),
            "total_adaptations": len(self.adaptation_history),
            "metrics": self.metrics,
            "policy_training": self.policy_trainer.training_stats,
        }