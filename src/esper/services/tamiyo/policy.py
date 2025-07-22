"""
Enhanced GNN-based Policy Model for Strategic Morphogenetic Control

This module implements the core intelligence of the Tamiyo Strategic Controller,
using advanced Graph Neural Networks with uncertainty quantification, attention
mechanisms, and safety regularization for production deployment.

Production Features:
- Uncertainty quantification for decision confidence
- Multi-head attention for complex graph analysis
- Safety regularization to prevent dangerous adaptations
- Integration with health collector and graph builder
"""

import importlib.util
import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool

from esper.contracts.operational import AdaptationDecision

from .model_graph_builder import ModelGraphState

# Initialize logger
logger = logging.getLogger(__name__)

# Detect torch-scatter acceleration availability
SCATTER_AVAILABLE = importlib.util.find_spec("torch_scatter") is not None
if SCATTER_AVAILABLE:
    logger.info("torch-scatter acceleration enabled")
else:
    logger.info("torch-scatter not available, using fallback pooling")


@dataclass
class PolicyConfig:
    """Enhanced configuration for production Tamiyo policy model."""

    # Enhanced GNN Architecture
    node_feature_dim: int = 20  # Matches ModelGraphBuilder with gradient features
    edge_feature_dim: int = 8  # Matches ModelGraphBuilder default
    hidden_dim: int = 128
    num_gnn_layers: int = 4
    num_attention_heads: int = 4
    attention_dropout: float = 0.1

    # Uncertainty quantification
    enable_uncertainty: bool = True
    uncertainty_samples: int = 10
    epistemic_weight: float = 0.1

    # Safety and decision thresholds
    health_threshold: float = 0.3
    adaptation_confidence_threshold: float = 0.75
    uncertainty_threshold: float = 0.2
    max_adaptations_per_epoch: int = 2
    safety_margin: float = 0.1

    # Training parameters
    learning_rate: float = 1e-4
    replay_buffer_size: int = 50000  # Increased for better experience diversity
    batch_size: int = 64
    target_update_freq: int = 100
    gradient_clip_norm: float = 1.0

    # Temporal analysis
    temporal_window_size: int = 100
    trend_weight: float = 0.2
    stability_weight: float = 0.3


class MultiHeadGraphAttention(nn.Module):
    """Multi-head attention mechanism for graph neural networks."""

    def __init__(
        self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.attention_layers = nn.ModuleList(
            [GATConv(in_dim, self.head_dim, dropout=dropout) for _ in range(num_heads)]
        )

        self.output_projection = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to graph."""
        # Apply each attention head
        head_outputs = []
        for attention_layer in self.attention_layers:
            head_out = attention_layer(x, edge_index)
            head_outputs.append(head_out)

        # Concatenate heads
        concat_output = torch.cat(head_outputs, dim=-1)

        # Apply output projection and layer norm
        output = self.output_projection(concat_output)
        output = self.layer_norm(output)

        return output


class UncertaintyQuantification(nn.Module):
    """Epistemic uncertainty quantification using Monte Carlo dropout."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout for uncertainty estimation
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate epistemic uncertainty using MC dropout.

        Returns:
            mean_prediction: Average prediction across samples
            uncertainty: Estimated epistemic uncertainty
        """
        if not self.training:
            # Enable dropout during inference for uncertainty estimation
            self.train()

        predictions = []
        for _ in range(num_samples):
            pred = self.uncertainty_head(x)
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        return mean_prediction, uncertainty


class SafetyRegularizer(nn.Module):
    """Safety regularization module to prevent dangerous adaptations."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.safety_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, graph_repr: torch.Tensor) -> torch.Tensor:
        """Compute safety score for proposed adaptation."""
        return self.safety_classifier(graph_repr)

    def safety_penalty(
        self, safety_score: torch.Tensor, threshold: float = 0.8
    ) -> torch.Tensor:
        """Compute penalty for unsafe adaptations."""
        # Higher penalty for lower safety scores
        penalty = torch.clamp(threshold - safety_score, min=0.0)
        return penalty.mean()


class EnhancedTamiyoPolicyGNN(nn.Module):
    """
    Enhanced Graph Neural Network policy for strategic morphogenetic control.

    Production features:
    - Multi-head attention for complex topology analysis
    - Uncertainty quantification for decision confidence
    - Safety regularization to prevent dangerous adaptations
    - Integration with health signal analysis and temporal trends
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Device handling - use CPU for testing, CUDA for production
        self.device = torch.device("cpu")  # Default to CPU for compatibility

        # Enhanced node encoder with residual connections
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        # Multi-head attention layers for complex topology analysis
        self.attention_layers = nn.ModuleList(
            [
                MultiHeadGraphAttention(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.num_attention_heads,
                    config.attention_dropout,
                )
                for _ in range(config.num_gnn_layers)
            ]
        )

        # Traditional GNN layers for comparison and ensemble
        self.gnn_layers = nn.ModuleList(
            [
                GCNConv(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_gnn_layers)
            ]
        )

        # Enhanced decision head with multiple outputs
        graph_repr_dim = config.hidden_dim * 2  # mean + max pooling
        self.decision_head = nn.Sequential(
            nn.Linear(graph_repr_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(
                config.hidden_dim // 2, 4
            ),  # [adapt_prob, layer_priority, urgency, risk_assessment]
        )

        # Value head for RL training with temporal consideration
        self.value_head = nn.Sequential(
            nn.Linear(graph_repr_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Temporal analysis head for trend consideration
        self.temporal_head = nn.Sequential(
            nn.Linear(graph_repr_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(
                config.hidden_dim // 2, 3
            ),  # [health_trend, stability_trend, urgency_trend]
        )

        # Advanced components
        if config.enable_uncertainty:
            self.uncertainty_module = UncertaintyQuantification(
                graph_repr_dim, config.hidden_dim
            )

        self.safety_regularizer = SafetyRegularizer(graph_repr_dim)

        # Log acceleration status
        if SCATTER_AVAILABLE:
            logger.info("Enhanced TamiyoPolicyGNN: Using torch-scatter acceleration")
        else:
            logger.info(
                "Enhanced TamiyoPolicyGNN: Using fallback pooling (install torch-scatter for 2-10x speedup)"
            )

        # Move model to specified device
        self.to(self.device)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with uncertainty quantification and safety analysis.

        Args:
            node_features: [num_nodes, node_feature_dim] tensor of layer features
            edge_index: [2, num_edges] tensor defining graph connectivity
            batch: Optional batch assignment for multiple graphs
            return_uncertainty: Whether to compute uncertainty estimates

        Returns:
            Dictionary containing:
            - adaptation_prob: Probability of adaptation
            - layer_priority: Priority score for layer selection
            - urgency_score: Urgency of adaptation
            - risk_assessment: Risk score for proposed adaptation
            - value_estimate: Value function estimate
            - temporal_analysis: Temporal trend analysis
            - safety_score: Safety assessment
            - uncertainty (optional): Epistemic uncertainty estimates
        """
        # Move inputs to same device as model
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)

        # Encode node features
        x = self.node_encoder(node_features)

        # Apply both attention and traditional GNN layers with ensemble
        attention_features = x
        traditional_features = x

        # Multi-head attention pathway
        for attention_layer in self.attention_layers:
            attention_new = attention_layer(attention_features, edge_index)
            attention_features = (
                attention_features + attention_new
            )  # Residual connection
            attention_features = F.dropout(attention_features, training=self.training)

        # Traditional GNN pathway
        for gnn_layer in self.gnn_layers:
            traditional_new = F.relu(gnn_layer(traditional_features, edge_index))
            traditional_features = (
                traditional_features + traditional_new
            )  # Residual connection
            traditional_features = F.dropout(
                traditional_features, training=self.training
            )

        # Ensemble combination
        x = 0.6 * attention_features + 0.4 * traditional_features

        # Global pooling to get graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)

        # Generate enhanced decisions
        decision_logits = self.decision_head(graph_repr)
        adaptation_prob = torch.sigmoid(decision_logits[:, 0])
        layer_priority = torch.sigmoid(decision_logits[:, 1])
        urgency_score = torch.sigmoid(decision_logits[:, 2])
        risk_assessment = torch.sigmoid(decision_logits[:, 3])

        # Value estimate for RL training
        value_estimate = self.value_head(graph_repr)

        # Temporal analysis
        temporal_logits = self.temporal_head(graph_repr)
        temporal_analysis = {
            "health_trend": torch.tanh(temporal_logits[:, 0]),
            "stability_trend": torch.tanh(temporal_logits[:, 1]),
            "urgency_trend": torch.tanh(temporal_logits[:, 2]),
        }

        # Safety assessment
        safety_score = self.safety_regularizer(graph_repr)

        # Prepare outputs
        outputs = {
            "adaptation_prob": adaptation_prob,
            "layer_priority": layer_priority,
            "urgency_score": urgency_score,
            "risk_assessment": risk_assessment,
            "value_estimate": value_estimate,
            "temporal_analysis": temporal_analysis,
            "safety_score": safety_score,
        }

        # Uncertainty quantification (optional)
        if return_uncertainty and hasattr(self, "uncertainty_module"):
            _, uncertainty = self.uncertainty_module(
                graph_repr, self.config.uncertainty_samples
            )
            outputs["uncertainty"] = uncertainty
            outputs["epistemic_confidence"] = 1.0 - uncertainty

        return outputs

    def make_decision(
        self, model_graph_state: ModelGraphState
    ) -> Optional[AdaptationDecision]:
        """
        Make an intelligent adaptation decision based on complete model graph state.

        Args:
            model_graph_state: Complete model graph state from ModelGraphBuilder

        Returns:
            AdaptationDecision if adaptation is recommended, None otherwise
        """
        # Extract graph data from model state
        graph_data = model_graph_state.graph_data

        # Run inference with uncertainty quantification
        with torch.no_grad():
            outputs = self.forward(
                node_features=graph_data.x,
                edge_index=graph_data.edge_index,
                batch=None,
                return_uncertainty=self.config.enable_uncertainty,
            )

        adaptation_prob = outputs["adaptation_prob"].item()
        safety_score = outputs["safety_score"].item()
        urgency_score = outputs["urgency_score"].item()
        risk_assessment = outputs["risk_assessment"].item()

        # Enhanced decision logic with multiple criteria
        confidence_met = adaptation_prob > self.config.adaptation_confidence_threshold
        safety_acceptable = safety_score > (1.0 - self.config.safety_margin)

        # Uncertainty check if available
        uncertainty_acceptable = True
        if "uncertainty" in outputs:
            uncertainty = outputs["uncertainty"].item()
            uncertainty_acceptable = uncertainty < self.config.uncertainty_threshold

        # Risk assessment
        risk_acceptable = risk_assessment < 0.7  # High risk threshold

        # Temporal analysis consideration
        temporal = outputs["temporal_analysis"]
        health_trend = temporal["health_trend"].item()
        stability_trend = temporal["stability_trend"].item()

        # Decision with comprehensive criteria
        should_adapt = (
            confidence_met
            and safety_acceptable
            and uncertainty_acceptable
            and risk_acceptable
            and health_trend < -0.2  # Declining health trend
        )

        if not should_adapt:
            return None

        # Find target layer using problematic layers from model state
        if not model_graph_state.problematic_layers:
            return None

        # Select the most problematic layer based on health trends
        target_layer = self._select_target_layer(
            model_graph_state.problematic_layers, model_graph_state.health_trends
        )

        # Determine adaptation type based on analysis
        adaptation_type = self._determine_adaptation_type(
            target_layer, model_graph_state, outputs
        )

        # Build comprehensive metadata
        metadata = {
            "reason": "intelligent_policy_decision",
            "confidence": adaptation_prob,
            "safety_score": safety_score,
            "risk_assessment": risk_assessment,
            "health_trend": health_trend,
            "stability_trend": stability_trend,
            "temporal_analysis": temporal,
            "problematic_layers_count": len(model_graph_state.problematic_layers),
        }

        if "uncertainty" in outputs:
            metadata["epistemic_uncertainty"] = outputs["uncertainty"].item()

        return AdaptationDecision(
            layer_name=target_layer,
            adaptation_type=adaptation_type,
            confidence=adaptation_prob,
            urgency=urgency_score,
            metadata=metadata,
        )

    def _select_target_layer(
        self, problematic_layers: List[str], health_trends: Dict[str, List[float]]
    ) -> str:
        """Select the most critical layer for adaptation."""
        if len(problematic_layers) == 1:
            return problematic_layers[0]

        # Score layers based on health trend deterioration
        layer_scores = {}
        for layer in problematic_layers:
            trend = health_trends.get(layer, [0.5])
            if len(trend) > 1:
                # Calculate trend slope (more negative = worse)
                recent_trend = sum(trend[-5:]) / len(trend[-5:])  # Last 5 signals
                older_trend = (
                    sum(trend[-10:-5]) / len(trend[-10:-5])
                    if len(trend) >= 10
                    else recent_trend
                )
                trend_slope = recent_trend - older_trend
                layer_scores[layer] = trend_slope
            else:
                layer_scores[layer] = -trend[0]  # Single point, use negative health

        # Return layer with worst (most negative) trend
        return min(layer_scores.items(), key=lambda x: x[1])[0]

    def _determine_adaptation_type(
        self,
        target_layer: str,
        model_state: ModelGraphState,
        outputs: Dict[str, torch.Tensor],
    ) -> str:
        """Determine the most appropriate adaptation type based on gradient analysis."""
        # Default adaptation types based on analysis
        urgency = outputs["urgency_score"].item()
        risk = outputs["risk_assessment"].item()

        # Extract gradient information from global metrics
        gradient_health = model_state.global_metrics.get("gradient_health", 0.5)
        training_stability = model_state.global_metrics.get("training_stability", 0.5)
        avg_gradient_norm = model_state.global_metrics.get("avg_gradient_norm", 1.0)

        # Gradient-informed adaptation decisions
        if avg_gradient_norm > 10.0:
            # Exploding gradients - need immediate stabilization
            return "add_gradient_clipping"
        elif avg_gradient_norm < 0.01:
            # Vanishing gradients - need activation changes
            return "add_residual_connection"
        elif training_stability < 0.3:
            # Unstable training - add normalization
            return "add_batch_normalization"
        elif gradient_health < 0.3 and urgency > 0.6:
            # Poor gradient flow with urgency
            return "add_skip_connection"
        elif urgency > 0.8:
            return "emergency_stabilization"
        elif risk > 0.6:
            return "conservative_optimization"
        else:
            return "optimize_parameters"

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive policy model statistics."""
        return {
            "model_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "attention_heads": self.config.num_attention_heads,
            "gnn_layers": self.config.num_gnn_layers,
            "hidden_dim": self.config.hidden_dim,
            "uncertainty_enabled": self.config.enable_uncertainty,
            "safety_regularization": True,
            "acceleration_available": SCATTER_AVAILABLE,
        }

    def compute_safety_penalty(self, graph_repr: torch.Tensor) -> torch.Tensor:
        """Compute safety penalty for training regularization."""
        safety_score = self.safety_regularizer(graph_repr)
        return self.safety_regularizer.safety_penalty(safety_score, threshold=0.8)

    @property
    def acceleration_status(self) -> Dict[str, Any]:
        """
        Report current acceleration status.

        Returns:
            Dictionary containing acceleration availability and status
        """
        return {
            "torch_scatter_available": SCATTER_AVAILABLE,
            "acceleration_enabled": SCATTER_AVAILABLE,
            "fallback_mode": not SCATTER_AVAILABLE,
            "attention_layers": len(self.attention_layers),
            "uncertainty_quantification": hasattr(self, "uncertainty_module"),
        }


class EnhancedPolicyTrainingState:
    """Enhanced training state management with prioritized experience replay."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.experience_buffer: List[Dict[str, Any]] = []
        self.priority_weights: List[float] = []
        self.alpha = 0.6  # Prioritization exponent
        self.beta = 0.4  # Importance sampling exponent
        self.epsilon = 1e-6  # Small constant to avoid zero priorities

    def add_experience(
        self,
        state: ModelGraphState,
        action: AdaptationDecision,
        reward: float,
        next_state: ModelGraphState,
        td_error: Optional[float] = None,
    ) -> None:
        """Add an experience tuple to the prioritized replay buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": len(self.experience_buffer),
            "td_error": td_error or 1.0,  # Default high priority for new experiences
        }

        # Calculate priority based on TD error
        priority = (abs(td_error) + self.epsilon) ** self.alpha if td_error else 1.0

        self.experience_buffer.append(experience)
        self.priority_weights.append(priority)

        # Maintain buffer size
        if len(self.experience_buffer) > self.config.replay_buffer_size:
            self.experience_buffer.pop(0)
            self.priority_weights.pop(0)

    def sample_batch(
        self, batch_size: Optional[int] = None, use_prioritization: bool = True
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """Sample a batch of experiences with optional prioritization."""
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.experience_buffer) < batch_size:
            experiences = self.experience_buffer.copy()
            weights = torch.ones(len(experiences))
            return experiences, weights

        if use_prioritization and self.priority_weights:
            # Prioritized sampling
            priorities = torch.tensor(self.priority_weights)
            probabilities = priorities / priorities.sum()

            indices = torch.multinomial(probabilities, batch_size, replacement=True)

            # Importance sampling weights
            weights = (len(self.experience_buffer) * probabilities[indices]) ** (
                -self.beta
            )
            weights = weights / weights.max()  # Normalize for stability

            experiences = [self.experience_buffer[i] for i in indices]
        else:
            # Uniform sampling
            indices = torch.randperm(len(self.experience_buffer))[:batch_size]
            experiences = [self.experience_buffer[i] for i in indices]
            weights = torch.ones(batch_size)

        return experiences, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priority_weights):
                self.priority_weights[idx] = (
                    abs(td_error) + self.epsilon
                ) ** self.alpha


# Backward compatibility alias
TamiyoPolicyGNN = EnhancedTamiyoPolicyGNN
PolicyTrainingState = EnhancedPolicyTrainingState
