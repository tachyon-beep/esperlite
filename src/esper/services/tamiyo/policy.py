"""
GNN-based policy model for strategic morphogenetic control.

This module implements the core intelligence of the Tamiyo Strategic Controller,
using Graph Neural Networks to analyze model topology and performance metrics.
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
import torch.nn.functional
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import ModelGraphState

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
    """Configuration for Tamiyo policy model."""

    # GNN Architecture
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    hidden_dim: int = 128
    num_gnn_layers: int = 3

    # Decision thresholds
    health_threshold: float = 0.3
    adaptation_confidence_threshold: float = 0.7
    max_adaptations_per_epoch: int = 2

    # Training parameters
    learning_rate: float = 1e-4
    replay_buffer_size: int = 10000
    batch_size: int = 32


class TamiyoPolicyGNN(nn.Module):
    """
    Graph Neural Network policy for strategic morphogenetic control.

    This network analyzes the topology and performance characteristics of a host
    model to make intelligent decisions about when, where, and how to adapt.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Node encoder - processes layer-level features
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # GNN layers for topology-aware processing
        self.gnn_layers = nn.ModuleList(
            [
                GCNConv(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_gnn_layers)
            ]
        )

        # Decision head - outputs adaptation decisions
        self.decision_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim * 2, config.hidden_dim
            ),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [should_adapt, layer_priority_logits, urgency_score]
        )

        # Value head for policy training
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Log acceleration status
        if SCATTER_AVAILABLE:
            logger.info("TamiyoPolicyGNN: Using torch-scatter acceleration")
        else:
            logger.info(
                "TamiyoPolicyGNN: Using fallback pooling (install torch-scatter for 2-10x speedup)"
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Args:
            node_features: [num_nodes, node_feature_dim] tensor of layer features
            edge_index: [2, num_edges] tensor defining graph connectivity
            batch: Optional batch assignment for multiple graphs

        Returns:
            Tuple of (adaptation_decision, layer_priorities, urgency_score, value_estimate)
        """
        # Encode node features
        x = self.node_encoder(node_features)

        # Apply GNN layers with residual connections
        for gnn_layer in self.gnn_layers:
            x_new = torch.nn.functional.relu(gnn_layer(x, edge_index))
            x = x + x_new  # Residual connection
            x = torch.nn.functional.dropout(x, training=self.training)

        # Global pooling to get graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)

        # Generate decisions
        decision_logits = self.decision_head(graph_repr)
        adaptation_prob = torch.sigmoid(decision_logits[:, 0])  # Should adapt?
        layer_priorities = torch.nn.functional.softmax(
            decision_logits[:, 1:2], dim=-1
        )  # Which layer?
        urgency_score = torch.sigmoid(decision_logits[:, 2])  # How urgent?

        # Value estimate for RL training
        value_estimate = self.value_head(graph_repr)

        return adaptation_prob, layer_priorities, urgency_score, value_estimate

    def make_decision(
        self, model_state: ModelGraphState, layer_health: Dict[str, float]
    ) -> Optional[AdaptationDecision]:
        """
        Make an adaptation decision based on current model state.

        Args:
            model_state: Current state of the model graph
            layer_health: Health scores for each layer

        Returns:
            AdaptationDecision if adaptation is recommended, None otherwise
        """
        # Convert model state to graph representation
        node_features, edge_index = self._state_to_graph(model_state, layer_health)

        # Run inference
        with torch.no_grad():
            adapt_prob, _, urgency, _ = self.forward(node_features, edge_index)

        # Make decision based on thresholds
        should_adapt = adapt_prob.item() > self.config.adaptation_confidence_threshold

        if not should_adapt:
            return None

        # Find the layer with highest priority and poor health
        unhealthy_layers = [
            name
            for name, health in layer_health.items()
            if health < self.config.health_threshold
        ]

        if not unhealthy_layers:
            return None

        # Select target layer (pick the one with lowest health)
        target_layer = min(unhealthy_layers, key=lambda name: layer_health[name])

        return AdaptationDecision(
            layer_name=target_layer,
            adaptation_type="optimize_parameters",  # Fixed to match contract pattern
            confidence=adapt_prob.item(),
            urgency=urgency.item(),
            metadata={"reason": "health_threshold_breach"},
        )

    def _state_to_graph(
        self, model_state: ModelGraphState, layer_health: Dict[str, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert model state to graph tensors for GNN processing.

        Args:
            model_state: Current model state
            layer_health: Health scores per layer

        Returns:
            Tuple of (node_features, edge_index)
        """
        # For MVP: create simple linear graph based on layer order
        layer_names = list(layer_health.keys())
        num_layers = len(layer_names)

        # Create node features (simplified)
        node_features = []
        for layer_name in layer_names:
            health = layer_health.get(layer_name, 1.0)
            # Create basic feature vector [health, position, dummy_features...]
            features = [health, len(node_features) / max(1, num_layers - 1)]
            features.extend([0.0] * (self.config.node_feature_dim - 2))
            node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Create simple sequential edge connectivity
        edge_index = []
        for i in range(num_layers - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])  # bidirectional

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Single node graph
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return node_features, edge_index

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
        }


class PolicyTrainingState:
    """Manages training state and experience replay for policy learning."""

    def __init__(self, config: PolicyConfig):
        self.config = config
        self.experience_buffer: List[Dict[str, Any]] = []

    def add_experience(
        self,
        state: ModelGraphState,
        action: AdaptationDecision,
        reward: float,
        next_state: ModelGraphState,
    ) -> None:
        """Add an experience tuple to the replay buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "timestamp": torch.tensor(len(self.experience_buffer)),
        }

        self.experience_buffer.append(experience)

        # Maintain buffer size
        if len(self.experience_buffer) > self.config.replay_buffer_size:
            self.experience_buffer.pop(0)

    def sample_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sample a random batch of experiences."""
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self.experience_buffer) < batch_size:
            return self.experience_buffer.copy()

        indices = torch.randperm(len(self.experience_buffer))[:batch_size]
        return [self.experience_buffer[i] for i in indices]
