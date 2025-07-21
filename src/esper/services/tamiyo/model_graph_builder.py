"""
Model Graph Builder for GNN Analysis

This module builds graph representations of model state for GNN processing,
converting health signals into graph structures that capture both
architectural relationships and performance characteristics.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from esper.contracts.operational import HealthSignal

logger = logging.getLogger(__name__)


@dataclass
class ModelTopology:
    """Represents the architectural topology of a model."""
    
    layer_names: List[str]
    layer_types: Dict[str, str]  # layer_name -> type
    layer_shapes: Dict[str, Tuple[int, ...]]  # layer_name -> shape
    connections: List[Tuple[str, str]]  # (source, target) pairs
    parameter_counts: Dict[str, int]  # layer_name -> param count
    
    def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a layer."""
        return {
            "type": self.layer_types.get(layer_name, "unknown"),
            "shape": self.layer_shapes.get(layer_name, ()),
            "parameters": self.parameter_counts.get(layer_name, 0),
            "connections": [
                target for source, target in self.connections 
                if source == layer_name
            ]
        }


@dataclass
class ModelGraphState:
    """Complete graph representation of model state."""
    
    graph_data: Data  # PyTorch Geometric data object
    timestamp: float
    health_signals: List[HealthSignal]
    topology: ModelTopology
    global_metrics: Dict[str, float]
    health_trends: Dict[str, List[float]]
    problematic_layers: List[str]
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.graph_data.x.size(0)
    
    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return self.graph_data.edge_index.size(1)
    
    def get_layer_health(self, layer_name: str) -> float:
        """Get average health score for a specific layer."""
        layer_id = hash(layer_name) % 1000  # Convert to numeric ID
        layer_signals = [s for s in self.health_signals if s.layer_id == layer_id]
        
        if not layer_signals:
            return 0.5  # Default neutral health
        
        return np.mean([s.health_score for s in layer_signals])


class LayerFeatureExtractor:
    """Extracts features from individual layers for graph nodes."""
    
    def __init__(self):
        self.layer_type_encoding = {
            "linear": [1, 0, 0, 0, 0],
            "conv2d": [0, 1, 0, 0, 0],
            "attention": [0, 0, 1, 0, 0],
            "normalization": [0, 0, 0, 1, 0],
            "activation": [0, 0, 0, 0, 1],
            "unknown": [0, 0, 0, 0, 0]
        }
    
    def extract_features(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        layer_signals: List[HealthSignal],
        topology: ModelTopology
    ) -> List[float]:
        """Extract comprehensive features for a layer."""
        features = []
        
        # 1. Health and performance metrics
        if layer_signals:
            avg_health = np.mean([s.health_score for s in layer_signals])
            avg_latency = np.mean([s.execution_latency for s in layer_signals])
            avg_error_rate = np.mean([s.error_count for s in layer_signals])
            variance_activation = np.mean([s.activation_variance for s in layer_signals])
            dead_neuron_ratio = np.mean([s.dead_neuron_ratio for s in layer_signals])
            correlation = np.mean([s.avg_correlation for s in layer_signals])
        else:
            avg_health = 0.5
            avg_latency = 0.0
            avg_error_rate = 0.0
            variance_activation = 0.0
            dead_neuron_ratio = 0.0
            correlation = 0.5
        
        features.extend([
            avg_health,
            min(avg_latency / 10.0, 1.0),  # Normalize to [0,1]
            min(avg_error_rate, 1.0),
            variance_activation,
            dead_neuron_ratio,
            correlation
        ])
        
        # 2. Architectural features
        layer_type = layer_info.get("type", "unknown")
        type_encoding = self.layer_type_encoding.get(layer_type, self.layer_type_encoding["unknown"])
        features.extend(type_encoding)
        
        # 3. Layer properties
        param_count = layer_info.get("parameters", 0)
        shape = layer_info.get("shape", ())
        
        features.extend([
            np.log10(param_count + 1),  # Log scale for parameters
            len(shape),  # Dimensionality
            np.prod(shape) if shape else 0,  # Total size
        ])
        
        # 4. Topological features
        in_degree = len([s for s, t in topology.connections if t == layer_name])
        out_degree = len([s for s, t in topology.connections if s == layer_name])
        
        features.extend([
            in_degree,
            out_degree,
            in_degree + out_degree  # Total degree
        ])
        
        return features


class ConnectionFeatureExtractor:
    """Extracts features for connections between layers."""
    
    def extract_edge_features(
        self,
        source_layer: str,
        target_layer: str,
        topology: ModelTopology
    ) -> List[float]:
        """Extract features for edge between two layers."""
        
        source_info = topology.get_layer_info(source_layer)
        target_info = topology.get_layer_info(target_layer)
        
        features = []
        
        # 1. Shape compatibility
        source_shape = source_info.get("shape", ())
        target_shape = target_info.get("shape", ())
        
        if source_shape and target_shape:
            # Simple compatibility measure
            shape_compatibility = 1.0 if source_shape[-1:] == target_shape[:1] else 0.0
        else:
            shape_compatibility = 0.5
        
        features.append(shape_compatibility)
        
        # 2. Parameter ratio
        source_params = source_info.get("parameters", 0)
        target_params = target_info.get("parameters", 0)
        
        if source_params + target_params > 0:
            param_ratio = source_params / (source_params + target_params)
        else:
            param_ratio = 0.5
        
        features.append(param_ratio)
        
        # 3. Connection type encoding
        source_type = source_info.get("type", "unknown")
        target_type = target_info.get("type", "unknown")
        
        # Common connection patterns
        connection_patterns = {
            ("linear", "linear"): [1, 0, 0],
            ("conv2d", "conv2d"): [0, 1, 0],
            ("attention", "linear"): [0, 0, 1],
        }
        
        pattern_key = (source_type, target_type)
        pattern_encoding = connection_patterns.get(pattern_key, [0, 0, 0])
        features.extend(pattern_encoding)
        
        return features


class PerformanceFeatureExtractor:
    """Extracts performance-related features from health signals."""
    
    def extract_temporal_features(
        self,
        health_signals: List[HealthSignal],
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Extract temporal performance features."""
        
        if not health_signals:
            return {
                "health_trend": 0.0,
                "latency_trend": 0.0,
                "error_trend": 0.0,
                "stability": 0.5
            }
        
        # Sort by timestamp
        sorted_signals = sorted(health_signals, key=lambda s: s.timestamp)
        recent_signals = sorted_signals[-window_size:]
        
        # Extract time series
        timestamps = [s.timestamp for s in recent_signals]
        health_scores = [s.health_score for s in recent_signals]
        latencies = [s.execution_latency for s in recent_signals]
        error_counts = [s.error_count for s in recent_signals]
        
        # Compute trends
        health_trend = self._compute_trend(timestamps, health_scores)
        latency_trend = self._compute_trend(timestamps, latencies)
        error_trend = self._compute_trend(timestamps, error_counts)
        
        # Compute stability (inverse of variance)
        health_stability = 1.0 / (np.var(health_scores) + 0.01)
        latency_stability = 1.0 / (np.var(latencies) + 0.01)
        
        overall_stability = min((health_stability + latency_stability) / 2, 1.0)
        
        return {
            "health_trend": health_trend,
            "latency_trend": latency_trend,
            "error_trend": error_trend,
            "stability": overall_stability,
            "signal_count": len(recent_signals),
            "time_span": timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        }
    
    def _compute_trend(self, timestamps: List[float], values: List[float]) -> float:
        """Compute trend using simple linear regression."""
        if len(timestamps) < 2:
            return 0.0
        
        # Normalize timestamps
        t_norm = np.array(timestamps) - timestamps[0]
        v_array = np.array(values)
        
        # Simple linear regression
        if np.std(t_norm) > 0:
            correlation = np.corrcoef(t_norm, v_array)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0


class ModelGraphBuilder:
    """
    Builds graph representations of model state for GNN processing.
    
    Converts health signals into graph structures that capture both
    architectural relationships and performance characteristics.
    """
    
    def __init__(self, node_feature_dim: int = 16, edge_feature_dim: int = 8):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # Feature extractors
        self.layer_extractor = LayerFeatureExtractor()
        self.connection_extractor = ConnectionFeatureExtractor()
        self.performance_extractor = PerformanceFeatureExtractor()
        
        # Cache for model topologies
        self.topology_cache = {}
        
        logger.info(
            f"Initialized ModelGraphBuilder with node_dim={node_feature_dim}, "
            f"edge_dim={edge_feature_dim}"
        )

    def build_model_graph(
        self,
        health_signals: List[HealthSignal],
        model_topology: Optional[ModelTopology] = None,
        window_size: int = 100
    ) -> ModelGraphState:
        """Build graph representation from health signals and topology."""
        
        if model_topology is None:
            model_topology = self._infer_topology_from_signals(health_signals)
        
        # Extract node features (layers/components)
        node_features = self._extract_node_features(health_signals, model_topology)
        
        # Extract edge features (connections between layers)
        edge_features, edge_index = self._extract_edge_features(model_topology)
        
        # Aggregate temporal information
        temporal_features = self.performance_extractor.extract_temporal_features(
            health_signals, window_size
        )
        
        # Identify problematic layers
        problematic_layers = self._identify_problematic_layers(health_signals)
        
        # Compute global metrics
        global_metrics = self._compute_global_metrics(health_signals)
        
        # Compute health trends
        health_trends = self._compute_health_trends(health_signals, model_topology)
        
        # Build PyTorch Geometric data object
        graph_data = Data(
            x=node_features,           # Node features [num_nodes, node_feature_dim]
            edge_index=edge_index,     # Edge connectivity [2, num_edges]
            edge_attr=edge_features,   # Edge features [num_edges, edge_feature_dim]
            
            # Additional graph-level information
            global_features=torch.tensor(
                [temporal_features["stability"], temporal_features["health_trend"]],
                dtype=torch.float32
            ),
            num_nodes=len(model_topology.layer_names)
        )
        
        return ModelGraphState(
            graph_data=graph_data,
            timestamp=time.time(),
            health_signals=health_signals,
            topology=model_topology,
            global_metrics=global_metrics,
            health_trends=health_trends,
            problematic_layers=problematic_layers
        )

    def _extract_node_features(
        self,
        health_signals: List[HealthSignal],
        topology: ModelTopology
    ) -> torch.Tensor:
        """Extract per-layer features for graph nodes."""
        
        features = []
        
        for layer_name in topology.layer_names:
            # Get health signals for this layer
            layer_id = hash(layer_name) % 1000  # Convert to numeric
            layer_signals = [s for s in health_signals if s.layer_id == layer_id]
            
            # Get layer information
            layer_info = topology.get_layer_info(layer_name)
            
            # Extract comprehensive features
            layer_features = self.layer_extractor.extract_features(
                layer_name, layer_info, layer_signals, topology
            )
            
            # Pad or truncate to target dimension
            if len(layer_features) < self.node_feature_dim:
                layer_features.extend([0.0] * (self.node_feature_dim - len(layer_features)))
            elif len(layer_features) > self.node_feature_dim:
                layer_features = layer_features[:self.node_feature_dim]
            
            features.append(layer_features)
        
        return torch.tensor(features, dtype=torch.float32)

    def _extract_edge_features(
        self,
        topology: ModelTopology
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge features and connectivity."""
        
        if not topology.connections:
            # No connections - return empty tensors
            return (
                torch.empty((0, self.edge_feature_dim), dtype=torch.float32),
                torch.empty((2, 0), dtype=torch.long)
            )
        
        # Create layer name to index mapping
        layer_to_idx = {name: i for i, name in enumerate(topology.layer_names)}
        
        edge_features = []
        edge_indices = []
        
        for source_layer, target_layer in topology.connections:
            if source_layer in layer_to_idx and target_layer in layer_to_idx:
                # Extract edge features
                edge_feat = self.connection_extractor.extract_edge_features(
                    source_layer, target_layer, topology
                )
                
                # Pad or truncate to target dimension
                if len(edge_feat) < self.edge_feature_dim:
                    edge_feat.extend([0.0] * (self.edge_feature_dim - len(edge_feat)))
                elif len(edge_feat) > self.edge_feature_dim:
                    edge_feat = edge_feat[:self.edge_feature_dim]
                
                edge_features.append(edge_feat)
                
                # Add edge indices
                source_idx = layer_to_idx[source_layer]
                target_idx = layer_to_idx[target_layer]
                edge_indices.append([source_idx, target_idx])
        
        if not edge_indices:
            return (
                torch.empty((0, self.edge_feature_dim), dtype=torch.float32),
                torch.empty((2, 0), dtype=torch.long)
            )
        
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        return edge_features_tensor, edge_index_tensor

    def _infer_topology_from_signals(
        self,
        health_signals: List[HealthSignal]
    ) -> ModelTopology:
        """Infer basic topology from health signals when explicit topology unavailable."""
        
        # Extract unique layers from signals
        layer_ids = list(set(s.layer_id for s in health_signals))
        layer_names = [f"layer_{lid}" for lid in sorted(layer_ids)]
        
        # Create basic topology
        topology = ModelTopology(
            layer_names=layer_names,
            layer_types={name: "unknown" for name in layer_names},
            layer_shapes={name: (64, 64) for name in layer_names},  # Default shape
            connections=[(layer_names[i], layer_names[i+1]) for i in range(len(layer_names)-1)],
            parameter_counts={name: 1000 for name in layer_names}  # Default param count
        )
        
        return topology

    def _identify_problematic_layers(
        self,
        health_signals: List[HealthSignal]
    ) -> List[str]:
        """Identify layers with performance issues."""
        
        layer_health = {}
        for signal in health_signals:
            layer_name = f"layer_{signal.layer_id}"
            if layer_name not in layer_health:
                layer_health[layer_name] = []
            layer_health[layer_name].append(signal.health_score)
        
        problematic = []
        for layer_name, scores in layer_health.items():
            avg_score = np.mean(scores)
            if avg_score < 0.3:  # Threshold for problematic
                problematic.append(layer_name)
        
        return problematic

    def _compute_global_metrics(
        self,
        health_signals: List[HealthSignal]
    ) -> Dict[str, float]:
        """Compute global model metrics."""
        
        if not health_signals:
            return {
                "overall_health": 0.5,
                "avg_latency": 0.0,
                "total_errors": 0.0,
                "signal_count": 0
            }
        
        return {
            "overall_health": np.mean([s.health_score for s in health_signals]),
            "avg_latency": np.mean([s.execution_latency for s in health_signals]),
            "total_errors": sum(s.error_count for s in health_signals),
            "signal_count": len(health_signals)
        }

    def _compute_health_trends(
        self,
        health_signals: List[HealthSignal],
        topology: ModelTopology
    ) -> Dict[str, List[float]]:
        """Compute health trends for each layer."""
        
        trends = {}
        
        for layer_name in topology.layer_names:
            layer_id = hash(layer_name) % 1000
            layer_signals = [s for s in health_signals if s.layer_id == layer_id]
            
            if layer_signals:
                # Sort by timestamp and get recent scores
                sorted_signals = sorted(layer_signals, key=lambda s: s.timestamp)
                recent_scores = [s.health_score for s in sorted_signals[-20:]]  # Last 20 signals
                trends[layer_name] = recent_scores
            else:
                trends[layer_name] = [0.5]  # Default neutral trend
        
        return trends

    def create_test_topology(self, num_layers: int = 5) -> ModelTopology:
        """Create test topology for development and testing."""
        
        layer_names = [f"layer_{i}" for i in range(num_layers)]
        layer_types = {
            f"layer_{i}": "linear" if i % 2 == 0 else "activation"
            for i in range(num_layers)
        }
        
        layer_shapes = {
            f"layer_{i}": (64, 64) for i in range(num_layers)
        }
        
        connections = [
            (f"layer_{i}", f"layer_{i+1}") for i in range(num_layers-1)
        ]
        
        parameter_counts = {
            f"layer_{i}": 1000 * (i + 1) for i in range(num_layers)
        }
        
        return ModelTopology(
            layer_names=layer_names,
            layer_types=layer_types,
            layer_shapes=layer_shapes,
            connections=connections,
            parameter_counts=parameter_counts
        )

    def get_graph_statistics(self, graph_state: ModelGraphState) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph state."""
        
        return {
            "num_nodes": graph_state.num_nodes,
            "num_edges": graph_state.num_edges,
            "node_feature_dim": graph_state.graph_data.x.size(1),
            "edge_feature_dim": graph_state.graph_data.edge_attr.size(1) if graph_state.graph_data.edge_attr is not None else 0,
            "problematic_layers": len(graph_state.problematic_layers),
            "health_signals_count": len(graph_state.health_signals),
            "global_metrics": graph_state.global_metrics,
            "timestamp": graph_state.timestamp
        }