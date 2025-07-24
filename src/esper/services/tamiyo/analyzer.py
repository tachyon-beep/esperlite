"""
Model state analysis and graph construction for Tamiyo.

This module processes telemetry from KasminaLayers and constructs graph
representations that the GNN policy can analyze.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from esper.contracts.operational import HealthSignal
from esper.contracts.operational import ModelGraphState


@dataclass
class LayerNode:
    """Represents a layer in the model graph."""

    layer_name: str
    layer_type: str

    # Performance metrics
    health_score: float
    execution_latency: float
    error_count: int
    active_seeds: int
    total_seeds: int

    # Architectural properties
    input_size: int
    output_size: int
    parameter_count: int

    # Temporal dynamics
    health_trend: float  # Rate of change in health
    adaptation_history: List[str] = field(default_factory=list)


@dataclass
class GraphTopology:
    """Represents the topology and connectivity of a model."""

    nodes: Dict[str, LayerNode]
    edges: List[Tuple[str, str]]  # (source, target) pairs
    execution_order: List[str]


class ModelGraphAnalyzer:
    """
    Analyzes model state and constructs graph representations for Tamiyo.

    This class processes telemetry signals from KasminaLayers and builds
    structured graph representations that capture both topology and dynamics.
    """

    def __init__(self, health_history_window: int = 10):
        self.health_history_window = health_history_window
        self.health_history: Dict[str, List[float]] = defaultdict(list)
        self.performance_baselines: Dict[str, float] = {}
        self.topology_cache: Optional[GraphTopology] = None
        self.last_analysis_time: float = 0

    def analyze_model_state(
        self,
        health_signals: Dict[str, HealthSignal],
        model_topology: Optional[Dict[str, Any]] = None,
    ) -> ModelGraphState:
        """
        Analyze current model state and construct graph representation.

        Args:
            health_signals: Dictionary mapping layer names to health signals
            model_topology: Optional topology information for the model

        Returns:
            ModelGraphState representing the current model condition
        """
        current_time = time.time()

        # Update health history for trend analysis
        self._update_health_history(health_signals)

        # Build or update topology
        if model_topology is not None or self.topology_cache is None:
            self.topology_cache = self._build_topology(health_signals, model_topology)

        # Calculate health trends
        health_trends = self._calculate_health_trends()

        # Identify problematic layers
        problematic_layers = self._identify_problematic_layers(health_signals)

        # Assess overall system health
        overall_health = self._calculate_overall_health(health_signals)

        # Create graph state
        graph_state = ModelGraphState(
            topology=self.topology_cache,
            health_signals=health_signals,
            health_trends=health_trends,
            problematic_layers=problematic_layers,
            overall_health=overall_health,
            analysis_timestamp=current_time,
        )

        self.last_analysis_time = current_time
        return graph_state

    def _update_health_history(self, health_signals: Dict[str, HealthSignal]) -> None:
        """Update the rolling health history for each layer."""
        for layer_name, signal in health_signals.items():
            history = self.health_history[layer_name]
            history.append(signal.health_score)

            # Maintain window size
            if len(history) > self.health_history_window:
                history.pop(0)

    def _build_topology(
        self,
        health_signals: Dict[str, HealthSignal],
        model_topology: Optional[Dict[str, Any]] = None,
    ) -> GraphTopology:
        """
        Build the graph topology from available information.

        Args:
            health_signals: Current health signals
            model_topology: Optional explicit topology information

        Returns:
            GraphTopology representing the model structure
        """
        nodes = {}

        # Create nodes from health signals
        for layer_name, signal in health_signals.items():
            # Calculate health trend
            health_trend = self._calculate_layer_trend(layer_name)

            # Estimate layer properties (simplified for MVP)
            layer_type = self._infer_layer_type(layer_name)

            node = LayerNode(
                layer_name=layer_name,
                layer_type=layer_type,
                health_score=signal.health_score,
                execution_latency=signal.execution_latency,
                error_count=signal.error_count,
                active_seeds=signal.active_seeds,
                total_seeds=signal.total_seeds,
                input_size=self._estimate_layer_size(layer_name, "input"),
                output_size=self._estimate_layer_size(layer_name, "output"),
                parameter_count=self._estimate_parameter_count(layer_name),
                health_trend=health_trend,
            )
            nodes[layer_name] = node

        # Infer edges (simplified sequential assumption for MVP)
        edges = self._infer_edges(list(nodes.keys()), model_topology)

        # Determine execution order
        execution_order = self._determine_execution_order(edges, list(nodes.keys()))

        return GraphTopology(nodes=nodes, edges=edges, execution_order=execution_order)

    def _calculate_health_trends(self) -> Dict[str, float]:
        """Calculate health trends for all layers."""
        trends = {}
        for layer_name in self.health_history:
            trends[layer_name] = self._calculate_layer_trend(layer_name)
        return trends

    def _calculate_layer_trend(self, layer_name: str) -> float:
        """Calculate the health trend for a specific layer."""
        history = self.health_history[layer_name]
        if len(history) < 2:
            return 0.0

        # Simple linear trend calculation
        recent_values = history[-min(5, len(history)) :]
        if len(recent_values) < 2:
            return 0.0

        # Calculate slope
        x = np.arange(len(recent_values))
        y = np.array(recent_values)

        if np.var(x) == 0:
            return 0.0

        slope = np.cov(x, y)[0, 1] / np.var(x)
        return float(slope)

    def _identify_problematic_layers(
        self, health_signals: Dict[str, HealthSignal]
    ) -> Set[str]:
        """Identify layers that are experiencing problems."""
        problematic = set()

        for layer_name, signal in health_signals.items():
            # Health-based criteria
            if signal.health_score < 0.3:
                problematic.add(layer_name)
                continue

            # Error-based criteria
            if signal.error_count > 5:
                problematic.add(layer_name)
                continue

            # Trend-based criteria
            trend = self._calculate_layer_trend(layer_name)
            if trend < -0.1:  # Rapidly declining health
                problematic.add(layer_name)
                continue

            # Latency-based criteria (if we have baselines)
            baseline = self.performance_baselines.get(layer_name)
            if baseline is not None and signal.execution_latency > baseline * 2.0:
                problematic.add(layer_name)

        return problematic

    def _calculate_overall_health(
        self, health_signals: Dict[str, HealthSignal]
    ) -> float:
        """Calculate overall system health score."""
        if not health_signals:
            return 1.0

        # Weighted average based on layer importance
        total_weight = 0
        weighted_sum = 0

        for _, signal in health_signals.items():
            # For MVP, treat all layers equally
            weight = 1.0
            weighted_sum += signal.health_score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 1.0

    def _infer_layer_type(self, layer_name: str) -> str:
        """Infer the type of layer from its name."""
        layer_name_lower = layer_name.lower()

        if "conv" in layer_name_lower:
            return "convolutional"
        elif "linear" in layer_name_lower or "fc" in layer_name_lower:
            return "linear"
        elif "attention" in layer_name_lower or "attn" in layer_name_lower:
            return "attention"
        elif "norm" in layer_name_lower or "bn" in layer_name_lower:
            return "normalization"
        elif "dropout" in layer_name_lower:
            return "dropout"
        elif "relu" in layer_name_lower or "activation" in layer_name_lower:
            return "activation"
        else:
            return "unknown"

    def _estimate_layer_size(self, layer_name: str, size_type: str) -> int:
        """Estimate the input/output size of a layer."""
        # Default reasonable sizes based on common architectures
        layer_type = self._infer_layer_type(layer_name)
        
        if layer_type == "linear":
            # Common FC layer sizes
            if "fc1" in layer_name or "classifier" in layer_name:
                return 512 if size_type == "output" else 2048
            elif "fc2" in layer_name or "output" in layer_name:
                return 10 if size_type == "output" else 512
            else:
                return 256  # Default for other linear layers
        elif layer_type == "convolutional":
            # Common conv layer channel sizes
            if "conv1" in layer_name:
                return 64 if size_type == "output" else 3
            elif "conv2" in layer_name:
                return 128 if size_type == "output" else 64
            elif "conv3" in layer_name:
                return 256 if size_type == "output" else 128
            else:
                return 128  # Default
        else:
            return 128  # Default for unknown types

    def _estimate_parameter_count(self, layer_name: str) -> int:
        """Estimate the parameter count of a layer."""
        layer_type = self._infer_layer_type(layer_name)
        
        if layer_type == "linear":
            # Estimate based on common layer configurations
            input_size = self._estimate_layer_size(layer_name, "input")
            output_size = self._estimate_layer_size(layer_name, "output")
            # params = input * output + bias
            return input_size * output_size + output_size
        elif layer_type == "convolutional":
            # Assume 3x3 kernels as default
            kernel_size = 3
            in_channels = self._estimate_layer_size(layer_name, "input")
            out_channels = self._estimate_layer_size(layer_name, "output")
            # params = kernel_h * kernel_w * in_channels * out_channels + bias
            return kernel_size * kernel_size * in_channels * out_channels + out_channels
        elif layer_type == "normalization":
            # BatchNorm/LayerNorm have 2 * num_features parameters (gamma, beta)
            features = self._estimate_layer_size(layer_name, "output")
            return 2 * features
        else:
            # Default for unknown types
            return 1024

    def _infer_edges(
        self, layer_names: List[str], model_topology: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, str]]:
        """
        Infer connectivity edges between layers.

        For MVP, assumes sequential connectivity.
        In a full implementation, this would analyze the actual model graph.
        """
        if model_topology and "edges" in model_topology:
            return model_topology["edges"]

        # Simple sequential assumption
        edges = []
        for i in range(len(layer_names) - 1):
            edges.append((layer_names[i], layer_names[i + 1]))

        return edges

    def _determine_execution_order(
        self, edges: List[Tuple[str, str]], layer_names: List[str]
    ) -> List[str]:
        """
        Determine the execution order of layers.

        For MVP, returns the original order.
        In a full implementation, this would perform topological sorting.
        """
        if not edges:
            return layer_names

        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all nodes
        for name in layer_names:
            in_degree[name] = 0

        # Build graph
        for source, target in edges:
            graph[source].append(target)
            in_degree[target] += 1

        # Topological sort
        queue = [name for name in layer_names if in_degree[name] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If not all nodes were processed, fall back to original order
        if len(result) != len(layer_names):
            return layer_names

        return result

    def update_performance_baseline(self, layer_name: str, latency: float) -> None:
        """Update the performance baseline for a layer."""
        current_baseline = self.performance_baselines.get(layer_name, latency)
        # Exponential moving average
        alpha = 0.1
        self.performance_baselines[layer_name] = (
            alpha * latency + (1 - alpha) * current_baseline
        )
