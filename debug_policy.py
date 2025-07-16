#!/usr/bin/env python3
"""Debug script to understand why the policy test is failing."""

import sys
from pathlib import Path
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from unittest.mock import Mock

def debug_policy_decision():
    """Debug the policy decision making process."""
    
    # Create policy with default config
    config = PolicyConfig()
    policy = TamiyoPolicyGNN(config)
    
    print(f"Policy config:")
    print(f"  health_threshold: {config.health_threshold}")
    print(f"  adaptation_confidence_threshold: {config.adaptation_confidence_threshold}")
    print()
    
    # Mock model state (not used by _state_to_graph)
    mock_state = Mock()
    layer_health = {"layer1": 0.2, "layer2": 0.8}  # layer1 is unhealthy
    
    print(f"Layer health: {layer_health}")
    print(f"Unhealthy layers (< {config.health_threshold}): {[name for name, health in layer_health.items() if health < config.health_threshold]}")
    print()
    
    # Convert to graph representation
    node_features, edge_index = policy._state_to_graph(mock_state, layer_health)
    print(f"Node features shape: {node_features.shape}")
    print(f"Node features:\n{node_features}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index:\n{edge_index}")
    print()
    
    # Run forward pass
    with torch.no_grad():
        adapt_prob, layer_priorities, urgency, value = policy.forward(node_features, edge_index)
    
    print(f"Forward pass outputs:")
    print(f"  adapt_prob: {adapt_prob.item():.4f}")
    print(f"  layer_priorities: {layer_priorities}")
    print(f"  urgency: {urgency.item():.4f}")
    print(f"  value: {value.item():.4f}")
    print()
    
    # Check decision logic
    should_adapt = adapt_prob.item() > config.adaptation_confidence_threshold
    print(f"Should adapt? {should_adapt} (prob {adapt_prob.item():.4f} > threshold {config.adaptation_confidence_threshold})")
    
    if should_adapt:
        unhealthy_layers = [
            name for name, health in layer_health.items() 
            if health < config.health_threshold
        ]
        print(f"Unhealthy layers: {unhealthy_layers}")
        
        if unhealthy_layers:
            target_layer = min(unhealthy_layers, key=lambda name: layer_health[name])
            print(f"Target layer: {target_layer}")
        else:
            print("No unhealthy layers found!")
    else:
        print("Adaptation confidence too low, no decision made")
    
    print()
    
    # Try the actual make_decision method
    decision = policy.make_decision(mock_state, layer_health)
    print(f"Final decision: {decision}")
    
    return decision

if __name__ == "__main__":
    debug_policy_decision()
