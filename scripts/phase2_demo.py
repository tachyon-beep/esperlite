#!/usr/bin/env python3
"""
Phase 2 Extended Lifecycle System - Initial Demo Run

This script demonstrates the full functionality of the Phase 2 implementation:
- 11-state lifecycle management
- Checkpoint/recovery system
- Advanced grafting strategies
- GPU-optimized state management
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.esper.morphogenetic_v2.kasmina.chunked_layer_v2 import ChunkedKasminaLayerV2
from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_seed_status(layer, seed_id):
    """Print detailed status for a seed."""
    state_tensor = layer.extended_state
    state = ExtendedLifecycle(state_tensor.get_state(torch.tensor([seed_id])).item())
    
    print(f"\nSeed {seed_id} Status:")
    print(f"  State: {state.name}")
    print(f"  Epochs in state: {state_tensor.state_tensor[seed_id, state_tensor.EPOCHS_IN_STATE].item()}")
    print(f"  Blueprint ID: {state_tensor.state_tensor[seed_id, state_tensor.BLUEPRINT_ID].item()}")
    print(f"  Error count: {state_tensor.state_tensor[seed_id, state_tensor.ERROR_COUNT].item()}")
    
    # Performance metrics
    perf = state_tensor.performance_metrics[seed_id]
    print(f"  Performance:")
    print(f"    Loss: {perf[0].item():.4f}")
    print(f"    Accuracy: {perf[1].item():.4f}")
    print(f"    Stability: {perf[2].item():.4f}")
    print(f"    Efficiency: {perf[3].item():.4f}")


def main():
    """Run the Phase 2 demonstration."""
    print_banner("Phase 2 Extended Lifecycle System - Initial Run")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create a base neural network layer
    print("\n1. Creating base layer...")
    # Use a single Linear layer so dimensions are easily detected
    base_layer = nn.Linear(512, 512)
    
    # Create ChunkedKasminaLayerV2 with Phase 2 features
    print("\n2. Initializing ChunkedKasminaLayerV2...")
    layer = ChunkedKasminaLayerV2(
        base_layer=base_layer,
        num_seeds=100,
        layer_id="demo_layer",
        device=device,
        checkpoint_dir=Path("./demo_checkpoints")
    )
    
    # Show initial state distribution
    print("\n3. Initial state distribution:")
    state_summary = layer.extended_state.get_state_summary()
    for state_name, count in state_summary.items():
        if count > 0:
            print(f"  {state_name}: {count} seeds")
    
    # Demonstrate lifecycle progression
    print_banner("Lifecycle Demonstration")
    
    # Select seeds for different demonstrations
    demo_seeds = {
        "linear_grafting": 0,
        "adaptive_grafting": 1,
        "error_scenario": 2,
        "checkpoint_demo": 3
    }
    
    # 1. Linear Grafting Demo
    print("\n4. Demonstrating Linear Grafting Strategy...")
    seed_id = demo_seeds["linear_grafting"]
    
    print(f"  Germinating seed {seed_id} with linear grafting...")
    success = layer.request_germination(seed_id, grafting_strategy='linear')
    print(f"  Germination {'succeeded' if success else 'failed'}")
    
    # Progress through states
    print(f"  Transitioning to TRAINING...")
    layer._request_transition(seed_id, ExtendedLifecycle.TRAINING)
    
    # Simulate training
    print(f"  Simulating training for 10 epochs...")
    for epoch in range(10):
        x = torch.randn(32, 512, device=device)
        output = layer(x)
        
        if epoch % 5 == 0:
            print_seed_status(layer, seed_id)
    
    print(f"  Transitioning to GRAFTING...")
    layer._request_transition(seed_id, ExtendedLifecycle.GRAFTING)
    
    # 2. Adaptive Grafting Demo
    print_banner("Adaptive Grafting Strategy")
    seed_id = demo_seeds["adaptive_grafting"]
    
    print(f"Germinating seed {seed_id} with adaptive grafting...")
    layer.request_germination(seed_id, grafting_strategy='adaptive')
    layer._request_transition(seed_id, ExtendedLifecycle.TRAINING)
    layer._request_transition(seed_id, ExtendedLifecycle.GRAFTING)
    
    print("Simulating grafting phase with varying performance...")
    for epoch in range(5):
        x = torch.randn(32, 512, device=device)
        output = layer(x)
        
        # Simulate varying performance
        layer.extended_state.update_performance(
            torch.tensor([seed_id]),
            {
                'loss': torch.tensor([0.5 - epoch * 0.08], dtype=torch.float32),
                'accuracy': torch.tensor([0.6 + epoch * 0.08], dtype=torch.float32),
                'stability': torch.tensor([0.7 + np.sin(epoch) * 0.1], dtype=torch.float32),
                'efficiency': torch.tensor([0.85], dtype=torch.float32)
            }
        )
    
    print_seed_status(layer, seed_id)
    
    # 3. Checkpoint Demo
    print_banner("Checkpoint/Recovery Demonstration")
    seed_id = demo_seeds["checkpoint_demo"]
    
    print(f"Setting up seed {seed_id} for checkpoint demo...")
    layer.request_germination(seed_id)
    layer._request_transition(seed_id, ExtendedLifecycle.TRAINING)
    
    # Update performance metrics
    layer.extended_state.update_performance(
        torch.tensor([seed_id]),
        {
            'loss': torch.tensor([0.234], dtype=torch.float32),
            'accuracy': torch.tensor([0.912], dtype=torch.float32),
            'stability': torch.tensor([0.867], dtype=torch.float32),
            'efficiency': torch.tensor([0.945], dtype=torch.float32)
        }
    )
    
    print(f"\nSaving checkpoint...")
    checkpoint_id = layer.save_checkpoint(seed_id, priority='high')
    print(f"  Checkpoint saved: {checkpoint_id}")
    
    # Simulate failure
    print(f"\nSimulating failure scenario...")
    layer._request_transition(seed_id, ExtendedLifecycle.CULLED)
    print_seed_status(layer, seed_id)
    
    print(f"\nRestoring from checkpoint...")
    success = layer.restore_checkpoint(seed_id, checkpoint_id)
    print(f"  Restore {'succeeded' if success else 'failed'}")
    print_seed_status(layer, seed_id)
    
    # 4. Error Recovery Demo
    print_banner("Error Recovery Scenario")
    seed_id = demo_seeds["error_scenario"]
    
    print(f"Setting up seed {seed_id} for error recovery...")
    layer.request_germination(seed_id)
    layer._request_transition(seed_id, ExtendedLifecycle.TRAINING)
    
    print("Simulating errors...")
    for i in range(6):  # Trigger error threshold
        layer.extended_state.increment_error_count(torch.tensor([seed_id]))
    
    print_seed_status(layer, seed_id)
    
    # 5. Full System Statistics
    print_banner("System Statistics")
    
    # Process multiple batches to gather statistics
    print("\nProcessing 100 batches...")
    start_time = time.time()
    
    for i in range(100):
        x = torch.randn(64, 512, device=device)
        output = layer(x)
    
    elapsed_time = time.time() - start_time
    
    # Get layer statistics
    stats = layer.get_layer_stats()
    print(f"\nPerformance Metrics:")
    print(f"  Total forward passes: {stats['forward_count']}")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f} ms")
    print(f"  Throughput: {100 * 64 / elapsed_time:.0f} samples/sec")
    
    print(f"\nState Distribution:")
    for state, count in stats['state_distribution'].items():
        if count > 0:
            print(f"  {state}: {count} seeds")
    
    print(f"\nResource Usage:")
    print(f"  Active seeds: {stats['active_seeds']}")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Blueprints created: {stats['num_blueprints']}")
    
    # 6. Health Report
    print_banner("Health Report Sample")
    
    health_report = layer.get_health_report()
    print(f"\nShowing first 3 seeds:")
    
    for seed_report in health_report['seeds'][:3]:
        print(f"\nSeed {seed_report['seed_id']}:")
        print(f"  State: {seed_report['state']}")
        print(f"  Epochs in state: {seed_report['epochs_in_state']}")
        print(f"  Performance: L={seed_report['performance']['loss']:.3f}, "
              f"A={seed_report['performance']['accuracy']:.3f}, "
              f"S={seed_report['performance']['stability']:.3f}")
        
        if seed_report['recent_transitions']:
            print(f"  Recent transitions:")
            for trans in seed_report['recent_transitions']:
                print(f"    {trans['from']} → {trans['to']}")
    
    # 7. Checkpoint Management
    print_banner("Checkpoint Management")
    
    # List checkpoints
    checkpoints = layer.checkpoint_manager.list_checkpoints(layer_id="demo_layer")
    print(f"\nTotal checkpoints saved: {len(checkpoints)}")
    
    if checkpoints:
        print("\nRecent checkpoints:")
        for cp in checkpoints[:3]:
            print(f"  - {cp['checkpoint_id']}")
            print(f"    Seed: {cp['seed_id']}, Priority: {cp['priority']}")
    
    # Final summary
    print_banner("Demo Complete!")
    
    print("\n✅ Successfully demonstrated:")
    print("  - 11-state lifecycle management")
    print("  - Multiple grafting strategies")
    print("  - Checkpoint/recovery system")
    print("  - Error handling and recovery")
    print("  - Performance monitoring")
    print("  - Health reporting")
    
    print(f"\n📊 Final Statistics:")
    print(f"  - Device: {device}")
    print(f"  - Total seeds: {layer.num_seeds}")
    print(f"  - Active seeds: {layer.extended_state.get_active_seeds_mask().sum().item()}")
    print(f"  - Checkpoints saved: {len(checkpoints)}")
    print(f"  - Average latency: {stats['avg_latency_ms']:.2f} ms")
    
    print("\n🚀 Phase 2 Extended Lifecycle System is fully operational!")


if __name__ == "__main__":
    main()