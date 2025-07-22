#!/usr/bin/env python3
"""Basic training example for Esper morphogenetic neural networks.

This script demonstrates the simplest way to use Esper to enable
autonomous architecture evolution in a PyTorch model during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from esper import wrap


def create_dummy_dataset(num_samples=1000, input_dim=784, num_classes=10):
    """Create a dummy dataset for demonstration purposes."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def main():
    """Run basic training with morphogenetic capabilities."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define a simple feedforward model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # Wrap with morphogenetic capabilities
    print("Wrapping model with morphogenetic capabilities...")
    morphable_model = wrap(
        model,
        target_layers=[nn.Linear],  # Target Linear layers for adaptation
        seeds_per_layer=4,  # Number of adaptation seeds per layer
        device=device,
    )

    # Move to device
    morphable_model = morphable_model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(morphable_model.parameters(), lr=0.001)

    # Create dummy data
    train_dataset = create_dummy_dataset(num_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 10
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        morphable_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = morphable_model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(
                    f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")

        # Get adaptation statistics
        if hasattr(morphable_model, "get_model_stats"):
            stats = morphable_model.get_model_stats()
            print(f"  Active Seeds: {stats.get('active_seeds', 'N/A')}")
            print(f"  Total Adaptations: {stats.get('total_kernel_executions', 'N/A')}")

    print("\nTraining completed!")

    # Final model statistics
    if hasattr(morphable_model, "get_model_stats"):
        print("\nFinal Model Statistics:")
        stats = morphable_model.get_model_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
