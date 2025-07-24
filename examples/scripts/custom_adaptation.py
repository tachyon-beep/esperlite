#!/usr/bin/env python3
"""Custom adaptation example for Esper morphogenetic neural networks.

This script demonstrates advanced usage of Esper, including:
- Custom layer targeting strategies
- Monitoring adaptation events
- Custom adaptation policies
- Performance optimization techniques
"""

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from esper import wrap


class CustomModel(nn.Module):
    """A custom model with named layers for targeted adaptation."""

    def __init__(self, input_dim=784, hidden_dims=None, num_classes=10):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Build encoder layers
        self.encoder = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, hidden_dim))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output layer
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        """Forward pass through the model."""
        # Pass through encoder
        for layer in self.encoder:
            x = layer(x)

        # Classification
        return self.classifier(x)


def custom_layer_selector(module: nn.Module, name: str) -> bool:
    """Custom function to select which layers to make morphable.

    Args:
        module: The module to check
        name: The name of the module

    Returns:
        bool: Whether to make this layer morphable
        module: The PyTorch module to check
        name: The name of the module in the model

    Returns:
        bool: True if the layer should be made morphable
    """
    # Only target Linear layers in the encoder, not the classifier
    if isinstance(module, nn.Linear) and "encoder" in name:
        print(f"  Targeting layer: {name}")
        return True
    return False


def create_datasets(num_train=10000, num_val=2000, input_dim=784, num_classes=10):
    """Create training and validation datasets."""
    # Training data
    x_train = torch.randn(num_train, input_dim)
    y_train = torch.randint(0, num_classes, (num_train,))
    train_dataset = TensorDataset(x_train, y_train)

    # Validation data
    x_val = torch.randn(num_val, input_dim)
    y_val = torch.randint(0, num_classes, (num_val,))
    val_dataset = TensorDataset(x_val, y_val)

    return train_dataset, val_dataset


def validate(model, val_loader, criterion, device):
    """Validate the model and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    """Run training with custom adaptation configuration."""
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = CustomModel(input_dim=784, hidden_dims=[512, 256, 128], num_classes=10)

    # Print model structure
    print("\nModel Architecture:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {module}")

    # Wrap with custom morphogenetic configuration
    print("\nApplying morphogenetic capabilities with custom configuration...")
    morphable_model = wrap(
        model,
        target_layers=custom_layer_selector,  # Use custom selector function
        seeds_per_layer=8,  # More seeds for experimentation
    )

    morphable_model = morphable_model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(morphable_model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

    # Create datasets
    train_dataset, val_dataset = create_datasets()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    # Training loop with validation
    num_epochs = 20
    best_val_acc = 0.0
    adaptation_history = []

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Training phase
        morphable_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
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

        # Training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation phase
        val_loss, val_acc = validate(morphable_model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("  New best validation accuracy!")

        # Get and display adaptation statistics
        if hasattr(morphable_model, "get_model_stats"):
            stats = morphable_model.get_model_stats()
            print("  Adaptation Stats:")
            print(f"    Active Seeds: {stats.get('active_seeds', 'N/A')}")
            print(
                f"    Total Kernel Executions: {stats.get('total_kernel_executions', 'N/A')}"
            )
            print(
                f"    Average Kernel Performance: {stats.get('avg_kernel_performance', 'N/A')}"
            )

            # Store adaptation history
            adaptation_history.append(
                {
                    "epoch": epoch + 1,
                    "active_seeds": stats.get("active_seeds", 0),
                    "kernel_executions": stats.get("total_kernel_executions", 0),
                }
            )

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    # Analyze adaptation patterns
    if adaptation_history:
        print("\nAdaptation Summary:")
        total_adaptations = sum(h["kernel_executions"] for h in adaptation_history)
        print(f"  Total adaptations across training: {total_adaptations}")

        # Find most active epoch
        most_active = max(adaptation_history, key=lambda x: x["kernel_executions"])
        print(
            f"  Most active epoch: {most_active['epoch']} "
            f"({most_active['kernel_executions']} executions)"
        )

    # Display final model analysis
    if hasattr(morphable_model, "get_layer_stats"):
        print("\nPer-Layer Adaptation Analysis:")
        layer_stats = morphable_model.get_layer_stats()
        for layer_name, stats in layer_stats.items():
            print(f"  {layer_name}:")
            print(f"    Seeds: {stats.get('seeds', 'N/A')}")
            print(f"    Executions: {stats.get('executions', 'N/A')}")
            print(f"    Performance: {stats.get('performance', 'N/A')}")


if __name__ == "__main__":
    main()
