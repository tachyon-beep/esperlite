#!/bin/bash
# Install acceleration dependencies after base dependencies are installed

# This script handles the installation of torch-scatter which requires
# PyTorch to be already installed

set -e

echo "Installing torch-scatter for GNN acceleration..."

# Check if PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch must be installed first"
    exit 1
fi

# Get PyTorch version and CUDA version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print('cpu' if not torch.cuda.is_available() else torch.version.cuda.replace('.', ''))")

echo "Detected PyTorch version: $TORCH_VERSION"
echo "Detected CUDA version: $CUDA_VERSION"

# Install torch-scatter
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
else
    pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
fi

echo "torch-scatter installation complete!"