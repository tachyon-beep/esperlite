#!/bin/bash

# Demo script using CPU-only PyTorch to avoid long build times

set -e

echo "
🚀 Esper Demo - CPU Mode
========================

This demo runs with CPU-only PyTorch for faster setup.
"

# Ensure virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Please activate your virtual environment first:"
    echo "   source venv/bin/activate"
    exit 1
fi

# Install CPU-only PyTorch
echo "📦 Installing CPU-only dependencies..."
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html --quiet
pip install -e . --quiet

echo "✅ Dependencies installed"

# Generate credentials if needed
if [ ! -f .env.dev-demo ]; then
    echo "🔐 Generating credentials..."
    python scripts/generate-credentials.py
fi

# Start minimal infrastructure
echo "🚀 Starting infrastructure..."

# Use simplified compose
cat > docker-compose.minimal.yml << 'EOF'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: esper_dev
      POSTGRES_USER: esper_dev
      POSTGRES_PASSWORD: esper_dev_password
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass esper_dev_password
    ports:
      - "6379:6379"

  demo-ui:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./docker/nginx/html:/usr/share/nginx/html:ro
EOF

docker compose -f docker-compose.minimal.yml up -d

echo "
✅ Demo is ready!

🎯 Quick Test:
   python examples/scripts/basic_training.py

🌐 Demo UI:
   http://localhost:8080

📚 Examples:
   - Basic: examples/scripts/basic_training.py
   - Advanced: examples/scripts/custom_adaptation.py

🛑 To stop:
   docker compose -f docker-compose.minimal.yml down
"