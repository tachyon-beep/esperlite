#!/bin/bash

# Quick demo setup - uses CPU-only PyTorch for faster builds

set -e

echo "🚀 Esper Quick Demo Setup"
echo "========================"

# Check for .env.dev-demo
if [ ! -f .env.dev-demo ]; then
    echo "⚠️  Environment file not found. Generating credentials..."
    python scripts/generate-credentials.py
fi

# Source environment
source .env.dev-demo

# Create simplified docker-compose for demo
cat > docker/docker-compose.quick-demo.yml << 'EOF'
# Quick demo configuration - CPU only, minimal build time

services:
  # Infrastructure
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD", "redis-cli", "--pass", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Simple demo API
  demo-api:
    image: python:3.12-slim
    command: python -m http.server 8000
    working_dir: /app
    volumes:
      - ./src:/app/src
      - ./configs:/app/configs
    environment:
      PYTHONPATH: /app
    ports:
      - "0.0.0.0:8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  # Demo dashboard
  nginx:
    image: nginx:alpine
    ports:
      - "0.0.0.0:80:80"
    volumes:
      - ./docker/nginx/demo.conf:/etc/nginx/conf.d/default.conf:ro
      - ./docker/nginx/html:/usr/share/nginx/html:ro

volumes:
  postgres_data:

networks:
  default:
    name: esper-demo-network
EOF

echo "📦 Starting infrastructure services..."
docker compose -f docker/docker-compose.quick-demo.yml up -d

echo ""
echo "✅ Quick demo is running!"
echo ""
echo "🌐 Access points:"
echo "   - Demo Dashboard: http://localhost"
echo "   - API: http://localhost:8000"
echo ""
echo "📝 To run a training example:"
echo "   python train.py --quick-start cifar10"
echo ""
echo "🛑 To stop the demo:"
echo "   docker compose -f docker/docker-compose.quick-demo.yml down"