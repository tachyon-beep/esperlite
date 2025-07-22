#!/bin/bash

# Minimal demo setup - just infrastructure and a simple API

echo "
🚀 Esper Minimal Demo
====================

This sets up just the infrastructure services for testing.
"

# Ensure we have the env file
if [ ! -f .env.dev-demo ]; then
    echo "🔐 Generating credentials..."
    python scripts/generate-credentials.py
fi

# Start only infrastructure services
echo "📦 Starting infrastructure..."
docker compose --env-file .env.dev-demo -f docker/docker-compose.dev-demo.yml up -d \
    postgres redis minio grafana prometheus nginx

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check status
echo "
✅ Infrastructure is ready!

🌐 Access Points:
   - Grafana: http://localhost:3000 (admin / see .env.dev-demo)
   - MinIO: http://localhost:9001
   - Prometheus: http://localhost:9090

📊 To test locally:
   python examples/scripts/basic_training.py

🛑 To stop:
   docker compose -f docker/docker-compose.dev-demo.yml down
"