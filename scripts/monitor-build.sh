#!/bin/bash

# Monitor Docker build progress and start services when ready

echo "
🏗️  Building Esper GPU-Enabled Demo
===================================

This will take several minutes as we're downloading full GPU support...
"

# Monitor build
while true; do
    if docker compose --env-file .env.dev-demo -f docker/docker-compose.dev-demo.yml ps 2>/dev/null | grep -q "docker-"; then
        echo "✅ Build appears to be complete!"
        break
    fi
    echo -n "."
    sleep 5
done

echo "
🚀 Starting services...
"

# Start all services
docker compose --env-file .env.dev-demo -f docker/docker-compose.dev-demo.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check status
docker compose --env-file .env.dev-demo -f docker/docker-compose.dev-demo.yml ps

echo "
✅ Demo is ready!

🌐 Access Points:
   - Demo Dashboard: http://localhost
   - Grafana: http://localhost:3000
   - MinIO Console: http://localhost:9001
   - Prometheus: http://localhost:9090
   - Urza API: http://localhost:8000
   - Tamiyo API: http://localhost:8001
   - Tolaria API: http://localhost:8080

🧪 Test GPU Support:
   docker exec esper-demo-tamiyo python -c 'import torch; print(f\"GPU Available: {torch.cuda.is_available()}\")'

📊 Run Example:
   python examples/scripts/custom_adaptation.py

🛑 To stop:
   docker compose -f docker/docker-compose.dev-demo.yml down
"