# Esper Tech Demo Quick Start Guide

## 🚀 Quick Start (5 minutes)

1. **Generate secure credentials** (one-time setup):
   ```bash
   python scripts/generate-credentials.py
   ```

2. **Configure firewall for external access** (if needed):
   ```bash
   sudo ./scripts/configure-firewall.sh
   # Select option 1 for full demo access
   ```

3. **Start the demo environment**:
   ```bash
   ./scripts/start-demo.sh
   ```

4. **Access the platform**:
   - 🌐 Demo Dashboard: http://localhost or http://YOUR-SERVER-IP
   - 📊 Grafana Metrics: http://localhost:3000 or http://YOUR-SERVER-IP:3000
   - 🗄️ MinIO Console: http://localhost:9001 or http://YOUR-SERVER-IP:9001

## 🎯 Demo Scenarios

### Scenario 1: Basic Morphogenetic Training

```bash
# Start a training session with morphogenetic adaptations
curl -X POST http://localhost/api/tolaria/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "config": "demo/tolaria.yaml",
    "experiment_name": "demo_morphogenetic_cifar10"
  }'
```

### Scenario 2: Monitor Real-time Adaptations

1. Open Grafana dashboard: http://localhost:3000
2. Navigate to "Esper Platform Overview"
3. Watch real-time metrics:
   - Active morphogenetic seeds
   - Kernel execution rate
   - GNN decision latency
   - Adaptation success rate

### Scenario 3: Inspect Model Evolution

```python
# Example: Query adaptation history
import requests

# Get model adaptation history
response = requests.get("http://localhost/api/tamiyo/adaptations/history")
adaptations = response.json()

for adaptation in adaptations[-5:]:  # Last 5 adaptations
    print(f"Layer: {adaptation['layer_name']}")
    print(f"Decision: {adaptation['decision_type']}")
    print(f"Confidence: {adaptation['confidence']:.2%}")
    print(f"Impact: {adaptation['performance_delta']:.4f}")
    print("---")
```

## 📋 Service Health Check

```bash
# Check all services
curl http://localhost/api/urza/health
curl http://localhost/api/tamiyo/health
curl http://localhost/api/tolaria/health
```

## 🔧 Useful Commands

### View Logs
```bash
# All services
docker-compose -f docker/docker-compose.dev-demo.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.dev-demo.yml logs -f tamiyo
```

### Access Service Shell
```bash
# Tamiyo service
docker exec -it esper-demo-tamiyo /bin/bash

# Check GPU availability
docker exec -it esper-demo-tamiyo python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Performance Monitoring
```bash
# Real-time resource usage
docker stats

# Database connections
docker exec -it esper-demo-postgres psql -U esper_dev -d esper_dev -c "SELECT count(*) FROM pg_stat_activity;"
```

## 🎨 Demo Features to Showcase

1. **Dynamic Architecture Evolution**
   - Models adapt their structure during training
   - Real-time kernel loading and execution
   - Alpha blending for smooth transitions

2. **GNN-based Intelligence**
   - Graph neural network analyzes model state
   - Strategic decisions with confidence scores
   - Safety regularization prevents dangerous adaptations

3. **Production-Grade Infrastructure**
   - Comprehensive error recovery
   - Circuit breakers for resilience
   - Distributed caching with Redis
   - Persistent storage with PostgreSQL/MinIO

4. **Observability**
   - Real-time metrics with Prometheus/Grafana
   - Distributed tracing capabilities
   - Health monitoring across all services

## ⚠️ Demo Limitations

- CPU-only by default (change `DEVICE=cuda` in `.env.dev-demo` for GPU)
- Limited to small models for quick demonstrations
- Synthetic kernel adaptations (Phase 3 Karn not implemented)

## 🛑 Stopping the Demo

```bash
# Stop all services
docker-compose -f docker/docker-compose.dev-demo.yml down

# Reset to clean state (removes all data)
./scripts/reset-demo.sh
```

## 📚 Next Steps

1. Review the [full documentation](docs/project/ai/LLM_CODEBASE_GUIDE.md)
2. Explore the [API documentation](http://localhost/api/urza/docs)
3. Check the [deployment guide](docs/deployment/README.md) for production setup

## 🆘 Troubleshooting

### Services not starting
```bash
# Check logs
docker-compose -f docker/docker-compose.dev-demo.yml logs [service-name]

# Verify credentials were generated
ls -la .env.dev-demo
```

### Port conflicts
```bash
# Check what's using the ports
sudo lsof -i :80    # nginx
sudo lsof -i :8000  # urza
sudo lsof -i :8001  # tamiyo
sudo lsof -i :3000  # grafana
```

### Memory issues
```bash
# Increase Docker memory allocation
# Docker Desktop: Preferences > Resources > Memory

# Or reduce service memory limits in docker-compose.dev-demo.yml
```

---

For questions or issues, please check the [GitHub repository](https://github.com/esper/esperlite).