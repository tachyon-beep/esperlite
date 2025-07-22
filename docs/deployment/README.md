# Esper Production Deployment Guide

This guide covers deploying the Esper Morphogenetic Training Platform to production environments.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Options](#deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Infrastructure Requirements](#infrastructure-requirements)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Troubleshooting](#troubleshooting)

## Overview

Esper consists of multiple microservices that work together to provide morphogenetic neural network training capabilities:

- **Tolaria**: Training orchestrator
- **Tamiyo**: Strategic controller with GNN-based decision making
- **Urza**: Central library for kernel artifacts
- **Infrastructure**: PostgreSQL, Redis, MinIO/S3

## Prerequisites

### Software Requirements

- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- kubectl and kustomize
- Helm 3.0+ (optional)

### Hardware Requirements

#### Minimum (Development)
- 16GB RAM
- 4 CPU cores
- 50GB storage
- CPU-only inference

#### Recommended (Production)
- 64GB RAM
- 16 CPU cores
- 500GB SSD storage
- NVIDIA GPU with 16GB+ VRAM (for Tamiyo GNN acceleration)

## Deployment Options

### 1. Docker Compose (Small-Scale Production)

Best for:
- Single-node deployments
- Small teams
- Development/staging environments

```bash
# Clone the repository
git clone https://github.com/esper/esperlite.git
cd esperlite

# Copy and configure environment variables
cp .env.production .env
# Edit .env with your production values

# Build and start services
docker-compose -f docker/docker-compose.prod.yml up -d

# Check service health
docker-compose -f docker/docker-compose.prod.yml ps
docker-compose -f docker/docker-compose.prod.yml logs -f
```

### 2. Kubernetes (Large-Scale Production)

Best for:
- Multi-node clusters
- High availability requirements
- Auto-scaling needs

## Docker Deployment

### Configuration

1. **Environment Variables**

Create a `.env` file with production values:

```bash
# Database
POSTGRES_PASSWORD=<secure-password>

# Redis
REDIS_PASSWORD=<secure-password>

# MinIO/S3
MINIO_ROOT_USER=<access-key>
MINIO_ROOT_PASSWORD=<secret-key>

# Monitoring
GRAFANA_PASSWORD=<admin-password>

# Services
LOG_LEVEL=INFO
DEVICE=cuda  # or cpu
```

2. **Build Images**

```bash
# Build all service images
docker-compose -f docker/docker-compose.prod.yml build

# Or build individually
docker build -f docker/Dockerfile.tamiyo -t esper/tamiyo:latest .
docker build -f docker/Dockerfile.tolaria -t esper/tolaria:latest .
docker build -f docker/Dockerfile.urza -t esper/urza:latest .
```

3. **Start Services**

```bash
# Start infrastructure first
docker-compose -f docker/docker-compose.prod.yml up -d postgres redis minio

# Wait for infrastructure to be ready
sleep 30

# Start Esper services
docker-compose -f docker/docker-compose.prod.yml up -d urza tamiyo tolaria

# Start monitoring (optional)
docker-compose -f docker/docker-compose.prod.yml up -d prometheus grafana
```

### Data Persistence

Ensure proper volume management:

```bash
# Backup volumes
docker run --rm -v esper_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data

# Restore volumes
docker run --rm -v esper_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /
```

## Kubernetes Deployment

### Prerequisites

1. **Install Required Tools**

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

2. **Prepare Cluster**

```bash
# Create namespace
kubectl create namespace esper

# Install NGINX ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Deployment Steps

1. **Configure Secrets**

```bash
# Create secrets file
cat > k8s/overlays/production/secrets.env << EOF
postgres-password=<secure-password>
redis-password=<secure-password>
minio-access-key=<access-key>
minio-secret-key=<secret-key>
EOF

# Apply base configuration
kubectl apply -k k8s/base/

# Apply production overlay
kubectl apply -k k8s/overlays/production/
```

2. **Configure Ingress**

Edit `k8s/overlays/production/ingress.yaml` with your domain:

```yaml
spec:
  tls:
    - hosts:
        - api.your-domain.com
      secretName: esper-tls
  rules:
    - host: api.your-domain.com
```

3. **Deploy Services**

```bash
# Deploy all services
kubectl apply -k k8s/overlays/production/

# Check deployment status
kubectl -n esper get pods
kubectl -n esper get services
kubectl -n esper get ingress
```

### Scaling

```bash
# Scale Tamiyo replicas
kubectl -n esper scale deployment tamiyo --replicas=5

# Enable HPA (Horizontal Pod Autoscaler)
kubectl -n esper autoscale deployment tamiyo --cpu-percent=70 --min=2 --max=10
```

## Infrastructure Requirements

### PostgreSQL

For production, consider using managed database services:

- **AWS**: Amazon RDS for PostgreSQL
- **GCP**: Cloud SQL for PostgreSQL
- **Azure**: Azure Database for PostgreSQL

Configuration:
```sql
-- Recommended settings
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

### Redis

For production, consider:

- **AWS**: Amazon ElastiCache for Redis
- **GCP**: Memorystore for Redis
- **Azure**: Azure Cache for Redis

Enable persistence:
```
appendonly yes
appendfsync everysec
```

### Object Storage

Production options:
- **AWS S3**: Native support
- **GCP Cloud Storage**: S3-compatible API
- **Azure Blob Storage**: Use MinIO gateway
- **On-premise**: MinIO cluster

## Security Considerations

### Network Security

1. **Use Network Policies** (Kubernetes)
```yaml
# Applied automatically with production overlay
kubectl apply -f k8s/overlays/production/network-policy.yaml
```

2. **TLS/SSL**
- Use cert-manager for automatic certificate management
- Enforce HTTPS for all external endpoints
- Use TLS for database connections

3. **Secrets Management**
- Use Kubernetes Secrets or external secret managers (Vault, AWS Secrets Manager)
- Rotate credentials regularly
- Never commit secrets to version control

### Authentication & Authorization

1. **API Authentication**
```python
# Configure in service environment
API_KEY_HEADER = "X-API-Key"
REQUIRE_API_KEY = True
```

2. **RBAC** (Kubernetes)
```yaml
# Create service accounts with minimal permissions
kubectl create serviceaccount esper-services -n esper
```

## Monitoring and Observability

### Metrics Collection

The platform exposes Prometheus metrics on `/metrics` endpoints:

- `esper_forward_calls_total`: Total forward passes
- `esper_kernel_executions_total`: Kernel execution count
- `tamiyo_gnn_forward_duration_seconds`: GNN inference latency
- `esper_active_seeds_total`: Active morphogenetic seeds

### Grafana Dashboards

Import provided dashboards:

```bash
# Access Grafana
kubectl port-forward -n esper svc/grafana 3000:3000
# Default login: admin / <GRAFANA_PASSWORD>
```

### Logging

Configure centralized logging:

```yaml
# Fluentd/Fluent Bit configuration
<source>
  @type tail
  path /var/log/containers/esper-*.log
  tag esper.*
  <parse>
    @type json
  </parse>
</source>
```

### Alerts

Key alerts to configure:

1. **Service Health**
   - Service down > 5 minutes
   - High error rate > 5%
   - Response time > 1s (p99)

2. **Resource Usage**
   - Memory usage > 90%
   - CPU usage > 80%
   - Disk usage > 85%

3. **Application Metrics**
   - Kernel execution failures > 10%
   - GNN inference latency > 1s
   - Low adaptation rate

## Troubleshooting

### Common Issues

1. **Services Not Starting**
```bash
# Check logs
kubectl -n esper logs -f deployment/tamiyo
docker-compose -f docker/docker-compose.prod.yml logs tamiyo

# Check resource constraints
kubectl -n esper describe pod <pod-name>
```

2. **Database Connection Issues**
```bash
# Test connectivity
kubectl -n esper exec -it deployment/urza -- pg_isready -h postgres-service

# Check credentials
kubectl -n esper get secret esper-secrets -o yaml
```

3. **Redis Connection Issues**
```bash
# Test Redis connection
kubectl -n esper exec -it deployment/tamiyo -- redis-cli -h redis-service ping
```

4. **GPU Not Available**
```bash
# Check CUDA availability
kubectl -n esper exec -it deployment/tamiyo -- python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Tuning

1. **Tamiyo GNN Acceleration**
```bash
# Verify torch-scatter is installed
kubectl -n esper exec -it deployment/tamiyo -- python -c "import torch_scatter; print('Acceleration enabled')"
```

2. **Database Performance**
```sql
-- Check slow queries
SELECT query, calls, mean_time, max_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

3. **Redis Performance**
```bash
# Monitor Redis performance
redis-cli --latency
redis-cli --bigkeys
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
kubectl -n esper exec postgresql-0 -- pg_dump -U esper urza | gzip > backup_${DATE}.sql.gz

# Upload to S3
aws s3 cp backup_${DATE}.sql.gz s3://esper-backups/postgres/
```

### Disaster Recovery

1. **Multi-Region Setup**
   - Use database replication
   - Implement cross-region S3 replication
   - Deploy services in multiple availability zones

2. **Recovery Time Objective (RTO)**
   - Database: < 1 hour
   - Services: < 15 minutes
   - Full system: < 2 hours

## Next Steps

1. Set up continuous deployment pipeline
2. Implement comprehensive monitoring
3. Configure auto-scaling policies
4. Set up regular backup schedules
5. Conduct load testing and capacity planning

For additional support, please refer to the [GitHub issues](https://github.com/esper/esperlite/issues) or contact the development team.