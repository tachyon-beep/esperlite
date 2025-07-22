#!/bin/bash
# Infrastructure Validation Script for Esper Platform

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Esper Infrastructure Validation${NC}"
echo "================================="

# Check Docker files
echo -e "\n${YELLOW}Checking Docker infrastructure...${NC}"

docker_files=(
    "docker/Dockerfile.base"
    "docker/Dockerfile.tamiyo"
    "docker/Dockerfile.tolaria"
    "docker/Dockerfile.urza"
    "docker/docker-compose.yml"
    "docker/docker-compose.prod.yml"
)

for file in "${docker_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

# Check Kubernetes files
echo -e "\n${YELLOW}Checking Kubernetes infrastructure...${NC}"

k8s_files=(
    "k8s/base/namespace.yaml"
    "k8s/base/configmap.yaml"
    "k8s/base/secrets.yaml"
    "k8s/base/redis.yaml"
    "k8s/base/postgresql.yaml"
    "k8s/base/tamiyo.yaml"
    "k8s/base/kustomization.yaml"
    "k8s/overlays/production/kustomization.yaml"
    "k8s/overlays/production/ingress.yaml"
)

for file in "${k8s_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

# Check monitoring files
echo -e "\n${YELLOW}Checking monitoring setup...${NC}"

monitoring_files=(
    "monitoring/prometheus/prometheus.yml"
    "monitoring/prometheus/alerts/esper-alerts.yml"
    "monitoring/grafana/datasources/prometheus.yaml"
    "monitoring/grafana/dashboards/esper-overview.json"
)

for file in "${monitoring_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
    fi
done

# Check documentation
echo -e "\n${YELLOW}Checking deployment documentation...${NC}"

if [ -f "docs/deployment/README.md" ]; then
    echo -e "${GREEN}✓${NC} Deployment documentation exists"
    lines=$(wc -l < "docs/deployment/README.md")
    echo "  Documentation has $lines lines"
else
    echo -e "${RED}✗${NC} Deployment documentation missing"
fi

# Check scripts
echo -e "\n${YELLOW}Checking deployment scripts...${NC}"

if [ -f "scripts/deploy.sh" ] && [ -x "scripts/deploy.sh" ]; then
    echo -e "${GREEN}✓${NC} Deploy script exists and is executable"
else
    echo -e "${RED}✗${NC} Deploy script missing or not executable"
fi

# Summary
echo -e "\n${GREEN}Infrastructure Validation Complete!${NC}"
echo "================================="
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Configure environment variables in .env files"
echo "2. Update Kubernetes secrets in k8s/overlays/production/secrets.env"
echo "3. Run deployment: ./scripts/deploy.sh [docker|k8s] [environment]"
echo "4. Monitor services using Grafana dashboards"
echo ""
echo "For detailed instructions, see docs/deployment/README.md"