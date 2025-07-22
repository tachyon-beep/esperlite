#!/bin/bash
# Esper Platform Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}

echo -e "${GREEN}Esper Platform Deployment Script${NC}"
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Environment: $ENVIRONMENT"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Docker is not installed${NC}"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}Docker Compose is not installed${NC}"
            exit 1
        fi
    elif [ "$DEPLOYMENT_TYPE" == "k8s" ] || [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}kubectl is not installed${NC}"
            exit 1
        fi
        
        if ! command -v kustomize &> /dev/null; then
            echo -e "${RED}kustomize is not installed${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Deploy with Docker Compose
deploy_docker() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"
    
    # Check for .env file
    if [ ! -f .env ]; then
        echo -e "${YELLOW}No .env file found. Copying from .env.${ENVIRONMENT}${NC}"
        if [ -f ".env.${ENVIRONMENT}" ]; then
            cp ".env.${ENVIRONMENT}" .env
        else
            echo -e "${RED}No .env.${ENVIRONMENT} file found${NC}"
            exit 1
        fi
    fi
    
    # Build images
    echo -e "${YELLOW}Building Docker images...${NC}"
    docker-compose -f docker/docker-compose.prod.yml build
    
    # Start infrastructure
    echo -e "${YELLOW}Starting infrastructure services...${NC}"
    docker-compose -f docker/docker-compose.prod.yml up -d postgres redis minio
    
    # Wait for infrastructure
    echo -e "${YELLOW}Waiting for infrastructure to be ready...${NC}"
    sleep 30
    
    # Start Esper services
    echo -e "${YELLOW}Starting Esper services...${NC}"
    docker-compose -f docker/docker-compose.prod.yml up -d urza tamiyo tolaria
    
    # Start monitoring (optional)
    read -p "Deploy monitoring stack? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker/docker-compose.prod.yml up -d prometheus grafana
    fi
    
    # Show status
    echo -e "${GREEN}Deployment complete!${NC}"
    docker-compose -f docker/docker-compose.prod.yml ps
}

# Deploy to Kubernetes
deploy_kubernetes() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    
    # Check if namespace exists
    if ! kubectl get namespace esper &> /dev/null; then
        echo -e "${YELLOW}Creating namespace 'esper'...${NC}"
        kubectl create namespace esper
    fi
    
    # Check for secrets
    if [ ! -f "k8s/overlays/${ENVIRONMENT}/secrets.env" ]; then
        echo -e "${RED}No secrets.env file found at k8s/overlays/${ENVIRONMENT}/secrets.env${NC}"
        echo "Please create this file with the following content:"
        echo "postgres-password=<secure-password>"
        echo "redis-password=<secure-password>"
        echo "minio-access-key=<access-key>"
        echo "minio-secret-key=<secret-key>"
        exit 1
    fi
    
    # Apply base configuration
    echo -e "${YELLOW}Applying base configuration...${NC}"
    kubectl apply -k k8s/base/
    
    # Apply environment overlay
    echo -e "${YELLOW}Applying ${ENVIRONMENT} overlay...${NC}"
    kubectl apply -k "k8s/overlays/${ENVIRONMENT}/"
    
    # Wait for deployments
    echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
    kubectl -n esper wait --for=condition=available --timeout=300s deployment --all
    
    # Show status
    echo -e "${GREEN}Deployment complete!${NC}"
    kubectl -n esper get pods
    kubectl -n esper get services
    kubectl -n esper get ingress
}

# Health check
health_check() {
    echo -e "${YELLOW}Running health checks...${NC}"
    
    if [ "$DEPLOYMENT_TYPE" == "docker" ]; then
        # Check Docker services
        services=("urza:8000" "tamiyo:8001" "tolaria:8080")
        for service in "${services[@]}"; do
            IFS=':' read -r name port <<< "$service"
            if curl -f "http://localhost:$port/health" &> /dev/null; then
                echo -e "${GREEN}✓ $name is healthy${NC}"
            else
                echo -e "${RED}✗ $name is not responding${NC}"
            fi
        done
    elif [ "$DEPLOYMENT_TYPE" == "k8s" ] || [ "$DEPLOYMENT_TYPE" == "kubernetes" ]; then
        # Check Kubernetes services
        kubectl -n esper get pods --no-headers | while read line; do
            pod=$(echo $line | awk '{print $1}')
            status=$(echo $line | awk '{print $3}')
            if [ "$status" == "Running" ]; then
                echo -e "${GREEN}✓ $pod is running${NC}"
            else
                echo -e "${RED}✗ $pod is in state: $status${NC}"
            fi
        done
    fi
}

# Main execution
main() {
    check_prerequisites
    
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        k8s|kubernetes)
            deploy_kubernetes
            ;;
        *)
            echo -e "${RED}Unknown deployment type: $DEPLOYMENT_TYPE${NC}"
            echo "Usage: $0 [docker|k8s|kubernetes] [environment]"
            exit 1
            ;;
    esac
    
    # Run health check after a delay
    echo -e "${YELLOW}Waiting 30 seconds before health check...${NC}"
    sleep 30
    health_check
}

# Run main function
main