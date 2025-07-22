#!/bin/bash
# Esper Tech Demo Startup Script
# This script sets up and launches the complete Esper platform for demonstrations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Demo configuration
DEMO_NAME="Esper Morphogenetic Training Platform"
COMPOSE_FILE="docker/docker-compose.dev-demo.yml"
ENV_FILE=".env.dev-demo"
COMPOSE_CMD=""  # Will be set by check_prerequisites

# ASCII Art Banner
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ___________                           
    \_   _____/ ____________   ___________
     |    __)_ /  ___/\____ \_/ __ \_  __ \
     |        \\___ \ |  |_> >  ___/|  | \/
    /_______  /____  >|   __/ \___  >__|   
            \/     \/ |__|        \/       
    
    Morphogenetic Neural Network Training Platform
EOF
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    local missing=0
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}   ✗ Docker is not installed${NC}"
        missing=1
    else
        echo -e "${GREEN}   ✓ Docker $(docker --version | cut -d' ' -f3 | tr -d ',')${NC}"
    fi
    
    # Check Docker Compose (v1 or v2)
    if command -v docker-compose &> /dev/null; then
        # Docker Compose v1 (standalone)
        COMPOSE_CMD="docker-compose"
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f4 | tr -d ',')
        echo -e "${GREEN}   ✓ Docker Compose $COMPOSE_VERSION (standalone)${NC}"
    elif docker compose version &> /dev/null; then
        # Docker Compose v2 (plugin)
        COMPOSE_CMD="docker compose"
        COMPOSE_VERSION=$(docker compose version | cut -d' ' -f4)
        echo -e "${GREEN}   ✓ Docker Compose $COMPOSE_VERSION (plugin)${NC}"
    else
        echo -e "${RED}   ✗ Docker Compose is not installed${NC}"
        echo -e "${YELLOW}     Install with: docker plugin install compose${NC}"
        missing=1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}   ✗ Python 3 is not installed${NC}"
        missing=1
    else
        echo -e "${GREEN}   ✓ Python $(python3 --version | cut -d' ' -f2)${NC}"
    fi
    
    # Check environment file
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}   ✗ Environment file $ENV_FILE not found${NC}"
        echo -e "${YELLOW}     Run: python scripts/generate-credentials.py${NC}"
        missing=1
    else
        echo -e "${GREEN}   ✓ Environment configuration found${NC}"
    fi
    
    if [ $missing -eq 1 ]; then
        echo -e "\n${RED}Please install missing prerequisites before continuing.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites satisfied${NC}\n"
}

# Stop any running containers
cleanup_existing() {
    echo -e "${YELLOW}🧹 Cleaning up existing containers...${NC}"
    
    # Stop existing demo containers
    $COMPOSE_CMD -f "$COMPOSE_FILE" down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove any orphaned containers
    docker container prune -f > /dev/null 2>&1
    
    echo -e "${GREEN}✅ Cleanup complete${NC}\n"
}

# Build Docker images
build_images() {
    echo -e "${YELLOW}🔨 Building Docker images...${NC}"
    echo -e "${CYAN}   This may take several minutes on first run...${NC}"
    
    # Build all services (use .env.build to suppress warnings)
    $COMPOSE_CMD -f "$COMPOSE_FILE" --env-file ".env.build" build --parallel
    
    echo -e "${GREEN}✅ Images built successfully${NC}\n"
}

# Start infrastructure services
start_infrastructure() {
    echo -e "${YELLOW}🚀 Starting infrastructure services...${NC}"
    
    # Start infrastructure services
    $COMPOSE_CMD -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        postgres redis minio minio-init nginx
    
    # Wait for services to be healthy
    echo -e "${CYAN}   Waiting for infrastructure to be ready...${NC}"
    
    local retries=30
    local count=0
    
    while [ $count -lt $retries ]; do
        if $COMPOSE_CMD -f "$COMPOSE_FILE" ps | grep -q "unhealthy\|starting"; then
            echo -ne "\r   Progress: [$(printf '%-30s' $(printf '#%.0s' $(seq 1 $count)))] $count/$retries"
            sleep 2
            ((count++))
        else
            echo -ne "\r   Progress: [##############################] Ready!     \n"
            break
        fi
    done
    
    echo -e "${GREEN}✅ Infrastructure services started${NC}\n"
}

# Start Esper services
start_esper_services() {
    echo -e "${YELLOW}🧬 Starting Esper services...${NC}"
    
    # Start core services
    $COMPOSE_CMD -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        urza tamiyo tolaria
    
    # Wait for services to be ready
    echo -e "${CYAN}   Waiting for Esper services to initialize...${NC}"
    sleep 10
    
    # Check service health
    local services=("urza:8000" "tamiyo:8001" "tolaria:8080")
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -sf "http://localhost:$port/health" > /dev/null; then
            echo -e "${GREEN}   ✓ $name service is healthy${NC}"
        else
            echo -e "${YELLOW}   ⚠ $name service is starting...${NC}"
        fi
    done
    
    echo -e "${GREEN}✅ Esper services started${NC}\n"
}

# Start monitoring stack
start_monitoring() {
    echo -e "${YELLOW}📊 Starting monitoring services...${NC}"
    
    $COMPOSE_CMD -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        prometheus grafana
    
    echo -e "${GREEN}✅ Monitoring services started${NC}\n"
}

# Display access information
show_access_info() {
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}🎉 Esper Tech Demo is Ready!${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
    
    echo -e "${CYAN}📍 Access Points:${NC}"
    echo -e "   ${GREEN}Demo Dashboard:${NC}  http://localhost"
    echo -e "   ${GREEN}Grafana:${NC}        http://localhost:3000 (admin / check .env.dev-demo)"
    echo -e "   ${GREEN}MinIO Console:${NC}  http://localhost:9001"
    echo -e "   ${GREEN}Prometheus:${NC}     http://localhost:9090"
    echo -e ""
    echo -e "${CYAN}🔌 API Endpoints:${NC}"
    echo -e "   ${GREEN}Urza API:${NC}       http://localhost/api/urza/"
    echo -e "   ${GREEN}Tamiyo API:${NC}     http://localhost/api/tamiyo/"
    echo -e "   ${GREEN}Tolaria API:${NC}    http://localhost/api/tolaria/"
    echo -e ""
    echo -e "${CYAN}🛠️  Management Commands:${NC}"
    echo -e "   ${YELLOW}View logs:${NC}      $COMPOSE_CMD -f $COMPOSE_FILE logs -f [service]"
    echo -e "   ${YELLOW}Stop demo:${NC}      $COMPOSE_CMD -f $COMPOSE_FILE down"
    echo -e "   ${YELLOW}Restart:${NC}        $COMPOSE_CMD -f $COMPOSE_FILE restart [service]"
    echo -e ""
    echo -e "${CYAN}📚 Quick Start:${NC}"
    echo -e "   1. Open ${GREEN}http://localhost${NC} in your browser"
    echo -e "   2. Check service health on the dashboard"
    echo -e "   3. View real-time metrics in Grafana"
    echo -e "   4. Start a training session via Tolaria API"
    echo -e ""
    echo -e "${YELLOW}⚠️  Note:${NC} Services may take 1-2 minutes to fully initialize"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Monitor services (optional)
monitor_services() {
    echo -e "\n${CYAN}Would you like to monitor service logs? (y/n)${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Showing service logs (Ctrl+C to exit)...${NC}\n"
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f --tail=50
    fi
}

# Main execution
main() {
    clear
    show_banner
    
    echo -e "${BLUE}Starting $DEMO_NAME Tech Demo${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
    
    # Run setup steps
    check_prerequisites
    cleanup_existing
    build_images
    start_infrastructure
    start_esper_services
    start_monitoring
    
    # Show access information
    show_access_info
    
    # Optional monitoring
    monitor_services
}

# Trap to handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Demo startup interrupted${NC}"; exit 1' INT

# Run main function
main