#!/bin/bash
# Reset demo environment to clean state

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🔄 Resetting Esper Demo Environment${NC}"
echo "This will:"
echo "  - Stop all demo containers"
echo "  - Remove demo volumes (data will be lost)"
echo "  - Clean up Docker resources"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Reset cancelled${NC}"
    exit 0
fi

echo -e "\n${YELLOW}Stopping services...${NC}"
# Detect Docker Compose version
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo -e "${RED}Docker Compose not found${NC}"
    exit 1
fi

$COMPOSE_CMD -f docker/docker-compose.dev-demo.yml down -v --remove-orphans

echo -e "${YELLOW}Cleaning up volumes...${NC}"
docker volume ls | grep demo | awk '{print $2}' | xargs -r docker volume rm || true

echo -e "${YELLOW}Pruning unused resources...${NC}"
docker system prune -f --volumes

echo -e "\n${GREEN}✅ Demo environment reset complete${NC}"
echo -e "${YELLOW}Run ./scripts/start-demo.sh to start fresh${NC}"