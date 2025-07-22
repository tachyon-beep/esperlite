#!/bin/bash
# Check Esper demo accessibility from external network

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🌐 Esper Demo Network Accessibility Check${NC}"
echo -e "${CYAN}=========================================${NC}\n"

# Get local IPs
echo -e "${YELLOW}Local Network Information:${NC}"
echo -e "Hostname: ${GREEN}$(hostname)${NC}"

# Get all network interfaces
echo -e "\n${YELLOW}Network Interfaces:${NC}"
ip -4 addr show | grep -E "inet " | grep -v "127.0.0.1" | awk '{print "  " $NF ": " $2}'

# Get external IP
echo -e "\n${YELLOW}External IP Address:${NC}"
EXTERNAL_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s https://icanhazip.com 2>/dev/null || echo "Could not determine")
echo -e "  ${GREEN}$EXTERNAL_IP${NC}"

# Check if services are running
echo -e "\n${YELLOW}Service Status:${NC}"
SERVICES=("nginx:80" "grafana:3000" "urza:8000" "tamiyo:8001" "tolaria:8080" "minio:9001" "prometheus:9090")

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if nc -z localhost $port 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $name (port $port) - Running"
    else
        echo -e "  ${RED}✗${NC} $name (port $port) - Not accessible"
    fi
done

# Check firewall status
echo -e "\n${YELLOW}Firewall Status:${NC}"
if command -v ufw &> /dev/null; then
    if sudo ufw status | grep -q "Status: active"; then
        echo -e "  UFW is ${GREEN}active${NC}"
        echo -e "\n  ${CYAN}Esper-related rules:${NC}"
        sudo ufw status numbered | grep -E "(80|3000|8000|8001|8080|9000|9001|9090)" | sed 's/^/  /'
    else
        echo -e "  UFW is ${YELLOW}inactive${NC}"
    fi
else
    echo -e "  ${YELLOW}UFW not installed${NC}"
fi

# Test external accessibility
echo -e "\n${YELLOW}Testing External Accessibility:${NC}"
echo -e "${CYAN}(This requires services to be running)${NC}\n"

# Function to test URL
test_url() {
    local url=$1
    local service=$2
    if curl -sf -o /dev/null --connect-timeout 3 "$url"; then
        echo -e "  ${GREEN}✓${NC} $service is accessible at: $url"
        return 0
    else
        echo -e "  ${RED}✗${NC} $service is not accessible at: $url"
        return 1
    fi
}

# Test localhost
echo -e "${CYAN}Local Access Test:${NC}"
test_url "http://localhost" "Demo Dashboard"
test_url "http://localhost:3000" "Grafana"

# Test external IP if available
if [ "$EXTERNAL_IP" != "Could not determine" ]; then
    echo -e "\n${CYAN}External Access Test:${NC}"
    test_url "http://$EXTERNAL_IP" "Demo Dashboard"
    test_url "http://$EXTERNAL_IP:3000" "Grafana"
    test_url "http://$EXTERNAL_IP:9001" "MinIO Console"
fi

# Docker network info
echo -e "\n${YELLOW}Docker Network Information:${NC}"
if docker network ls | grep -q esper-demo-network; then
    echo -e "  ${GREEN}✓${NC} esper-demo-network exists"
    # Show containers on network
    echo -e "\n  ${CYAN}Containers on network:${NC}"
    docker ps --filter "network=esper-demo-network" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed 's/^/  /'
else
    echo -e "  ${RED}✗${NC} esper-demo-network not found"
fi

# Access URLs
echo -e "\n${GREEN}📍 Access URLs:${NC}"
echo -e "\n${CYAN}From this machine:${NC}"
echo -e "  Demo Dashboard:  http://localhost"
echo -e "  Grafana:        http://localhost:3000"
echo -e "  MinIO Console:  http://localhost:9001"
echo -e "  Prometheus:     http://localhost:9090"

if [ "$EXTERNAL_IP" != "Could not determine" ]; then
    echo -e "\n${CYAN}From external network:${NC}"
    echo -e "  Demo Dashboard:  http://$EXTERNAL_IP"
    echo -e "  Grafana:        http://$EXTERNAL_IP:3000"
    echo -e "  MinIO Console:  http://$EXTERNAL_IP:9001"
    echo -e "  Prometheus:     http://$EXTERNAL_IP:9090"
fi

# Troubleshooting tips
echo -e "\n${YELLOW}💡 Troubleshooting Tips:${NC}"
echo "1. If services are not accessible externally:"
echo "   - Run: ${CYAN}sudo ./scripts/configure-firewall.sh${NC}"
echo "   - Check cloud provider security groups/firewall rules"
echo "   - Ensure Docker is binding to 0.0.0.0 (already configured)"
echo ""
echo "2. If services are not running:"
echo "   - Run: ${CYAN}./scripts/start-demo.sh${NC}"
echo "   - Check logs: ${CYAN}docker-compose -f docker/docker-compose.dev-demo.yml logs${NC}"
echo ""
echo "3. For cloud deployments, also check:"
echo "   - AWS: Security Groups allow inbound traffic"
echo "   - GCP: Firewall rules are configured"
echo "   - Azure: Network Security Groups permit access"