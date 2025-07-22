#!/bin/bash
# Configure UFW firewall for Esper demo access

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔥 Esper Demo Firewall Configuration${NC}"
echo -e "${CYAN}=====================================${NC}\n"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}This script must be run with sudo${NC}"
    echo -e "${YELLOW}Usage: sudo ./scripts/configure-firewall.sh${NC}"
    exit 1
fi

# Check if UFW is installed
if ! command -v ufw &> /dev/null; then
    echo -e "${YELLOW}UFW is not installed. Installing...${NC}"
    apt-get update && apt-get install -y ufw
fi

# Display current status
echo -e "${YELLOW}Current UFW status:${NC}"
ufw status numbered
echo ""

# Define ports
declare -A PORTS=(
    ["80"]="HTTP (Nginx/Demo Dashboard)"
    ["3000"]="Grafana"
    ["8000"]="Urza API"
    ["8001"]="Tamiyo API"
    ["8080"]="Tolaria API"
    ["9000"]="MinIO S3 API"
    ["9001"]="MinIO Console"
    ["9090"]="Prometheus"
    ["5432"]="PostgreSQL (optional)"
    ["6379"]="Redis (optional)"
)

# Core ports that should be open for demo
CORE_PORTS=(80 3000 8000 8001 8080 9001 9090)

# Optional ports (databases)
OPTIONAL_PORTS=(5432 6379 9000)

echo -e "${CYAN}The following ports need to be configured:${NC}"
echo -e "\n${GREEN}Core Demo Ports:${NC}"
for port in "${CORE_PORTS[@]}"; do
    echo -e "  - ${GREEN}$port${NC}: ${PORTS[$port]}"
done

echo -e "\n${YELLOW}Optional Ports (for direct DB access):${NC}"
for port in "${OPTIONAL_PORTS[@]}"; do
    echo -e "  - ${YELLOW}$port${NC}: ${PORTS[$port]}"
done

echo -e "\n${CYAN}Choose configuration option:${NC}"
echo "1) Open all ports (recommended for demo)"
echo "2) Open core ports only"
echo "3) Custom configuration"
echo "4) Check current rules only"
echo "5) Exit"
echo ""
read -p "Select option (1-5): " OPTION

case $OPTION in
    1)
        echo -e "\n${YELLOW}Opening all demo ports...${NC}"
        
        # Enable UFW if not already
        ufw --force enable
        
        # Allow SSH first (important!)
        ufw allow ssh
        
        # Open all ports
        for port in "${!PORTS[@]}"; do
            echo -e "Opening port ${GREEN}$port${NC} (${PORTS[$port]})"
            ufw allow $port/tcp comment "Esper Demo - ${PORTS[$port]}"
        done
        
        echo -e "\n${GREEN}✅ All ports configured successfully${NC}"
        ;;
        
    2)
        echo -e "\n${YELLOW}Opening core demo ports only...${NC}"
        
        # Enable UFW if not already
        ufw --force enable
        
        # Allow SSH first
        ufw allow ssh
        
        # Open core ports only
        for port in "${CORE_PORTS[@]}"; do
            echo -e "Opening port ${GREEN}$port${NC} (${PORTS[$port]})"
            ufw allow $port/tcp comment "Esper Demo - ${PORTS[$port]}"
        done
        
        echo -e "\n${GREEN}✅ Core ports configured successfully${NC}"
        ;;
        
    3)
        echo -e "\n${CYAN}Custom configuration${NC}"
        echo "Select ports to open (space-separated):"
        echo "Available: ${!PORTS[@]}"
        read -p "Ports: " -a CUSTOM_PORTS
        
        # Enable UFW if not already
        ufw --force enable
        
        # Allow SSH first
        ufw allow ssh
        
        for port in "${CUSTOM_PORTS[@]}"; do
            if [[ -n "${PORTS[$port]}" ]]; then
                echo -e "Opening port ${GREEN}$port${NC} (${PORTS[$port]})"
                ufw allow $port/tcp comment "Esper Demo - ${PORTS[$port]}"
            else
                echo -e "${RED}Unknown port: $port${NC}"
            fi
        done
        
        echo -e "\n${GREEN}✅ Custom ports configured${NC}"
        ;;
        
    4)
        echo -e "\n${CYAN}Current firewall rules:${NC}"
        ;;
        
    5)
        echo -e "${YELLOW}Exiting without changes${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

# Show final status
echo -e "\n${CYAN}Final UFW status:${NC}"
ufw status numbered

# Additional recommendations
echo -e "\n${YELLOW}📋 Additional Security Recommendations:${NC}"
echo "1. Restrict source IPs for production:"
echo "   ${CYAN}sudo ufw allow from 192.168.1.0/24 to any port 80${NC}"
echo ""
echo "2. Remove rules when demo is complete:"
echo "   ${CYAN}sudo ufw status numbered${NC}"
echo "   ${CYAN}sudo ufw delete [rule_number]${NC}"
echo ""
echo "3. For cloud deployments, also configure:"
echo "   - Security groups (AWS)"
echo "   - Firewall rules (GCP)"
echo "   - Network security groups (Azure)"

# Check if services are accessible
echo -e "\n${CYAN}🔍 Testing accessibility:${NC}"
echo -e "${YELLOW}Note: Services must be running for tests to pass${NC}\n"

# Get the machine's external IP
EXTERNAL_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || echo "unknown")
if [ "$EXTERNAL_IP" != "unknown" ]; then
    echo -e "Your external IP: ${GREEN}$EXTERNAL_IP${NC}"
    echo -e "\nYou can access the demo at:"
    echo -e "  ${GREEN}http://$EXTERNAL_IP${NC} - Demo Dashboard"
    echo -e "  ${GREEN}http://$EXTERNAL_IP:3000${NC} - Grafana"
    echo -e "  ${GREEN}http://$EXTERNAL_IP:9001${NC} - MinIO Console"
else
    echo -e "${YELLOW}Could not determine external IP${NC}"
fi

echo -e "\n${GREEN}✅ Firewall configuration complete!${NC}"
echo -e "${CYAN}Run ${YELLOW}./scripts/start-demo.sh${CYAN} to start the demo${NC}"