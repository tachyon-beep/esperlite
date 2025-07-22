#!/bin/bash

# Fix Docker DNS resolution issues

echo "🔧 Fixing Docker DNS configuration..."

# Check if we have sudo access
if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo access to modify Docker configuration"
    exit 1
fi

# Create Docker daemon configuration with Google DNS
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-opts": ["timeout:3", "attempts:3"]
}
EOF

echo "✅ Docker DNS configuration updated"
echo "🔄 Restarting Docker service..."

# Restart Docker
sudo systemctl restart docker

# Wait for Docker to be ready
sleep 5

# Verify Docker is running
if docker info >/dev/null 2>&1; then
    echo "✅ Docker service restarted successfully"
else
    echo "❌ Docker service failed to restart"
    exit 1
fi

echo "
✨ Docker DNS fix complete!

You can now run the demo startup script again:
  ./scripts/start-demo.sh
"