#!/bin/bash
#
# Setup test services (Redis, etc.) for running the test suite
#

set -e

echo "Setting up test services..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1 || netstat -an | grep -q ":$1.*LISTEN"
}

# Redis setup
setup_redis() {
    echo "Setting up Redis..."
    
    # Check if Redis is already running on port 6379
    if port_in_use 6379; then
        echo "✓ Redis appears to be running on port 6379"
        return 0
    fi
    
    # Try Docker first
    if command_exists docker; then
        echo "Using Docker to run Redis..."
        
        # Stop any existing test-redis container
        docker stop test-redis 2>/dev/null || true
        docker rm test-redis 2>/dev/null || true
        
        # Run Redis in Docker
        docker run \
            --name test-redis \
            -d \
            -p 6379:6379 \
            redis:7-alpine \
            redis-server --save "" --appendonly no
        
        # Wait for Redis to be ready
        echo -n "Waiting for Redis to start..."
        for i in {1..30}; do
            if docker exec test-redis redis-cli ping >/dev/null 2>&1; then
                echo " ✓"
                echo "Redis is running in Docker (container: test-redis)"
                return 0
            fi
            echo -n "."
            sleep 0.5
        done
        
        echo " ✗"
        echo "Failed to start Redis in Docker"
        return 1
    fi
    
    # Try redis-server
    if command_exists redis-server; then
        echo "Using redis-server..."
        
        # Create a minimal config
        cat > /tmp/redis-test.conf <<EOF
port 6379
save ""
appendonly no
EOF
        
        # Start Redis in background
        redis-server /tmp/redis-test.conf --daemonize yes
        
        # Wait for Redis to be ready
        echo -n "Waiting for Redis to start..."
        for i in {1..10}; do
            if redis-cli ping >/dev/null 2>&1; then
                echo " ✓"
                echo "Redis is running (redis-server)"
                return 0
            fi
            echo -n "."
            sleep 0.5
        done
        
        echo " ✗"
        echo "Failed to start redis-server"
        return 1
    fi
    
    echo "Neither Docker nor redis-server found!"
    echo "Please install one of:"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    echo "  - Redis: https://redis.io/download"
    return 1
}

# Install Python dependencies
install_dependencies() {
    echo "Checking Python dependencies..."
    
    # Check if redis async is installed
    if ! python -c "import redis.asyncio" 2>/dev/null; then
        echo "Installing redis with async support..."
        pip install "redis[hiredis]"
    else
        echo "✓ redis[async] is installed"
    fi
    
    # Check if redis-py is installed
    if ! python -c "import redis" 2>/dev/null; then
        echo "Installing redis..."
        pip install redis
    else
        echo "✓ redis-py is installed"
    fi
}

# Stop services
stop_services() {
    echo "Stopping test services..."
    
    # Stop Docker Redis
    if command_exists docker; then
        docker stop test-redis 2>/dev/null || true
        docker rm test-redis 2>/dev/null || true
    fi
    
    # Stop redis-server
    if command_exists redis-cli; then
        redis-cli shutdown 2>/dev/null || true
    fi
    
    echo "Services stopped"
}

# Main
case "${1:-start}" in
    start)
        setup_redis && install_dependencies
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Test services are ready!"
            echo ""
            echo "You can now run the tests with:"
            echo "  pytest"
            echo ""
            echo "To stop services:"
            echo "  $0 stop"
        else
            exit 1
        fi
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 1
        setup_redis && install_dependencies
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac