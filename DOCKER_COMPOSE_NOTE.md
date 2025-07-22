# Docker Compose Version Note

This project supports both Docker Compose v1 (standalone) and v2 (plugin).

## Your System

You have Docker Compose v2.38.2 installed as a Docker plugin.

## Usage

All scripts have been updated to automatically detect and use the correct version:
- Docker Compose v1: `docker-compose` command
- Docker Compose v2: `docker compose` command (your version)

## Manual Commands

If you need to run Docker Compose commands manually, use:
```bash
docker compose -f docker/docker-compose.dev-demo.yml [command]
```

Example:
```bash
# View logs
docker compose -f docker/docker-compose.dev-demo.yml logs -f

# Stop services
docker compose -f docker/docker-compose.dev-demo.yml down

# List containers
docker compose -f docker/docker-compose.dev-demo.yml ps
```

The scripts will handle this automatically for you.