#!/usr/bin/env python3
"""
Generate secure random credentials for Esper development environment
"""

import secrets
import string
import json
import os
from pathlib import Path

def generate_password(length=32):
    """Generate a secure random password"""
    # Avoid shell-problematic characters like $, &, and quotes
    alphabet = string.ascii_letters + string.digits + "!@#%^*_+-="
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_api_key(length=48):
    """Generate a secure API key"""
    return secrets.token_urlsafe(length)

def generate_credentials():
    """Generate all required credentials"""
    credentials = {
        "postgres": {
            "password": generate_password(32),
            "user": "esper_dev",
            "database": "esper_dev"
        },
        "redis": {
            "password": generate_password(32),
        },
        "minio": {
            "access_key": generate_password(20).upper(),
            "secret_key": generate_password(40),
        },
        "services": {
            "urza_api_key": generate_api_key(),
            "tamiyo_api_key": generate_api_key(),
            "tolaria_api_key": generate_api_key(),
        },
        "monitoring": {
            "grafana_admin_password": generate_password(24),
            "prometheus_basic_auth": generate_password(24),
        },
        "jwt": {
            "secret": generate_api_key(64),
        },
        "encryption": {
            "master_key": secrets.token_hex(32),
        }
    }
    
    return credentials

def write_env_file(credentials, filepath):
    """Write credentials to .env file"""
    env_content = f"""# Esper Development Environment Configuration
# Generated securely for tech demo
# DO NOT COMMIT THIS FILE TO VERSION CONTROL

# Database Configuration
POSTGRES_USER={credentials['postgres']['user']}
POSTGRES_PASSWORD={credentials['postgres']['password']}
POSTGRES_DB={credentials['postgres']['database']}
DATABASE_URL=postgresql://{credentials['postgres']['user']}:{credentials['postgres']['password']}@postgres:5432/{credentials['postgres']['database']}

# Redis Configuration
REDIS_PASSWORD={credentials['redis']['password']}
REDIS_URL=redis://:{credentials['redis']['password']}@redis:6379/0

# MinIO/S3 Configuration
MINIO_ROOT_USER={credentials['minio']['access_key']}
MINIO_ROOT_PASSWORD={credentials['minio']['secret_key']}
MINIO_ACCESS_KEY={credentials['minio']['access_key']}
MINIO_SECRET_KEY={credentials['minio']['secret_key']}
S3_ENDPOINT=http://minio:9000
S3_BUCKET=esper-artifacts
S3_REGION=us-east-1

# Service API Keys
URZA_API_KEY={credentials['services']['urza_api_key']}
TAMIYO_API_KEY={credentials['services']['tamiyo_api_key']}
TOLARIA_API_KEY={credentials['services']['tolaria_api_key']}

# Service URLs
URZA_URL=http://urza:8000
TAMIYO_URL=http://tamiyo:8001
OONA_REDIS_URL=redis://:{credentials['redis']['password']}@redis:6379/0

# Monitoring
GRAFANA_ADMIN_PASSWORD={credentials['monitoring']['grafana_admin_password']}
PROMETHEUS_BASIC_AUTH_PASSWORD={credentials['monitoring']['prometheus_basic_auth']}

# Security
JWT_SECRET={credentials['jwt']['secret']}
ENCRYPTION_MASTER_KEY={credentials['encryption']['master_key']}

# Environment Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG_MODE=false
DEVICE=cpu  # Change to 'cuda' if GPU available

# Performance Settings
CACHE_SIZE_MB=256
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20
REQUEST_TIMEOUT=300

# Feature Flags
ENABLE_TELEMETRY=true
ENABLE_PROFILING=false
ENABLE_ASYNC_COMPILATION=true
SAFETY_CHECKS_ENABLED=true
"""
    
    with open(filepath, 'w') as f:
        f.write(env_content)
    
    # Set restrictive permissions
    os.chmod(filepath, 0o600)

def write_secrets_json(credentials, filepath):
    """Write credentials to JSON file for programmatic access"""
    with open(filepath, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    # Set restrictive permissions
    os.chmod(filepath, 0o600)

def write_k8s_secrets(credentials, filepath):
    """Write Kubernetes secrets file"""
    k8s_secrets = f"""# Kubernetes Secrets for Development Environment
# Generated securely - DO NOT COMMIT
postgres-password={credentials['postgres']['password']}
redis-password={credentials['redis']['password']}
minio-access-key={credentials['minio']['access_key']}
minio-secret-key={credentials['minio']['secret_key']}
grafana-admin-password={credentials['monitoring']['grafana_admin_password']}
jwt-secret={credentials['jwt']['secret']}
"""
    
    with open(filepath, 'w') as f:
        f.write(k8s_secrets)
    
    os.chmod(filepath, 0o600)

def main():
    """Generate all credential files"""
    print("🔐 Generating secure credentials for Esper development environment...")
    
    # Generate credentials
    credentials = generate_credentials()
    
    # Create secrets directory
    secrets_dir = Path("secrets")
    secrets_dir.mkdir(exist_ok=True)
    
    # Write files
    write_env_file(credentials, ".env.dev-demo")
    write_secrets_json(credentials, "secrets/credentials.json")
    write_k8s_secrets(credentials, "k8s/overlays/development/secrets.env")
    
    print("✅ Credentials generated successfully!")
    print("\nFiles created:")
    print("  - .env.dev-demo (Docker Compose environment)")
    print("  - secrets/credentials.json (Programmatic access)")
    print("  - k8s/overlays/development/secrets.env (Kubernetes secrets)")
    print("\n⚠️  Remember:")
    print("  - These files contain sensitive data")
    print("  - Do NOT commit them to version control")
    print("  - Add them to .gitignore")
    print("\n🚀 Ready for tech demo deployment!")

if __name__ == "__main__":
    main()