### **Definitive Implementation Guide: Esper Phase 0 (Foundation Layer)**

**Objective:** To establish a production-ready, version-controlled foundation for the Esper project. This phase implements all shared contracts, configurations, and infrastructure-as-code required for local development. By its completion, the project will have a fully configured, testable, and robust bedrock for all subsequent development phases.

-----

### **1. Finalized Technology Stack & Versions**

This stack incorporates the peer review feedback for a more precise and realistic setup.

| Category | Technology | Version | Rationale |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `~3.12` | Confirmed. Modern and feature-rich. |
| **Deep Learning**| PyTorch | `>=2.2.0,<3.0`| **(Revised)** Adopts the reviewer's suggestion for a realistic version range, ensuring compatibility while allowing for minor updates. |
| **Data Contracts**| Pydantic | `~2.8` | Confirmed. Ideal for enforcing contracts between services. |
| **Web Framework**| FastAPI | `~0.112` | Confirmed. High-performance async API development. |
| **Code Quality** | Black, Ruff, Pytype | `latest` | Confirmed. A best-in-class stack for formatting, linting, and static type analysis. |
| **Databases** | PostgreSQL, Redis | `16`, `7.2` | Confirmed. Stable and production-proven versions. |
| **Storage** | MinIO | `latest` | Confirmed. S3-compatible standard. |

-----

### **2. Finalized Project Structure**

This structure includes all new files recommended by the peer review.

```plaintext
esper-morphogen/
├── .github/
│   └── workflows/
│       └── ci.yml             # ✅ NEW: GitHub Actions CI/CD pipeline
├── configs/
│   └── phase1_mvp.yaml
├── docker/
│   └── docker-compose.yml     # ✅ NEW: Infrastructure-as-code for local dev
├── scripts/
│   └── train_tamiyo.py
├── src/
│   └── esper/
│       ├── __init__.py        # ✅ NEW: Will contain API_VERSION
│       ├── contracts/
│       │   ├── __init__.py
│       │   ├── assets.py      # Core asset models
│       │   ├── enums.py       # All status enums
│       │   ├── messages.py    # ✅ NEW: Oona message bus contracts
│       │   ├── operational.py # ✅ REVISED: Operational models (rich HealthSignal)
│       │   └── validators.py  # ✅ NEW: Pydantic custom validators
│       ├── configs.py         # Pydantic models for parsing YAML
│       ├── services/
│       ├── core/
│       └── utils/
│           └── logging.py     # ✅ NEW: Centralized logging setup
├── tests/
│   └── contracts/
├── .env.example               # ✅ NEW: Example environment variables
├── .gitignore
├── pyproject.toml             # ✅ REVISED: Now includes full test config
└── README.md
```

-----

### **3. Environment & Configuration**

#### **3.1. Docker Compose (`docker/docker-compose.yml`)**

This file defines the project's local infrastructure stack. Create a `.env` file from the `.env.example` below before running `docker-compose up`.

```yaml
version: '3.8'

services:
  redis:
    image: redis:7.2-alpine
    container_name: esper_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  postgres:
    image: postgres:16-alpine
    container_name: esper_postgres
    environment:
      POSTGRES_DB: urza_db
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d urza_db"]
      interval: 10s
      timeout: 5s
      retries: 5
      
  minio:
    image: minio/minio:latest
    container_name: esper_minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  redis_data:
  postgres_data:
  minio_data:
```

#### **3.2. Environment Variables (`.env.example`)**

Create this file at the project root. Users will copy it to `.env` and fill in their secrets.

```dotenv
# .env.example - Copy to .env and configure for local development

# === DATABASE (PostgreSQL for Urza Metadata) ===
POSTGRES_USER=esper_admin
POSTGRES_PASSWORD=replace_with_a_strong_password

# === OBJECT STORAGE (MinIO for Artifacts) ===
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=replace_with_a_strong_password
S3_BUCKET_NAME=esper-artifacts
S3_ENDPOINT_URL=http://localhost:9000

# === MESSAGE BUS (Redis Streams for Oona) ===
REDIS_URL=redis://localhost:6379/0
```

-----

### **4. Code Quality & CI/CD**

#### **4.1. Project & Tool Configuration (`pyproject.toml`)**

This file is now complete with testing and coverage configuration.

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "esper-morphogen"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.2.0,<3.0",
    "pydantic~=2.8",
    "fastapi~=0.112",
    "uvicorn[standard]~=0.30",
    "pyyaml~=6.0",
    "redis~=5.0",
    "psycopg2-binary~=2.9",
    "boto3~=1.34",
    "numpy~=1.26",
]

[project.optional-dependencies]
dev = [
    "black~=24.8",
    "ruff~=0.5",
    "pytype~=2025.7.1",
    "pytest~=8.2",
    "pytest-cov~=5.0",
    "httpx~=0.27",
]

[tool.black]
line-length = 88
target-version = ['py312']

[tool.ruff]
line-length = 88
select = ["E", "W", "F", "I", "C90", "N"]
ignore = ["E501"]

[tool.ruff.isort]
force-single-line = true

[tool.pytype]
inputs = ["src/esper"]
python_version = "3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "--cov=src/esper --cov-report=term-missing --cov-report=html"

[tool.coverage.run]
source = ["src/esper"]
omit = ["*/tests/*", "*/__main__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

#### **4.2. CI Pipeline (`.github/workflows/ci.yml`)**

This workflow automates quality checks for every code change.

```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .[dev]
      
      - name: Check formatting with Black and Ruff
        run: |
          black --check src tests
          ruff check src tests
      
      - name: Run static type analysis with Pytype
        run: pytype
      
      - name: Run tests with Pytest
        run: pytest
```

-----

### **5. Core Data Contracts Implementation**

Implement the following revised and new files in the `src/esper/contracts/` directory.

#### **`src/esper/contracts/__init__.py`**

```python
API_VERSION = "v1"
```

#### **`src/esper/contracts/operational.py` (Revised)**

```python
from typing import Dict, Tuple

from pydantic import BaseModel, Field, field_validator
from .validators import validate_seed_id

class HealthSignal(BaseModel):
    """High-frequency health signal from a KasminaSeed."""
    layer_id: int
    seed_id: int
    chunk_id: int
    epoch: int
    activation_variance: float
    dead_neuron_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_correlation: float = Field(..., ge=-1.0, le=1.0)
    is_ready_for_transition: bool = False # Critical for state machine sync

    @field_validator('seed_id', 'layer_id')
    def _validate_ids(cls, v):
        if v < 0:
            raise ValueError("IDs must be non-negative")
        return v
        
# ... other operational contracts like SystemStatePacket, etc. from previous plan
```

#### **`src/esper/contracts/messages.py` (New)**

```python
from datetime import datetime
from enum import Enum
from typing import Any, Dict
import uuid

from pydantic import BaseModel, Field

class TopicNames(str, Enum):
    """Centralized definition of all Oona message bus topics."""
    TELEMETRY_SEED_HEALTH = "telemetry.seed.health"
    CONTROL_KASMINA_COMMANDS = "control.kasmina.commands"
    COMPILATION_BLUEPRINT_SUBMITTED = "compilation.blueprint.submitted"
    COMPILATION_KERNEL_READY = "compilation.kernel.ready"
    VALIDATION_KERNEL_CHARACTERIZED = "validation.kernel.characterized"
    SYSTEM_EVENTS_EPOCH = "system.events.epoch"
    INNOVATION_FIELD_REPORTS = "innovation.field_reports"

class OonaMessage(BaseModel):
    """Base envelope for all messages published on the Oona bus."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str # e.g., 'Tezzeret-Worker-5', 'Tamiyo-Controller'
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str # To trace a request across multiple services
    topic: TopicNames
    payload: Dict[str, Any]
```

#### **`src/esper/contracts/validators.py` (New)**

```python
from typing import Tuple

def validate_seed_id(v: Tuple[int, int]) -> Tuple[int, int]:
    """Ensures a seed_id tuple (layer_id, seed_idx) is valid."""
    if not isinstance(v, tuple) or len(v) != 2:
        raise ValueError("Seed ID must be a tuple of (layer_id, seed_idx)")
    
    layer_id, seed_idx = v
    if not isinstance(layer_id, int) or not isinstance(seed_idx, int):
        raise TypeError("Seed ID elements must be integers")

    if layer_id < 0 or seed_idx < 0:
        raise ValueError("Seed ID elements must be non-negative")
    return v
```

-----

### **6. Shared Utilities**

#### **`src/esper/utils/logging.py` (New)**

```python
import logging
import sys

def setup_logging(service_name: str, level=logging.INFO):
    """Configures structured logging for a service."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        f'%(asctime)s - {service_name} - %(levelname)s - [%(name)s:%(lineno)s] - %(message)s'
    )
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    return logging.getLogger(service_name)
```

-----

### **7. Phase 0 Definition of Done**

This phase is complete when:

1. All specified files (`pyproject.toml`, `docker-compose.yml`, `.env.example`, `.github/workflows/ci.yml`, and all Python source files) have been created and committed.
2. The command `pip install -e .[dev]` installs all dependencies without error.
3. The CI pipeline passes successfully, confirming that all formatting, linting, type checks, and unit tests for the contracts are green.
4. The command `docker-compose up` successfully starts the Redis, PostgreSQL, and MinIO containers, and they pass their health checks.
5. The project README.md is updated with instructions for setting up the local development environment using `.env` and `docker-compose`.
