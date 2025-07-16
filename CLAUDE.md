# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]
```

### Code Quality
```bash
# Format code (required before commits)
black src tests

# Lint code (must pass)
ruff check src tests

# Type checking (must pass)
pytype
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/esper --cov-report=html

# Run specific test file
pytest tests/core/test_model_wrapper.py

# Run integration tests only
pytest tests/integration/
```

### Infrastructure
```bash
# Start infrastructure services (Postgres, Redis, MinIO)
docker-compose -f docker/docker-compose.yml up -d

# Stop services
docker-compose -f docker/docker-compose.yml down

# View service logs
docker-compose -f docker/docker-compose.yml logs -f
```

## Architecture Overview

Esper is a morphogenetic training platform that enables neural networks to autonomously evolve their architecture during training. The system is organized into specialized components across three functional planes:

### Core Components

**Training Plane:**
- `Tolaria` (src/esper/services/tolaria/): Training orchestrator that manages the main training loop
- `Kasmina` (src/esper/execution/): Execution layer that wraps PyTorch models and handles dynamic kernel loading

**Control Plane:**
- `Tamiyo` (src/esper/services/tamiyo/): Strategic controller using GNN-based policy networks to analyze models and decide when/where to intervene
- `Urza` (src/esper/services/urza/): Central library that manages kernel artifacts and metadata using PostgreSQL

**Innovation Plane:**
- `Tezzeret` (src/esper/services/tezzeret/): Compilation forge that transforms Blueprint IR into compiled kernel artifacts
- `Oona` (src/esper/services/oona_client.py): Message bus client for inter-component communication using Redis

### Key Data Contracts

All inter-service communication is governed by Pydantic data contracts in `src/esper/contracts/`:
- `assets.py`: Core data structures for blueprints, kernels, and artifacts
- `messages.py`: Message formats for the Oona message bus
- `operational.py`: Operational state and lifecycle management
- `enums.py`: System-wide enumerations and constants

### Entry Points

The main public API is exposed through `src/esper/__init__.py`:
- `esper.wrap()`: Wraps a PyTorch model to enable morphogenetic capabilities
- `KasminaLayer`: Core execution layer for dynamic kernel loading
- `MorphableModel`: Enhanced model wrapper with adaptation capabilities

## Engineering Standards

### Code Quality Requirements
- All code must pass `black`, `ruff`, and `pytype` checks
- Functions require full type hints (avoid `typing.Any`)
- All public APIs need Google-style docstrings
- Unit tests required for all non-trivial logic (>90% coverage target)

### Import Policy
- All imports must be firm - no conditional `try/except ImportError` blocks
- Dependencies must be declared in `pyproject.toml`
- Use `force-single-line` import style (enforced by ruff)

### Testing Philosophy
- Unit tests in `tests/` mirror source structure
- Integration tests in `tests/integration/` for component interactions
- No test-specific code in production (`src/`) directory
- Tests must be independent and deterministic

### Development Workflow
1. Create feature branch from main
2. Implement changes following coding standards
3. Ensure all quality checks pass locally
4. Run relevant tests
5. CI pipeline must be green before merge

## Configuration

Project uses YAML configuration files in `configs/` directory:
- `development.yaml`: Local development settings
- `phase1_mvp.yaml`: MVP phase configuration
- Experiment-specific configs for CIFAR-10/100

Configuration is loaded through `src/esper/configs.py` using the `EsperConfig` class.