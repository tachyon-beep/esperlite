# Esper - Morphogenetic Training Platform

A neural network training system that enables autonomous architectural evolution through morphogenetic architectures.

## Overview

Esper is a revolutionary approach to neural network training that enables models to evolve their own architecture during training. Instead of static, pre-defined architectures, Esper uses **morphogenetic architectures** - neural networks that can autonomously detect computational bottlenecks and patch them by grafting specialized sub-networks called **Blueprints**.

## Key Features

- **Autonomous Architecture Evolution**: Models can modify their own structure during training
- **Strategic Controller (Tamiyo)**: AI-powered system that analyzes models and identifies where improvements are needed
- **Generative Architect (Karn)**: AI system that continuously invents new architectural primitives
- **Zero Training Disruption**: Asynchronous compilation pipeline ensures training never stops
- **Parameter Efficiency**: Targeted capacity injection is dramatically more efficient than full model retraining

## Architecture

The system consists of 11 specialized components organized into three functional planes:

### Training Plane

- **Tolaria**: Training orchestrator
- **Kasmina**: Execution layer

### Control Plane

- **Tamiyo**: Strategic controller (AI policy network)
- **Simic**: Policy training environment
- **Emrakul**: Architectural sculptor

### Innovation Plane

- **Karn**: Generative architect (AI design network)
- **Tezzeret**: Compilation forge
- **Urabrask**: Evaluation engine
- **Urza**: Central library

### Infrastructure

- **Oona**: Message bus for inter-component communication
- **Nissa**: Observability and auditing platform

## Development Setup

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd esperlite
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   pip install -e .[dev]
   ```

5. **Start infrastructure services**

   ```bash
   cd docker
   docker-compose up -d
   ```

6. **Run tests**

   ```bash
   pytest
   ```

### Development Workflow

1. **Code Quality Checks**

   ```bash
   # Format code
   black src tests
   
   # Lint code
   ruff check src tests
   
   # Type checking
   pytype
   ```

2. **Testing**

   ```bash
   # Run all tests
   pytest
   
   # Run tests with coverage
   pytest --cov=src/esper --cov-report=html
   ```

3. **Infrastructure Management**

   ```bash
   # Start services
   docker-compose -f docker/docker-compose.yml up -d
   
   # Stop services
   docker-compose -f docker/docker-compose.yml down
   
   # View logs
   docker-compose -f docker/docker-compose.yml logs -f
   ```

## Performance Optimization

### GNN Acceleration

For maximum performance in Tamiyo strategic decision-making, install the acceleration dependencies:

```bash
pip install -e .[acceleration]
```

This enables 2-10x faster graph operations via torch-scatter acceleration. The system gracefully falls back to standard implementations if not installed.

#### System Requirements

- **CUDA Toolkit**: Required for torch-scatter compilation
- **Compatible PyTorch**: Version 2.2+ (see requirements)
- **GPU**: Recommended for optimal performance

#### Installation Examples

```bash
# Development with acceleration
pip install -e .[dev,acceleration]

# Production deployment
pip install -e .[acceleration]

# Base installation (fallback mode)
pip install -e .
```

#### Verifying Acceleration Status

Check if acceleration is active in your Python code:

```python
from src.esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig

config = PolicyConfig()
policy = TamiyoPolicyGNN(config)
status = policy.acceleration_status

print(f"Acceleration enabled: {status['acceleration_enabled']}")
print(f"Fallback mode: {status['fallback_mode']}")
```

#### Troubleshooting

**Compilation Errors:**

- Ensure CUDA toolkit is installed and matches PyTorch version
- Set `CUDA_HOME` environment variable
- Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

**Version Conflicts:**

- torch-scatter 2.1.0+ requires PyTorch 2.2+
- Verify versions: `pip list | grep torch`

**Performance Issues:**

- Check logs for "torch-scatter acceleration enabled" message
- Monitor forward pass latency improvements (2-10x expected)
- Ensure GPU is available for optimal performance

## Project Structure

```plaintext
esperlite/
├── .github/workflows/       # CI/CD pipelines
├── configs/                 # Configuration files
├── docker/                  # Docker infrastructure
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
├── src/esper/              # Main source code
│   ├── contracts/          # Data contracts and models
│   ├── services/           # Service implementations
│   ├── core/               # Core functionality
│   └── utils/              # Shared utilities
├── tests/                  # Test suite
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Current Status

**Phase 0: Foundation Layer** ✅

- [x] Project structure established
- [x] Core data contracts defined
- [x] Infrastructure as code (Docker Compose)
- [x] CI/CD pipeline configured
- [x] Development environment setup

**Phase 1: Core Morphogenetic System** ✅

- [x] Real kernel execution with PyTorch JIT/pickle support
- [x] Enhanced GPU-resident kernel caching with metadata validation
- [x] Comprehensive error recovery with circuit breakers
- [x] KasminaLayer execution engine with state management
- [x] Basic morphogenetic mechanics (wrap/unwrap, dynamic loading)
- [x] Extensive test coverage (2,500+ lines)

**Phase 2: Intelligence Layer** ✅

- [x] Strategic Controller (Tamiyo) with GNN-based policy network
- [x] Multi-head attention graph neural networks
- [x] Uncertainty quantification and safety regularization
- [x] Production health monitoring and telemetry
- [x] Autonomous decision-making service
- [x] Comprehensive safety testing

**Phase 3: Advanced Features** 🚧

- [ ] Generative Architect (Karn) neural network
- [ ] Asynchronous compilation pipeline (Tezzeret)
- [ ] Architectural sculptor (Emrakul)
- [ ] Evaluation engine (Urabrask)
- [ ] Full production deployment infrastructure

## Infrastructure Status

⚠️ **Important**: While the core system is implemented and tested, production deployment infrastructure is currently limited to development configurations. Production deployment requires:

- Containerization of services (Dockerfiles)
- Orchestration setup (Kubernetes/Helm charts)
- Production Redis and PostgreSQL clusters
- Monitoring and observability stack
- Load balancing and scaling configurations

See the [deployment guide](docs/deployment/README.md) for current deployment options and roadmap.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use Esper in your research, please cite:

```bibtex
@software{esper2025,
  title={Esper: Morphogenetic Training Platform},
  author={John Morrissey},
  year={2025},
  url={https://github.com/johnmorrissey/esperlite}
}
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the development team.
