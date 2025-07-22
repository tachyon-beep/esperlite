# Esper - Morphogenetic Training Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready platform that enables neural networks to autonomously evolve their architecture during training.

## Key Features

- 🧬 **Autonomous Architecture Evolution** - Models adapt their structure in real-time
- 🧠 **GNN-based Strategic Controller** - AI-powered decision making for adaptations  
- ⚡ **Zero Training Disruption** - Asynchronous compilation maintains training flow
- 📈 **Parameter Efficiency** - Targeted capacity injection vs full retraining
- 🛡️ **Production Ready** - Comprehensive error recovery, monitoring, and safety

## Quick Start

```bash
# Clone and setup
git clone https://github.com/esper/esperlite
cd esperlite
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .[dev]

# Start infrastructure (one-time)
./scripts/start-demo.sh

# Train with morphogenetic capabilities
python train.py --quick-start cifar10
```

See [DEMO_QUICKSTART.md](DEMO_QUICKSTART.md) for detailed demo instructions.

## Usage

```python
from esper import wrap
import torch.nn as nn

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Enable morphogenetic capabilities
morphable_model = wrap(
    model,
    target_layers=[nn.Linear],
    seeds_per_layer=4
)

# Train normally - adaptations happen automatically
# The model will evolve its architecture as needed
```

## Architecture

Esper uses a three-plane architecture with specialized components:

- **Training Plane**: Tolaria (orchestrator) + Kasmina (execution)
- **Control Plane**: Tamiyo (GNN strategic controller) + Emrakul (sculptor)
- **Innovation Plane**: Karn (generative architect) + Tezzeret (compiler)
- **Infrastructure**: Oona (message bus) + Urza (kernel library)

See [Architecture Overview](docs/architecture/README.md) for details.

## Documentation

- [API Reference](docs/api/README.md)
- [Development Guide](CONTRIBUTING.md)
- [Deployment Guide](docs/deployment/README.md)
- [Configuration Reference](configs/README.md)
- [Examples](examples/README.md)

## Performance

For GPU acceleration and 2-10x faster GNN operations:

```bash
pip install -e .[acceleration]
```

This enables torch-scatter optimizations with automatic fallback if unavailable.

## Status

- ✅ **Phase 1 & 2 Complete** - Core morphogenetic system with GNN-based intelligence
- ✅ **Production Infrastructure** - Docker, Kubernetes, monitoring, and deployment tools
- ✅ **Comprehensive Testing** - 90%+ coverage with unit and integration tests
- 🚧 **Phase 3 In Progress** - Generative kernel synthesis and advanced compilation

## Production Deployment

Full production infrastructure is available:

- Docker containerization with multi-stage builds
- Kubernetes manifests with Kustomize overlays
- Prometheus/Grafana monitoring stack
- Health checks and circuit breakers
- Secure credential management

See [PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md) for the complete checklist.

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

For questions or issues, please open an issue on GitHub.

---

<p align="center">
  <i>Evolving neural architectures, one adaptation at a time.</i>
</p>
