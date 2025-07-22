# Changelog

All notable changes to the Esper Morphogenetic Training Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Production deployment infrastructure (Docker, Kubernetes)
- Comprehensive tech demo environment
- Network accessibility configuration
- Firewall management scripts
- Enhanced monitoring with Prometheus and Grafana

### Changed

- Updated README to reflect actual implementation status
- Improved Docker Compose configurations for external access
- Enhanced documentation structure

### Fixed

- Docker Compose v2 compatibility
- External network access configuration

## [0.2.0] - 2024-07-22

### Added

- Phase 2: Intelligence Layer (Tamiyo GNN implementation)
- Graph Neural Network with multi-head attention
- Uncertainty quantification and safety regularization
- Production health monitoring
- Autonomous decision-making service
- Comprehensive safety testing

### Changed

- Enhanced error recovery system
- Improved kernel execution performance
- Updated test coverage

## [0.1.0] - 2024-07-15

### Added

- Phase 1: Core Morphogenetic System
- Real kernel execution with PyTorch JIT/pickle support
- Enhanced GPU-resident kernel caching with metadata validation
- Comprehensive error recovery with circuit breakers
- KasminaLayer execution engine with state management
- Basic morphogenetic mechanics (wrap/unwrap, dynamic loading)
- Extensive test coverage (2,500+ lines)

### Changed

- Moved from placeholder to real kernel execution
- Implemented production-grade error handling
- Added metadata-aware caching

## [0.0.1] - 2024-07-01

### Added

- Initial project structure
- Core data contracts
- Basic infrastructure setup
- Development environment configuration
- CI/CD pipeline

[Unreleased]: https://github.com/esper/esperlite/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/esper/esperlite/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/esper/esperlite/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/esper/esperlite/releases/tag/v0.0.1
