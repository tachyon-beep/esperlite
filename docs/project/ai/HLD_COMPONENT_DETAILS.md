# HLD Component Architecture Details

This document provides detailed information about each of the 11 subsystems in the Esper architecture.

## 7.1 Major Subsystems

### 7.1.1 Tolaria - The Training Orchestrator

**Purpose**: Master orchestrator managing the primary training loop while coordinating morphogenetic adaptations.

**Key Responsibilities**:
- Freezes base model parameters during grafting operations
- Rebuilds optimizer to include newly grafted parameters  
- Provides system "heartbeat" by invoking Tamiyo at epoch end
- Acts as final authority on system stability with emergency rollback capability
- Manages checkpointing/restoration of both host model and Tamiyo state

**Core Logic**: 
- Standard training loop followed by validation loop
- End-of-Epoch Hook assembles SystemStatePacket for Tamiyo
- DynamicOptimizerManager rebuilds optimizer for new parameters

**Key Interfaces**:
- Invokes `tamiyo.step(SystemStatePacket)`
- Consumes `AdaptationSignal` to drive actions
- Listens to `telemetry.model.health` for stability

### 7.1.2 Kasmina - The Execution Layer  

**Purpose**: Streamlined, high-performance pure executor loading pre-compiled kernels.

**Key Responsibilities**:
- Manages 11-stage lifecycle state machine for each Seed
- Monitors network chunks and generates execution telemetry
- Receives commands from Tamiyo with kernel_artifact_id
- Loads kernels from Urza with GPU-resident LRU cache
- Executes loaded kernels with maximum performance

**Core Logic**:
- Chunked Architecture splitting layer activations
- KasminaSeed as stateful agent executing controller commands
- Robust error handling with graceful fallback

**Key Interfaces**:
- Publishes health signals to `telemetry.seed.health`
- Exposes `request_germination()` and `cancel_germination()`

### 7.1.3 Tamiyo - The Strategic Controller

**Purpose**: Strategic decision-making brain determining if/when/where/how to adapt.

**Key Responsibilities**:
- Analyzes metrics to detect plateaus and weaknesses
- Can issue force rollback for critical instability
- Selects optimal dormant Seed for intervention
- Queries Urza for validated kernels by tags
- Calculates tamiyo_interest scores for blueprint importance
- Selects GraftingStrategy based on conditions
- Generates FieldReports for Karn

**Core Logic**:
- Pluggable intelligence (heuristics → GNN)
- Heterogeneous Graph Neural Network modeling host network
- Rich feature extraction and strategic planning

**Key Interfaces**:
- Implements `step(SystemStatePacket) → AdaptationSignal`
- Queries Urza via `search_kernels_by_tags()`
- Publishes to `control.kasmina.commands`

### 7.1.4 Karn - The Generative Architect  

**Purpose**: Autonomous R&D engine inventing novel architectural solutions.

**Key Responsibilities**:
- Generates diverse BlueprintIR designs
- Learns from FieldReports to improve designs
- Maintains novelty through exploration
- Ensures generated designs meet constraints

**Core Logic**:
- Generative model producing architectural graphs
- Reward model trained on performance outcomes
- Exploration/exploitation balance

**Key Interfaces**:
- Consumes FieldReports from Tamiyo
- Produces BlueprintIR structures
- Publishes to Urza library

### 7.1.5 Tezzeret - The Compilation Forge

**Purpose**: Transforms blueprints into optimized kernel artifacts asynchronously.

**Key Responsibilities**:
- Polls Urza for UNVALIDATED blueprints
- Performs static analysis and safety checks
- Compiles to multiple optimization targets
- Manages compilation worker pool
- Pushes compiled kernels to Urza

**Core Logic**:
- Multi-tier compilation pipeline
- Hardware-specific optimizations
- Caching and incremental compilation

**Key Interfaces**:
- Polls blueprints from Urza
- Pushes CompiledKernelArtifacts
- Reports compilation metrics

### 7.1.6 Urabrask - The Evaluation Engine

**Purpose**: Rigorous testing and characterization of compiled kernels.

**Key Responsibilities**:
- Polls PENDING_BENCHMARKING kernels
- Runs comprehensive test suites
- Performs empirical characterization
- Generates performance tags
- Validates safety properties

**Core Logic**:
- Multi-dimensional benchmarking
- Statistical significance testing
- Tag generation from results

**Key Interfaces**:
- Polls kernels from Urza
- Pushes ValidationReports
- Updates kernel status

### 7.1.7 Urza - The Central Library

**Purpose**: Centralized repository for all morphogenetic assets.

**Key Responsibilities**:
- Stores BlueprintIR designs
- Manages CompiledKernelArtifacts
- Tracks asset lifecycle states
- Provides rich query interface
- Maintains asset lineage

**Core Logic**:
- ACID-compliant asset management
- Tag-based search capabilities
- Version control and lineage

**Key Interfaces**:
- REST API for asset operations
- Pub/sub for state changes
- Query interface for Tamiyo

### 7.1.8 Simic - The Policy Sculptor

**Purpose**: Trains Tamiyo's decision-making policies through RL.

**Key Responsibilities**:
- Simulates morphogenetic scenarios
- Trains controller policies
- Evaluates policy performance
- Manages experience replay

**Core Logic**:
- RL environment for architecture evolution
- Policy gradient methods
- Safe exploration strategies

### 7.1.9 Emrakul - The Architectural Sculptor

**Purpose**: Trains Karn's generative capabilities.

**Key Responsibilities**:
- Curates training data
- Trains generative models
- Evaluates design quality
- Manages model versions

**Core Logic**:
- VAE/GAN architectures
- Quality metrics
- Diversity preservation

### 7.1.10 Oona - The Message Bus

**Purpose**: Event-driven communication backbone.

**Key Responsibilities**:
- Reliable message delivery
- Topic-based routing
- Event ordering guarantees
- Message persistence

**Key Topics**:
- `telemetry.seed.health`
- `control.kasmina.commands`
- `events.adaptation.*`
- `metrics.system.*`

### 7.1.11 Nissa - The Observability Platform

**Purpose**: Comprehensive monitoring and auditing.

**Key Responsibilities**:
- Collects system-wide metrics
- Maintains audit trail
- Provides dashboards
- Enables debugging

**Core Features**:
- Real-time metrics
- Historical analysis
- Anomaly detection
- Compliance reporting

## Data Flow Summary

1. **Telemetry Flow**: Seeds → Oona → Tamiyo/Nissa
2. **Control Flow**: Tamiyo → Oona → Kasmina Seeds  
3. **Compilation Flow**: Karn → Urza → Tezzeret → Urza → Urabrask → Urza
4. **Innovation Flow**: Tamiyo → Karn (FieldReports) → New Blueprints

## State Management

- **Tolaria**: Training state, checkpoints, optimizer
- **Tamiyo**: Policy state, decision history, metrics
- **Kasmina**: Seed lifecycles, health buffers, GPU cache
- **Urza**: Asset catalog, lifecycle states, lineage

Synchronization happens at epoch boundaries via Tolaria's EndOfEpoch event.