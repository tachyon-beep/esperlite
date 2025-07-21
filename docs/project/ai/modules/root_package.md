# Root Package Module (`src/esper/`)

## Overview

The root package serves as the main entry point for the Esper Morphogenetic Training Platform, providing the primary public API that end users interact with to transform standard PyTorch models into morphogenetic models.

## Files

### `__init__.py` - Main Package Entry Point

**Purpose:** Defines the public API surface and exports core functionality for end users.

**Key Exports:**
- `wrap` - Primary function to transform PyTorch models into morphogenetic models
- `unwrap` - Function to extract original model from MorphableModel
- `MorphableModel` - Enhanced model class with morphogenetic capabilities  
- `KasminaLayer` - Core execution layer for kernel loading
- `SeedLifecycleState` - State enumeration for seed lifecycle management
- `EsperConfig` - Configuration management class

**Metadata:**
- **Version:** 0.2.0
- **API Version:** v1

**Implementation Status:** Fully functional with working model wrapping, kernel execution (placeholder), and service integration.

**Dependencies:**
```python
from .core.model_wrapper import wrap, MorphableModel, unwrap
from .execution.kasmina_layer import KasminaLayer
from .execution.state_layout import SeedLifecycleState
from .configs import EsperConfig
```

**Usage Pattern:**
```python
import esper

# Transform a standard PyTorch model
morphable_model = esper.wrap(pytorch_model)

# Access configuration
config = esper.EsperConfig(...)
```

### `configs.py` - Configuration System

**Purpose:** Provides Pydantic-based configuration models for parsing YAML configuration files and managing system-wide settings.

#### Key Classes

**`DatabaseConfig`**
```python
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "urza_db"
    username: str
    password: str
    ssl_mode: str = "prefer"
```
- **Purpose:** PostgreSQL connection configuration
- **Features:** Default values for development, SSL support

**`RedisConfig`**
```python
class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
```
- **Purpose:** Redis message bus configuration
- **Features:** Optional authentication, database selection

**`StorageConfig`**
```python
class StorageConfig(BaseModel):
    endpoint_url: str = "http://localhost:9000"
    access_key: str
    secret_key: str
    bucket_name: str = "esper-artifacts"
    region: str = "us-east-1"
```
- **Purpose:** S3/MinIO object storage configuration
- **Features:** MinIO development defaults, AWS compatibility

**`ComponentConfig`**
```python
class ComponentConfig(BaseModel):
    enabled: bool = True
    replicas: int = 1
    resources: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
```
- **Purpose:** Individual service component configuration
- **Features:** Scaling parameters, resource allocation

**`EsperConfig`** (Master Configuration)
```python
class EsperConfig(BaseModel):
    name: str
    version: str = "0.1.0"
    environment: str = "development"
    database: DatabaseConfig
    redis: RedisConfig
    storage: StorageConfig
    components: Dict[str, ComponentConfig] = Field(default_factory=dict)
    training: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(
        default_factory=lambda: {"level": "INFO", "format": "structured"}
    )
```
- **Purpose:** Top-level configuration orchestrator
- **Features:** Environment-specific settings, nested configuration validation

#### Configuration Usage Patterns

**YAML Configuration File:**
```yaml
name: "esper-training"
version: "0.2.0"
environment: "development"

database:
  host: "localhost"
  port: 5432
  database: "esper_db"
  username: "${POSTGRES_USER}"
  password: "${POSTGRES_PASSWORD}"

redis:
  host: "localhost"
  port: 6379

storage:
  endpoint_url: "http://localhost:9000"
  access_key: "${MINIO_ACCESS_KEY}"
  secret_key: "${MINIO_SECRET_KEY}"
  bucket_name: "esper-artifacts"

components:
  tamiyo:
    enabled: true
    replicas: 1
    config:
      policy_model_path: "/models/tamiyo_policy.pt"
  
training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 100
```

**Python Usage:**
```python
from esper import EsperConfig
import yaml

# Load from YAML
with open('config.yaml') as f:
    config_data = yaml.safe_load(f)
    config = EsperConfig(**config_data)

# Access nested configuration
db_config = config.database
redis_config = config.redis
```

## Architecture Integration

The root package serves as the primary integration point between:

1. **User Code** → `esper.wrap()` → **Core Module**
2. **Configuration Files** → `EsperConfig` → **All Services**
3. **Public API** → Internal modules for execution and orchestration

## Dependencies

**External:**
- `pydantic` - Configuration validation and serialization
- `typing` - Type hints and annotations

**Internal:**
- `esper.core.model_wrapper` - Model transformation functionality
- `esper.execution.kasmina_layer` - Core execution engine
- `esper.execution.state_layout` - State management
- All service modules indirectly through configuration

## Best Practices

### Configuration Management
- Use environment variables for sensitive data (passwords, keys)
- Provide sensible defaults for development environments
- Validate all configuration at startup using Pydantic
- Support both YAML files and programmatic configuration

### API Design
- Keep the public API surface minimal and stable
- Export only essential classes and functions
- Maintain backward compatibility in version updates
- Provide clear type hints for all public functions

## Common Usage Patterns

### Basic Model Wrapping
```python
import torch
import torch.nn as nn
import esper

# Create a standard PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Transform to morphogenetic model
morphable_model = esper.wrap(
    model,
    target_layers=[nn.Linear],
    seeds_per_layer=4,
    cache_size_mb=128
)

# Use like a normal PyTorch model
output = morphable_model(input_tensor)
```

### Configuration-Driven Setup
```python
import esper
import yaml

# Load configuration
with open('production.yaml') as f:
    config = esper.EsperConfig(**yaml.safe_load(f))

# Access service configurations
database_url = f"postgresql://{config.database.username}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.database}"
```

## Error Handling

### Configuration Errors
- Pydantic validation errors for invalid configuration values
- Clear error messages for missing required fields
- Environment variable substitution errors

### API Usage Errors
- Type validation for function parameters
- Clear exceptions for unsupported model types
- Helpful error messages for common mistakes

## Performance Considerations

- Configuration parsing is done once at startup
- Minimal overhead in public API functions
- Lazy loading of heavy dependencies
- Efficient import structure to minimize startup time

## Future Enhancements

1. **Configuration Validation**
   - Add more sophisticated validation rules
   - Support for configuration schemas
   - Runtime configuration updates

2. **API Extensions**
   - Additional model transformation options
   - More granular control over morphogenetic behavior
   - Integration with popular ML frameworks beyond PyTorch

3. **Developer Experience**
   - Better error messages and debugging information
   - Configuration templates for common use cases
   - Interactive configuration builders