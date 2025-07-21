# Services Module (`src/esper/services/`)

## Overview

The services module implements the distributed system components of the Esper platform, organized according to the three-plane architecture: Training Plane (Tolaria), Control Plane (Tamiyo, Urza), and Innovation Plane (Tezzeret, Oona). Each service is independently deployable with well-defined APIs and clear separation of concerns.

## Implementation Status Summary

### Fully Operational Services
- **Oona Client** - Production-ready Redis Streams message bus
- **Urza Service** - Complete REST API with PostgreSQL backend
- **Tezzeret Worker** - Functional blueprint compilation service
- **Tolaria Trainer** - Complete training orchestration with real integrations
- **Service Clients** - Production HTTP clients with circuit breakers

### Partially Implemented
- **Tamiyo Service** - Core GNN policy works, using simulated data for training

## Architecture Summary

### Three-Plane Architecture
- **Training Plane:** Tolaria (orchestrator) + Kasmina (execution engine)
- **Control Plane:** Tamiyo (strategic controller) + Urza (artifact storage)
- **Innovation Plane:** Tezzeret (compilation forge) + Oona (message bus)

### Service Communication Patterns
- **Message Bus:** Redis Streams via Oona for real-time coordination
- **REST APIs:** Heavy data transfer and artifact management
- **Event-Driven:** Pub/sub for system-wide coordination

## Files Overview

### `__init__.py` - Services Module Initialization

**Purpose:** Placeholder for service layer module.

**Contents:**
```python
"""
Service layer for the Esper system.
Each service is a standalone component that can be deployed independently.
"""
```

**Status:** Minimal implementation - serves as namespace for service components.

### `contracts.py` - Service API Contracts

**Purpose:** Simplified MVP contracts for Phase 1 inter-service communication.

#### Key Classes

**`SimpleBlueprintContract`** - Minimal Blueprint Representation
```python
class SimpleBlueprintContract(BaseModel):
    """Simplified blueprint contract for Phase 1 MVP."""
    blueprint_id: str
    name: str
    architecture_ir: str  # JSON representation of architecture
    status: str = "unvalidated"  # Simple string status for MVP
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**`SimpleCompiledKernelContract`** - Kernel Metadata
```python
class SimpleCompiledKernelContract(BaseModel):
    """Simplified compiled kernel contract for Phase 1 MVP."""
    kernel_id: str
    blueprint_id: str
    kernel_binary_ref: str  # S3 URI to binary artifact
    status: KernelStatus = KernelStatus.VALIDATED
    compilation_time: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

**API Request/Response Models:**
```python
class BlueprintSubmissionRequest(BaseModel):
    """Request to submit a new blueprint."""
    name: str
    architecture_ir: str
    submitted_by: str = "system"

class BlueprintSubmissionResponse(BaseModel):
    """Response after blueprint submission."""
    blueprint_id: str
    status: str
    message: str
```

**Features:**
- Pydantic validation for type safety
- Simplified status management for MVP
- UTC timestamp handling
- Clear request/response patterns

**Usage Pattern:**
```python
# Submit blueprint
request = BlueprintSubmissionRequest(
    name="Attention Module",
    architecture_ir='{"type": "attention", "heads": 8}',
    submitted_by="Karn-Architect-1"
)

# Process in service
response = BlueprintSubmissionResponse(
    blueprint_id=str(uuid4()),
    status="submitted",
    message="Blueprint queued for compilation"
)
```

### `oona_client.py` - Message Bus Client

**Purpose:** Redis Streams client providing reliable message bus functionality for inter-service communication.

#### Key Components

**`OonaClient`** - Redis Streams Pub/Sub Client
```python
class OonaClient:
    """
    Redis Streams client for the Oona message bus.
    
    Provides publish/subscribe functionality for inter-service communication
    with consumer group support and automatic retry logic.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Oona client.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client = None
        self._connected = False
        
        # Connection management
        self._connection_lock = asyncio.Lock()
        self._max_retries = 3
        self._retry_delay = 1.0
```

**Core Methods:**

**Connection Management:**
```python
async def connect(self) -> None:
    """Establish connection to Redis with retry logic."""
    async with self._connection_lock:
        if self._connected:
            return
            
        for attempt in range(self._max_retries):
            try:
                import redis.asyncio as redis
                
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                await self.redis_client.ping()
                self._connected = True
                return
                
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise ConnectionError(f"Failed to connect to Redis after {self._max_retries} attempts: {e}")
                await asyncio.sleep(self._retry_delay * (2 ** attempt))
```

**Message Publishing:**
```python
async def publish(self, message: OonaMessage) -> None:
    """
    Publish a message to the appropriate stream.
    
    Args:
        message: OonaMessage to publish
        
    Raises:
        ConnectionError: If not connected to Redis
        PublishError: If message publishing fails
    """
    if not self._connected:
        await self.connect()
    
    try:
        # Convert message to Redis stream format
        stream_name = f"esper:{message.topic.value}"
        message_data = {
            "event_id": message.event_id,
            "sender_id": message.sender_id,
            "timestamp": message.timestamp.isoformat(),
            "trace_id": message.trace_id,
            "payload": json.dumps(message.payload)
        }
        
        # Publish to stream
        message_id = await self.redis_client.xadd(stream_name, message_data)
        
        logger.debug(f"Published message {message.event_id} to {stream_name} with ID {message_id}")
        
    except Exception as e:
        logger.error(f"Failed to publish message {message.event_id}: {e}")
        raise PublishError(f"Message publishing failed: {e}")
```

**Message Consumption:**
```python
async def consume(
    self, 
    streams: List[str], 
    consumer_group: str, 
    consumer_name: str,
    count: int = 10,
    timeout: int = 1000
) -> List[OonaMessage]:
    """
    Consume messages from streams using consumer groups.
    
    Args:
        streams: List of stream names to consume from
        consumer_group: Consumer group name
        consumer_name: Individual consumer name
        count: Maximum messages to read
        timeout: Timeout in milliseconds
        
    Returns:
        List of consumed OonaMessage objects
    """
    if not self._connected:
        await self.connect()
    
    try:
        # Ensure consumer groups exist
        for stream in streams:
            stream_name = f"esper:{stream}"
            try:
                await self.redis_client.xgroup_create(
                    stream_name, consumer_group, id="$", mkstream=True
                )
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
        
        # Read from streams
        stream_dict = {f"esper:{stream}": ">" for stream in streams}
        
        response = await self.redis_client.xreadgroup(
            consumer_group,
            consumer_name,
            stream_dict,
            count=count,
            block=timeout
        )
        
        # Convert to OonaMessage objects
        messages = []
        for stream_name, stream_messages in response:
            for message_id, fields in stream_messages:
                try:
                    # Parse message fields
                    oona_message = OonaMessage(
                        event_id=fields["event_id"],
                        sender_id=fields["sender_id"],
                        timestamp=datetime.fromisoformat(fields["timestamp"]),
                        trace_id=fields["trace_id"],
                        topic=self._extract_topic_from_stream(stream_name),
                        payload=json.loads(fields["payload"])
                    )
                    
                    # Acknowledge message
                    await self.redis_client.xack(stream_name, consumer_group, message_id)
                    
                    messages.append(oona_message)
                    
                except Exception as e:
                    logger.error(f"Failed to parse message {message_id}: {e}")
                    # Could implement dead letter queue here
        
        return messages
        
    except Exception as e:
        logger.error(f"Failed to consume messages: {e}")
        raise ConsumptionError(f"Message consumption failed: {e}")
```

**Health Checking:**
```python
async def health_check(self) -> Dict[str, Any]:
    """
    Check connection health and return status.
    
    Returns:
        Dictionary containing health information
    """
    try:
        if not self._connected:
            return {"status": "disconnected", "error": "Not connected to Redis"}
        
        # Test Redis connectivity
        start_time = time.time()
        pong = await self.redis_client.ping()
        latency = (time.time() - start_time) * 1000  # ms
        
        if pong:
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "latency_ms": round(latency, 2),
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients")
            }
        else:
            return {"status": "unhealthy", "error": "Redis ping failed"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

**Features:**
- **Consumer Groups:** Reliable message distribution with acknowledgment
- **Automatic Retry:** Connection retry with exponential backoff
- **Health Monitoring:** Connection status and performance metrics
- **Error Handling:** Comprehensive error recovery and logging
- **Stream Management:** Automatic stream and consumer group creation

**Integration Pattern:**
```python
# Service initialization
oona_client = OonaClient("redis://localhost:6379")
await oona_client.connect()

# Publishing
message = OonaMessage(
    sender_id="Tamiyo-Controller-1",
    trace_id="trace-123",
    topic=TopicNames.CONTROL_KASMINA_COMMANDS,
    payload={"command": "load_kernel", "layer": "transformer.0"}
)
await oona_client.publish(message)

# Consuming
messages = await oona_client.consume(
    streams=["telemetry.seed.health"],
    consumer_group="tamiyo-controllers",
    consumer_name="controller-1"
)

for message in messages:
    await process_health_signal(message.payload)
```

---

# Urza Service - Central Asset Hub (`src/esper/services/urza/`)

## Overview

Urza serves as the central library and source of truth for all blueprints and compiled kernels in the Esper system. It provides REST APIs for blueprint submission, kernel retrieval, and artifact management with PostgreSQL metadata storage and S3 binary storage.

### `__init__.py` - Urza Service Initialization

**Purpose:** Service module initialization.

**Contents:** Minimal initialization for Urza service components.

### `database.py` - Database Configuration

**Purpose:** PostgreSQL connection management and session configuration for Urza service.

#### Key Components

**`DatabaseConfig`** - Connection Configuration
```python
class DatabaseConfig:
    """Database configuration for Urza service."""
    
    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = os.getenv("POSTGRES_DB", "urza_db")
        self.username = os.getenv("POSTGRES_USER", "urza")
        self.password = os.getenv("POSTGRES_PASSWORD", "")
        self.ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
        
        # Connection pooling
        self.pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "10"))
        self.pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
        
    @property
    def url(self) -> str:
        """Get SQLAlchemy connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"
```

**Database Engine Creation:**
```python
def create_engine() -> Engine:
    """
    Create SQLAlchemy engine with connection pooling.
    
    Returns:
        Configured SQLAlchemy engine
    """
    config = DatabaseConfig()
    
    engine = sqlalchemy.create_engine(
        config.url,
        # Connection pooling configuration
        poolclass=StaticPool,  # Note: Consider QueuePool for production
        pool_size=config.pool_size,
        max_overflow=config.max_overflow,
        pool_timeout=config.pool_timeout,
        pool_pre_ping=True,  # Verify connections before use
        
        # Performance settings
        echo=False,  # Set to True for SQL logging in development
        future=True,  # Use SQLAlchemy 2.0 API
    )
    
    return engine

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Yields:
        SQLAlchemy session
    """
    engine = create_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

**Features:**
- Environment-based configuration
- Connection pooling with configurable parameters
- FastAPI dependency injection support
- Pre-ping for connection validation

**Issues Identified:**
- **StaticPool:** May not be suitable for production workloads
- **Recommendation:** Use QueuePool for production deployments

### `models.py` - Database Models

**Purpose:** SQLAlchemy ORM models for blueprint and kernel metadata storage.

#### Key Models

**`Blueprint`** - Blueprint Metadata Table
```python
class Blueprint(Base):
    """Blueprint metadata table."""
    
    __tablename__ = "blueprints"
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Blueprint information
    name = Column(String, nullable=False)
    architecture_ir = Column(Text, nullable=False)  # JSON IR representation
    status = Column(String, nullable=False, default="unvalidated")
    
    # Metadata
    submitted_by = Column(String, nullable=False, default="system")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    compiled_kernels = relationship("CompiledKernel", back_populates="blueprint", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Blueprint(id={self.id}, name={self.name}, status={self.status})>"
```

**`CompiledKernel`** - Kernel Artifact Table
```python
class CompiledKernel(Base):
    """Compiled kernel metadata table."""
    
    __tablename__ = "compiled_kernels"
    
    # Primary key
    id = Column(String, primary_key=True)
    
    # Foreign key to blueprint
    blueprint_id = Column(String, ForeignKey("blueprints.id"), nullable=False)
    
    # Kernel information
    kernel_binary_ref = Column(Text, nullable=False)  # S3 URI
    status = Column(String, nullable=False, default="validated")
    compilation_time = Column(Float, nullable=False, default=0.0)
    
    # Performance metadata (JSON)
    performance_metrics = Column(JSON, nullable=True)
    validation_report = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    blueprint = relationship("Blueprint", back_populates="compiled_kernels")
    
    def __repr__(self):
        return f"<CompiledKernel(id={self.id}, blueprint_id={self.blueprint_id}, status={self.status})>"
```

**Database Initialization:**
```python
def create_tables(engine: Engine) -> None:
    """
    Create all tables in the database.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(bind=engine)

def drop_tables(engine: Engine) -> None:
    """
    Drop all tables from the database.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.drop_all(bind=engine)
```

**Features:**
- **Foreign Key Relationships:** Proper normalization with cascade deletes
- **JSON Fields:** Flexible metadata storage for performance metrics
- **Automatic Timestamps:** Created/updated timestamp tracking
- **Status Tracking:** Simple string-based status for MVP

### `main.py` - REST API Service

**Purpose:** FastAPI service providing REST endpoints for blueprint and kernel management.

#### Key Components

**FastAPI Application Setup:**
```python
app = FastAPI(
    title="Urza API",
    description="Central library service for Esper blueprints and kernels",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )
    
    return response
```

**Public API Endpoints:**

**Blueprint Management:**
```python
@app.post("/api/v1/blueprints/", response_model=BlueprintSubmissionResponse)
async def submit_blueprint(
    request: BlueprintSubmissionRequest,
    session: Session = Depends(get_session)
):
    """
    Submit a new blueprint for compilation.
    
    Args:
        request: Blueprint submission request
        session: Database session
        
    Returns:
        Blueprint submission response
    """
    try:
        # Create blueprint record
        blueprint = Blueprint(
            id=str(uuid4()),
            name=request.name,
            architecture_ir=request.architecture_ir,
            status="unvalidated",
            submitted_by=request.submitted_by
        )
        
        session.add(blueprint)
        session.commit()
        
        logger.info(f"Blueprint {blueprint.id} submitted by {request.submitted_by}")
        
        return BlueprintSubmissionResponse(
            blueprint_id=blueprint.id,
            status="submitted",
            message="Blueprint submitted successfully"
        )
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to submit blueprint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/blueprints/{blueprint_id}")
async def get_blueprint(
    blueprint_id: str,
    session: Session = Depends(get_session)
):
    """
    Get blueprint by ID.
    
    Args:
        blueprint_id: Blueprint identifier
        session: Database session
        
    Returns:
        Blueprint information
    """
    blueprint = session.query(Blueprint).filter(Blueprint.id == blueprint_id).first()
    
    if not blueprint:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    
    return {
        "blueprint_id": blueprint.id,
        "name": blueprint.name,
        "architecture_ir": blueprint.architecture_ir,
        "status": blueprint.status,
        "submitted_by": blueprint.submitted_by,
        "created_at": blueprint.created_at.isoformat(),
        "updated_at": blueprint.updated_at.isoformat()
    }
```

**Kernel Management:**
```python
@app.get("/api/v1/kernels/{kernel_id}")
async def get_kernel(
    kernel_id: str,
    session: Session = Depends(get_session)
):
    """
    Get compiled kernel by ID.
    
    Args:
        kernel_id: Kernel identifier
        session: Database session
        
    Returns:
        Kernel metadata including S3 binary reference
    """
    kernel = session.query(CompiledKernel).filter(CompiledKernel.id == kernel_id).first()
    
    if not kernel:
        raise HTTPException(status_code=404, detail="Kernel not found")
    
    return {
        "kernel_id": kernel.id,
        "blueprint_id": kernel.blueprint_id,
        "kernel_binary_ref": kernel.kernel_binary_ref,
        "status": kernel.status,
        "compilation_time": kernel.compilation_time,
        "performance_metrics": kernel.performance_metrics,
        "created_at": kernel.created_at.isoformat()
    }

@app.post("/api/v1/kernels/")
async def submit_compiled_kernel(
    kernel_data: dict,
    session: Session = Depends(get_session)
):
    """
    Submit a compiled kernel (called by Tezzeret).
    
    Args:
        kernel_data: Kernel metadata
        session: Database session
        
    Returns:
        Kernel submission response
    """
    try:
        kernel = CompiledKernel(
            id=str(uuid4()),
            blueprint_id=kernel_data["blueprint_id"],
            kernel_binary_ref=kernel_data["kernel_binary_ref"],
            status=kernel_data.get("status", "validated"),
            compilation_time=kernel_data.get("compilation_time", 0.0),
            performance_metrics=kernel_data.get("performance_metrics"),
            validation_report=kernel_data.get("validation_report")
        )
        
        session.add(kernel)
        
        # Update blueprint status
        blueprint = session.query(Blueprint).filter(Blueprint.id == kernel.blueprint_id).first()
        if blueprint:
            blueprint.status = "compiled"
            blueprint.updated_at = datetime.utcnow()
        
        session.commit()
        
        return {"kernel_id": kernel.id, "status": "submitted"}
        
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to submit kernel: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**Internal API Endpoints (for Tezzeret worker):**
```python
@app.get("/internal/unvalidated-blueprints")
async def get_unvalidated_blueprints(
    limit: int = 10,
    session: Session = Depends(get_session)
):
    """
    Get unvalidated blueprints for compilation (Tezzeret polling endpoint).
    
    Args:
        limit: Maximum number of blueprints to return
        session: Database session
        
    Returns:
        List of unvalidated blueprints
    """
    blueprints = (
        session.query(Blueprint)
        .filter(Blueprint.status == "unvalidated")
        .order_by(Blueprint.created_at)
        .limit(limit)
        .all()
    )
    
    return [
        {
            "blueprint_id": bp.id,
            "name": bp.name,
            "architecture_ir": bp.architecture_ir,
            "created_at": bp.created_at.isoformat()
        }
        for bp in blueprints
    ]

@app.post("/internal/blueprints/{blueprint_id}/status")
async def update_blueprint_status(
    blueprint_id: str,
    status_data: dict,
    session: Session = Depends(get_session)
):
    """
    Update blueprint status (called by Tezzeret).
    
    Args:
        blueprint_id: Blueprint identifier
        status_data: New status information
        session: Database session
        
    Returns:
        Update confirmation
    """
    blueprint = session.query(Blueprint).filter(Blueprint.id == blueprint_id).first()
    
    if not blueprint:
        raise HTTPException(status_code=404, detail="Blueprint not found")
    
    blueprint.status = status_data["status"]
    blueprint.updated_at = datetime.utcnow()
    
    session.commit()
    
    return {"blueprint_id": blueprint_id, "status": blueprint.status}
```

**Health Check Endpoint:**
```python
@app.get("/health")
async def health_check(session: Session = Depends(get_session)):
    """
    Health check endpoint.
    
    Returns:
        Service health information
    """
    try:
        # Test database connectivity
        session.execute("SELECT 1")
        
        # Get basic statistics
        blueprint_count = session.query(Blueprint).count()
        kernel_count = session.query(CompiledKernel).count()
        
        return {
            "status": "healthy",
            "service": "urza",
            "database": "connected",
            "blueprints": blueprint_count,
            "kernels": kernel_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "urza",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
```

**Features:**
- **OpenAPI Documentation:** Automatic API documentation with FastAPI
- **Error Handling:** Comprehensive HTTP error responses
- **Logging:** Request/response logging middleware
- **CORS Support:** Cross-origin request handling
- **Health Monitoring:** Database connectivity checks
- **Separation of Concerns:** Public vs internal API endpoints

---

# Tezzeret Service - Compilation Forge (`src/esper/services/tezzeret/`)

## Overview

Tezzeret handles the asynchronous compilation of blueprint IR into executable PyTorch modules. It operates as a background worker that polls Urza for unvalidated blueprints, compiles them using torch.compile, and uploads the resulting artifacts to S3 storage.

### `__init__.py` - Tezzeret Service Initialization

**Purpose:** Service module initialization for Tezzeret compilation service.

### `main.py` - Service Orchestration

**Purpose:** Minimal service wrapper and entry point for Tezzeret worker.

**Contents:**
```python
"""
Tezzeret service main entry point.
Manages the compilation worker lifecycle.
"""

# Placeholder for service orchestration
# Main service coordination will be implemented when needed
```

**Status:** Placeholder implementation - focuses on worker functionality for MVP.

### `worker.py` - Blueprint Compilation Worker

**Purpose:** Background worker implementing the complete blueprint compilation pipeline from IR to executable artifacts.

#### Key Components

**`TezzeretWorker`** - Compilation Pipeline Worker
```python
class TezzeretWorker:
    """
    Background worker for compiling blueprints into executable kernels.
    
    This worker polls Urza for unvalidated blueprints, compiles them using
    PyTorch, and uploads the resulting artifacts to S3 storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Tezzeret worker.
        
        Args:
            config: Worker configuration
        """
        self.config = config
        self.urza_url = config.get("urza_url", "http://localhost:8000")
        self.s3_bucket = config.get("s3_bucket", "esper-artifacts")
        self.polling_interval = config.get("polling_interval", 5)  # seconds
        self.max_concurrent_jobs = config.get("max_concurrent_jobs", 2)
        
        # State
        self.running = False
        self.active_jobs = 0
        self.total_compiled = 0
        self.total_failed = 0
        
        # S3 client
        self.s3_client = get_s3_client()
        
        logger.info(f"Initialized TezzeretWorker with config: {config}")
```

**Main Worker Loop:**
```python
async def run(self) -> None:
    """
    Main worker loop - polls for blueprints and compiles them.
    """
    self.running = True
    logger.info("TezzeretWorker starting...")
    
    while self.running:
        try:
            # Check if we can take on more work
            if self.active_jobs >= self.max_concurrent_jobs:
                await asyncio.sleep(self.polling_interval)
                continue
            
            # Poll Urza for unvalidated blueprints
            blueprints = self._fetch_unvalidated_blueprints()
            
            if not blueprints:
                await asyncio.sleep(self.polling_interval)
                continue
            
            # Process blueprints
            for blueprint in blueprints[:self.max_concurrent_jobs - self.active_jobs]:
                # Start compilation task
                task = asyncio.create_task(self._compile_blueprint(blueprint))
                # Don't await - let it run concurrently
                
            await asyncio.sleep(self.polling_interval)
            
        except Exception as e:
            logger.error(f"Worker loop error: {e}")
            await asyncio.sleep(self.polling_interval)
    
    logger.info("TezzeretWorker stopped")

def _fetch_unvalidated_blueprints(self) -> List[Dict[str, Any]]:
    """
    Fetch unvalidated blueprints from Urza.
    
    Returns:
        List of blueprint data
    """
    try:
        import requests
        
        response = requests.get(
            f"{self.urza_url}/internal/unvalidated-blueprints",
            params={"limit": self.max_concurrent_jobs},
            timeout=10
        )
        
        if response.status_code == 200:
            blueprints = response.json()
            logger.debug(f"Fetched {len(blueprints)} unvalidated blueprints")
            return blueprints
        else:
            logger.warning(f"Failed to fetch blueprints: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching blueprints: {e}")
        return []
```

**Blueprint Compilation Pipeline:**
```python
async def _compile_blueprint(self, blueprint: Dict[str, Any]) -> None:
    """
    Compile a single blueprint through the complete pipeline.
    
    Args:
        blueprint: Blueprint data from Urza
    """
    blueprint_id = blueprint["blueprint_id"]
    self.active_jobs += 1
    
    try:
        logger.info(f"Starting compilation of blueprint {blueprint_id}")
        
        # Update status to compiling
        self._update_blueprint_status(blueprint_id, "compiling")
        
        # Parse IR and create PyTorch module
        start_time = time.time()
        architecture_ir = blueprint["architecture_ir"]
        pytorch_module = self._ir_to_pytorch_module(architecture_ir)
        
        # Compile with torch.compile
        compiled_module = torch.compile(pytorch_module)
        
        # Serialize compiled module
        serialized_artifact = self._serialize_module(compiled_module)
        
        # Upload to S3
        s3_uri = self._upload_to_s3(blueprint_id, serialized_artifact)
        
        compilation_time = time.time() - start_time
        
        # Submit compiled kernel to Urza
        kernel_data = {
            "blueprint_id": blueprint_id,
            "kernel_binary_ref": s3_uri,
            "status": "validated",
            "compilation_time": compilation_time,
            "performance_metrics": {
                "compilation_time_ms": compilation_time * 1000,
                "module_size_bytes": len(serialized_artifact)
            }
        }
        
        self._submit_compiled_kernel(kernel_data)
        
        # Update blueprint status to compiled
        self._update_blueprint_status(blueprint_id, "compiled")
        
        self.total_compiled += 1
        logger.info(f"Successfully compiled blueprint {blueprint_id} in {compilation_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to compile blueprint {blueprint_id}: {e}")
        
        # Update status to failed
        self._update_blueprint_status(blueprint_id, "failed")
        self.total_failed += 1
        
    finally:
        self.active_jobs -= 1

def _ir_to_pytorch_module(self, architecture_ir: str) -> nn.Module:
    """
    Convert IR to PyTorch module.
    
    Args:
        architecture_ir: JSON IR representation
        
    Returns:
        PyTorch module
    """
    try:
        import json
        
        # Parse IR (simplified for MVP)
        ir_data = json.loads(architecture_ir)
        
        # Extract architecture parameters
        arch_type = ir_data.get("type", "linear")
        input_size = ir_data.get("input_size", 512)
        output_size = ir_data.get("output_size", 512)
        
        # Create module based on type (simplified)
        if arch_type == "linear":
            return nn.Linear(input_size, output_size)
        elif arch_type == "attention":
            num_heads = ir_data.get("heads", 8)
            # Simplified attention module
            return nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        elif arch_type == "mlp":
            hidden_size = ir_data.get("hidden_size", 1024)
            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            # Default to linear layer
            logger.warning(f"Unknown architecture type {arch_type}, defaulting to linear")
            return nn.Linear(input_size, output_size)
            
    except Exception as e:
        logger.error(f"Failed to parse IR: {e}")
        # Fallback to simple linear layer
        return nn.Linear(512, 512)

def _serialize_module(self, module: nn.Module) -> bytes:
    """
    Serialize PyTorch module to bytes.
    
    Args:
        module: PyTorch module to serialize
        
    Returns:
        Serialized module as bytes
    """
    import io
    
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()

def _upload_to_s3(self, blueprint_id: str, artifact_data: bytes) -> str:
    """
    Upload compiled artifact to S3.
    
    Args:
        blueprint_id: Blueprint identifier
        artifact_data: Serialized artifact bytes
        
    Returns:
        S3 URI of uploaded artifact
    """
    try:
        # Generate S3 key
        timestamp = int(time.time())
        s3_key = f"kernels/{blueprint_id}/{timestamp}/kernel.pt"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=artifact_data,
            ContentType="application/octet-stream",
            Metadata={
                "blueprint_id": blueprint_id,
                "compiled_at": str(timestamp),
                "artifact_type": "pytorch_module"
            }
        )
        
        # Return S3 URI
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.debug(f"Uploaded artifact to {s3_uri}")
        
        return s3_uri
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise
```

**Urza Integration Methods:**
```python
def _update_blueprint_status(self, blueprint_id: str, status: str) -> None:
    """
    Update blueprint status in Urza.
    
    Args:
        blueprint_id: Blueprint identifier
        status: New status
    """
    try:
        import requests
        
        response = requests.post(
            f"{self.urza_url}/internal/blueprints/{blueprint_id}/status",
            json={"status": status},
            timeout=10
        )
        
        if response.status_code != 200:
            logger.warning(f"Failed to update blueprint status: HTTP {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error updating blueprint status: {e}")

def _submit_compiled_kernel(self, kernel_data: Dict[str, Any]) -> None:
    """
    Submit compiled kernel to Urza.
    
    Args:
        kernel_data: Kernel metadata
    """
    try:
        import requests
        
        response = requests.post(
            f"{self.urza_url}/api/v1/kernels/",
            json=kernel_data,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.debug("Successfully submitted compiled kernel to Urza")
        else:
            logger.warning(f"Failed to submit kernel: HTTP {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error submitting kernel: {e}")
```

**Worker Management:**
```python
def stop(self) -> None:
    """Stop the worker gracefully."""
    logger.info("Stopping TezzeretWorker...")
    self.running = False

def get_stats(self) -> Dict[str, Any]:
    """
    Get worker statistics.
    
    Returns:
        Worker performance statistics
    """
    return {
        "running": self.running,
        "active_jobs": self.active_jobs,
        "total_compiled": self.total_compiled,
        "total_failed": self.total_failed,
        "success_rate": (
            self.total_compiled / max(self.total_compiled + self.total_failed, 1)
        ),
        "max_concurrent_jobs": self.max_concurrent_jobs
    }
```

**Features:**
- **Asynchronous Processing:** Non-blocking compilation with configurable concurrency
- **Polling-based Architecture:** Regular checks for new work from Urza
- **Error Handling:** Comprehensive error recovery and status reporting
- **S3 Integration:** Artifact upload with metadata
- **Performance Tracking:** Compilation time and success rate monitoring
- **Graceful Shutdown:** Clean worker lifecycle management

**Issues Identified:**
1. **Synchronous HTTP:** Uses requests library in async context
2. **Simplified IR Parsing:** MVP-level architecture interpretation
3. **Limited Error Recovery:** Basic error handling without retry mechanisms

**Configuration:**
```python
# Example worker configuration
worker_config = {
    "urza_url": "http://localhost:8000",
    "s3_bucket": "esper-artifacts",
    "polling_interval": 5,
    "max_concurrent_jobs": 2
}

# Start worker
worker = TezzeretWorker(worker_config)
await worker.run()
```

---

This covers the foundational services (Oona, Urza, Tezzeret) and their core functionality. The remaining services (Tamiyo, Tolaria) and utils will be documented in the next continuation due to length limits. Would you like me to continue with the remaining services?