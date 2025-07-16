### **Implementation Guide: Phase 1 - The Core Asset Pipeline**

**Objective:** To build and integrate the services required to transform a manually submitted `BlueprintIR` into a `Validated` `CompiledKernelArtifact`. This phase creates the asynchronous "factory" at the heart of the Esper platform.

**Key Components to Implement:** `Oona`, `Urza`, `Tezzeret`.

-----

### **1. Oona: Message Bus Implementation**

**Task:** Create a reliable, typed client for interacting with the Redis Streams message bus.

#### **`src/esper/services/oona_client.py`**

This client will provide a simple, high-level interface for all other services.

```python
import redis
from typing import Dict, Any
import os

from esper.contracts.messages import OonaMessage, TopicNames

class OonaClient:
    """A client for publishing and consuming events on the Oona message bus."""

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        print("OonaClient connected to Redis.")

    def publish(self, message: OonaMessage):
        """Publishes a message to a specific topic (stream)."""
        try:
            stream_name = message.topic.value
            message_body = message.model_dump(mode="json")
            self.redis_client.xadd(stream_name, message_body)
            print(f"Published event {message.event_id} to topic {stream_name}")
        except Exception as e:
            # In a real system, add more robust error handling and logging
            print(f"Error publishing to Oona: {e}")

    # Note: Subscription logic will be implemented within each consumer service
    # as needed, using blocking reads (xread).
```

-----

### **2. Urza: Central Asset Hub Implementation**

**Task:** Implement the Urza service to manage the lifecycle and storage of blueprints and kernels.

#### **2.1. Database Schema (`scripts/init_db.py`)**

Create a script to initialize the PostgreSQL database schema.

```python
import psycopg2
import os

def init_database():
    """Initializes the Urza database tables."""
    conn = psycopg2.connect(
        dbname="urza_db",
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host="localhost" # Assumes running locally or docker networking
    )
    cur = conn.cursor()

    # Blueprints Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS blueprints (
            id VARCHAR(64) PRIMARY KEY,
            status VARCHAR(32) NOT NULL,
            architecture_ir_ref TEXT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
        );
    """)

    # Compiled Kernels Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS compiled_kernels (
            id VARCHAR(128) PRIMARY KEY,
            blueprint_id VARCHAR(64) REFERENCES blueprints(id),
            status VARCHAR(32) NOT NULL,
            compilation_pipeline VARCHAR(32) NOT NULL,
            kernel_binary_ref TEXT NOT NULL,
            validation_report JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Urza database initialized successfully.")

if __name__ == "__main__":
    init_database()
```

#### **2.2. S3/MinIO Bucket Initialization**

`Tezzeret` and `Urza` will need a utility to ensure the MinIO bucket exists.

#### **`src/esper/utils/s3_client.py`**

```python
import boto3
from botocore.client import Config
import os

def get_s3_client():
    """Configures and returns a boto3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket_exists(client, bucket_name: str):
    """Creates an S3 bucket if it doesn't already exist."""
    try:
        client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created.")
        else:
            raise
```

#### **2.3. Urza Service (`src/esper/services/urza/main.py`)**

Implement the FastAPI application.

```python
from fastapi import FastAPI, HTTPException
# ... other imports for db connection, etc.
from esper.contracts.assets import Blueprint
from esper.contracts.enums import BlueprintStatus
# ...

app = FastAPI()

# Note: In a real implementation, use a dependency injection system
# for managing database and S3 clients.

@app.post("/v1/blueprints", status_code=201)
async def submit_blueprint(blueprint: Blueprint):
    """Endpoint for developers (or Karn) to submit a new blueprint."""
    # 1. Store blueprint.architecture_ir_ref in MinIO
    # 2. Insert blueprint metadata into PostgreSQL blueprints table
    #    with status UNVALIDATED.
    # 3. Publish a 'COMPILATION_BLUEPRINT_SUBMITTED' message to Oona.
    #
    # This is a simplified placeholder for the actual logic.
    print(f"Received blueprint {blueprint.id}")
    return {"message": "Blueprint submitted for compilation.", "blueprint_id": blueprint.id}


@app.get("/v1/blueprints/unvalidated")
async def get_unvalidated_blueprints():
    """Internal endpoint for Tezzeret to poll for work."""
    # 1. Query PostgreSQL for all blueprints where status = 'UNVALIDATED'.
    # 2. Return a list of Blueprint objects.
    #
    # Placeholder logic
    return []

# ... other endpoints for Tezzeret to push compiled kernels, etc.
```

-----

### **3. Tezzeret: Compilation Forge Implementation**

**Task:** Create the background service that polls Urza, compiles blueprints, and pushes the resulting artifacts back.

#### **`src/esper/services/tezzeret/worker.py`**

```python
import time
import torch
import torch.nn as nn
import requests # to communicate with Urza API

# A placeholder for turning a blueprint IR into a runnable module
def ir_to_module(ir: dict) -> nn.Module:
    # In a real system, this would be a complex graph-to-code converter.
    # For the MVP, we can assume a simple structure.
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

class TezzeretWorker:
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.urza_api_url = "http://localhost:8000" # From config
        print(f"Tezzeret Worker {self.worker_id} started.")

    def run_fast_compilation(self, blueprint_ir: dict) -> bytes:
        """Compiles a blueprint using the 'Fast' pipeline."""
        module = ir_to_module(blueprint_ir)
        compiled_module = torch.compile(module)
        
        # To save, we need to script it first
        scripted_module = torch.jit.script(compiled_module)
        # In-memory buffer to save the model
        buffer = torch.jit.save(scripted_module)
        return buffer.getvalue()

    def process_one_blueprint(self):
        """Fetches one blueprint, compiles it, and pushes it back."""
        # 1. GET request to /v1/blueprints/unvalidated
        response = requests.get(f"{self.urza_api_url}/v1/blueprints/unvalidated")
        blueprints_to_process = response.json()

        if not blueprints_to_process:
            return

        blueprint_data = blueprints_to_process[0]
        blueprint_id = blueprint_data['id']

        try:
            # 2. Fetch the full IR from MinIO using blueprint_data['architecture_ir_ref']
            blueprint_ir = {} # Placeholder for fetched IR
            
            # 3. Compile it
            compiled_binary = self.run_fast_compilation(blueprint_ir)

            # 4. Push the binary to MinIO, get back the new ref (e.g., s3://.../kernel.pt)
            kernel_binary_ref = "s3://placeholder/kernel.pt"

            # 5. POST the new CompiledKernelArtifact metadata to Urza
            #    This includes the new ref, pipeline name, and blueprint_id.
            #    CRITICAL WORKAROUND: Set status to 'VALIDATED' directly since Urabrask is missing.
            print(f"Successfully compiled blueprint {blueprint_id}")

        except Exception as e:
            # 6. If compilation fails, update blueprint status in Urza to 'INVALID'.
            print(f"Failed to compile blueprint {blueprint_id}: {e}")

    def start_polling(self):
        """The main loop for the worker."""
        while True:
            self.process_one_blueprint()
            time.sleep(10) # Poll every 10 seconds
```

-----

### **4. Phase 1 Testing & Validation Strategy**

1. **Unit Tests:**
      * Test `OonaClient` can correctly format and publish messages.
      * Test `Urza`'s database interactions (mocking `psycopg2`).
      * Test `Tezzeret`'s `ir_to_module` conversion logic.
2. **Integration Test (The "Golden Path"):**
      * Write a single, comprehensive `pytest` integration test script (`tests/integration/test_phase1_pipeline.py`).
      * **Setup:** The test will start all services using `docker-compose`. It will call `init_db.py` and `ensure_bucket_exists`.
      * **Action:**
        1. The test will directly call the `Urza` API to `POST` a new, hand-crafted `blueprint.json`.
        2. The test will then poll another `Urza` endpoint (`GET /kernels/{blueprint_id}`) in a loop with a timeout.
      * **Assertion:** The test succeeds if, within the timeout, the polling endpoint returns data for a `CompiledKernelArtifact` with the status `VALIDATED`.

### **5. Definition of Done**

Phase 1 is complete when:

* ✅ All code for `OonaClient`, the `Urza` service, and the `TezzeretWorker` is implemented and unit-tested.
* ✅ The `init_db.py` script successfully creates the required PostgreSQL tables.
* ✅ The comprehensive integration test for the "Golden Path" passes reliably in the CI environment.
* ✅ A developer can manually submit a blueprint and observe the entire compilation process through service logs, resulting in a validated kernel in the `Urza` database and MinIO storage.
