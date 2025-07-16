### **High-Level Design: Oona - Message Bus Subsystem (v1.2)**

**Status:** Final | **Date:** July 7, 2025

#### **1. Introduction**

This document provides the formal detailed design for **Oona**, the asynchronous message bus and event-driven backbone of the Esper Morphogenetic Platform. Oona's sole purpose is to provide a reliable, scalable, and decoupled communication layer that enables Esper's specialized subsystems to interact without direct dependencies. It is the infrastructure that realizes the system's event-sourcing and auditable history principles.

#### **2. Architectural Goals & Principles**

* **Decoupling:** Oona serves as a pure intermediary. A publisher sends a message to a topic, and a consumer reads from it, with Oona managing the entire exchange.
* **Reliability & Durability:** The system must guarantee at-least-once message delivery. All critical system events are persisted to form an immutable log.
* **Scalability:** The architecture will scale from a lightweight single-node deployment (Phase 1) to a high-throughput, distributed system (Phase 2).
* **Observability:** The structure of messages and topics inherently supports a complete and transparent audit trail of all system actions and decisions.

#### **3. System Architecture**

##### **3.1. Technology Stack (Phased Implementation)**

* **Phase 1 (Single-Node MVP): Redis 8.0**

  * **Rationale:** Redis is lightweight, fast, and its Streams data type provides the necessary persistence, consumer group, and ordered-log semantics required for the MVP. It is simple to deploy via Docker and managed by our existing `docker-compose.yml`.

* **Phase 2 (Distributed Production): Apache Pulsar**

  * **Rationale:** Pulsar is purpose-built for large-scale event streaming and provides key features for a production Esper deployment, including multi-tenancy, tiered storage for indefinite event log retention, and a built-in schema registry.

##### **3.2. MVP Topic Architecture**

The following focused topic architecture will be implemented for the MVP. All topic names are centrally managed by the `TopicNames` enum in `src/esper/contracts/messages.py`.

| MVP Topic Name | Publisher(s) | Consumer(s) | Payload Data Model |
| :--- | :--- | :--- | :--- |
| **`compilation.blueprint.submitted`** | `Urza` (via API call) | `Tezzeret` | `BlueprintSubmitted` |
| **`telemetry.seed.health`** | `Kasmina` | `Tamiyo` | `HealthSignal` |
| **`control.kasmina.commands`** | `Tamiyo` | `Kasmina` | `KasminaControlCommand`|
| **`innovation.field_reports`** | `Tamiyo` | Offline `train_tamiyo.py` | `FieldReport` |

##### **3.3. Message Contracts & "Fat Envelope" Pattern**

All messages published to Oona **must** adhere to the `OonaMessage` Pydantic model. This "fat envelope" wraps the specific event `payload`, providing critical metadata for tracing, debugging, and observability. The authoritative schemas are the Pydantic models in `src/esper/contracts/`.

```json
{
  "event_id": "uuid-v4-string",
  "sender_id": "Urza-API-Service",
  "timestamp": "2025-07-07T09:55:13Z",
  "trace_id": "trace-id-for-a-specific-request",
  "topic": "compilation.blueprint.submitted",
  "payload": {
    "blueprint_id": "b3d4f5...",
    "submitted_by": "human_developer"
  }
}
```

-----

### **Implementation Guide: Phase 1 - The Core Asset Pipeline**

**Objective:** To build and integrate the services required to transform a manually submitted `BlueprintIR` into a `Validated` `CompiledKernelArtifact`. This phase creates the asynchronous "factory" at the heart of the Esper platform.

**Key Components to Implement:** `Oona`, `Urza`, `Tezzeret`.

#### **1. Oona: Message Bus Client**

**Task:** Implement the `OonaClient` to provide a standardized interface for interacting with Redis Streams.

* **File:** `src/esper/services/oona_client.py`
* **Details:**
  * The client should initialize a connection to Redis using the `REDIS_URL` environment variable.
  * Implement a `publish` method that takes an `OonaMessage`, serializes its `payload` to JSON, and uses the `XADD` command to add it to the appropriate stream (topic).

<!-- end list -->

```python
# src/esper/services/oona_client.py
import redis
import os
import json
from esper.contracts.messages import OonaMessage

class OonaClient:
    """A client for publishing events to the Oona message bus (Redis Streams)."""

    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.from_url(redis_url)
            # Check connection
            self.redis_client.ping()
            print("OonaClient connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            print(f"FATAL: Could not connect to Redis at {redis_url}. Error: {e}")
            raise

    def publish(self, message: OonaMessage):
        """Publishes a message to a specific topic (stream)."""
        try:
            stream_name = message.topic.value
            # Redis Streams expects a flat dictionary of bytes/strings.
            # We serialize the full message envelope.
            message_dict = message.model_dump(mode="json")
            self.redis_client.xadd(stream_name, message_dict)
        except Exception as e:
            # Add robust logging here
            print(f"Error publishing event {message.event_id} to Oona: {e}")

# Note: Consumer logic will be service-specific.
```

#### **2. Urza: Central Asset Hub**

**Task:** Implement the Urza service to manage the lifecycle and storage of blueprints and kernels.

##### **2.1. Database and Storage Setup**

1. **Database Schema Script:** Create `scripts/init_db.py`. This script will connect to the PostgreSQL container and execute `CREATE TABLE` statements for `blueprints` and `compiled_kernels` as detailed in the previous response.
2. **Bucket Initialization Utility:** Create `src/esper/utils/s3_client.py`. This will contain a `get_s3_client` function to connect to MinIO and an `ensure_bucket_exists` function that creates the `esper-artifacts` bucket if it's missing.

##### **2.2. Urza Service Implementation**

**Task:** Implement the FastAPI application that serves as the front door to Urza's data.

* **File:** `src/esper/services/urza/main.py`
* **Details:**
  * Create a FastAPI app instance.
  * On startup, it should initialize connections to PostgreSQL and the S3 client.
  * Implement the API endpoints below. Use dependency injection for database sessions.

<!-- end list -->

```python
# src/esper/services/urza/main.py (Skeleton)
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
# ... other imports
from esper.contracts.assets import Blueprint
from esper.contracts.messages import OonaMessage, TopicNames
from esper.services.oona_client import OonaClient

app = FastAPI()
oona = OonaClient()

# (Implement database dependency injection logic here using 'yield')

@app.on_event("startup")
async def startup_event():
    # Ensure MinIO bucket exists
    # ...
    pass

@app.post("/v1/blueprints", status_code=201, response_model=Blueprint)
async def submit_blueprint(blueprint: Blueprint, db: Session = Depends(get_db)):
    """Endpoint for developers to submit a new BlueprintIR."""
    # 1. Check if a blueprint with this ID (hash) already exists in DB.
    #    If so, return a 409 Conflict error.
    
    # 2. Upload the architecture IR file (referenced by blueprint.architecture_ir_ref)
    #    to MinIO/S3. The filename in the bucket should be the blueprint ID.
    
    # 3. Insert blueprint metadata into the 'blueprints' table in PostgreSQL.
    #    Set its status to UNVALIDATED.
    
    # 4. Create and publish an OonaMessage.
    msg = OonaMessage(
        sender_id="urza-api-service",
        trace_id="...", # Add tracing middleware later
        topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
        payload=blueprint.model_dump()
    )
    oona.publish(msg)
    
    return blueprint

# Add other internal endpoints for Tezzeret:
# - GET /internal/v1/blueprints/unvalidated
# - POST /internal/v1/kernels (for Tezzeret to submit compiled artifacts)
# - PUT /internal/v1/blueprints/{blueprint_id}/status
```

#### **3. Tezzeret: Compilation Forge**

**Task:** Create the background service that polls Urza, compiles blueprints, and pushes the results back.

* **File:** `src/esper/services/tezzeret/worker.py`
* **Details:**
  * The worker should be a long-running process.
  * It will not use FastAPI, but will be a script that instantiates a `TezzeretWorker` class and calls a `start()` method.
  * It needs to handle both fetching the `BlueprintIR` from MinIO and uploading the compiled kernel back.

<!-- end list -->

```python
# src/esper/services/tezzeret/main.py (Entrypoint)
from .worker import TezzeretWorker

def main():
    worker = TezzeretWorker(worker_id="tezzeret-worker-01")
    worker.start_polling()

if __name__ == "__main__":
    main()

# src/esper/services/tezzeret/worker.py (Logic)
import time
import requests
import torch
# ...

class TezzeretWorker:
    # ... (init method) ...

    def process_one_blueprint(self):
        # 1. Poll Urza's internal API for an unvalidated blueprint.
        #    GET http://urza-service:8000/internal/v1/blueprints/unvalidated
        
        if not blueprint_to_process:
            return

        # 2. Update status in Urza to COMPILING via API to prevent other workers
        #    from picking up the same job.
        #    PUT http://urza-service:8000/internal/v1/blueprints/{id}/status
        
        try:
            # 3. Download the IR file from MinIO.
            # 4. Run the 'Fast' compilation pipeline (`torch.compile`).
            # 5. Upload the compiled binary to MinIO.
            # 6. Submit the new CompiledKernelArtifact metadata to Urza via API.
            #    CRITICAL WORKAROUND: In the submission payload, set the kernel
            #    status directly to 'VALIDATED'.
            #    POST http://urza-service:8000/internal/v1/kernels
            
        except Exception as e:
            # 7. If any step fails, update the blueprint status in Urza to 'INVALID'.
            print(f"ERROR: Failed to compile {blueprint_id}. Error: {e}")

    def start_polling(self):
        print("Tezzeret worker polling for jobs...")
        while True:
            self.process_one_blueprint()
            time.sleep(5) # Poll every 5 seconds
```

#### **4. Phase 1 Definition of Done**

1. **Code Complete:** All specified files and classes for `OonaClient`, the `Urza` API, and the `TezzeretWorker` are implemented.
2. **Infrastructure Ready:** The `init_db.py` script runs successfully. The S3 bucket is created on startup.
3. **Unit Tests Passed:** All new business logic (e.g., API request handling, database interaction logic) is unit-tested.
4. **Integration Test Passed:** A `pytest` integration test successfully:
      * Starts all services.
      * `POST`s a new blueprint to the `Urza` API.
      * Confirms an event is published on the `compilation.blueprint.submitted` topic in Redis.
      * Polls an `Urza` endpoint until the corresponding `CompiledKernelArtifact` record appears with the status `VALIDATED`.
5. **Manual Verification:** A developer can manually perform the integration test steps and observe the correct behavior in the service logs, PostgreSQL database, and MinIO console.
