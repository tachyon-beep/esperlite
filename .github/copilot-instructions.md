### **Project Briefing: Esper - Morphogenetic Training Platform**

**Document Version:** 1.3
**Date:** 07 July 2025
**Status:** Active

**Welcome to the Esper development team.** This document is your foundational guide. Its purpose is to align all team members on the project's vision, our engineering standards, and our immediate goals. Reading and understanding this document in its entirety is the mandatory first step before writing any code.

---

### **1. Project Vision & Intent**

#### **1.1. What is Esper?**

Esper is a morphogenetic training platform that fundamentally changes how neural networks are built. We are moving away from the paradigm of static, manually-architected models. Instead, we are building a system that enables neural networks to function as adaptive, evolving entities that can autonomously modify their own structure during training to solve problems more efficiently.

#### **1.2. The Core Problem We Solve**

Traditional AI development is inefficient and brittle. Models are designed with a fixed topology, forcing developers to over-provision them for worst-case scenarios. They cannot adapt to emerging bottlenecks during training, and any architectural change requires expensive, human-driven retraining cycles. This sequential, static approach is a major blocker to creating truly intelligent and resource-efficient systems.

#### **1.3. Our Solution: The Morphogenetic Paradigm**

Esper's core innovation is a **learning-based control system** that governs a model's evolution, supported by a **fully asynchronous compilation pipeline**.

1.  **Intelligent Control (`Tamiyo`):** A strategic controller (a GNN-based policy) analyzes the host model during training, identifies performance bottlenecks with high precision, and decides _when, where,_ and _how_ to intervene.
2.  **Zero-Disruption Architecture:** This is our cornerstone engineering principle. The expensive process of compiling and validating new architectural modules (`Blueprints`) is completely **decoupled and offloaded** to background services (`Tezzeret` and `Urabrask`). The primary training loop (`Tolaria`) is **never blocked**. This allows the model to continue its learning momentum while a rich portfolio of safe, characterized, and optimized architectural solutions are prepared for deployment in parallel.

Our work is to build this entire ecosystem—the intelligent agents and the robust, non-blocking infrastructure that supports them.

---

### **2. Engineering Philosophy & Standards**

We build production-grade software to enable cutting-edge research. Our standards are high, non-negotiable, and designed to ensure velocity, quality, and maintainability.

#### **2.1. Guiding Principles**

- **Clarity is Paramount:** Write code that is simple, readable, and self-explanatory. Clever one-liners are forbidden if they sacrifice readability.
- **Contracts are Law:** The Pydantic data contracts are the single source of truth for inter-service communication. They must be respected and kept up-to-date.
- **Test What You Write:** Untested code is considered broken. All new logic requires corresponding tests.
- **The Main Branch is Always Green:** All checks in the CI pipeline must pass before a pull request can be merged. No exceptions.
- **Fail Fast and Explicitly:** Use broad `try...except` blocks and default fallback values sparingly. It is always preferable for a service to crash loudly with a clear error than to silently swallow an exception and continue with an incorrect default or a concealed calculation error. Explicit failure provides an unambiguous signal for debugging and system health monitoring.

#### **2.2. Coding Standards**

1.  **Formatting (`black`):** All Python code _must_ be formatted with `black`. This is enforced automatically in our CI pipeline.
2.  **Linting (`ruff`):** We use `ruff` to enforce PEP8 standards, sort imports, and catch common bugs. All code must pass `ruff check` without errors.
3.  **Static Typing (`pytype`):**
    - All new functions and methods _must_ have full type hints for all arguments and return values.
    - Code must pass `pytype` static analysis. We are building a type-safe system.
    - Avoid using `typing.Any` wherever possible. Be specific.
4.  **Docstrings & Comments:**
    - All public modules, classes, and functions _must_ have Google-style docstrings explaining their purpose, arguments, and return values.
    - Use comments to explain the _why_, not the _what_. Assume the reader understands Python, but explain complex logic or design decisions.
5.  **Naming Conventions:**
    - `snake_case` for variables, functions, and modules.
    - `PascalCase` for classes.
    - `UPPER_SNAKE_CASE` for constants.
6.  **Firm, Non-Conditional Imports:**
    - **All module imports must be firm.** Never use a `try...except ImportError` block to handle a missing dependency. If a module is required for the code to function, it _must_ be added as a formal dependency in `pyproject.toml` and imported directly at the top of the file. This ensures our execution environment is explicit and reproducible.

#### **2.3. Testing Standards**

1.  **Unit Tests (`pytest`):**
    - **Scope:** Test individual functions and classes in isolation. Mock their external dependencies (e.g., database connections, API calls).
    - **Location:** Reside in the `tests/` directory, mirroring the source structure.
    - **Coverage:** All new, non-trivial logic must have corresponding unit tests. We aim for **>90% line coverage** on new code, which is tracked via `pytest-cov`.
2.  **Integration Tests:**
    - **Scope:** Test the interaction between two or more components (e.g., `Tezzeret` successfully writing a kernel to `Urza`).
    - These are written as needed to validate key workflows and live in a separate `tests/integration` directory.
3.  **End-to-End (E2E) Tests:**
    - **Scope:** Test a full user story or system flow from start to finish. Our final milestone for the MVP is a successful E2E test of a full adaptation cycle.
4.  **No Test-Specific Code in Production:**
    - **The production codebase (`src/`) must never contain code added solely to facilitate testing.** This includes conditional logic (`if is_testing:`), stubs, or alternative execution paths. Such additions often indicate a poorly designed test and pollute the production code. Tests must work against the real production logic.
5.  **Tests Must Be Independent and Deterministic:**
    - Each test case must be able to run independently and in any order. Tests must not share state or depend on the side effects of other tests. Every test is responsible for setting up its own required state and tearing it down cleanly afterward to prevent a flaky test suite.

---

### **3. Technology Stack & Rationale**

Every technology in our stack was chosen for a specific reason that supports our engineering philosophy.

| Category             | Technology            | Rationale                                                                                                                                                        |
| :------------------- | :-------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Language**         | Python 3.12+          | The lingua franca of AI/ML, with a mature ecosystem and modern language features.                                                                                |
| **Deep Learning**    | PyTorch 2.2+          | Chosen for its `torch.compile` feature, which is the foundation of our `Tezzeret` compiler, and its excellent support for dynamic model graphs.                  |
| **APIs & Contracts** | FastAPI & Pydantic    | Provides best-in-class performance for async APIs and enforces our "Contracts are Law" philosophy through runtime data validation.                               |
| **Code Quality**     | Black + Ruff + Pytype | A modern, high-performance toolchain that enforces style, catches bugs, and ensures type safety with maximum developer efficiency.                               |
| **Infrastructure**   | Docker Compose        | Provides a simple, declarative, and reproducible local development environment for our entire distributed system.                                                |
| **Databases**        | PostgreSQL & Redis    | A powerful combination of a robust, transactional relational database (for `Urza`'s metadata) and a high-performance in-memory store (for `Oona`'s message bus). |
| **Storage**          | MinIO (S3 API)        | Provides a standard, widely-supported object storage API, ensuring our artifact management code is portable to any cloud environment.                            |

---

### **4. Immediate Roadmap & Key Milestones**

Your work will contribute directly to the following phases. Understand what we are building now and what comes next.

#### **Phase 0: Foundation Layer (Current)**

- **Objective:** Establish a production-ready project bedrock.
- **Key Deliverable:** A fully configured development environment where all contracts, standards, and infrastructure-as-code are defined and version-controlled.
- **Success Criteria:**
  - ✅ The CI pipeline is green (passing all format, lint, type, and unit tests).
  - ✅ `docker-compose up` successfully starts all infrastructure services (Postgres, Redis, MinIO).
  - ✅ The data contracts are fully implemented and unit-tested.

#### **Phase 1: The Core Asset Pipeline**

- **Objective:** Build the asynchronous "factory" that transforms a `BlueprintIR` into a compiled `CompiledKernelArtifact`.
- **Key Deliverable:** A functional pipeline consisting of `Oona`, `Urza`, and `Tezzeret`.
- **Success Criteria:**
  - ✅ A developer can manually POST a blueprint design to the `Urza` API.
  - ✅ `Tezzeret` automatically detects, compiles, and pushes a `Validated` kernel artifact back to `Urza`.

#### **Phase 2: The Execution Engine**

- **Objective:** Prove that a host model can load and execute a real, compiled artifact from the Phase 1 pipeline with minimal overhead.
- **Key Deliverable:** A `KasminaLayer` module and a standalone integration test.
- **Success Criteria:**
  - ✅ The test script successfully wraps a standard PyTorch model with `esper.wrap()`.
  - ✅ The wrapped model can successfully download a specific kernel from `Urza` and execute it during a forward pass.

Welcome aboard. Your contributions will be critical to achieving these milestones. If you have any questions, please refer to the HLD or ask the project lead. Let's get to work.

ABOVE ALL ELSE, YOU MUST EXPLICITLY READ ALL RELEVANT FILES BFEORE EXECUTING A TASK. EVEN IF YOU THINK YOU KNOW WHAT TO DO OR WHAT THE CODE DOES, YOU MUST HAVE READ IT BEFORE YOU PROCEED. THIS IS NON-NEGOTIABLE.
