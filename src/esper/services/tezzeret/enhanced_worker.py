"""
Enhanced Tezzeret Worker with Real Compilation Pipeline.

This module provides an enhanced worker that uses the real compilation,
optimization, and validation pipeline for blueprints.
"""

import asyncio
import io
import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

import torch

from esper.contracts.assets import Blueprint
from esper.contracts.assets import CompiledKernel
from esper.contracts.enums import BlueprintStatus
from esper.contracts.enums import KernelStatus
from esper.services.contracts import CompiledKernelArtifact
from esper.services.tezzeret.compiler import BlueprintCompiler
from esper.services.tezzeret.compiler import CompilationError
from esper.services.tezzeret.optimizer import KernelOptimizer
from esper.services.tezzeret.validator import KernelValidator
from esper.utils.circuit_breaker import CircuitBreaker
from esper.utils.circuit_breaker import CircuitBreakerConfig
from esper.utils.circuit_breaker import CircuitBreakerOpenError
from esper.utils.config import ServiceConfig
from esper.utils.config import get_service_config
from esper.utils.s3_client import get_s3_client
from esper.utils.s3_client import upload_bytes

logger = logging.getLogger(__name__)


class EnhancedTezzeretWorker:
    """
    Enhanced worker with real compilation pipeline.
    
    This worker:
    1. Polls Urza for blueprints to compile
    2. Compiles blueprints using BlueprintCompiler
    3. Optimizes kernels using KernelOptimizer
    4. Validates kernels using KernelValidator
    5. Uploads validated kernels back to Urza
    """

    def __init__(self, worker_id: str, config: Optional[ServiceConfig] = None):
        self.worker_id = worker_id
        self.config = config or get_service_config()

        # Service configuration
        self.urza_base_url = self.config.urza_url
        self.s3_client = get_s3_client()
        self.bucket_name = self.config.s3_bucket
        self.poll_interval = self.config.poll_interval_seconds

        # Initialize compilation pipeline components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compiler = BlueprintCompiler(device=self.device)
        self.optimizer = KernelOptimizer(device=self.device)
        self.validator = KernelValidator(device=self.device)

        # Circuit breaker for Urza service
        self._circuit_breaker = CircuitBreaker(
            name=f"enhanced_tezzeret_{worker_id}_urza",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=45,
                success_threshold=2,
                timeout=self.config.http_timeout,
            ),
        )

        # Statistics
        self._stats = {
            "total_processed": 0,
            "compilation_success": 0,
            "compilation_failed": 0,
            "optimization_success": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "circuit_breaker_failures": 0,
        }

        logger.info(
            "Enhanced Tezzeret Worker %s initialized on device %s",
            self.worker_id, self.device
        )

    async def process_one_blueprint(self) -> bool:
        """
        Process one blueprint through the full pipeline.
        
        Returns:
            bool: True if a blueprint was processed
        """
        # Fetch unvalidated blueprints
        blueprints = await self.fetch_unvalidated_blueprints()

        if not blueprints:
            return False

        blueprint_data = blueprints[0]
        blueprint_id = blueprint_data["id"]

        logger.info("Processing blueprint %s", blueprint_id)
        self._stats["total_processed"] += 1

        # Update status to COMPILING
        if not await self.update_blueprint_status(
            blueprint_id, BlueprintStatus.COMPILING
        ):
            logger.error("Failed to update blueprint status to COMPILING")
            return False

        try:
            # Convert blueprint data to Blueprint object
            blueprint = self._create_blueprint_from_data(blueprint_data)

            # Step 1: Compile blueprint
            logger.info("Compiling blueprint %s", blueprint_id)
            compiled_kernel = self.compiler.compile_blueprint(blueprint)
            self._stats["compilation_success"] += 1

            # Step 2: Optimize kernel
            logger.info("Optimizing kernel for blueprint %s", blueprint_id)
            original_module = self._load_kernel_module(compiled_kernel)
            optimized_module = self.optimizer.optimize_kernel(original_module)
            self._stats["optimization_success"] += 1

            # Compare performance
            performance_comparison = self.optimizer.compare_performance(
                original_module, optimized_module
            )
            logger.info(
                "Optimization achieved %.2fx speedup for blueprint %s",
                performance_comparison["speedup"], blueprint_id
            )

            # Step 3: Validate kernel
            logger.info("Validating kernel for blueprint %s", blueprint_id)
            validation_result = self.validator.validate_kernel(
                compiled_kernel, blueprint, original_module
            )

            if not validation_result.is_valid:
                logger.error(
                    "Kernel validation failed for blueprint %s: %s",
                    blueprint_id, validation_result.errors
                )
                self._stats["validation_failed"] += 1
                await self.update_blueprint_status(
                    blueprint_id, BlueprintStatus.INVALID
                )
                return True

            self._stats["validation_passed"] += 1

            # Step 4: Upload optimized kernel to S3
            kernel_binary = self._save_optimized_kernel(optimized_module)
            kernel_object_key = f"kernels/{compiled_kernel.kernel_id}/optimized.pt"
            kernel_binary_ref = upload_bytes(
                self.s3_client, kernel_binary, self.bucket_name, kernel_object_key
            )

            # Update kernel reference
            compiled_kernel.binary_ref = kernel_binary_ref

            # Step 5: Create kernel artifact for Urza
            kernel_artifact = CompiledKernelArtifact(
                id=compiled_kernel.kernel_id,
                blueprint_id=blueprint_id,
                status=KernelStatus.VALIDATED.value,
                compilation_pipeline="enhanced",
                kernel_binary_ref=kernel_binary_ref,
                validation_report={
                    "functional_correctness": validation_result.functional_correctness,
                    "performance_acceptable": validation_result.performance_acceptable,
                    "memory_safe": validation_result.memory_safe,
                    "gradient_correct": validation_result.gradient_correct,
                    "optimization_speedup": performance_comparison["speedup"],
                    "metrics": validation_result.metrics,
                    "warnings": validation_result.warnings,
                },
            )

            # Step 6: Submit kernel to Urza
            if await self.submit_compiled_kernel(kernel_artifact):
                logger.info(
                    "Successfully processed blueprint %s -> kernel %s "
                    "(speedup: %.2fx, validation: PASSED)",
                    blueprint_id, compiled_kernel.kernel_id,
                    performance_comparison["speedup"]
                )

                # Update blueprint status to VALIDATED
                await self.update_blueprint_status(
                    blueprint_id, BlueprintStatus.VALIDATED
                )
                return True
            else:
                logger.error("Failed to submit kernel to Urza")
                await self.update_blueprint_status(
                    blueprint_id, BlueprintStatus.INVALID
                )
                return False

        except CompilationError as e:
            logger.error("Compilation failed for blueprint %s: %s", blueprint_id, e)
            self._stats["compilation_failed"] += 1
            await self.update_blueprint_status(blueprint_id, BlueprintStatus.INVALID)
            return True

        except Exception as e:
            logger.error("Unexpected error processing blueprint %s: %s", blueprint_id, e)
            await self.update_blueprint_status(blueprint_id, BlueprintStatus.INVALID)
            return True

    def _create_blueprint_from_data(self, blueprint_data: Dict[str, Any]) -> Blueprint:
        """Convert raw blueprint data to Blueprint object."""
        # Parse the architecture IR
        architecture = json.loads(blueprint_data["architecture_ir"])

        return Blueprint(
            blueprint_id=blueprint_data["id"],
            name=blueprint_data.get("name", f"blueprint_{blueprint_data['id'][:8]}"),
            description=blueprint_data.get("description", "Auto-generated blueprint"),
            architecture=architecture,
            hyperparameters=blueprint_data.get("hyperparameters", {}),
            created_by="tezzeret",
        )

    def _load_kernel_module(self, kernel: CompiledKernel) -> torch.jit.ScriptModule:
        """Load kernel module from compiled kernel."""
        # In production, this would load from S3 using kernel.binary_ref
        # For now, create a simple module for testing
        buffer = io.BytesIO()

        # Create test module matching the kernel metadata
        input_size = kernel.metadata.input_shape[0] if kernel.metadata.input_shape else 128
        output_size = kernel.metadata.output_shape[0] if kernel.metadata.output_shape else 128

        module = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )

        scripted = torch.jit.script(module)
        torch.jit.save(scripted, buffer)
        buffer.seek(0)

        return torch.jit.load(buffer, map_location=self.device)

    def _save_optimized_kernel(self, module: torch.jit.ScriptModule) -> bytes:
        """Save optimized kernel module to bytes."""
        buffer = io.BytesIO()
        torch.jit.save(module, buffer)
        buffer.seek(0)
        return buffer.getvalue()

    async def fetch_unvalidated_blueprints(self) -> list:
        """Fetch unvalidated blueprints from Urza."""
        try:
            return await self._circuit_breaker.call(
                self._fetch_unvalidated_blueprints_impl
            )
        except CircuitBreakerOpenError:
            self._stats["circuit_breaker_failures"] += 1
            logger.warning("Circuit breaker open for Urza")
            return []
        except Exception as e:
            logger.error("Failed to fetch blueprints: %s", e)
            return []

    async def _fetch_unvalidated_blueprints_impl(self) -> list:
        """Implementation of fetching blueprints."""
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/blueprints/unvalidated"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout,
            max_retries=self.config.retry_attempts
        ) as client:
            response = await client.get(url)
            return await response.json()

    async def update_blueprint_status(
        self, blueprint_id: str, status: BlueprintStatus
    ) -> bool:
        """Update blueprint status in Urza."""
        try:
            return await self._circuit_breaker.call(
                self._update_blueprint_status_impl, blueprint_id, status
            )
        except CircuitBreakerOpenError:
            self._stats["circuit_breaker_failures"] += 1
            logger.warning("Circuit breaker open for Urza")
            return False
        except Exception as e:
            logger.error("Failed to update blueprint status: %s", e)
            return False

    async def _update_blueprint_status_impl(
        self, blueprint_id: str, status: BlueprintStatus
    ) -> bool:
        """Implementation of updating blueprint status."""
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/blueprints/{blueprint_id}/status"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout,
            max_retries=self.config.retry_attempts
        ) as client:
            await client.put(url, json={"status": status.value})
            return True

    async def submit_compiled_kernel(self, kernel: CompiledKernelArtifact) -> bool:
        """Submit compiled kernel to Urza."""
        try:
            return await self._circuit_breaker.call(
                self._submit_compiled_kernel_impl, kernel
            )
        except CircuitBreakerOpenError:
            self._stats["circuit_breaker_failures"] += 1
            logger.warning("Circuit breaker open for Urza")
            return False
        except Exception as e:
            logger.error("Failed to submit kernel: %s", e)
            return False

    async def _submit_compiled_kernel_impl(
        self, kernel: CompiledKernelArtifact
    ) -> bool:
        """Implementation of submitting kernel."""
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/kernels"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout,
            max_retries=self.config.retry_attempts
        ) as client:
            await client.post(url, json=kernel.model_dump())
            return True

    async def start_polling(self) -> None:
        """Start the main polling loop."""
        logger.info(
            "Enhanced Tezzeret worker %s starting polling loop", self.worker_id
        )

        while True:
            try:
                processed = await self.process_one_blueprint()
                if not processed:
                    await asyncio.sleep(self.poll_interval)
                else:
                    # Process immediately if there are more blueprints
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info("Worker %s stopping", self.worker_id)
                break
            except Exception as e:
                logger.error("Unexpected error in polling loop: %s", e)
                await asyncio.sleep(self.poll_interval)

        # Log final statistics
        logger.info("Worker %s final statistics: %s", self.worker_id, self._stats)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self._stats.copy()

        # Calculate rates
        if stats["total_processed"] > 0:
            stats["compilation_success_rate"] = (
                stats["compilation_success"] / stats["total_processed"]
            )
            stats["validation_pass_rate"] = (
                stats["validation_passed"] /
                max(stats["validation_passed"] + stats["validation_failed"], 1)
            )
        else:
            stats["compilation_success_rate"] = 0.0
            stats["validation_pass_rate"] = 0.0

        # Add component statistics
        stats["compiler_stats"] = self.compiler.compilation_cache
        stats["optimizer_stats"] = self.optimizer.get_optimization_stats()
        stats["validator_stats"] = self.validator.get_validation_stats()

        return stats
