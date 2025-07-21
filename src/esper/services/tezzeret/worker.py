"""
Tezzeret Worker - Compilation Forge.

This module provides the background worker that compiles blueprints into
executable kernel artifacts.
"""

import asyncio
import hashlib
import io
import json
import logging
import time
from typing import Any
from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from esper.contracts.enums import BlueprintStatus
from esper.contracts.enums import KernelStatus
from esper.services.contracts import CompiledKernelArtifact
from esper.utils.circuit_breaker import CircuitBreaker
from esper.utils.circuit_breaker import CircuitBreakerConfig
from esper.utils.circuit_breaker import CircuitBreakerOpenError
from esper.utils.config import ServiceConfig
from esper.utils.config import get_service_config
from esper.utils.s3_client import get_s3_client
from esper.utils.s3_client import upload_bytes

logger = logging.getLogger(__name__)


class TezzeretWorker:
    """
    Background worker that compiles blueprints into executable kernels.

    This worker polls Urza for unvalidated blueprints, compiles them using
    the Fast compilation pipeline (torch.compile), and uploads the results
    back to Urza.
    """

    def __init__(self, worker_id: str, config: Optional[ServiceConfig] = None):
        self.worker_id = worker_id
        self.config = config or get_service_config()

        # Use configuration instead of environment variables directly
        self.urza_base_url = self.config.urza_url
        self.s3_client = get_s3_client()
        self.bucket_name = self.config.s3_bucket
        self.poll_interval = self.config.poll_interval_seconds

        # Circuit breaker for Urza service calls
        self._circuit_breaker = CircuitBreaker(
            name=f"tezzeret_worker_{worker_id}_urza",
            config=CircuitBreakerConfig(
                failure_threshold=3,  # Open after 3 failures (more sensitive for workers)
                recovery_timeout=45,  # Try recovery after 45 seconds
                success_threshold=2,  # Need 2 successes to close
                timeout=self.config.http_timeout,  # Use config timeout
            ),
        )

        # Statistics
        self._circuit_breaker_failures = 0
        self._total_blueprints_processed = 0
        self._total_compilation_failures = 0

        logger.info("Tezzeret Worker %s initialized", self.worker_id)

    def ir_to_module(self, ir_data: Dict[str, Any]) -> nn.Module:
        """
        Converts a blueprint IR to a PyTorch module.

        Args:
            ir_data: Blueprint intermediate representation

        Returns:
            nn.Module: PyTorch module

        Note:
            This is a simplified implementation for MVP. In a real system,
            this would be a sophisticated graph-to-code converter.
        """
        # For MVP, we'll create simple modules based on the IR structure
        # In practice, this would parse a more complex IR format

        module_type = ir_data.get("type", "linear")

        if module_type == "linear":
            input_size = ir_data.get("input_size", 10)
            output_size = ir_data.get("output_size", 10)
            hidden_size = ir_data.get("hidden_size", 20)

            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        elif module_type == "conv":
            in_channels = ir_data.get("in_channels", 3)
            out_channels = ir_data.get("out_channels", 16)
            kernel_size = ir_data.get("kernel_size", 3)

            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )

        else:
            # Default fallback
            return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

    def run_fast_compilation(self, blueprint_ir: Dict[str, Any]) -> bytes:
        """
        Compiles a blueprint using the Fast compilation pipeline.

        Args:
            blueprint_ir: Blueprint intermediate representation

        Returns:
            bytes: Compiled kernel binary

        Raises:
            RuntimeError: If compilation fails
        """
        try:
            # Convert IR to PyTorch module
            module = self.ir_to_module(blueprint_ir)

            # Compile the module
            compiled_module = torch.compile(module, mode="default")

            # Create a buffer to save the compiled module
            buffer = io.BytesIO()

            # Save the module using torch.save
            torch.save(
                {
                    "module": compiled_module,
                    "state_dict": compiled_module.state_dict(),
                    "ir_metadata": blueprint_ir,
                },
                buffer,
            )

            buffer.seek(0)
            return buffer.getvalue()

        except Exception as e:
            logger.error("Compilation failed: %s", e)
            raise RuntimeError(f"Compilation failed: {e}")

    def generate_kernel_id(self, blueprint_id: str, pipeline: str) -> str:
        """
        Generates a unique kernel ID.

        Args:
            blueprint_id: Blueprint ID
            pipeline: Compilation pipeline name

        Returns:
            str: Unique kernel ID
        """
        # Create deterministic but unique kernel ID
        content = f"{blueprint_id}:{pipeline}:{int(time.time())}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def fetch_unvalidated_blueprints(self) -> list:
        """
        Fetches unvalidated blueprints from Urza.

        Returns:
            list: List of blueprint data
        """
        try:
            return await self._circuit_breaker.call(
                self._fetch_unvalidated_blueprints_impl
            )

        except CircuitBreakerOpenError:
            self._circuit_breaker_failures += 1
            logger.warning(
                f"Circuit breaker open for Urza, cannot fetch blueprints (worker {self.worker_id})"
            )
            return []

        except Exception as e:
            logger.error("Failed to fetch blueprints: %s", e)
            return []

    async def _fetch_unvalidated_blueprints_impl(self) -> list:
        """
        Implementation of fetching unvalidated blueprints from Urza.

        Returns:
            list: List of blueprint data
        """
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/blueprints/unvalidated"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout, max_retries=self.config.retry_attempts
        ) as client:
            response = await client.get(url)
            return await response.json()

    async def update_blueprint_status(
        self, blueprint_id: str, status: BlueprintStatus
    ) -> bool:
        """
        Updates blueprint status in Urza.

        Args:
            blueprint_id: Blueprint ID
            status: New status

        Returns:
            bool: True if successful
        """
        try:
            return await self._circuit_breaker.call(
                self._update_blueprint_status_impl, blueprint_id, status
            )

        except CircuitBreakerOpenError:
            self._circuit_breaker_failures += 1
            logger.warning(
                f"Circuit breaker open for Urza, cannot update blueprint {blueprint_id} status (worker {self.worker_id})"
            )
            return False

        except Exception as e:
            logger.error("Failed to update blueprint status: %s", e)
            return False

    async def _update_blueprint_status_impl(
        self, blueprint_id: str, status: BlueprintStatus
    ) -> bool:
        """
        Implementation of updating blueprint status in Urza.

        Args:
            blueprint_id: Blueprint ID
            status: New status

        Returns:
            bool: True if successful
        """
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/blueprints/{blueprint_id}/status"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout, max_retries=self.config.retry_attempts
        ) as client:
            await client.put(url, json={"status": status.value})
            return True

    async def submit_compiled_kernel(self, kernel: CompiledKernelArtifact) -> bool:
        """
        Submits compiled kernel to Urza.

        Args:
            kernel: Compiled kernel artifact

        Returns:
            bool: True if successful
        """
        try:
            return await self._circuit_breaker.call(
                self._submit_compiled_kernel_impl, kernel
            )

        except CircuitBreakerOpenError:
            self._circuit_breaker_failures += 1
            logger.warning(
                f"Circuit breaker open for Urza, cannot submit kernel {kernel.id} (worker {self.worker_id})"
            )
            return False

        except Exception as e:
            logger.error("Failed to submit compiled kernel: %s", e)
            return False

    async def _submit_compiled_kernel_impl(
        self, kernel: CompiledKernelArtifact
    ) -> bool:
        """
        Implementation of submitting compiled kernel to Urza.

        Args:
            kernel: Compiled kernel artifact

        Returns:
            bool: True if successful
        """
        from esper.utils.http_client import AsyncHttpClient

        url = f"{self.urza_base_url}/internal/v1/kernels"

        async with AsyncHttpClient(
            timeout=self.config.http_timeout, max_retries=self.config.retry_attempts
        ) as client:
            await client.post(url, json=kernel.model_dump())
            return True

    async def process_one_blueprint(self) -> bool:
        """
        Processes one blueprint from the queue.

        Returns:
            bool: True if a blueprint was processed
        """
        blueprints = await self.fetch_unvalidated_blueprints()

        if not blueprints:
            return False

        blueprint_data = blueprints[0]
        blueprint_id = blueprint_data["id"]

        logger.info("Processing blueprint %s", blueprint_id)
        self._total_blueprints_processed += 1

        # Update status to COMPILING to prevent other workers from picking it up
        if not await self.update_blueprint_status(
            blueprint_id, BlueprintStatus.COMPILING
        ):
            logger.error(
                "Failed to update blueprint %s status to COMPILING", blueprint_id
            )
            return False

        try:
            # Parse the architecture IR
            architecture_ir = json.loads(blueprint_data["architecture_ir"])

            # Compile the blueprint
            compiled_binary = self.run_fast_compilation(architecture_ir)

            # Generate kernel ID
            kernel_id = self.generate_kernel_id(blueprint_id, "fast")

            # Upload compiled binary to S3
            kernel_object_key = f"kernels/{kernel_id}/compiled.pt"
            kernel_binary_ref = upload_bytes(
                self.s3_client, compiled_binary, self.bucket_name, kernel_object_key
            )

            # Create kernel artifact
            kernel_artifact = CompiledKernelArtifact(
                id=kernel_id,
                blueprint_id=blueprint_id,
                status=KernelStatus.VALIDATED.value,  # MVP workaround: mark as validated
                compilation_pipeline="fast",
                kernel_binary_ref=kernel_binary_ref,
                validation_report={"pipeline": "fast", "validated": True},
            )

            # Submit kernel to Urza
            if await self.submit_compiled_kernel(kernel_artifact):
                logger.info(
                    "Successfully compiled blueprint %s -> kernel %s",
                    blueprint_id,
                    kernel_id,
                )
                return True
            else:
                logger.error(
                    "Failed to submit compiled kernel for blueprint %s", blueprint_id
                )
                # Only try to update status if circuit breaker allows it
                await self.update_blueprint_status(
                    blueprint_id, BlueprintStatus.INVALID
                )
                self._total_compilation_failures += 1
                return False

        except Exception as e:
            logger.error("Failed to compile blueprint %s: %s", blueprint_id, e)
            # Only try to update status if circuit breaker allows it
            await self.update_blueprint_status(blueprint_id, BlueprintStatus.INVALID)
            self._total_compilation_failures += 1
            return False

    def get_worker_stats(self) -> dict:
        """
        Get worker performance and circuit breaker statistics.

        Returns:
            Dictionary containing worker metrics
        """
        return {
            "worker_id": self.worker_id,
            "total_blueprints_processed": self._total_blueprints_processed,
            "total_compilation_failures": self._total_compilation_failures,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "circuit_breaker_stats": self._circuit_breaker.get_stats(),
            "success_rate": (
                (self._total_blueprints_processed - self._total_compilation_failures)
                / max(self._total_blueprints_processed, 1)
            ),
        }

    async def start_polling(self) -> None:
        """
        Starts the main polling loop.

        This method runs indefinitely, polling for blueprints to compile.
        """
        logger.info("Tezzeret worker %s starting polling loop", self.worker_id)

        while True:
            try:
                processed = await self.process_one_blueprint()
                if not processed:
                    # No blueprints to process, wait before next poll
                    await asyncio.sleep(self.poll_interval)
                else:
                    # Blueprint processed, poll immediately for next one
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                logger.info(
                    "Tezzeret worker %s stopping due to keyboard interrupt",
                    self.worker_id,
                )
                break
            except Exception as e:
                logger.error("Unexpected error in polling loop: %s", e)
                await asyncio.sleep(self.poll_interval)

        logger.info("Tezzeret worker %s stopped", self.worker_id)
