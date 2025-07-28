"""
KasminaLayer: High-performance execution layer for morphogenetic kernels.

This module implements the core execution engine that loads and runs
pre-compiled kernel artifacts with minimal overhead.
"""

import asyncio
import logging
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames
from esper.contracts.operational import HealthSignal
from esper.services.oona_client import OonaClient
from esper.utils.circuit_breaker import CircuitBreakerOpenError

from .enhanced_kernel_cache import EnhancedKernelCache
from .error_recovery import ErrorRecoveryManager
from .error_recovery import ErrorType
from .error_recovery import classify_kernel_error
from .error_recovery import create_error_context
from .kernel_executor import KernelExecutionError
from .kernel_executor import RealKernelExecutor
from .state_layout import KasminaStateLayout
from .state_layout import SeedLifecycleState

logger = logging.getLogger(__name__)


class KasminaLayer(nn.Module):
    """
    High-performance execution layer for morphogenetic kernels.

    This layer acts as a pure executor, loading and running pre-compiled
    kernel artifacts from Urza with GPU-resident caching.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_seeds: int = 4,
        cache_size_mb: int = 128,
        telemetry_enabled: bool = True,
        layer_name: str = "kasmina_layer",
        oona_client: Optional[OonaClient] = None,
    ):
        """
        Initialize KasminaLayer.

        Args:
            input_size: Input tensor size
            output_size: Output tensor size
            num_seeds: Number of morphogenetic seeds
            cache_size_mb: Kernel cache size in MB
            telemetry_enabled: Whether to collect telemetry
            layer_name: Name of this layer for telemetry
            oona_client: Optional pre-configured OonaClient instance
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_seeds = num_seeds
        self.layer_name = layer_name
        self.telemetry_enabled = telemetry_enabled

        # Default transformation (preserves original model behavior)
        self.default_transform = nn.Linear(input_size, output_size)

        # State management
        device = (
            next(self.parameters()).device
            if list(self.parameters())
            else torch.device("cpu")
        )
        self.state_layout = KasminaStateLayout(num_seeds, device)

        # Enhanced kernel cache with metadata support
        self.kernel_cache = EnhancedKernelCache(
            max_cache_size_mb=cache_size_mb,
            max_entries=min(num_seeds * 4, 64),  # Reasonable default
        )

        # Real kernel executor
        self.kernel_executor = RealKernelExecutor(
            device=device,
            max_kernel_cache_size=min(num_seeds * 2, 32),
            enable_validation=True,
        )

        # Error recovery system
        self.error_recovery = ErrorRecoveryManager()

        # Telemetry
        self.oona_client = oona_client
        self._telemetry_available = False
        if telemetry_enabled:
            if oona_client is not None:
                # Use provided client
                self.oona_client = oona_client
                self._telemetry_available = True
            else:
                # Try to create new client
                try:
                    self.oona_client = OonaClient()
                    self._telemetry_available = True
                except ImportError as e:
                    # Missing dependency is ok, just disable telemetry
                    logger.warning("Oona client not available, telemetry disabled: %s", e)
                    self.telemetry_enabled = False
                except (ConnectionError, RuntimeError) as e:
                    # Connection/runtime errors are critical - fail fast
                    logger.error("Failed to connect to Oona service: %s", e, exc_info=True)
                    raise RuntimeError(f"Telemetry initialization failed: {e}") from e

        # Performance tracking
        self.total_forward_calls = 0
        self.total_kernel_executions = 0

        logger.info("Initialized KasminaLayer '%s' with %d seeds", layer_name, num_seeds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass with morphogenetic kernel execution.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        self.total_forward_calls += 1
        start_time = time.perf_counter()

        # Fast path for dormant seeds - check if we have any active seeds at all
        # This avoids expensive GPU operations when all seeds are dormant (common case)
        if not self.state_layout.has_active_seeds():
            # All seeds dormant - use default transform only
            output = self.default_transform(x)
        else:
            # Default transformation (always computed for blending)
            default_output = self.default_transform(x)

            # Execute with kernels if any are active
            active_seeds = self.state_layout.get_active_seeds()

            # Handle async kernel execution in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, we can't use run_until_complete
                    # Fall back to sync execution
                    kernel_output = self._execute_with_kernels_sync(x, active_seeds)
                else:
                    # No running loop, safe to use run_until_complete
                    kernel_output = loop.run_until_complete(
                        self._execute_with_kernels(x, active_seeds)
                    )
            except RuntimeError:
                # Fallback to sync execution if async fails
                kernel_output = self._execute_with_kernels_sync(x, active_seeds)

            output = self._blend_outputs(default_output, kernel_output, active_seeds)

        # Update telemetry
        if self.telemetry_enabled and self._telemetry_available:
            exec_time_us = int((time.perf_counter() - start_time) * 1_000_000)
            # For telemetry, we need the actual count - use a lightweight check
            active_count = (
                self.state_layout.get_active_count()
                if self.state_layout.has_active_seeds()
                else 0
            )
            self._update_telemetry(exec_time_us, active_count)

        return output

    def _execute_with_kernels_sync(
        self, x: torch.Tensor, active_seeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Synchronous kernel execution fallback.

        Used when async execution is not available or fails.
        This should not use placeholder kernels in production/tests.

        Args:
            x: Input tensor
            active_seeds: Boolean mask of active seeds

        Returns:
            Output tensor from kernel execution
        """
        # In sync mode, we still need to apply active kernels
        # but without async execution
        # TODO: Implement proper sync kernel execution
        # For now, just return default transform
        return self.default_transform(x)

    async def _execute_with_kernels(
        self, x: torch.Tensor, active_seeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute forward pass using active kernels.

        Args:
            x: Input tensor
            active_seeds: Boolean mask of active seeds

        Returns:
            Output tensor from kernel execution
        """
        batch_size = x.size(0)
        output = torch.zeros(
            batch_size, self.output_size, device=x.device, dtype=x.dtype
        )

        for seed_idx in range(self.num_seeds):
            if not active_seeds[seed_idx]:
                continue

            try:
                # Execute kernel for this seed using real kernel execution
                kernel_output = await self._execute_kernel_real(x, seed_idx)

                # Accumulate output (the blending is now handled in kernel executor)
                alpha = self.state_layout.alpha_blend[seed_idx].item()
                output += alpha * kernel_output

                self.total_kernel_executions += 1

                # Update telemetry for this seed
                health_score = self._compute_health_score(kernel_output)
                self.state_layout.update_telemetry(
                    seed_idx, 100, health_score
                )

            except Exception as e:
                logger.warning("Kernel execution failed for seed %d: %s", seed_idx, e)
                error_count = self.state_layout.increment_error_count(seed_idx)

                if error_count >= 3:
                    logger.error(
                        f"Seed {seed_idx} moved to error recovery after {error_count} failures"
                    )

        return output

    async def _execute_kernel_real(
        self, x: torch.Tensor, seed_idx: int
    ) -> torch.Tensor:
        """
        Real kernel execution using compiled artifacts with metadata validation.

        Args:
            x: Input tensor
            seed_idx: Index of the seed

        Returns:
            Kernel output tensor

        Raises:
            KernelExecutionError: If kernel execution fails
        """
        # Get kernel artifact ID from state
        kernel_id = int(self.state_layout.active_kernel_id[seed_idx].item())
        if kernel_id == 0:
            # No kernel loaded, fallback to default
            return self.default_transform(x)

        try:
            # Load kernel with validation using enhanced cache
            kernel_data = await self.kernel_cache.load_kernel_with_validation(
                artifact_id=str(kernel_id),
                target_shape=x.shape,
                device=x.device,
                batch_size=x.size(0),
            )

            if kernel_data is None:
                logger.warning(
                    f"Kernel {kernel_id} not compatible or not found for seed {seed_idx}"
                )
                # Fallback to default transformation
                return self.default_transform(x)

            _, metadata = kernel_data

            # Get kernel bytes for execution
            kernel_artifact = await self.kernel_cache.get_kernel_bytes(str(kernel_id))
            if kernel_artifact is None:
                logger.warning(
                    f"Kernel bytes {kernel_id} not found for seed {seed_idx}"
                )
                return self.default_transform(x)

            # Get alpha blending factor
            alpha = self.state_layout.alpha_blend[seed_idx].item()

            # Execute kernel with real executor
            result = await self.kernel_executor.execute_kernel(
                kernel_artifact=kernel_artifact,
                input_tensor=x,
                original_shape=x.shape,
                blend_alpha=alpha,
                kernel_id=str(kernel_id),
            )

            # Note: Success metrics are tracked via telemetry
            # Error counts are reset when transitioning to ACTIVE state

            # Log performance info
            logger.debug(
                f"Executed kernel {kernel_id} for seed {seed_idx}: "
                f"{metadata.parameter_count} params, {metadata.memory_footprint_mb:.1f}MB"
            )

            return result

        except KernelExecutionError as e:
            # Create error context for recovery system
            error_context = create_error_context(
                error_type=ErrorType.KERNEL_EXECUTION,
                component="kasmina_layer",
                layer_name=self.layer_name,
                exception=e,
                seed_idx=seed_idx,
                kernel_id=str(kernel_id),
            )

            # Handle error through recovery system
            await self.error_recovery.handle_error(
                error_context, fallback_action=lambda: self.default_transform(x)
            )

            # Increment error count and handle circuit breaker
            error_count = self.state_layout.increment_error_count(seed_idx)

            if error_count >= 3:
                logger.warning(
                    f"Seed {seed_idx} exceeded error threshold, unloading kernel"
                )
                # Unload problematic kernel
                await self.unload_kernel(seed_idx)

            # Fallback to default transformation
            return self.default_transform(x)

        except Exception as e:
            # Classify and handle unexpected errors
            error_type = classify_kernel_error(e)
            error_context = create_error_context(
                error_type=error_type,
                component="kasmina_layer",
                layer_name=self.layer_name,
                exception=e,
                seed_idx=seed_idx,
                kernel_id=str(kernel_id) if kernel_id else None,
            )

            # Handle through recovery system
            await self.error_recovery.handle_error(
                error_context, fallback_action=lambda: self.default_transform(x)
            )

            # Fallback to default transformation
            return self.default_transform(x)


    def _blend_outputs(
        self,
        default_output: torch.Tensor,
        kernel_output: torch.Tensor,
        active_seeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Blend default and kernel outputs using alpha blending.

        Args:
            default_output: Output from default transformation
            kernel_output: Output from kernel execution
            active_seeds: Boolean mask of active seeds

        Returns:
            Blended output tensor
        """
        # Compute overall alpha (how much to blend in kernel output)
        total_alpha = 0.0
        for seed_idx in range(self.num_seeds):
            if active_seeds[seed_idx]:
                total_alpha += self.state_layout.alpha_blend[seed_idx].item()

        # Clamp alpha to [0, 1]
        total_alpha = min(total_alpha, 1.0)

        # Blend outputs
        return (1.0 - total_alpha) * default_output + total_alpha * kernel_output

    def _compute_health_score(self, output: torch.Tensor) -> float:
        """
        Compute health score for kernel output.

        Args:
            output: Kernel output tensor

        Returns:
            Health score between 0.0 and 1.0
        """
        # Simple health metric: inverse of gradient norm
        # In production, this would be more sophisticated
        try:
            grad_norm = torch.norm(output, dim=None).item()
            return 1.0 / (1.0 + grad_norm)
        except Exception:
            return 0.5  # Neutral score if computation fails

    def _update_telemetry(self, exec_time_us: int, _active_seed_count: int) -> None:
        """
        Update and publish telemetry data.

        Args:
            exec_time_us: Execution time in microseconds
            _active_seed_count: Number of active seeds (unused)
        """
        try:
            # Collect layer statistics
            state_stats = self.state_layout.get_stats()

            # Create health signal
            health_signal = HealthSignal(
                layer_id=hash(self.layer_name) % 10000,  # Simple layer ID
                seed_id=0,  # Representative seed
                chunk_id=0,  # Not applicable for execution telemetry
                epoch=self.total_forward_calls,  # Use forward call count as epoch proxy
                activation_variance=state_stats["avg_health"],
                dead_neuron_ratio=min(
                    state_stats["total_errors"] / max(state_stats["num_seeds"], 1), 1.0
                ),
                avg_correlation=state_stats["avg_health"],
                is_ready_for_transition=False,
            )

            # Publish via Oona (async)
            if self.oona_client:
                try:
                    # Create task and let it run in background
                    task = asyncio.create_task(self._publish_health_signal(health_signal))
                    # Add exception handler to prevent warnings
                    task.add_done_callback(lambda t: None if not t.exception() else logger.debug("Telemetry publish error: %s", t.exception()))
                except RuntimeError:
                    # If no event loop is running, skip telemetry
                    pass

        except Exception as e:
            logger.warning("Failed to update telemetry: %s", e)

    async def _publish_health_signal(self, health_signal: HealthSignal) -> None:
        """
        Publish health signal to Oona message bus.

        Args:
            health_signal: Health signal to publish
        """
        try:
            message = OonaMessage(
                sender_id=f"kasmina.{self.layer_name}",
                trace_id=f"telemetry-{int(time.time())}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload=health_signal.model_dump(),
            )
            await self.oona_client.publish(message)
        except Exception as e:
            logger.warning("Failed to publish health signal: %s", e)

    async def load_kernel(self, seed_idx: int, artifact_id: str) -> bool:
        """
        Load a compiled kernel for a specific seed.

        Args:
            seed_idx: Index of the seed
            artifact_id: ID of the kernel artifact

        Returns:
            True if kernel was loaded successfully
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        # Enhanced error context
        error_context = {
            "layer_name": self.layer_name,
            "seed_idx": seed_idx,
            "artifact_id": artifact_id,
            "operation": "load_kernel",
        }

        try:
            # Transition to loading state
            self.state_layout.transition_seed_state(
                seed_idx, SeedLifecycleState.LOADING
            )

            # Load kernel with enhanced validation
            # First, try to get a dummy tensor to determine current shape requirements
            dummy_input = torch.zeros(
                1, self.input_size, device=self.state_layout.device
            )

            kernel_data = await self.kernel_cache.load_kernel_with_validation(
                artifact_id=artifact_id,
                target_shape=dummy_input.shape,
                device=self.state_layout.device,
                batch_size=32,  # Default batch size for estimation
            )

            if kernel_data is not None:
                _, metadata = kernel_data

                # Transition to active state
                kernel_id = hash(artifact_id)
                self.state_layout.transition_seed_state(
                    seed_idx, SeedLifecycleState.ACTIVE, kernel_id
                )

                # Set blend factor based on kernel confidence (could be configured)
                # Use metadata to determine optimal blending
                confidence_factor = metadata.performance_profile.get("confidence", 0.5)
                self.state_layout.alpha_blend[seed_idx] = min(
                    confidence_factor * 0.6, 0.5
                )

                logger.info(
                    f"Loaded kernel {artifact_id} for seed {seed_idx}: "
                    f"{metadata.parameter_count} params, alpha={self.state_layout.alpha_blend[seed_idx].item():.2f}"
                )
                await self._publish_success_telemetry(error_context)
                return True
            else:
                # Failed to load or incompatible, apply recovery strategy
                await self._handle_kernel_load_failure(
                    error_context, "kernel_incompatible", None
                )
                return False

        except CircuitBreakerOpenError as e:
            # Create error context for circuit breaker
            recovery_error_context = create_error_context(
                error_type=ErrorType.CIRCUIT_BREAKER,
                component="kernel_cache",
                layer_name=self.layer_name,
                exception=e,
                seed_idx=seed_idx,
                kernel_id=artifact_id,
            )

            # Handle through recovery system
            await self.error_recovery.handle_error(recovery_error_context)

            # Apply specific recovery strategy
            await self._handle_kernel_load_failure(
                error_context, "circuit_breaker_open", None
            )
            return False

        except Exception as e:
            # Classify and handle unexpected errors
            error_type = classify_kernel_error(e)
            recovery_error_context = create_error_context(
                error_type=error_type,
                component="kasmina_layer",
                layer_name=self.layer_name,
                exception=e,
                seed_idx=seed_idx,
                kernel_id=artifact_id,
            )

            # Handle through recovery system
            await self.error_recovery.handle_error(recovery_error_context)

            # Apply general recovery strategy
            await self._handle_kernel_load_failure(error_context, "unexpected_error", e)
            return False

    async def _handle_kernel_load_failure(
        self,
        error_context: dict,
        failure_reason: str,
        exception: Optional[Exception] = None,
    ) -> None:
        """
        Centralized error handling for kernel loading failures.

        Args:
            error_context: Context information about the error
            failure_reason: Categorized reason for failure
            exception: Optional exception that caused the failure
        """
        seed_idx = error_context["seed_idx"]
        artifact_id = error_context["artifact_id"]

        # Apply recovery strategy based on failure reason
        if failure_reason == "circuit_breaker_open":
            # Urza service unavailable - transition to dormant and wait
            self.state_layout.transition_seed_state(
                seed_idx, SeedLifecycleState.DORMANT
            )
            logger.warning(
                f"Circuit breaker open for kernel {artifact_id}, seed {seed_idx} staying dormant"
            )

        elif failure_reason == "kernel_not_found":
            # Kernel doesn't exist - transition to dormant
            self.state_layout.transition_seed_state(
                seed_idx, SeedLifecycleState.DORMANT
            )
            logger.warning(
                f"Kernel {artifact_id} not found, seed {seed_idx} remaining dormant"
            )

        elif failure_reason == "kernel_incompatible":
            # Kernel incompatible with current requirements - transition to dormant
            self.state_layout.transition_seed_state(
                seed_idx, SeedLifecycleState.DORMANT
            )
            logger.warning(
                f"Kernel {artifact_id} incompatible with current requirements, seed {seed_idx} remaining dormant"
            )

        elif failure_reason == "unexpected_error":
            # Unknown error - increment error count and potentially go to error recovery
            error_count = self.state_layout.increment_error_count(seed_idx)
            if error_count >= 3:
                self.state_layout.transition_seed_state(
                    seed_idx, SeedLifecycleState.ERROR_RECOVERY
                )
                logger.error(
                    f"Seed {seed_idx} moved to ERROR_RECOVERY after {error_count} failures"
                )
            else:
                self.state_layout.transition_seed_state(
                    seed_idx, SeedLifecycleState.DORMANT
                )
                logger.warning(
                    f"Kernel load failed for seed {seed_idx} (attempt {error_count}/3): {exception}"
                )

        # Publish error telemetry
        await self._publish_error_telemetry(error_context, failure_reason, exception)

    async def _publish_success_telemetry(self, context: dict) -> None:
        """Publish success telemetry for kernel operations."""
        if not self.telemetry_enabled or not self._telemetry_available:
            return

        try:
            message = OonaMessage(
                sender_id=f"{self.layer_name}_success",
                trace_id=f"kernel_load_{context['artifact_id']}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload={
                    "event_type": "kernel_load_success",
                    "layer_name": context["layer_name"],
                    "seed_idx": context["seed_idx"],
                    "artifact_id": context["artifact_id"],
                    "timestamp": time.time(),
                },
            )
            await self.oona_client.publish(message)
        except Exception as e:
            logger.debug("Failed to publish success telemetry: %s", e)

    async def _publish_error_telemetry(
        self, context: dict, failure_reason: str, exception: Optional[Exception] = None
    ) -> None:
        """Publish error telemetry for kernel operations."""
        if not self.telemetry_enabled or not self._telemetry_available:
            return

        try:
            payload = {
                "event_type": "kernel_load_failure",
                "layer_name": context["layer_name"],
                "seed_idx": context["seed_idx"],
                "artifact_id": context["artifact_id"],
                "failure_reason": failure_reason,
                "timestamp": time.time(),
            }

            if exception:
                payload.update(
                    {
                        "exception_type": type(exception).__name__,
                        "exception_message": str(exception),
                    }
                )

            message = OonaMessage(
                sender_id=f"{self.layer_name}_error",
                trace_id=f"kernel_load_error_{context['artifact_id']}",
                topic=TopicNames.TELEMETRY_SEED_HEALTH,
                payload=payload,
            )
            await self.oona_client.publish(message)
        except Exception as e:
            logger.debug("Failed to publish error telemetry: %s", e)

    async def unload_kernel(self, seed_idx: int) -> bool:
        """
        Unload kernel from a specific seed.

        Args:
            seed_idx: Index of the seed

        Returns:
            True if kernel was unloaded successfully
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        try:
            self.state_layout.transition_seed_state(
                seed_idx, SeedLifecycleState.DORMANT
            )
            self.state_layout.alpha_blend[seed_idx] = 0.0

            logger.info("Unloaded kernel from seed %d", seed_idx)
            return True

        except Exception as e:
            logger.error("Error unloading kernel from seed %d: %s", seed_idx, e)
            return False

    def get_layer_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive layer statistics.

        Returns:
            Dictionary containing layer statistics
        """
        state_stats = self.state_layout.get_stats()
        cache_stats = self.kernel_cache.get_enhanced_stats()

        return {
            "layer_name": self.layer_name,
            "total_forward_calls": self.total_forward_calls,
            "total_kernel_executions": self.total_kernel_executions,
            "kernel_execution_ratio": (
                self.total_kernel_executions / max(self.total_forward_calls, 1)
            ),
            "state_stats": state_stats,
            "cache_stats": cache_stats,
            "error_recovery_stats": self.error_recovery.get_recovery_stats(),
            "telemetry_enabled": self.telemetry_enabled,
        }

    def find_compatible_kernels(
        self, max_memory_mb: Optional[float] = None
    ) -> List[Tuple[str, Any]]:
        """
        Find all kernels compatible with this layer's requirements.

        Args:
            max_memory_mb: Maximum allowed memory usage per kernel

        Returns:
            List of (artifact_id, metadata) tuples for compatible kernels
        """
        # Create dummy input to determine shape requirements
        dummy_input = torch.zeros(1, self.input_size, device=self.state_layout.device)

        return self.kernel_cache.find_compatible_kernels(
            target_shape=dummy_input.shape,
            device=self.state_layout.device,
            max_memory_mb=max_memory_mb,
        )

    def set_seed_alpha(self, seed_idx: int, alpha: float) -> None:
        """
        Set the alpha blend factor for a specific seed.

        Args:
            seed_idx: Index of the seed
            alpha: Blend factor (0.0 to 1.0)
        """
        if seed_idx < 0 or seed_idx >= self.num_seeds:
            raise ValueError(f"Invalid seed index: {seed_idx}")

        alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        self.state_layout.alpha_blend[seed_idx] = alpha

        logger.debug("Set alpha=%s for seed %d", alpha, seed_idx)

    def to(self, device: torch.device) -> "KasminaLayer":
        """
        Move layer to specified device.

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        super().to(device)

        # Move state tensors to device
        self.state_layout.lifecycle_states = self.state_layout.lifecycle_states.to(
            device
        )
        self.state_layout.active_kernel_id = self.state_layout.active_kernel_id.to(
            device
        )
        self.state_layout.alpha_blend = self.state_layout.alpha_blend.to(device)
        self.state_layout.health_accumulator = self.state_layout.health_accumulator.to(
            device
        )
        self.state_layout.last_update_epoch = self.state_layout.last_update_epoch.to(
            device
        )
        self.state_layout.exec_latency_us = self.state_layout.exec_latency_us.to(device)
        self.state_layout.error_count = self.state_layout.error_count.to(device)
        self.state_layout.fallback_active = self.state_layout.fallback_active.to(device)

        self.state_layout.device = device

        return self
