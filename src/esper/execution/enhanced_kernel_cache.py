"""
Enhanced GPU-resident LRU cache for kernel artifacts with metadata validation.

This module extends the basic KernelCache with metadata tracking, shape validation,
and compatibility checking for better kernel management.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch

from esper.contracts.assets import KernelMetadata
from esper.execution.kernel_cache import KernelCache
from esper.utils.circuit_breaker import CircuitBreakerOpenError

logger = logging.getLogger(__name__)


class KernelValidator:
    """Validates kernel compatibility and safety requirements."""

    def __init__(self):
        self.max_parameter_count = 10_000_000  # 10M parameters max
        self.max_memory_mb = 2048  # 2GB max per kernel
        self.supported_devices = {"cpu", "cuda"}

    def validate_compatibility(
        self,
        metadata: KernelMetadata,
        target_shape: torch.Size,
        device: torch.device,
        max_memory_mb: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Validate kernel compatibility with execution requirements.

        Args:
            metadata: Kernel metadata to validate
            target_shape: Target input tensor shape
            device: Target execution device
            max_memory_mb: Maximum allowed memory usage

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check parameter count
        if metadata.parameter_count > self.max_parameter_count:
            return (
                False,
                f"Too many parameters: {metadata.parameter_count} > {self.max_parameter_count}",
            )

        # Check shape compatibility
        target_shape_list = list(target_shape[1:])  # Exclude batch dimension
        if not metadata.is_compatible_with_shape(target_shape_list):
            return (
                False,
                f"Shape mismatch: kernel expects {metadata.input_shape}, got {target_shape_list}",
            )

        # Check device requirements
        device_str = str(device).split(":")[0]  # Extract device type
        if device_str not in self.supported_devices:
            return False, f"Unsupported device: {device_str}"

        if (
            metadata.device_requirements
            and device_str not in metadata.device_requirements
        ):
            return (
                False,
                f"Device {device_str} not in requirements: {metadata.device_requirements}",
            )

        # Check memory requirements
        max_allowed = max_memory_mb or self.max_memory_mb
        if metadata.memory_footprint_mb > max_allowed:
            return (
                False,
                f"Memory footprint too large: {metadata.memory_footprint_mb}MB > {max_allowed}MB",
            )

        return True, ""

    def estimate_execution_memory(
        self, metadata: KernelMetadata, batch_size: int
    ) -> float:
        """Estimate total memory usage for kernel execution."""
        base_memory = metadata.get_memory_estimate(batch_size)

        # Add overhead for intermediate tensors (rough estimate)
        overhead_factor = 1.5  # 50% overhead
        return base_memory * overhead_factor


class EnhancedKernelCache(KernelCache):
    """
    Enhanced kernel cache with metadata tracking and validation.

    Extends the basic KernelCache with:
    - Kernel metadata caching
    - Shape and device compatibility validation
    - Memory usage tracking
    - Performance profiling
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Metadata cache
        self.metadata_cache: OrderedDict[str, KernelMetadata] = OrderedDict()

        # Enhanced statistics
        self.compatibility_checks = 0
        self.compatibility_failures = 0
        self.memory_usage_estimates: Dict[str, float] = {}

        # Validator
        self.validator = KernelValidator()

        logger.info("Initialized EnhancedKernelCache with metadata support")

    async def load_kernel_with_validation(
        self,
        artifact_id: str,
        target_shape: torch.Size,
        device: torch.device,
        batch_size: int = 32,
    ) -> Optional[Tuple[torch.Tensor, KernelMetadata]]:
        """
        Load kernel with shape and device validation.

        Args:
            artifact_id: ID of the kernel artifact
            target_shape: Expected input tensor shape
            device: Target execution device
            batch_size: Batch size for memory estimation

        Returns:
            Tuple of (kernel_tensor, metadata) or None if not compatible
        """
        async with self._lock:
            self.compatibility_checks += 1

            # Check if we have metadata cached
            if artifact_id in self.metadata_cache:
                metadata = self.metadata_cache[artifact_id]

                # Validate compatibility
                is_valid, error_msg = self.validator.validate_compatibility(
                    metadata, target_shape, device, self.max_cache_size_mb
                )

                if not is_valid:
                    self.compatibility_failures += 1
                    logger.warning(
                        f"Kernel {artifact_id} compatibility failed: {error_msg}"
                    )
                    return None

                # Check if kernel tensor is cached
                if artifact_id in self._cache:
                    # Move to end (most recently used)
                    kernel_tensor = self._cache.pop(artifact_id)
                    self._cache[artifact_id] = kernel_tensor
                    self._cache_info[artifact_id]["last_accessed"] = time.time()
                    self._hits += 1

                    logger.debug("Enhanced cache hit for kernel %s", artifact_id)
                    return kernel_tensor, metadata

            # Cache miss - fetch from Urza with metadata
            self._misses += 1
            logger.debug(
                f"Enhanced cache miss for kernel {artifact_id}, fetching with metadata"
            )

            kernel_data = await self._fetch_kernel_with_metadata(artifact_id)
            if kernel_data is None:
                return None

            kernel_tensor, metadata = kernel_data

            # Validate compatibility with fetched metadata
            is_valid, error_msg = self.validator.validate_compatibility(
                metadata, target_shape, device, self.max_cache_size_mb
            )

            if not is_valid:
                self.compatibility_failures += 1
                logger.warning(
                    f"Fetched kernel {artifact_id} compatibility failed: {error_msg}"
                )
                return None

            # Estimate memory usage
            estimated_memory = self.validator.estimate_execution_memory(
                metadata, batch_size
            )
            self.memory_usage_estimates[artifact_id] = estimated_memory

            # Add to cache with metadata
            self._add_to_cache_with_metadata(artifact_id, kernel_tensor, metadata)

            return kernel_tensor, metadata

    async def _fetch_kernel_with_metadata(
        self, artifact_id: str
    ) -> Optional[Tuple[torch.Tensor, KernelMetadata]]:
        """
        Fetch kernel and metadata from Urza.

        Args:
            artifact_id: ID of the kernel artifact

        Returns:
            Tuple of (kernel_tensor, metadata) or None if not found
        """
        try:
            return await self._circuit_breaker.call(
                self._fetch_kernel_with_metadata_impl, artifact_id
            )
        except CircuitBreakerOpenError:
            self._circuit_breaker_failures += 1
            logger.warning("Circuit breaker open, cannot fetch kernel %s", artifact_id)
            return None
        except Exception as e:
            logger.error("Failed to fetch kernel with metadata %s: %s", artifact_id, e)
            return None

    async def _fetch_kernel_with_metadata_impl(
        self, artifact_id: str
    ) -> Optional[Tuple[torch.Tensor, KernelMetadata]]:
        """
        Implementation of kernel+metadata fetching from Urza.

        Args:
            artifact_id: ID of the kernel artifact

        Returns:
            Tuple of (kernel_tensor, metadata) or None if not found
        """
        from esper.utils.http_client import AsyncHttpClient

        # Use configured Urza URL
        urza_url = self.config.get_urza_api_url()

        async with AsyncHttpClient(
            timeout=self.config.http_timeout, max_retries=self.config.retry_attempts
        ) as client:
            # Fetch compiled kernel info from Urza API
            response = await client.get(f"{urza_url}/kernels/{artifact_id}")
            if response.status == 404:
                logger.warning("Kernel %s not found in Urza", artifact_id)
                return None

            kernel_info = await response.json()

            # Parse metadata
            try:
                metadata = KernelMetadata(**kernel_info["metadata"])
            except KeyError as e:
                logger.error("Missing metadata field for kernel %s: %s", artifact_id, e)
                return None
            except Exception as e:
                logger.error("Failed to parse metadata for kernel %s: %s", artifact_id, e)
                return None

            # Get binary reference
            binary_ref = kernel_info.get("binary_ref")
            if not binary_ref:
                logger.error("No binary reference found for kernel %s", artifact_id)
                return None

            # Download kernel binary
            s3_response = await client.get(binary_ref)
            kernel_data = await s3_response.read()

            # Verify checksum if available
            if metadata.checksum:
                actual_checksum = hashlib.sha256(kernel_data).hexdigest()
                if actual_checksum != metadata.checksum:
                    logger.error("Checksum mismatch for kernel %s", artifact_id)
                    return None

            # Deserialize kernel tensor
            try:
                # Create writable copy to avoid PyTorch warning
                writable_data = bytearray(kernel_data)
                kernel_tensor = torch.frombuffer(
                    writable_data, dtype=torch.float32
                ).clone()

                # Move to GPU if available
                if torch.cuda.is_available():
                    kernel_tensor = kernel_tensor.cuda()

                logger.debug("Successfully fetched kernel %s with metadata", artifact_id)
                return kernel_tensor, metadata

            except Exception as e:
                logger.error("Failed to deserialize kernel %s: %s", artifact_id, e)
                return None

    def _add_to_cache_with_metadata(
        self, artifact_id: str, kernel_tensor: torch.Tensor, metadata: KernelMetadata
    ) -> None:
        """
        Add kernel and metadata to cache with LRU eviction.

        Args:
            artifact_id: ID of the kernel artifact
            kernel_tensor: Compiled kernel tensor
            metadata: Kernel metadata
        """
        # Calculate tensor size in MB
        tensor_size_mb = (
            kernel_tensor.numel() * kernel_tensor.element_size() / (1024 * 1024)
        )

        # Check if we need to evict entries
        while (
            len(self._cache) >= self.max_entries
            or self.total_size_mb + tensor_size_mb > self.max_cache_size_mb
        ) and len(self._cache) > 0:
            self._evict_lru_with_metadata()

        # Add to caches
        self._cache[artifact_id] = kernel_tensor
        self.metadata_cache[artifact_id] = metadata

        self._cache_info[artifact_id] = {
            "size_mb": tensor_size_mb,
            "added_at": time.time(),
            "last_accessed": time.time(),
            "parameter_count": metadata.parameter_count,
            "compatibility_version": metadata.compatibility_version,
        }

        self.total_size_mb += tensor_size_mb

        logger.debug(
            f"Cached kernel {artifact_id} ({tensor_size_mb:.2f} MB, "
            f"{metadata.parameter_count} params)"
        )

    def _evict_lru_with_metadata(self) -> None:
        """Evict the least recently used entry from cache and metadata."""
        if not self._cache:
            return

        # Remove oldest entry (first in OrderedDict)
        lru_key = next(iter(self._cache))
        self._cache.pop(lru_key)
        self.metadata_cache.pop(lru_key, None)  # Safe removal
        cache_info = self._cache_info.pop(lru_key)
        self.memory_usage_estimates.pop(lru_key, None)

        self.total_size_mb -= cache_info["size_mb"]
        self._evictions += 1

        logger.debug("Evicted kernel %s (%.2f MB)", lru_key, cache_info['size_mb'])

    def get_kernel_metadata(self, artifact_id: str) -> Optional[KernelMetadata]:
        """
        Get cached metadata for a kernel.

        Args:
            artifact_id: ID of the kernel artifact

        Returns:
            Kernel metadata or None if not cached
        """
        return self.metadata_cache.get(artifact_id)

    def find_compatible_kernels(
        self,
        target_shape: torch.Size,
        device: torch.device,
        max_memory_mb: Optional[float] = None,
    ) -> List[Tuple[str, KernelMetadata]]:
        """
        Find all cached kernels compatible with given requirements.

        Args:
            target_shape: Target input tensor shape
            device: Target execution device
            max_memory_mb: Maximum allowed memory usage

        Returns:
            List of (artifact_id, metadata) tuples for compatible kernels
        """
        compatible = []

        for artifact_id, metadata in self.metadata_cache.items():
            is_valid, _ = self.validator.validate_compatibility(
                metadata, target_shape, device, max_memory_mb
            )

            if is_valid:
                compatible.append((artifact_id, metadata))

        # Sort by performance score (if available) or recency
        compatible.sort(
            key=lambda x: (
                x[1].performance_profile.get("score", 0.0),
                self._cache_info.get(x[0], {}).get("last_accessed", 0),
            ),
            reverse=True,
        )

        return compatible

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        Get enhanced cache statistics including metadata info.

        Returns:
            Dictionary containing enhanced cache statistics
        """
        base_stats = self.get_cache_stats()

        # Calculate compatibility statistics
        compatibility_rate = (
            self.compatibility_checks - self.compatibility_failures
        ) / max(self.compatibility_checks, 1)

        # Memory usage statistics
        total_estimated_memory = sum(self.memory_usage_estimates.values())
        avg_kernel_params = sum(
            info.get("parameter_count", 0) for info in self._cache_info.values()
        ) / max(len(self._cache_info), 1)

        # Metadata cache statistics
        metadata_cache_size = len(self.metadata_cache)

        enhanced_stats = {
            **base_stats,
            "metadata_cache_size": metadata_cache_size,
            "compatibility_checks": self.compatibility_checks,
            "compatibility_failures": self.compatibility_failures,
            "compatibility_rate": compatibility_rate,
            "total_estimated_memory_mb": total_estimated_memory,
            "average_kernel_parameters": avg_kernel_params,
            "kernels_with_metadata": len(
                [k for k in self._cache_info.keys() if k in self.metadata_cache]
            ),
        }

        return enhanced_stats

    def clear_cache(self) -> None:
        """Clear all cached kernels and metadata."""
        super().clear_cache()
        self.metadata_cache.clear()
        self.memory_usage_estimates.clear()

        logger.info("Enhanced kernel cache and metadata cleared")

    def remove_kernel(self, artifact_id: str) -> bool:
        """
        Remove a specific kernel and its metadata from cache.

        Args:
            artifact_id: ID of the kernel to remove

        Returns:
            True if kernel was removed, False if not found
        """
        success = super().remove_kernel(artifact_id)

        if success:
            self.metadata_cache.pop(artifact_id, None)
            self.memory_usage_estimates.pop(artifact_id, None)

        return success
