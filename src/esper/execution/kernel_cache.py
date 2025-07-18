"""
GPU-resident LRU cache for pre-compiled kernel artifacts.

This module provides high-performance caching of kernel artifacts to achieve
microsecond-latency execution.
"""

import asyncio
import logging
import hashlib
import time
import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Any

import torch

from esper.services.contracts import SimpleCompiledKernelContract

logger = logging.getLogger(__name__)


class KernelCache:
    """
    GPU-resident LRU cache for pre-compiled kernel artifacts.

    This cache maintains compiled kernel tensors in GPU memory for
    microsecond-latency execution.
    """

    def __init__(self, max_cache_size_mb: int = 512, max_entries: int = 128):
        """
        Initialize the kernel cache.

        Args:
            max_cache_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.max_entries = max_entries
        self.total_size_mb = 0.0

        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_info: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"Initialized KernelCache with {max_cache_size_mb}MB max size, {max_entries} max entries"
        )

    async def load_kernel(self, artifact_id: str) -> Optional[torch.Tensor]:
        """
        Load a kernel from cache or fetch from Urza if not cached.

        Args:
            artifact_id: ID of the kernel artifact to load

        Returns:
            Compiled kernel tensor, or None if not found
        """
        async with self._lock:
            # Check cache first
            if artifact_id in self._cache:
                # Move to end (most recently used)
                kernel_tensor = self._cache.pop(artifact_id)
                self._cache[artifact_id] = kernel_tensor
                self._cache_info[artifact_id]["last_accessed"] = time.time()
                self._hits += 1

                logger.debug(f"Cache hit for kernel {artifact_id}")
                return kernel_tensor

            # Cache miss - fetch from Urza
            self._misses += 1
            logger.debug(f"Cache miss for kernel {artifact_id}, fetching from Urza")

            kernel_tensor = self._fetch_from_urza(artifact_id)
            if kernel_tensor is not None:
                self._add_to_cache(artifact_id, kernel_tensor)

            return kernel_tensor

    def _fetch_from_urza(self, artifact_id: str) -> Optional[torch.Tensor]:
        """
        Fetch kernel binary from Urza and load to GPU.

        Args:
            artifact_id: ID of the kernel artifact

        Returns:
            Compiled kernel tensor, or None if not found
        """
        try:
            import requests
            import json

            # Get Urza API URL from environment or use default
            urza_url = os.getenv("URZA_API_URL", "http://localhost:8000")

            # Fetch kernel metadata from Urza API
            response = requests.get(
                f"{urza_url}/api/v1/kernels/{artifact_id}", timeout=5
            )

            if response.status_code == 404:
                logger.warning(f"Kernel {artifact_id} not found in Urza")
                return None

            response.raise_for_status()
            kernel_metadata = response.json()

            # Extract S3 binary reference
            binary_ref = kernel_metadata.get("kernel_binary_ref")
            if not binary_ref:
                logger.error(f"No binary reference found for kernel {artifact_id}")
                return None

            # Download actual kernel binary from S3
            s3_response = requests.get(binary_ref, timeout=10)
            s3_response.raise_for_status()

            # Deserialize the kernel tensor with writable copy
            kernel_data = s3_response.content
            # Create writable copy to avoid PyTorch warning about non-writable buffers
            writable_data = bytearray(kernel_data)
            kernel_tensor = torch.frombuffer(writable_data, dtype=torch.float32).clone()

            # Move to GPU if available
            if torch.cuda.is_available():
                kernel_tensor = kernel_tensor.cuda()

            logger.debug(f"Successfully fetched kernel {artifact_id} from Urza")
            return kernel_tensor

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching kernel {artifact_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch kernel {artifact_id}: {e}")
            return None

    def _add_to_cache(self, artifact_id: str, kernel_tensor: torch.Tensor) -> None:
        """
        Add kernel to cache with LRU eviction.

        Args:
            artifact_id: ID of the kernel artifact
            kernel_tensor: Compiled kernel tensor
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
            self._evict_lru()

        # Add to cache
        self._cache[artifact_id] = kernel_tensor
        self._cache_info[artifact_id] = {
            "size_mb": tensor_size_mb,
            "added_at": time.time(),
            "last_accessed": time.time(),
        }
        self.total_size_mb += tensor_size_mb

        logger.debug(f"Cached kernel {artifact_id} ({tensor_size_mb:.2f} MB)")

    def _evict_lru(self) -> None:
        """Evict the least recently used entry from cache."""
        if not self._cache:
            return

        # Remove oldest entry (first in OrderedDict)
        lru_key = next(iter(self._cache))
        self._cache.pop(lru_key)
        cache_info = self._cache_info.pop(lru_key)

        self.total_size_mb -= cache_info["size_mb"]
        self._evictions += 1

        logger.debug(f"Evicted kernel {lru_key} ({cache_info['size_mb']:.2f} MB)")

    async def preload_kernels(self, artifact_ids: list[str]) -> None:
        """
        Preload a list of kernels into the cache.

        Args:
            artifact_ids: List of kernel artifact IDs to preload.
        """
        for artifact_id in artifact_ids:
            await self.load_kernel(artifact_id)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary containing cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "total_size_mb": self.total_size_mb,
            "max_size_mb": self.max_cache_size_mb,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "cache_keys": list(self._cache.keys()),
        }

    def clear_cache(self) -> None:
        """Clear all cached kernels."""
        self._cache.clear()
        self._cache_info.clear()
        self.total_size_mb = 0.0
        self._evictions += len(self._cache)

        logger.info("Kernel cache cleared")

    def remove_kernel(self, artifact_id: str) -> bool:
        """
        Remove a specific kernel from cache.

        Args:
            artifact_id: ID of the kernel to remove

        Returns:
            True if kernel was removed, False if not found
        """
        if artifact_id in self._cache:
            self._cache.pop(artifact_id)
            cache_info = self._cache_info.pop(artifact_id)
            self.total_size_mb -= cache_info["size_mb"]

            logger.debug(f"Removed kernel {artifact_id} from cache")
            return True

        return False

    @property
    def is_gpu_resident(self) -> bool:
        """Check if cache is using GPU memory."""
        if not self._cache:
            return False

        # Check first cached tensor
        first_tensor = next(iter(self._cache.values()))
        return first_tensor.is_cuda if hasattr(first_tensor, "is_cuda") else False
