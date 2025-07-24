"""
Stream Manager for Multi-GPU Async Execution.

This module manages CUDA streams across multiple devices for efficient
async execution and proper synchronization.
"""

import asyncio
import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Manages CUDA streams for async execution across devices.
    
    This class provides efficient stream allocation and synchronization
    for multi-GPU async operations.
    """

    def __init__(self, num_streams_per_device: int = 4):
        """
        Initialize the stream manager.
        
        Args:
            num_streams_per_device: Number of streams to create per device
        """
        self.streams: Dict[int, List[torch.cuda.Stream]] = {}
        self.num_streams = num_streams_per_device
        self.stream_index: Dict[int, int] = {}
        self.default_streams: Dict[int, torch.cuda.Stream] = {}
        self._initialized_devices: set = set()
        
        logger.info(
            f"Initialized StreamManager with {num_streams_per_device} streams per device"
        )

    def get_stream(self, device: torch.device) -> torch.cuda.Stream:
        """
        Get next available stream for device (round-robin).
        
        Args:
            device: Target device
            
        Returns:
            CUDA stream for the device
        """
        if device.type != "cuda":
            raise ValueError(f"StreamManager only supports CUDA devices, got {device}")
        
        device_id = device.index or 0
        
        # Initialize streams for device if needed
        if device_id not in self.streams:
            self._initialize_device_streams(device_id)
        
        # Round-robin stream selection
        idx = self.stream_index[device_id]
        self.stream_index[device_id] = (idx + 1) % self.num_streams
        
        return self.streams[device_id][idx]

    def _initialize_device_streams(self, device_id: int):
        """Initialize streams for a specific device."""
        if device_id in self._initialized_devices:
            return
        
        # Create streams for the device
        with torch.cuda.device(device_id):
            self.streams[device_id] = [
                torch.cuda.Stream(device=device_id)
                for _ in range(self.num_streams)
            ]
            self.stream_index[device_id] = 0
            self.default_streams[device_id] = torch.cuda.default_stream(device_id)
        
        self._initialized_devices.add(device_id)
        logger.debug(f"Initialized {self.num_streams} streams for device {device_id}")

    def get_default_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Get the default stream for a device."""
        if device.type != "cuda":
            raise ValueError(f"StreamManager only supports CUDA devices, got {device}")
        
        device_id = device.index or 0
        
        if device_id not in self.default_streams:
            self._initialize_device_streams(device_id)
        
        return self.default_streams[device_id]

    async def synchronize_stream(self, stream: torch.cuda.Stream):
        """Asynchronously wait for stream completion."""
        event = torch.cuda.Event()
        event.record(stream)
        
        while not event.query():
            await asyncio.sleep(0)  # Yield to event loop

    async def synchronize_device(self, device: torch.device):
        """Synchronize all streams on a device."""
        if device.type != "cuda":
            return
        
        device_id = device.index or 0
        
        if device_id in self.streams:
            sync_tasks = [
                self.synchronize_stream(stream)
                for stream in self.streams[device_id]
            ]
            await asyncio.gather(*sync_tasks)

    async def synchronize_all(self):
        """Synchronize all streams across all devices."""
        sync_tasks = []
        
        for device_id, device_streams in self.streams.items():
            for stream in device_streams:
                sync_tasks.append(self.synchronize_stream(stream))
        
        if sync_tasks:
            await asyncio.gather(*sync_tasks)
            logger.debug(f"Synchronized {len(sync_tasks)} streams across all devices")

    def get_stats(self) -> Dict[str, any]:
        """Get stream manager statistics."""
        stats = {
            "num_devices": len(self._initialized_devices),
            "streams_per_device": self.num_streams,
            "total_streams": sum(len(streams) for streams in self.streams.values()),
            "devices": list(self._initialized_devices),
        }
        
        # Add per-device stats
        for device_id in self._initialized_devices:
            stats[f"device_{device_id}_current_stream"] = self.stream_index.get(device_id, 0)
        
        return stats

    def cleanup(self):
        """Clean up resources."""
        # Synchronize all streams before cleanup
        if self.streams:
            for device_streams in self.streams.values():
                for stream in device_streams:
                    stream.synchronize()
        
        self.streams.clear()
        self.stream_index.clear()
        self.default_streams.clear()
        self._initialized_devices.clear()
        
        logger.info("StreamManager cleaned up")


class StreamContext:
    """
    Context manager for stream-based execution.
    
    Ensures proper stream selection and synchronization for async operations.
    """

    def __init__(
        self,
        stream_manager: StreamManager,
        device: torch.device,
        synchronize_on_exit: bool = True
    ):
        """
        Initialize stream context.
        
        Args:
            stream_manager: StreamManager instance
            device: Target device
            synchronize_on_exit: Whether to synchronize on context exit
        """
        self.stream_manager = stream_manager
        self.device = device
        self.synchronize_on_exit = synchronize_on_exit
        self.stream = None
        self.original_stream = None

    def __enter__(self) -> torch.cuda.Stream:
        """Enter stream context."""
        if self.device.type == "cuda":
            self.original_stream = torch.cuda.current_stream(self.device)
            self.stream = self.stream_manager.get_stream(self.device)
            torch.cuda.set_stream(self.stream)
            return self.stream
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit stream context."""
        if self.device.type == "cuda" and self.stream:
            if self.synchronize_on_exit:
                self.stream.synchronize()
            
            # Restore original stream
            if self.original_stream:
                torch.cuda.set_stream(self.original_stream)

    async def __aenter__(self) -> torch.cuda.Stream:
        """Async enter stream context."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit stream context."""
        if self.device.type == "cuda" and self.stream and self.synchronize_on_exit:
            await self.stream_manager.synchronize_stream(self.stream)
        
        # Restore original stream
        if self.original_stream:
            torch.cuda.set_stream(self.original_stream)


# Global stream manager instance
_global_stream_manager: Optional[StreamManager] = None


def get_global_stream_manager(num_streams: int = 4) -> StreamManager:
    """
    Get or create global stream manager.
    
    Args:
        num_streams: Number of streams per device
        
    Returns:
        Global StreamManager instance
    """
    global _global_stream_manager
    
    if _global_stream_manager is None:
        _global_stream_manager = StreamManager(num_streams)
    
    return _global_stream_manager


def cleanup_global_stream_manager():
    """Clean up global stream manager."""
    global _global_stream_manager
    
    if _global_stream_manager is not None:
        _global_stream_manager.cleanup()
        _global_stream_manager = None