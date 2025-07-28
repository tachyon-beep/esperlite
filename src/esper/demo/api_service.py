"""Demo API service for the Esper tech demo portal."""

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import aiohttp
import redis
import redis.asyncio as aioredis
import torch
from aiohttp import web

logger = logging.getLogger(__name__)


class DemoAPIService:
    """API service for the demo portal."""

    def __init__(self):
        self.logs_buffer = deque(maxlen=1000)
        self.metrics_buffer = deque(maxlen=100)
        self.redis_client: Optional[redis.Redis] = None
        self.start_time = time.time()
        self.training_metrics = {}
        self.adaptation_history = []

    async def setup(self):
        """Initialize connections."""
        try:
            # Connect to Redis for pub/sub
            self.redis_client = await aioredis.from_url(
                "redis://redis:6379", decode_responses=True
            )
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)

    async def cleanup(self):
        """Cleanup connections."""
        if self.redis_client:
            await self.redis_client.close()

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            "uptime": int(time.time() - self.start_time),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": {},
            "infrastructure": {},
            "gpu": {},
        }

        # Check service health
        services = [
            ("urza", "http://urza:8000/health"),
            ("tamiyo", "http://tamiyo:8001/health"),
            ("tolaria", "http://tolaria:8080/health"),
        ]

        async with aiohttp.ClientSession() as session:
            for name, url in services:
                try:
                    async with session.get(url, timeout=2) as resp:
                        status["services"][name] = {
                            "status": "healthy" if resp.status == 200 else "unhealthy",
                            "response_time": 0,
                        }
                except Exception:
                    status["services"][name] = {"status": "error", "response_time": -1}

        # Check GPU status
        if torch.cuda.is_available():
            status["gpu"] = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,  # GB
                "utilization": self._get_gpu_utilization(),
            }
        else:
            status["gpu"] = {"available": False}

        # Infrastructure status
        status["infrastructure"] = {
            "redis": {"status": "healthy" if self.redis_client else "disconnected"},
            "postgres": {"status": "healthy"},  # Mock for now
            "minio": {"status": "healthy"},  # Mock for now
        }

        return status

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        # Try to fetch from Tolaria
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://tolaria:8080/status", timeout=2) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass

        # Return cached/mock data
        return {
            "active": True,
            "current_epoch": len(self.metrics_buffer),
            "total_epochs": 50,
            "current_loss": self.training_metrics.get("loss", 0.0),
            "current_accuracy": self.training_metrics.get("accuracy", 0.0),
            "adaptations_count": len(self.adaptation_history),
            "seeds_active": 0,
            "learning_rate": 0.001,
        }

    async def get_logs(
        self, service: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent logs."""
        logs = list(self.logs_buffer)

        if service:
            logs = [log for log in logs if log.get("service") == service]

        return logs[-limit:]

    async def stream_logs(self, request: web.Request) -> web.Response:
        """Stream logs via Server-Sent Events."""
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        await response.prepare(request)

        # Send historical logs first
        for log in list(self.logs_buffer)[-20:]:
            await response.write((f"data: {json.dumps(log)}\n\n").encode())

        # Subscribe to Redis for new logs
        if self.redis_client:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("logs:*")

            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        log_entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "channel": message["channel"],
                            "data": message["data"],
                        }
                        self.logs_buffer.append(log_entry)
                        await response.write(
                            (f"data: {json.dumps(log_entry)}\n\n").encode()
                        )
            finally:
                await pubsub.unsubscribe("logs:*")

        return response

    async def stream_metrics(self, request: web.Request) -> web.Response:
        """Stream metrics via Server-Sent Events."""
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        await response.prepare(request)

        while True:
            # Get current metrics
            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "training": await self.get_training_status(),
                "system": await self.get_system_status(),
            }

            await response.write((f"data: {json.dumps(metrics)}\n\n").encode())
            await asyncio.sleep(1)  # Update every second

        return response

    async def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history."""
        return self.adaptation_history[-50:]  # Last 50 adaptations

    async def get_kernel_stats(self) -> Dict[str, Any]:
        """Get kernel statistics."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://urza:8000/kernels/stats", timeout=2
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass

        # Return mock data
        return {
            "total_kernels": 42,
            "active_kernels": 16,
            "total_blueprints": 8,
            "cache_hit_rate": 0.94,
            "compilation_queue": 0,
        }


# API Routes
async def handle_status(request: web.Request) -> web.Response:
    """Handle system status request."""
    service = request.app["demo_service"]
    status = await service.get_system_status()
    return web.json_response(status)


async def handle_training(request: web.Request) -> web.Response:
    """Handle training status request."""
    service = request.app["demo_service"]
    status = await service.get_training_status()
    return web.json_response(status)


async def handle_logs(request: web.Request) -> web.Response:
    """Handle logs request."""
    service = request.app["demo_service"]
    service_filter = request.query.get("service")
    limit = int(request.query.get("limit", 100))
    logs = await service.get_logs(service_filter, limit)
    return web.json_response({"logs": logs})


async def handle_adaptations(request: web.Request) -> web.Response:
    """Handle adaptations history request."""
    service = request.app["demo_service"]
    history = await service.get_adaptation_history()
    return web.json_response({"adaptations": history})


async def handle_kernels(request: web.Request) -> web.Response:
    """Handle kernel stats request."""
    service = request.app["demo_service"]
    stats = await service.get_kernel_stats()
    return web.json_response(stats)


async def init_app() -> web.Application:
    """Initialize the web application."""
    app = web.Application()

    # Create demo service
    demo_service = DemoAPIService()
    await demo_service.setup()
    app["demo_service"] = demo_service

    # Add routes
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/training", handle_training)
    app.router.add_get("/api/logs", handle_logs)
    app.router.add_get("/api/logs/stream", demo_service.stream_logs)
    app.router.add_get("/api/metrics/stream", demo_service.stream_metrics)
    app.router.add_get("/api/adaptations", handle_adaptations)
    app.router.add_get("/api/kernels", handle_kernels)

    # Cleanup on shutdown
    async def cleanup(app):
        await app["demo_service"].cleanup()

    app.on_cleanup.append(cleanup)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(init_app(), host="0.0.0.0", port=8888)
