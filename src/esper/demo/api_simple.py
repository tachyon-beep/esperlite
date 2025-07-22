"""Simple demo API service for the Esper tech demo portal."""

import logging
import time
from datetime import datetime

import torch
from aiohttp import web

logger = logging.getLogger(__name__)


class SimpleDemoAPI:
    """Simple API service for the demo portal."""

    def __init__(self):
        self.start_time = time.time()
        self.training_epoch = 0
        self.training_loss = 2.3
        self.training_accuracy = 0.1

    async def get_status(self, request: web.Request) -> web.Response:
        """Get system status."""
        status = {
            "uptime": int(time.time() - self.start_time),
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "urza": {"status": "healthy", "response_time": 12},
                "tamiyo": {"status": "healthy", "response_time": 15},
                "tolaria": {"status": "healthy", "response_time": 8},
            },
            "infrastructure": {
                "redis": {"status": "healthy"},
                "postgres": {"status": "healthy"},
                "minio": {"status": "healthy"},
            },
            "gpu": {},
        }

        # Check GPU status
        if torch.cuda.is_available():
            status["gpu"] = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,
                "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,
                "utilization": 75.0,  # Mock utilization
            }
        else:
            status["gpu"] = {"available": False}

        return web.json_response(status)

    async def get_training(self, request: web.Request) -> web.Response:
        """Get training status."""
        # Simulate training progress
        self.training_epoch = (self.training_epoch + 1) % 50
        self.training_loss = max(0.1, self.training_loss * 0.98)
        self.training_accuracy = min(0.95, self.training_accuracy + 0.01)

        return web.json_response(
            {
                "active": True,
                "current_epoch": self.training_epoch,
                "total_epochs": 50,
                "current_loss": self.training_loss,
                "val_loss": self.training_loss * 1.1,
                "current_accuracy": self.training_accuracy,
                "adaptations_count": self.training_epoch // 5,
                "seeds_active": self.training_epoch // 10,
                "learning_rate": 0.001 * (0.95 ** (self.training_epoch // 10)),
            }
        )

    async def get_logs(self, request: web.Request) -> web.Response:
        """Get recent logs."""
        logs = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "service": "tolaria",
                "message": f"Epoch {self.training_epoch} completed - Loss: {self.training_loss:.4f}",
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "service": "tamiyo",
                "message": "Model analysis complete - No adaptation needed",
            },
        ]
        return web.json_response({"logs": logs})

    async def get_kernels(self, request: web.Request) -> web.Response:
        """Get kernel statistics."""
        return web.json_response(
            {
                "total_kernels": 42 + self.training_epoch,
                "active_kernels": 16 + (self.training_epoch // 5),
                "total_blueprints": 8,
                "cache_hit_rate": 0.94,
                "compilation_queue": 0,
            }
        )

    async def get_adaptations(self, request: web.Request) -> web.Response:
        """Get adaptation history."""
        adaptations = []
        for i in range(min(5, self.training_epoch // 5)):
            adaptations.append(
                {
                    "id": f"adapt-{i}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "layer": "layer4.1.conv2",
                    "type": "kernel_swap",
                    "confidence": 0.85 + i * 0.02,
                    "improvement": 0.02 + i * 0.005,
                }
            )
        return web.json_response({"adaptations": adaptations})


@web.middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


async def init_app() -> web.Application:
    """Initialize the web application."""
    app = web.Application(middlewares=[cors_middleware])
    api = SimpleDemoAPI()

    # Add routes
    app.router.add_get("/api/status", api.get_status)
    app.router.add_get("/api/training", api.get_training)
    app.router.add_get("/api/logs", api.get_logs)
    app.router.add_get("/api/kernels", api.get_kernels)
    app.router.add_get("/api/adaptations", api.get_adaptations)

    # Add health check
    app.router.add_get("/health", lambda r: web.Response(text="OK"))

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(init_app(), host="0.0.0.0", port=8888)
