"""Simple Tamiyo demo service for the tech demo."""

import logging

from aiohttp import web

logger = logging.getLogger(__name__)


async def health_check(_request):
    """Health check endpoint."""
    return web.json_response({"status": "healthy", "service": "tamiyo"})


async def analyze(request):
    """Analyze endpoint for demo."""
    _ = await request.json()
    # Simple demo response
    return web.json_response(
        {
            "decision": "no_adaptation",
            "confidence": 0.8,
            "reason": "Model is performing within expected parameters",
        }
    )


def create_app():
    """Create the web application."""
    app = web.Application()
    app.router.add_get("/health", health_check)
    app.router.add_post("/analyze", analyze)
    return app


if __name__ == "__main__":
    import sys

    port = 8001
    if len(sys.argv) > 2 and sys.argv[1] == "--port":
        port = int(sys.argv[2])

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Tamiyo demo service on port %d", port)

    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
