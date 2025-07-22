"""
Tezzeret client for blueprint compilation requests.
"""

import logging
from typing import Dict
from typing import Optional

from esper.contracts.assets import Blueprint
from esper.utils.http_client import AsyncHttpClient

logger = logging.getLogger(__name__)


class TezzeretClient:
    """Client for interacting with Tezzeret compilation service."""

    def __init__(self, urza_url: str, timeout: float = 30.0):
        self.urza_url = urza_url.rstrip('/')
        self.http_client = AsyncHttpClient(timeout=timeout)

    async def submit_blueprint(self, blueprint: Blueprint) -> Optional[str]:
        """
        Submit blueprint for compilation.
        
        Args:
            blueprint: Blueprint to compile
            
        Returns:
            Blueprint ID if successful
        """
        try:
            # Submit to Urza API
            response = await self.http_client.post(
                f"{self.urza_url}/api/v1/blueprints",
                json=blueprint.dict()
            )

            if response.status_code == 201:
                data = response.json()
                return data.get("id")
            else:
                logger.error(f"Blueprint submission failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error submitting blueprint: {e}")
            return None

    async def get_blueprint_status(self, blueprint_id: str) -> Optional[Dict]:
        """Get blueprint compilation status."""
        try:
            response = await self.http_client.get(
                f"{self.urza_url}/api/v1/blueprints/{blueprint_id}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting blueprint status: {e}")
            return None
