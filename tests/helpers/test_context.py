"""
Test context helpers for controlling mocking behavior.

This module provides context managers and decorators to selectively
disable auto-mocking for specific tests.
"""

import contextlib
import functools
from unittest.mock import patch


@contextlib.contextmanager
def real_oona_client():
    """
    Context manager to use real OonaClient instead of auto-mock.
    
    Usage:
        with real_oona_client():
            # Code here will use real OonaClient if available
    """
    # Temporarily disable the auto-mock by patching with the real class
    from esper.services.oona_client import OonaClient
    
    with patch('esper.execution.kasmina_layer.OonaClient', OonaClient):
        yield


@contextlib.contextmanager
def real_http_client():
    """
    Context manager to use real HTTP client instead of auto-mock.
    
    Usage:
        with real_http_client():
            # Code here will use real AsyncHttpClient
    """
    from esper.utils.http_client import AsyncHttpClient
    
    with patch('esper.utils.http_client.AsyncHttpClient', AsyncHttpClient):
        yield


def no_auto_mocks(test_func):
    """
    Decorator to disable all auto-mocks for a test function.
    
    Usage:
        @no_auto_mocks
        def test_something():
            # Test will run without auto-mocks
    """
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        with real_oona_client(), real_http_client():
            return test_func(*args, **kwargs)
    
    return wrapper


class RealComponentContext:
    """
    Context manager for tests that need real components.
    
    Provides fine-grained control over which components use real
    implementations vs mocks.
    """
    
    def __init__(
        self,
        use_real_oona: bool = False,
        use_real_http: bool = False,
        use_real_redis: bool = False
    ):
        self.use_real_oona = use_real_oona
        self.use_real_http = use_real_http
        self.use_real_redis = use_real_redis
        self._patches = []
    
    def __enter__(self):
        if self.use_real_oona:
            from esper.services.oona_client import OonaClient
            patch_oona = patch('esper.execution.kasmina_layer.OonaClient', OonaClient)
            self._patches.append(patch_oona)
            patch_oona.__enter__()
        
        if self.use_real_http:
            from esper.utils.http_client import AsyncHttpClient
            patch_http = patch('esper.utils.http_client.AsyncHttpClient', AsyncHttpClient)
            self._patches.append(patch_http)
            patch_http.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in reversed(self._patches):
            p.__exit__(exc_type, exc_val, exc_tb)
        self._patches.clear()


def with_real_components(**component_flags):
    """
    Decorator to selectively use real components in tests.
    
    Usage:
        @with_real_components(use_real_oona=True)
        def test_something():
            # Test will use real OonaClient
    """
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            with RealComponentContext(**component_flags):
                return test_func(*args, **kwargs)
        
        return wrapper
    
    return decorator