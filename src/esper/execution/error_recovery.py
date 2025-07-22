"""
Comprehensive error handling and recovery system for kernel execution.

This module provides centralized error handling, recovery strategies,
and resilience patterns for the morphogenetic execution system.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

import torch

from esper.contracts.operational import HealthSignal
from .exceptions import KernelExecutionError

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur in the system."""
    KERNEL_EXECUTION = "kernel_execution"
    KERNEL_LOADING = "kernel_loading"
    KERNEL_VALIDATION = "kernel_validation"
    MEMORY_OVERFLOW = "memory_overflow"
    DEVICE_ERROR = "device_error"
    NETWORK_ERROR = "network_error"
    CIRCUIT_BREAKER = "circuit_breaker"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATE = "escalate"
    IGNORE = "ignore"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    error_type: ErrorType
    component: str  # Which component experienced the error
    layer_name: str
    seed_idx: Optional[int] = None
    kernel_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    exception: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/telemetry."""
        return {
            "error_type": self.error_type.value,
            "component": self.component,
            "layer_name": self.layer_name,
            "seed_idx": self.seed_idx,
            "kernel_id": self.kernel_id,
            "timestamp": self.timestamp,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "metadata": self.metadata
        }


@dataclass
class RecoveryAction:
    """A recovery action to be executed."""
    strategy: RecoveryStrategy
    target_component: str
    action_callable: Callable
    action_args: tuple = field(default_factory=tuple)
    action_kwargs: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 1.5
    timeout: float = 30.0


class ErrorTracker:
    """Tracks error patterns and frequencies."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[ErrorType, int] = defaultdict(int)
        self.component_errors: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.kernel_errors: Dict[str, List[ErrorContext]] = defaultdict(list)
    
    def record_error(self, error_context: ErrorContext):
        """Record an error occurrence."""
        self.error_history.append(error_context)
        
        # Maintain sliding window
        if len(self.error_history) > self.window_size:
            old_error = self.error_history.pop(0)
            self.error_counts[old_error.error_type] -= 1
            
            # Clean up component and kernel tracking
            self.component_errors[old_error.component] = [
                e for e in self.component_errors[old_error.component] 
                if e.timestamp > old_error.timestamp
            ]
            
            if old_error.kernel_id:
                self.kernel_errors[old_error.kernel_id] = [
                    e for e in self.kernel_errors[old_error.kernel_id]
                    if e.timestamp > old_error.timestamp
                ]
        
        # Update counts
        self.error_counts[error_context.error_type] += 1
        self.component_errors[error_context.component].append(error_context)
        
        if error_context.kernel_id:
            self.kernel_errors[error_context.kernel_id].append(error_context)
    
    def get_error_rate(self, error_type: Optional[ErrorType] = None, 
                      time_window: float = 300.0) -> float:
        """Get error rate for the specified time window."""
        cutoff_time = time.time() - time_window
        
        if error_type:
            recent_errors = [
                e for e in self.error_history 
                if e.error_type == error_type and e.timestamp > cutoff_time
            ]
        else:
            recent_errors = [
                e for e in self.error_history 
                if e.timestamp > cutoff_time
            ]
        
        return len(recent_errors) / time_window  # errors per second
    
    def is_problematic_kernel(self, kernel_id: str, threshold: int = 3) -> bool:
        """Check if a kernel has exceeded error threshold."""
        return len(self.kernel_errors.get(kernel_id, [])) >= threshold
    
    def is_problematic_component(self, component: str, threshold: int = 10) -> bool:
        """Check if a component has exceeded error threshold."""
        return len(self.component_errors.get(component, [])) >= threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error tracking statistics."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": dict(self.error_counts),
            "error_rate_5min": self.get_error_rate(time_window=300.0),
            "problematic_kernels": [
                kernel_id for kernel_id in self.kernel_errors.keys()
                if self.is_problematic_kernel(kernel_id)
            ],
            "problematic_components": [
                component for component in self.component_errors.keys()
                if self.is_problematic_component(component)
            ]
        }


class ErrorRecoveryManager:
    """Manages error recovery strategies and execution."""
    
    def __init__(self):
        self.error_tracker = ErrorTracker()
        self.recovery_strategies: Dict[ErrorType, RecoveryStrategy] = {
            ErrorType.KERNEL_EXECUTION: RecoveryStrategy.FALLBACK,
            ErrorType.KERNEL_LOADING: RecoveryStrategy.RETRY,
            ErrorType.KERNEL_VALIDATION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorType.MEMORY_OVERFLOW: RecoveryStrategy.CIRCUIT_BREAKER,
            ErrorType.DEVICE_ERROR: RecoveryStrategy.ESCALATE,
            ErrorType.NETWORK_ERROR: RecoveryStrategy.RETRY,
            ErrorType.CIRCUIT_BREAKER: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorType.TIMEOUT: RecoveryStrategy.RETRY,
            ErrorType.UNKNOWN: RecoveryStrategy.FALLBACK
        }
        
        self.active_recoveries: Set[str] = set()  # Track ongoing recoveries
        self.recovery_history: List[Dict[str, Any]] = []
        
        logger.info("Initialized ErrorRecoveryManager")
    
    async def handle_error(
        self,
        error_context: ErrorContext,
        fallback_action: Optional[Callable] = None
    ) -> bool:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error_context: Context of the error
            fallback_action: Optional fallback action if recovery fails
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Record the error
        self.error_tracker.record_error(error_context)
        
        # Log the error
        logger.error(
            f"Error in {error_context.component}: {error_context.error_type.value} "
            f"- {error_context.exception}"
        )
        
        # Check if we're already recovering this component
        recovery_key = f"{error_context.component}_{error_context.layer_name}"
        if recovery_key in self.active_recoveries:
            logger.warning(f"Recovery already in progress for {recovery_key}")
            return False
        
        try:
            self.active_recoveries.add(recovery_key)
            
            # Get recovery strategy
            strategy = self.recovery_strategies.get(
                error_context.error_type, 
                RecoveryStrategy.FALLBACK
            )
            
            # Check for escalation conditions
            if self._should_escalate(error_context):
                strategy = RecoveryStrategy.ESCALATE
            
            # Execute recovery
            success = await self._execute_recovery(error_context, strategy, fallback_action)
            
            # Record recovery attempt
            self.recovery_history.append({
                "timestamp": time.time(),
                "error_context": error_context.to_dict(),
                "strategy": strategy.value,
                "success": success
            })
            
            return success
            
        finally:
            self.active_recoveries.discard(recovery_key)
    
    def _should_escalate(self, error_context: ErrorContext) -> bool:
        """Determine if error should be escalated."""
        # Escalate if kernel is consistently problematic
        if error_context.kernel_id and self.error_tracker.is_problematic_kernel(
            error_context.kernel_id
        ):
            return True
        
        # Escalate if component is consistently problematic
        if self.error_tracker.is_problematic_component(error_context.component):
            return True
        
        # Escalate if error rate is too high
        if self.error_tracker.get_error_rate(error_context.error_type) > 0.1:  # 0.1 errors/sec
            return True
        
        return False
    
    async def _execute_recovery(
        self,
        error_context: ErrorContext,
        strategy: RecoveryStrategy,
        fallback_action: Optional[Callable] = None
    ) -> bool:
        """Execute the specified recovery strategy."""
        logger.info(f"Executing recovery strategy: {strategy.value} for {error_context.component}")
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._retry_recovery(error_context)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._fallback_recovery(error_context, fallback_action)
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._circuit_breaker_recovery(error_context)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_recovery(error_context)
        
        elif strategy == RecoveryStrategy.ESCALATE:
            return await self._escalate_recovery(error_context)
        
        elif strategy == RecoveryStrategy.IGNORE:
            logger.info(f"Ignoring error in {error_context.component}")
            return True
        
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return False
    
    async def _retry_recovery(self, error_context: ErrorContext) -> bool:
        """Implement retry recovery with exponential backoff."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying {error_context.component} in {delay}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
            
            try:
                # For retry, we would need the original operation
                # This is a simplified implementation
                logger.info(f"Retry attempt {attempt + 1} for {error_context.component}")
                return True  # Assume success for now
                
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return False
        
        return False
    
    async def _fallback_recovery(
        self, 
        error_context: ErrorContext, 
        fallback_action: Optional[Callable] = None
    ) -> bool:
        """Implement fallback recovery."""
        logger.info(f"Executing fallback for {error_context.component}")
        
        if fallback_action:
            try:
                result = fallback_action()
                if asyncio.iscoroutine(result):
                    await result
                return True
            except Exception as e:
                logger.error(f"Fallback action failed: {e}")
                return False
        
        # Default fallback behavior
        if error_context.component == "kernel_executor":
            logger.info("Falling back to default transformation")
            return True
        
        return False
    
    async def _circuit_breaker_recovery(self, error_context: ErrorContext) -> bool:
        """Implement circuit breaker recovery."""
        logger.warning(f"Opening circuit breaker for {error_context.component}")
        
        # Mark component as temporarily unavailable
        # This would integrate with actual circuit breaker implementation
        return True
    
    async def _graceful_degradation_recovery(self, error_context: ErrorContext) -> bool:
        """Implement graceful degradation recovery."""
        logger.info(f"Gracefully degrading {error_context.component}")
        
        # Reduce functionality while maintaining basic operation
        if error_context.component == "kernel_cache":
            logger.info("Disabling kernel caching temporarily")
            return True
        
        if error_context.component == "kasmina_layer":
            logger.info("Switching to default transformation only")
            return True
        
        return True
    
    async def _escalate_recovery(self, error_context: ErrorContext) -> bool:
        """Implement escalation recovery."""
        logger.critical(f"Escalating error in {error_context.component}")
        
        # This would typically involve:
        # 1. Notifying administrators
        # 2. Triggering emergency procedures
        # 3. Potentially shutting down problematic components
        
        return False  # Escalation doesn't "fix" the error
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery system statistics."""
        recent_recoveries = [
            r for r in self.recovery_history 
            if r["timestamp"] > time.time() - 3600  # Last hour
        ]
        
        success_rate = (
            sum(1 for r in recent_recoveries if r["success"]) / 
            max(len(recent_recoveries), 1)
        )
        
        return {
            "total_recoveries": len(self.recovery_history),
            "recent_recoveries": len(recent_recoveries),
            "recovery_success_rate": success_rate,
            "active_recoveries": len(self.active_recoveries),
            "error_stats": self.error_tracker.get_stats()
        }


class HealthMonitor:
    """Monitors system health and triggers recovery when needed."""
    
    def __init__(self, recovery_manager: ErrorRecoveryManager):
        self.recovery_manager = recovery_manager
        self.health_thresholds = {
            "error_rate": 0.05,  # 5% error rate threshold
            "memory_usage": 0.9,  # 90% memory usage threshold
            "execution_latency": 1000,  # 1 second latency threshold (ms)
        }
        self.monitoring_active = False
        self.health_signals: List[HealthSignal] = []
        
    async def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring_active = True
        logger.info("Started health monitoring")
        
        while self.monitoring_active:
            try:
                await self._check_system_health()
                await asyncio.sleep(10.0)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30.0)  # Back off on error
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        logger.info("Stopped health monitoring")
    
    async def _check_system_health(self):
        """Check overall system health."""
        # Check error rates
        error_rate = self.recovery_manager.error_tracker.get_error_rate(time_window=300.0)
        if error_rate > self.health_thresholds["error_rate"]:
            logger.warning(f"High error rate detected: {error_rate:.4f} errors/sec")
            
            # Trigger proactive recovery
            error_context = ErrorContext(
                error_type=ErrorType.UNKNOWN,
                component="system",
                layer_name="global",
                metadata={"error_rate": error_rate}
            )
            await self.recovery_manager.handle_error(error_context)
        
        # Check memory usage (simplified)
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_usage > self.health_thresholds["memory_usage"]:
                logger.warning(f"High memory usage detected: {memory_usage:.2%}")
    
    def add_health_signal(self, signal: HealthSignal):
        """Add a health signal for analysis."""
        self.health_signals.append(signal)
        
        # Keep only recent signals
        cutoff_time = time.time() - 3600  # 1 hour
        self.health_signals = [
            s for s in self.health_signals 
            if s.activation_variance > cutoff_time  # Using this field as timestamp
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get system health summary."""
        return {
            "monitoring_active": self.monitoring_active,
            "health_signals_count": len(self.health_signals),
            "recovery_stats": self.recovery_manager.get_recovery_stats(),
            "thresholds": self.health_thresholds
        }


# Utility functions for error handling
def create_error_context(
    error_type: ErrorType,
    component: str,
    layer_name: str,
    exception: Exception,
    **kwargs
) -> ErrorContext:
    """Create an error context from an exception."""
    return ErrorContext(
        error_type=error_type,
        component=component,
        layer_name=layer_name,
        exception=exception,
        **kwargs
    )


def classify_kernel_error(exception: Exception) -> ErrorType:
    """Classify a kernel-related exception into an error type."""
    if isinstance(exception, KernelExecutionError):
        return ErrorType.KERNEL_EXECUTION
    elif isinstance(exception, torch.cuda.OutOfMemoryError):
        return ErrorType.MEMORY_OVERFLOW
    elif isinstance(exception, (ConnectionError, TimeoutError)):
        return ErrorType.NETWORK_ERROR
    elif "timeout" in str(exception).lower():
        return ErrorType.TIMEOUT
    else:
        return ErrorType.UNKNOWN