"""Command handlers for the morphogenetic message bus system.

This module provides handlers for processing control commands with
validation, authorization, and acknowledgment.
"""

import asyncio
import logging
import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..lifecycle import ExtendedLifecycle
from ..lifecycle import LifecycleManager
from .clients import MessageBusClient
from .schemas import BaseMessage
from .schemas import BatchCommand
from .schemas import BlueprintUpdateCommand
from .schemas import CommandResult
from .schemas import EmergencyStopCommand
from .schemas import LifecycleTransitionCommand
from .schemas import create_topic_name

logger = logging.getLogger(__name__)


class CommandPriority(Enum):
    """Command execution priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class CommandStatus(Enum):
    """Command execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class CommandContext:
    """Context for command execution."""
    command: BaseMessage
    layer_id: str
    seed_id: Optional[int] = None
    start_time: float = 0.0
    timeout_ms: int = 5000
    priority: CommandPriority = CommandPriority.NORMAL
    status: CommandStatus = CommandStatus.PENDING
    result: Optional[CommandResult] = None

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @property
    def is_timeout(self) -> bool:
        """Check if command has timed out."""
        return self.elapsed_ms > self.timeout_ms


class CommandProcessor(ABC):
    """Abstract base class for command processors."""

    @abstractmethod
    async def can_handle(self, command: BaseMessage) -> bool:
        """Check if this processor can handle the command."""
        pass

    @abstractmethod
    async def validate(self, context: CommandContext) -> Optional[str]:
        """Validate the command.
        
        Returns:
            None if valid, error message if invalid
        """
        pass

    @abstractmethod
    async def execute(self, context: CommandContext) -> CommandResult:
        """Execute the command."""
        pass

    async def rollback(self, context: CommandContext) -> None:
        """Rollback changes if execution failed."""
        pass


class LifecycleTransitionProcessor(CommandProcessor):
    """Processor for lifecycle transition commands."""

    def __init__(self, layer_registry: Dict[str, Any]):
        self.layer_registry = layer_registry
        # Note: LifecycleManager doesn't require num_seeds parameter
        # It manages transition validation, not seed storage
        self.lifecycle_manager = LifecycleManager()

    async def can_handle(self, command: BaseMessage) -> bool:
        return isinstance(command, LifecycleTransitionCommand)

    async def validate(self, context: CommandContext) -> Optional[str]:
        """Validate lifecycle transition command."""
        command: LifecycleTransitionCommand = context.command

        # Check layer exists
        if context.layer_id not in self.layer_registry:
            return f"Layer {context.layer_id} not found"

        layer = self.layer_registry[context.layer_id]

        # Check seed exists
        if context.seed_id is not None:
            if context.seed_id >= layer.num_seeds:
                return f"Seed {context.seed_id} not found in layer"

        # Validate target state
        try:
            target_state = ExtendedLifecycle[command.target_state.upper()]
        except KeyError:
            return f"Invalid target state: {command.target_state}"

        # Get current state
        if context.seed_id is not None:
            current_state = layer.get_seed_state(context.seed_id)

            # Check if transition is valid
            if not self.lifecycle_manager.can_transition(current_state, target_state):
                # Get valid transitions by checking all possible states
                valid_transitions = []
                for state in ExtendedLifecycle:
                    if self.lifecycle_manager.can_transition(current_state, state):
                        valid_transitions.append(state)
                valid_names = [s.name for s in valid_transitions]
                return f"Cannot transition from {current_state.name} to {target_state.name}. Valid transitions: {valid_names}"

        return None

    async def execute(self, context: CommandContext) -> CommandResult:
        """Execute lifecycle transition."""
        command: LifecycleTransitionCommand = context.command
        layer = self.layer_registry[context.layer_id]

        try:
            target_state = ExtendedLifecycle[command.target_state.upper()]

            if context.seed_id is not None:
                # Single seed transition
                success = await self._transition_seed(
                    layer, context.seed_id, target_state,
                    command.parameters, command.force
                )

                if success:
                    return CommandResult(
                        success=True,
                        details={
                            "seed_id": context.seed_id,
                            "new_state": target_state.name,
                            "transition_time_ms": context.elapsed_ms
                        },
                        execution_time_ms=context.elapsed_ms
                    )
                else:
                    return CommandResult(
                        success=False,
                        error="Transition failed - check layer logs",
                        execution_time_ms=context.elapsed_ms
                    )
            else:
                # Batch transition for all seeds
                results = await self._transition_all_seeds(
                    layer, target_state, command.parameters, command.force
                )

                success_count = sum(1 for r in results if r)

                return CommandResult(
                    success=success_count > 0,
                    details={
                        "total_seeds": len(results),
                        "successful_transitions": success_count,
                        "failed_transitions": len(results) - success_count,
                        "new_state": target_state.name
                    },
                    execution_time_ms=context.elapsed_ms
                )

        except Exception as e:
            logger.error("Lifecycle transition error: %s", e)
            return CommandResult(
                success=False,
                error=str(e),
                execution_time_ms=context.elapsed_ms
            )

    async def _transition_seed(self, layer: Any, seed_id: int,
                              target_state: ExtendedLifecycle,
                              parameters: Dict[str, Any],
                              force: bool) -> bool:
        """Transition a single seed."""
        try:
            # Get transition method from layer
            if hasattr(layer, 'transition_seed'):
                return await layer.transition_seed(
                    seed_id, target_state, parameters, force
                )
            else:
                # Fallback to direct state update
                current_state = layer.get_seed_state(seed_id)

                if force or self.lifecycle_manager.can_transition(current_state, target_state):
                    layer.set_seed_state(seed_id, target_state)
                    return True

            return False

        except Exception as e:
            logger.error("Error transitioning seed %s: %s", seed_id, e)
            return False

    async def _transition_all_seeds(self, layer: Any,
                                   target_state: ExtendedLifecycle,
                                   parameters: Dict[str, Any],
                                   force: bool) -> List[bool]:
        """Transition all seeds in a layer."""
        tasks = []

        for seed_id in range(layer.num_seeds):
            task = self._transition_seed(
                layer, seed_id, target_state, parameters, force
            )
            tasks.append(task)

        # Execute in batches to avoid overwhelming the system
        batch_size = 100
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            # Convert exceptions to False
            for r in batch_results:
                if isinstance(r, Exception):
                    logger.error("Batch transition error: %s", r)
                    results.append(False)
                else:
                    results.append(r)

        return results


class BlueprintUpdateProcessor(CommandProcessor):
    """Processor for blueprint update commands."""

    def __init__(self, layer_registry: Dict[str, Any],
                 blueprint_registry: Optional[Dict[str, Any]] = None):
        self.layer_registry = layer_registry
        self.blueprint_registry = blueprint_registry or {}

    async def can_handle(self, command: BaseMessage) -> bool:
        return isinstance(command, BlueprintUpdateCommand)

    async def validate(self, context: CommandContext) -> Optional[str]:
        """Validate blueprint update command."""
        command: BlueprintUpdateCommand = context.command

        # Check layer exists
        if context.layer_id not in self.layer_registry:
            return f"Layer {context.layer_id} not found"

        # Check seed exists
        if context.seed_id is None:
            return "Seed ID is required for blueprint update"

        layer = self.layer_registry[context.layer_id]
        if context.seed_id >= layer.num_seeds:
            return f"Seed {context.seed_id} not found"

        # Check blueprint exists
        if command.blueprint_id and command.blueprint_id not in self.blueprint_registry:
            logger.warning("Blueprint %s not in registry", command.blueprint_id)
            # Don't fail - layer might have its own blueprint management

        # Validate grafting strategy
        valid_strategies = ["immediate", "gradual", "conditional", "adaptive", "safe"]
        if command.grafting_strategy not in valid_strategies:
            return f"Invalid grafting strategy: {command.grafting_strategy}"

        # Check seed is in valid state for update
        current_state = layer.get_seed_state(context.seed_id)
        valid_states = [
            ExtendedLifecycle.TRAINING,
            ExtendedLifecycle.GRAFTING,
            ExtendedLifecycle.FINE_TUNING
        ]

        if current_state not in valid_states:
            return f"Cannot update blueprint in state {current_state.name}"

        return None

    async def execute(self, context: CommandContext) -> CommandResult:
        """Execute blueprint update."""
        command: BlueprintUpdateCommand = context.command
        layer = self.layer_registry[context.layer_id]

        try:
            # Store original blueprint for rollback
            original_blueprint = None
            if hasattr(layer, 'get_seed_blueprint'):
                original_blueprint = layer.get_seed_blueprint(context.seed_id)

            # Apply blueprint update
            success = await self._apply_blueprint(
                layer, context.seed_id, command
            )

            if success:
                # Validate metrics if specified
                if command.validation_metrics:
                    validation_passed = await self._validate_update(
                        layer, context.seed_id, command.validation_metrics
                    )

                    if not validation_passed and command.rollback_on_failure:
                        # Rollback
                        if original_blueprint:
                            await self._rollback_blueprint(
                                layer, context.seed_id, original_blueprint
                            )

                        return CommandResult(
                            success=False,
                            error="Validation metrics not met",
                            details={
                                "blueprint_id": command.blueprint_id,
                                "validation_failed": True,
                                "rolled_back": True
                            },
                            execution_time_ms=context.elapsed_ms
                        )

                return CommandResult(
                    success=True,
                    details={
                        "seed_id": context.seed_id,
                        "blueprint_id": command.blueprint_id,
                        "strategy": command.grafting_strategy,
                        "update_time_ms": context.elapsed_ms
                    },
                    execution_time_ms=context.elapsed_ms
                )
            else:
                return CommandResult(
                    success=False,
                    error="Blueprint update failed",
                    execution_time_ms=context.elapsed_ms
                )

        except Exception as e:
            logger.error("Blueprint update error: %s", e)
            return CommandResult(
                success=False,
                error=str(e),
                execution_time_ms=context.elapsed_ms
            )

    async def _apply_blueprint(self, layer: Any, seed_id: int,
                              command: BlueprintUpdateCommand) -> bool:
        """Apply blueprint to seed."""
        try:
            if hasattr(layer, 'update_seed_blueprint'):
                return await layer.update_seed_blueprint(
                    seed_id,
                    command.blueprint_id,
                    command.grafting_strategy,
                    command.configuration
                )
            else:
                # Fallback implementation
                logger.warning("Layer does not implement update_seed_blueprint")
                return False

        except Exception as e:
            logger.error("Error applying blueprint: %s", e)
            return False

    async def _validate_update(self, layer: Any, seed_id: int,
                              validation_metrics: Dict[str, float]) -> bool:
        """Validate blueprint update against metrics."""
        try:
            # Get current metrics
            if hasattr(layer, 'get_seed_metrics'):
                current_metrics = await layer.get_seed_metrics(seed_id)

                # Check each validation metric
                for metric_name, required_value in validation_metrics.items():
                    current_value = current_metrics.get(metric_name, 0.0)

                    # Simple threshold check - could be enhanced
                    if current_value < required_value:
                        logger.info("Validation failed: %s=%s < %s", metric_name, current_value, required_value)
                        return False

                return True
            else:
                logger.warning("Cannot validate - layer doesn't support metrics")
                return True

        except Exception as e:
            logger.error("Validation error: %s", e)
            return False

    async def _rollback_blueprint(self, layer: Any, seed_id: int,
                                 original_blueprint: Any) -> bool:
        """Rollback to original blueprint."""
        try:
            if hasattr(layer, 'rollback_seed_blueprint'):
                return await layer.rollback_seed_blueprint(seed_id, original_blueprint)
            else:
                logger.warning("Layer does not support blueprint rollback")
                return False

        except Exception as e:
            logger.error("Rollback error: %s", e)
            return False


class BatchCommandProcessor(CommandProcessor):
    """Processor for batch commands."""

    def __init__(self, processors: List[CommandProcessor]):
        self.processors = processors

    async def can_handle(self, command: BaseMessage) -> bool:
        return isinstance(command, BatchCommand)

    async def validate(self, context: CommandContext) -> Optional[str]:
        """Validate batch command."""
        command: BatchCommand = context.command

        # Check basic constraints
        error = self._validate_batch_constraints(command)
        if error:
            return error

        # Validate each command
        for i, sub_command in enumerate(command.commands):
            error = await self._validate_sub_command(i, sub_command)
            if error:
                return error

        return None

    def _validate_batch_constraints(self, command: BatchCommand) -> Optional[str]:
        """Validate basic batch constraints."""
        if not command.commands:
            return "Batch command is empty"

        if len(command.commands) > 1000:
            return "Batch too large (max 1000 commands)"

        return None

    async def _validate_sub_command(self, index: int,
                                   sub_command: BaseMessage) -> Optional[str]:
        """Validate a single sub-command."""
        # Find processor
        processor = None
        for p in self.processors:
            if await p.can_handle(sub_command):
                processor = p
                break

        if not processor:
            return f"No processor for command {index}: {type(sub_command).__name__}"

        # Create sub-context
        sub_context = CommandContext(
            command=sub_command,
            layer_id=sub_command.layer_id,
            seed_id=getattr(sub_command, 'seed_id', None)
        )

        # Validate
        error = await processor.validate(sub_context)
        if error:
            return f"Command {index} validation failed: {error}"

        return None

    async def execute(self, context: CommandContext) -> CommandResult:
        """Execute batch command."""
        command: BatchCommand = context.command
        results = []
        executed = []

        try:
            for i, sub_command in enumerate(command.commands):
                # Check timeout
                if context.is_timeout:
                    return self._create_timeout_result(context, executed, command.commands)

                # Process single command
                result = await self._execute_single_command(
                    i, sub_command, context, executed, results
                )

                # Handle failure
                if not result.success and command.stop_on_error:
                    return await self._handle_command_failure(
                        i, result, command, context, executed
                    )

            # All commands processed
            return self._create_batch_result(command.commands, results, context)

        except Exception as e:
            return await self._handle_batch_exception(e, command, context, executed)

    async def _execute_single_command(self, index: int, sub_command: BaseMessage,
                                    context: CommandContext, executed: List[tuple],
                                    results: List[CommandResult]) -> CommandResult:
        """Execute a single command in the batch."""
        # Find processor
        processor = await self._find_processor(sub_command)

        # Create sub-context
        sub_context = self._create_sub_context(sub_command, context)

        # Execute
        result = await processor.execute(sub_context)
        results.append(result)
        executed.append((index, sub_command, processor))

        return result

    async def _find_processor(self, command: BaseMessage) -> CommandProcessor:
        """Find processor for command."""
        for p in self.processors:
            if await p.can_handle(command):
                return p
        raise ValueError(f"No processor for {type(command).__name__}")

    def _create_sub_context(self, sub_command: BaseMessage,
                           parent_context: CommandContext) -> CommandContext:
        """Create context for sub-command."""
        return CommandContext(
            command=sub_command,
            layer_id=sub_command.layer_id,
            seed_id=getattr(sub_command, 'seed_id', None),
            start_time=time.time(),
            timeout_ms=min(
                getattr(sub_command, 'timeout_ms', 5000),
                parent_context.timeout_ms - parent_context.elapsed_ms
            )
        )

    def _create_timeout_result(self, context: CommandContext,
                              executed: List[tuple], commands: List) -> CommandResult:
        """Create timeout result."""
        return CommandResult(
            success=False,
            error="Batch timeout",
            details={
                "executed": len(executed),
                "total": len(commands)
            },
            execution_time_ms=context.elapsed_ms
        )

    async def _handle_command_failure(self, index: int, result: CommandResult,
                                    command: BatchCommand, context: CommandContext,
                                    executed: List[tuple]) -> CommandResult:
        """Handle failed command in batch."""
        if command.atomic:
            await self._rollback_executed(executed)

        return CommandResult(
            success=False,
            error=f"Command {index} failed: {result.error}",
            details={
                "failed_at": index,
                "executed": len(executed),
                "rolled_back": command.atomic
            },
            execution_time_ms=context.elapsed_ms
        )

    def _create_batch_result(self, commands: List[BaseMessage],
                           results: List[CommandResult],
                           context: CommandContext) -> CommandResult:
        """Create result for completed batch."""
        success_count = sum(1 for r in results if r.success)

        return CommandResult(
            success=success_count == len(commands),
            details={
                "total_commands": len(commands),
                "successful": success_count,
                "failed": len(commands) - success_count,
                "results": [r.to_dict() for r in results]
            },
            execution_time_ms=context.elapsed_ms
        )

    async def _handle_batch_exception(self, exception: Exception, command: BatchCommand,
                                    context: CommandContext, executed: List[tuple]) -> CommandResult:
        """Handle exception during batch execution."""
        if command.atomic and executed:
            await self._rollback_executed(executed)

        return CommandResult(
            success=False,
            error=str(exception),
            details={
                "executed": len(executed),
                "rolled_back": command.atomic
            },
            execution_time_ms=context.elapsed_ms
        )

    async def _rollback_executed(self, executed: List[tuple]):
        """Rollback executed commands in reverse order."""
        for i, sub_command, processor in reversed(executed):
            try:
                sub_context = CommandContext(
                    command=sub_command,
                    layer_id=sub_command.layer_id,
                    seed_id=getattr(sub_command, 'seed_id', None)
                )
                await processor.rollback(sub_context)

            except Exception as e:
                logger.error("Rollback error for command %s: %s", i, e)


class EmergencyStopProcessor(CommandProcessor):
    """Processor for emergency stop commands."""

    def __init__(self, layer_registry: Dict[str, Any]):
        self.layer_registry = layer_registry

    async def can_handle(self, command: BaseMessage) -> bool:
        return isinstance(command, EmergencyStopCommand)

    async def validate(self, context: CommandContext) -> Optional[str]:
        """Emergency stops are always valid."""
        return None

    async def execute(self, context: CommandContext) -> CommandResult:
        """Execute emergency stop."""
        command: EmergencyStopCommand = context.command
        stopped_layers = []
        stopped_seeds = []

        try:
            if command.layer_id:
                # Stop specific layer
                if command.layer_id in self.layer_registry:
                    layer = self.layer_registry[command.layer_id]

                    if command.seed_id is not None:
                        # Stop specific seed
                        await self._stop_seed(layer, command.seed_id)
                        stopped_seeds.append((command.layer_id, command.seed_id))
                    else:
                        # Stop all seeds in layer
                        await self._stop_layer(layer)
                        stopped_layers.append(command.layer_id)
            else:
                # Stop all layers
                for layer_id, layer in self.layer_registry.items():
                    await self._stop_layer(layer)
                    stopped_layers.append(layer_id)

            return CommandResult(
                success=True,
                details={
                    "stopped_layers": stopped_layers,
                    "stopped_seeds": stopped_seeds,
                    "reason": command.reason,
                    "stop_time_ms": context.elapsed_ms
                },
                execution_time_ms=context.elapsed_ms
            )

        except Exception as e:
            logger.error("Emergency stop error: %s", e)
            return CommandResult(
                success=False,
                error=str(e),
                details={
                    "partial_stop": True,
                    "stopped_layers": stopped_layers,
                    "stopped_seeds": stopped_seeds
                },
                execution_time_ms=context.elapsed_ms
            )

    async def _stop_layer(self, layer: Any):
        """Stop all processing in a layer."""
        if hasattr(layer, 'emergency_stop'):
            await layer.emergency_stop()
        else:
            # Transition all seeds to DORMANT
            for seed_id in range(layer.num_seeds):
                try:
                    layer.set_seed_state(seed_id, ExtendedLifecycle.DORMANT)
                except Exception as e:
                    logger.error("Error stopping seed %s: %s", seed_id, e)

    async def _stop_seed(self, layer: Any, seed_id: int):
        """Stop a specific seed."""
        if hasattr(layer, 'stop_seed'):
            await layer.stop_seed(seed_id)
        else:
            layer.set_seed_state(seed_id, ExtendedLifecycle.DORMANT)


class CommandHandler:
    """Main command handler that routes commands to processors."""

    def __init__(self, layer_registry: Dict[str, Any],
                 message_bus: Optional[MessageBusClient] = None):
        self.layer_registry = layer_registry
        self.message_bus = message_bus

        # Initialize processors
        self.processors: List[CommandProcessor] = [
            LifecycleTransitionProcessor(layer_registry),
            BlueprintUpdateProcessor(layer_registry),
            EmergencyStopProcessor(layer_registry)
        ]

        # Add batch processor with other processors
        self.processors.append(BatchCommandProcessor(self.processors))

        # Command queue for priority handling
        self.command_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

        # Active commands tracking
        self.active_commands: Dict[str, CommandContext] = {}

        # Stats
        self.stats = {
            "commands_received": 0,
            "commands_executed": 0,
            "commands_failed": 0,
            "commands_timeout": 0,
            "average_execution_ms": 0.0
        }

        # Background task
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start command handler."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_commands())
        logger.info("CommandHandler started")

    async def stop(self):
        """Stop command handler."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("CommandHandler stopped")

    async def handle_command(self, command: BaseMessage) -> CommandResult:
        """Process command with timeout and error handling."""
        self.stats["commands_received"] += 1

        # Create context
        context = self._create_command_context(command)

        # Find and validate processor
        processor = await self._find_and_validate_processor(command, context)
        if processor is None:
            return context.result  # Error result already set

        # Queue command for processing
        await self._queue_command(context, processor)

        # Wait for completion
        return await self._wait_for_completion(context)

    def _create_command_context(self, command: BaseMessage) -> CommandContext:
        """Create context for command execution."""
        return CommandContext(
            command=command,
            layer_id=getattr(command, 'layer_id', ''),
            seed_id=getattr(command, 'seed_id', None),
            start_time=time.time(),
            timeout_ms=getattr(command, 'timeout_ms', 5000),
            priority=self._get_priority(command)
        )

    async def _find_and_validate_processor(self, command: BaseMessage,
                                         context: CommandContext) -> Optional[CommandProcessor]:
        """Find processor and validate command."""
        # Find processor
        processor = None
        for p in self.processors:
            if await p.can_handle(command):
                processor = p
                break

        if not processor:
            context.result = CommandResult(
                success=False,
                error=f"No processor for command type: {type(command).__name__}",
                execution_time_ms=0.0
            )
            return None

        # Validate
        error = await processor.validate(context)
        if error:
            context.result = CommandResult(
                success=False,
                error=error,
                execution_time_ms=context.elapsed_ms
            )
            return None

        return processor

    async def _queue_command(self, context: CommandContext, processor: CommandProcessor):
        """Add command to processing queue."""
        await self.command_queue.put((
            -context.priority.value,  # Negative for max priority queue
            time.time(),
            context,
            processor
        ))

    async def _wait_for_completion(self, context: CommandContext) -> CommandResult:
        """Wait for command completion with timeout."""
        try:
            timeout_s = context.timeout_ms / 1000
            start = time.time()

            while time.time() - start < timeout_s:
                if context.status in (CommandStatus.COMPLETED, CommandStatus.FAILED):
                    return context.result

                await asyncio.sleep(0.01)

            # Timeout
            return self._create_timeout_result(context)

        except Exception as e:
            return CommandResult(
                success=False,
                error=str(e),
                execution_time_ms=context.elapsed_ms
            )

    def _create_timeout_result(self, context: CommandContext) -> CommandResult:
        """Create timeout result for command."""
        context.status = CommandStatus.TIMEOUT
        self.stats["commands_timeout"] += 1

        return CommandResult(
            success=False,
            error="Command timeout",
            execution_time_ms=context.timeout_ms
        )

    def _get_priority(self, command: BaseMessage) -> CommandPriority:
        """Determine command priority."""
        if isinstance(command, EmergencyStopCommand):
            return CommandPriority.CRITICAL

        priority_str = getattr(command, 'priority', 'normal')

        priority_map = {
            'low': CommandPriority.LOW,
            'normal': CommandPriority.NORMAL,
            'high': CommandPriority.HIGH,
            'critical': CommandPriority.CRITICAL
        }

        return priority_map.get(priority_str, CommandPriority.NORMAL)

    async def _process_commands(self):
        """Background task to process commands."""
        while self._running:
            try:
                # Get next command
                command_tuple = await self._get_next_command()
                if command_tuple is None:
                    continue

                priority, timestamp, context, processor = command_tuple

                # Process the command
                await self._process_single_command(context, processor)

            except Exception as e:
                logger.error("Command processor error: %s", e)
                await asyncio.sleep(0.1)

    async def _get_next_command(self) -> Optional[tuple]:
        """Get next command from queue with timeout."""
        try:
            return await asyncio.wait_for(
                self.command_queue.get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # No commands to process
            return None

    async def _process_single_command(self, context: CommandContext,
                                    processor: CommandProcessor):
        """Process a single command."""
        command_id = context.command.message_id

        # Track and update status
        self.active_commands[command_id] = context
        context.status = CommandStatus.RUNNING

        try:
            # Execute command
            await self._execute_command(context, processor)
        finally:
            # Always remove from active
            if command_id in self.active_commands:
                del self.active_commands[command_id]

    async def _execute_command(self, context: CommandContext,
                              processor: CommandProcessor):
        """Execute command with timeout and error handling."""
        remaining_time = context.timeout_ms - context.elapsed_ms

        if remaining_time <= 0:
            # Already timed out
            self._handle_pre_execution_timeout(context)
            return

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                processor.execute(context),
                timeout=remaining_time / 1000
            )

            # Handle successful execution
            await self._handle_execution_success(context, result)

        except asyncio.TimeoutError:
            self._handle_execution_timeout(context)

        except Exception as e:
            self._handle_execution_error(context, e)

    def _handle_pre_execution_timeout(self, context: CommandContext):
        """Handle command that timed out before execution."""
        context.status = CommandStatus.TIMEOUT
        context.result = CommandResult(
            success=False,
            error="Command timeout before execution",
            execution_time_ms=context.elapsed_ms
        )
        self.stats["commands_timeout"] += 1

    def _handle_execution_timeout(self, context: CommandContext):
        """Handle command execution timeout."""
        context.status = CommandStatus.TIMEOUT
        context.result = CommandResult(
            success=False,
            error="Command execution timeout",
            execution_time_ms=context.timeout_ms
        )
        self.stats["commands_timeout"] += 1

    def _handle_execution_error(self, context: CommandContext, error: Exception):
        """Handle command execution error."""
        context.status = CommandStatus.FAILED
        context.result = CommandResult(
            success=False,
            error=str(error),
            execution_time_ms=context.elapsed_ms
        )
        self.stats["commands_failed"] += 1
        logger.error("Command execution error: %s", error)

    async def _handle_execution_success(self, context: CommandContext,
                                      result: CommandResult):
        """Handle successful command execution."""
        context.result = result
        context.status = CommandStatus.COMPLETED

        # Update statistics
        self._update_execution_stats(context, result)

        # Publish acknowledgment
        if self.message_bus:
            await self._publish_ack(context)

    def _update_execution_stats(self, context: CommandContext, result: CommandResult):
        """Update execution statistics."""
        self.stats["commands_executed"] += 1

        if not result.success:
            self.stats["commands_failed"] += 1

        # Update average execution time
        avg = self.stats["average_execution_ms"]
        count = self.stats["commands_executed"]
        self.stats["average_execution_ms"] = (
            (avg * (count - 1) + context.elapsed_ms) / count
        )

    async def _publish_ack(self, context: CommandContext):
        """Publish command acknowledgment."""
        try:
            # Create ack topic
            topic = create_topic_name("control.ack", context.layer_id)

            # Include correlation ID for request/response
            context.result.metadata["correlation_id"] = context.command.message_id
            context.result.metadata["command_type"] = type(context.command).__name__

            await self.message_bus.publish(topic, context.result)

        except Exception as e:
            logger.error("Failed to publish ack: %s", e)

    async def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        stats = self.stats.copy()
        stats["queue_size"] = self.command_queue.qsize()
        stats["active_commands"] = len(self.active_commands)
        return stats


class CommandHandlerFactory:
    """Factory for creating command handlers with custom processors."""

    @staticmethod
    def create(layer_registry: Dict[str, Any],
               message_bus: Optional[MessageBusClient] = None,
               custom_processors: Optional[List[CommandProcessor]] = None) -> CommandHandler:
        """Create command handler with optional custom processors.
        
        Args:
            layer_registry: Registry of layers
            message_bus: Optional message bus client
            custom_processors: Optional list of custom processors
            
        Returns:
            Configured CommandHandler
        """
        handler = CommandHandler(layer_registry, message_bus)

        if custom_processors:
            # Add custom processors
            handler.processors.extend(custom_processors)

            # Recreate batch processor with all processors
            handler.processors = [p for p in handler.processors
                                 if not isinstance(p, BatchCommandProcessor)]
            handler.processors.append(BatchCommandProcessor(handler.processors))

        return handler
