"""Unit tests for message bus command handlers."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from src.esper.morphogenetic_v2.message_bus.handlers import (
    CommandContext, CommandStatus, CommandPriority, CommandResult,
    LifecycleTransitionProcessor, BlueprintUpdateProcessor,
    BatchCommandProcessor, EmergencyStopProcessor,
    CommandHandler, CommandHandlerFactory
)
from src.esper.morphogenetic_v2.message_bus.schemas import (
    LifecycleTransitionCommand, BlueprintUpdateCommand,
    BatchCommand, EmergencyStopCommand
)
from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle


class TestCommandContext:
    """Test CommandContext functionality."""
    
    def test_elapsed_ms(self):
        """Test elapsed time calculation."""
        cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        context = CommandContext(
            command=cmd,
            layer_id="test",
            seed_id=0,
            start_time=time.time() - 0.1  # 100ms ago
        )
        
        assert context.elapsed_ms >= 100
        assert context.elapsed_ms < 200
        
    def test_is_timeout(self):
        """Test timeout detection."""
        cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        context = CommandContext(
            command=cmd,
            layer_id="test",
            start_time=time.time() - 0.1,  # 100ms ago
            timeout_ms=50  # 50ms timeout
        )
        
        assert context.is_timeout
        
        # Not timed out
        context.timeout_ms = 200
        assert not context.is_timeout


class TestLifecycleTransitionProcessor:
    """Test LifecycleTransitionProcessor."""
    
    @pytest.fixture
    def mock_layer(self):
        """Create mock layer."""
        layer = Mock()
        layer.num_seeds = 100
        layer.get_seed_state = Mock(return_value=ExtendedLifecycle.DORMANT)
        layer.set_seed_state = Mock()
        layer.transition_seed = AsyncMock(return_value=True)
        return layer
        
    @pytest.fixture
    def processor(self, mock_layer):
        """Create processor with mock layer."""
        layer_registry = {"test_layer": mock_layer}
        return LifecycleTransitionProcessor(layer_registry)
        
    @pytest.mark.asyncio
    async def test_can_handle(self, processor):
        """Test command type detection."""
        cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        assert await processor.can_handle(cmd)
        
        other_cmd = BlueprintUpdateCommand(layer_id="test", seed_id=0, blueprint_id="bp1")
        assert not await processor.can_handle(other_cmd)
        
    @pytest.mark.asyncio
    async def test_validate_layer_not_found(self, processor):
        """Test validation when layer doesn't exist."""
        cmd = LifecycleTransitionCommand(
            layer_id="nonexistent",
            seed_id=0,
            target_state="TRAINING"
        )
        context = CommandContext(command=cmd, layer_id="nonexistent", seed_id=0)
        
        error = await processor.validate(context)
        assert error == "Layer nonexistent not found"
        
    @pytest.mark.asyncio
    async def test_validate_seed_not_found(self, processor, mock_layer):
        """Test validation when seed doesn't exist."""
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=999,  # Out of range
            target_state="TRAINING"
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=999)
        
        error = await processor.validate(context)
        assert error == "Seed 999 not found in layer"
        
    @pytest.mark.asyncio
    async def test_validate_invalid_target_state(self, processor, mock_layer):
        """Test validation with invalid target state."""
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=0,
            target_state="INVALID_STATE"
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=0)
        
        error = await processor.validate(context)
        assert "Invalid target state: INVALID_STATE" in error
        
    @pytest.mark.asyncio
    async def test_validate_invalid_transition(self, processor, mock_layer):
        """Test validation of invalid state transition."""
        mock_layer.get_seed_state.return_value = ExtendedLifecycle.DORMANT
        
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=0,
            target_state="GRAFTING"  # Can't go directly from DORMANT to GRAFTING
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=0)
        
        error = await processor.validate(context)
        assert "Cannot transition from DORMANT to GRAFTING" in error
        assert "Valid transitions:" in error
        
    @pytest.mark.asyncio
    async def test_validate_success(self, processor, mock_layer):
        """Test successful validation."""
        mock_layer.get_seed_state.return_value = ExtendedLifecycle.DORMANT
        
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=0,
            target_state="GERMINATED"  # Valid transition
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=0)
        
        error = await processor.validate(context)
        assert error is None
        
    @pytest.mark.asyncio
    async def test_execute_single_seed(self, processor, mock_layer):
        """Test executing transition for single seed."""
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=0,
            target_state="GERMINATED",
            parameters={"test": "param"}
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=0,
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["seed_id"] == 0
        assert result.details["new_state"] == "GERMINATED"
        mock_layer.transition_seed.assert_awaited_once_with(
            0, ExtendedLifecycle.GERMINATED, {"test": "param"}, False
        )
        
    @pytest.mark.asyncio
    async def test_execute_all_seeds(self, processor, mock_layer):
        """Test executing transition for all seeds."""
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=None,  # All seeds
            target_state="GERMINATED"
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=None,
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["total_seeds"] == 100
        assert result.details["successful_transitions"] == 100
        assert mock_layer.transition_seed.call_count == 100
        
    @pytest.mark.asyncio
    async def test_execute_with_failures(self, processor, mock_layer):
        """Test execution with some failures."""
        # Make some transitions fail
        mock_layer.transition_seed.side_effect = [True, False, True] * 34  # Mix of success/failure
        
        cmd = LifecycleTransitionCommand(
            layer_id="test_layer",
            seed_id=None,
            target_state="GERMINATED"
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=None,
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        # Should still be considered success if any succeeded
        assert result.success
        assert result.details["successful_transitions"] < 100
        assert result.details["failed_transitions"] > 0


class TestBlueprintUpdateProcessor:
    """Test BlueprintUpdateProcessor."""
    
    @pytest.fixture
    def mock_layer(self):
        """Create mock layer."""
        layer = Mock()
        layer.num_seeds = 100
        layer.get_seed_state = Mock(return_value=ExtendedLifecycle.TRAINING)
        layer.get_seed_blueprint = Mock(return_value="original_bp")
        layer.update_seed_blueprint = AsyncMock(return_value=True)
        layer.get_seed_metrics = AsyncMock(return_value={"loss": 0.1, "accuracy": 0.95})
        layer.rollback_seed_blueprint = AsyncMock(return_value=True)
        return layer
        
    @pytest.fixture
    def processor(self, mock_layer):
        """Create processor with mock layer."""
        layer_registry = {"test_layer": mock_layer}
        blueprint_registry = {"bp1": {"name": "Blueprint 1"}}
        return BlueprintUpdateProcessor(layer_registry, blueprint_registry)
        
    @pytest.mark.asyncio
    async def test_can_handle(self, processor):
        """Test command type detection."""
        cmd = BlueprintUpdateCommand(layer_id="test", seed_id=0, blueprint_id="bp1")
        assert await processor.can_handle(cmd)
        
        other_cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        assert not await processor.can_handle(other_cmd)
        
    @pytest.mark.asyncio
    async def test_validate_missing_seed_id(self, processor):
        """Test validation when seed_id is missing."""
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=None,
            blueprint_id="bp1"
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=None)
        
        error = await processor.validate(context)
        assert error == "Seed ID is required for blueprint update"
        
    @pytest.mark.asyncio
    async def test_validate_invalid_strategy(self, processor):
        """Test validation with invalid grafting strategy."""
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="invalid"
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=0)
        
        error = await processor.validate(context)
        assert "Invalid grafting strategy: invalid" in error
        
    @pytest.mark.asyncio
    async def test_validate_invalid_state(self, processor, mock_layer):
        """Test validation when seed is in invalid state."""
        mock_layer.get_seed_state.return_value = ExtendedLifecycle.DORMANT
        
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="immediate"
        )
        context = CommandContext(command=cmd, layer_id="test_layer", seed_id=0)
        
        error = await processor.validate(context)
        assert "Cannot update blueprint in state DORMANT" in error
        
    @pytest.mark.asyncio
    async def test_execute_success(self, processor, mock_layer):
        """Test successful blueprint update."""
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="immediate",
            configuration={"param": "value"}
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=0,
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["blueprint_id"] == "bp1"
        assert result.details["strategy"] == "immediate"
        mock_layer.update_seed_blueprint.assert_awaited_once_with(
            0, "bp1", "immediate", {"param": "value"}
        )
        
    @pytest.mark.asyncio
    async def test_execute_with_validation(self, processor, mock_layer):
        """Test blueprint update with validation metrics."""
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="immediate",
            validation_metrics={"accuracy": 0.9},  # Required accuracy
            rollback_on_failure=True
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=0,
            start_time=time.time()
        )
        
        # Current metrics meet requirement
        mock_layer.get_seed_metrics.return_value = {"accuracy": 0.95}
        
        result = await processor.execute(context)
        
        assert result.success
        mock_layer.get_seed_metrics.assert_awaited_once()
        
    @pytest.mark.asyncio
    async def test_execute_validation_failure_with_rollback(self, processor, mock_layer):
        """Test rollback when validation fails."""
        cmd = BlueprintUpdateCommand(
            layer_id="test_layer",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="immediate",
            validation_metrics={"accuracy": 0.9},
            rollback_on_failure=True
        )
        context = CommandContext(
            command=cmd,
            layer_id="test_layer",
            seed_id=0,
            start_time=time.time()
        )
        
        # Current metrics don't meet requirement
        mock_layer.get_seed_metrics.return_value = {"accuracy": 0.8}
        
        result = await processor.execute(context)
        
        assert not result.success
        assert "Validation metrics not met" in result.error
        assert result.details["rolled_back"]
        mock_layer.rollback_seed_blueprint.assert_awaited_once()


class TestBatchCommandProcessor:
    """Test BatchCommandProcessor."""
    
    @pytest.fixture
    def mock_processors(self):
        """Create mock processors."""
        lifecycle_proc = Mock()
        lifecycle_proc.can_handle = AsyncMock(side_effect=lambda cmd: isinstance(cmd, LifecycleTransitionCommand))
        lifecycle_proc.validate = AsyncMock(return_value=None)
        lifecycle_proc.execute = AsyncMock(return_value=CommandResult(success=True))
        
        blueprint_proc = Mock()
        blueprint_proc.can_handle = AsyncMock(side_effect=lambda cmd: isinstance(cmd, BlueprintUpdateCommand))
        blueprint_proc.validate = AsyncMock(return_value=None)
        blueprint_proc.execute = AsyncMock(return_value=CommandResult(success=True))
        
        return [lifecycle_proc, blueprint_proc]
        
    @pytest.fixture
    def processor(self, mock_processors):
        """Create batch processor."""
        return BatchCommandProcessor(mock_processors)
        
    @pytest.mark.asyncio
    async def test_can_handle(self, processor):
        """Test command type detection."""
        cmd = BatchCommand(commands=[])
        assert await processor.can_handle(cmd)
        
        other_cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        assert not await processor.can_handle(other_cmd)
        
    @pytest.mark.asyncio
    async def test_validate_empty_batch(self, processor):
        """Test validation of empty batch."""
        cmd = BatchCommand(commands=[])
        context = CommandContext(command=cmd, layer_id="")
        
        error = await processor.validate(context)
        assert error == "Batch command is empty"
        
    @pytest.mark.asyncio
    async def test_validate_batch_too_large(self, processor):
        """Test validation of oversized batch."""
        commands = [LifecycleTransitionCommand(layer_id="test", seed_id=i, target_state="TRAINING") 
                   for i in range(1001)]
        cmd = BatchCommand(commands=commands)
        context = CommandContext(command=cmd, layer_id="")
        
        error = await processor.validate(context)
        assert "Batch too large" in error
        
    @pytest.mark.asyncio
    async def test_validate_no_processor(self, processor):
        """Test validation when no processor available."""
        # Create command type that no processor handles
        unknown_cmd = Mock()
        unknown_cmd.__class__.__name__ = "UnknownCommand"
        
        cmd = BatchCommand(commands=[unknown_cmd])
        context = CommandContext(command=cmd, layer_id="")
        
        error = await processor.validate(context)
        assert "No processor for command 0: UnknownCommand" in error
        
    @pytest.mark.asyncio
    async def test_execute_success(self, processor, mock_processors):
        """Test successful batch execution."""
        cmd1 = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        cmd2 = BlueprintUpdateCommand(layer_id="test", seed_id=0, blueprint_id="bp1", grafting_strategy="immediate")
        
        batch_cmd = BatchCommand(commands=[cmd1, cmd2])
        context = CommandContext(
            command=batch_cmd,
            layer_id="",
            start_time=time.time(),
            timeout_ms=5000
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["total_commands"] == 2
        assert result.details["successful"] == 2
        assert result.details["failed"] == 0
        
    @pytest.mark.asyncio
    async def test_execute_with_failure_stop_on_error(self, processor, mock_processors):
        """Test batch execution with failure and stop_on_error."""
        # Make second command fail
        mock_processors[1].execute.return_value = CommandResult(success=False, error="Test error")
        
        cmd1 = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        cmd2 = BlueprintUpdateCommand(layer_id="test", seed_id=0, blueprint_id="bp1", grafting_strategy="immediate")
        
        batch_cmd = BatchCommand(
            commands=[cmd1, cmd2],
            stop_on_error=True,
            atomic=False
        )
        context = CommandContext(command=batch_cmd, layer_id="", start_time=time.time())
        
        result = await processor.execute(context)
        
        assert not result.success
        assert "Command 1 failed" in result.error
        assert result.details["failed_at"] == 1
        assert result.details["executed"] == 2
        
    @pytest.mark.asyncio
    async def test_execute_atomic_with_rollback(self, processor, mock_processors):
        """Test atomic batch with rollback on failure."""
        # Add rollback methods
        mock_processors[0].rollback = AsyncMock()
        mock_processors[1].rollback = AsyncMock()
        
        # Make second command fail
        mock_processors[1].execute.return_value = CommandResult(success=False, error="Test error")
        
        cmd1 = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        cmd2 = BlueprintUpdateCommand(layer_id="test", seed_id=0, blueprint_id="bp1", grafting_strategy="immediate")
        
        batch_cmd = BatchCommand(
            commands=[cmd1, cmd2],
            atomic=True,
            stop_on_error=True
        )
        context = CommandContext(command=batch_cmd, layer_id="", start_time=time.time())
        
        result = await processor.execute(context)
        
        assert not result.success
        assert result.details["rolled_back"]
        # Should rollback first command only (second failed, not executed)
        mock_processors[0].rollback.assert_awaited_once()
        
    @pytest.mark.asyncio
    async def test_execute_timeout(self, processor, mock_processors):
        """Test batch timeout handling."""
        cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        batch_cmd = BatchCommand(commands=[cmd])
        
        # Already timed out
        context = CommandContext(
            command=batch_cmd,
            layer_id="",
            start_time=time.time() - 1,  # 1 second ago
            timeout_ms=100  # 100ms timeout
        )
        
        result = await processor.execute(context)
        
        assert not result.success
        assert "Batch timeout" in result.error


class TestEmergencyStopProcessor:
    """Test EmergencyStopProcessor."""
    
    @pytest.fixture
    def mock_layer(self):
        """Create mock layer."""
        layer = Mock()
        layer.num_seeds = 10
        layer.emergency_stop = AsyncMock()
        layer.stop_seed = AsyncMock()
        layer.set_seed_state = Mock()
        return layer
        
    @pytest.fixture
    def processor(self, mock_layer):
        """Create processor with mock layer."""
        layer_registry = {
            "layer1": mock_layer,
            "layer2": Mock(emergency_stop=AsyncMock())
        }
        return EmergencyStopProcessor(layer_registry)
        
    @pytest.mark.asyncio
    async def test_can_handle(self, processor):
        """Test command type detection."""
        cmd = EmergencyStopCommand(reason="Test")
        assert await processor.can_handle(cmd)
        
        other_cmd = LifecycleTransitionCommand(layer_id="test", seed_id=0, target_state="TRAINING")
        assert not await processor.can_handle(other_cmd)
        
    @pytest.mark.asyncio
    async def test_validate_always_passes(self, processor):
        """Test that emergency stops always validate."""
        cmd = EmergencyStopCommand(layer_id="nonexistent", reason="Test")
        context = CommandContext(command=cmd, layer_id="nonexistent")
        
        error = await processor.validate(context)
        assert error is None
        
    @pytest.mark.asyncio
    async def test_execute_stop_all_layers(self, processor):
        """Test stopping all layers."""
        cmd = EmergencyStopCommand(reason="System emergency")
        context = CommandContext(
            command=cmd,
            layer_id="",
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert len(result.details["stopped_layers"]) == 2
        assert "layer1" in result.details["stopped_layers"]
        assert "layer2" in result.details["stopped_layers"]
        
    @pytest.mark.asyncio
    async def test_execute_stop_specific_layer(self, processor, mock_layer):
        """Test stopping specific layer."""
        cmd = EmergencyStopCommand(layer_id="layer1", reason="Layer failure")
        context = CommandContext(
            command=cmd,
            layer_id="layer1",
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["stopped_layers"] == ["layer1"]
        mock_layer.emergency_stop.assert_awaited_once()
        
    @pytest.mark.asyncio
    async def test_execute_stop_specific_seed(self, processor, mock_layer):
        """Test stopping specific seed."""
        cmd = EmergencyStopCommand(layer_id="layer1", seed_id=5, reason="Seed failure")
        context = CommandContext(
            command=cmd,
            layer_id="layer1",
            seed_id=5,
            start_time=time.time()
        )
        
        result = await processor.execute(context)
        
        assert result.success
        assert result.details["stopped_seeds"] == [("layer1", 5)]
        mock_layer.stop_seed.assert_awaited_once_with(5)
        
    @pytest.mark.asyncio
    async def test_execute_fallback_to_dormant(self, processor):
        """Test fallback to setting DORMANT state."""
        # Layer without emergency_stop method
        basic_layer = Mock()
        basic_layer.num_seeds = 5
        basic_layer.set_seed_state = Mock()
        
        processor.layer_registry["basic"] = basic_layer
        
        cmd = EmergencyStopCommand(layer_id="basic", reason="Test")
        context = CommandContext(command=cmd, layer_id="basic", start_time=time.time())
        
        result = await processor.execute(context)
        
        assert result.success
        # Should set all seeds to DORMANT
        assert basic_layer.set_seed_state.call_count == 5
        for i in range(5):
            basic_layer.set_seed_state.assert_any_call(i, ExtendedLifecycle.DORMANT)


class TestCommandHandler:
    """Test CommandHandler."""
    
    @pytest.fixture
    def mock_layer_registry(self):
        """Create mock layer registry."""
        return {"test_layer": Mock(num_seeds=10)}
        
    @pytest.fixture
    def mock_message_bus(self):
        """Create mock message bus client."""
        client = AsyncMock()
        client.publish = AsyncMock()
        return client
        
    @pytest.fixture
    def handler(self, mock_layer_registry, mock_message_bus):
        """Create command handler."""
        return CommandHandler(mock_layer_registry, mock_message_bus)
        
    @pytest.mark.asyncio
    async def test_start_stop(self, handler):
        """Test handler lifecycle."""
        assert not handler._running
        
        await handler.start()
        assert handler._running
        assert handler._processor_task is not None
        
        await handler.stop()
        assert not handler._running
        assert handler._processor_task is None
        
    @pytest.mark.asyncio
    async def test_handle_command_no_processor(self, handler):
        """Test handling command with no processor."""
        # Create unknown command type
        unknown_cmd = Mock()
        unknown_cmd.__class__.__name__ = "UnknownCommand"
        
        result = await handler.handle_command(unknown_cmd)
        
        assert not result.success
        assert "No processor for command type: UnknownCommand" in result.error
        
    @pytest.mark.asyncio
    async def test_handle_command_validation_failure(self, handler):
        """Test handling command that fails validation."""
        await handler.start()
        
        # Invalid layer
        cmd = LifecycleTransitionCommand(
            layer_id="nonexistent",
            seed_id=0,
            target_state="TRAINING"
        )
        
        result = await handler.handle_command(cmd)
        
        assert not result.success
        assert "not found" in result.error
        
        await handler.stop()
        
    @pytest.mark.asyncio
    async def test_get_priority(self, handler):
        """Test command priority determination."""
        # Emergency command
        emergency = EmergencyStopCommand(reason="Test")
        assert handler._get_priority(emergency) == CommandPriority.CRITICAL
        
        # Normal command with priority
        cmd = LifecycleTransitionCommand(
            layer_id="test",
            seed_id=0,
            target_state="TRAINING",
            priority="high"
        )
        assert handler._get_priority(cmd) == CommandPriority.HIGH
        
        # Default priority
        cmd_default = BlueprintUpdateCommand(
            layer_id="test",
            seed_id=0,
            blueprint_id="bp1",
            grafting_strategy="immediate"
        )
        assert handler._get_priority(cmd_default) == CommandPriority.NORMAL
        
    @pytest.mark.asyncio
    async def test_get_stats(self, handler):
        """Test statistics retrieval."""
        handler._stats = {
            "commands_received": 10,
            "commands_executed": 8,
            "commands_failed": 1,
            "commands_timeout": 1,
            "average_execution_ms": 50.0
        }
        handler.command_queue = Mock(qsize=Mock(return_value=3))
        handler.active_commands = {"cmd1": None, "cmd2": None}
        
        stats = await handler.get_stats()
        
        assert stats["commands_received"] == 10
        assert stats["commands_executed"] == 8
        assert stats["queue_size"] == 3
        assert stats["active_commands"] == 2


class TestCommandHandlerFactory:
    """Test CommandHandlerFactory."""
    
    def test_create_basic(self):
        """Test creating basic handler."""
        layer_registry = {"test": Mock()}
        
        handler = CommandHandlerFactory.create(layer_registry)
        
        assert isinstance(handler, CommandHandler)
        assert len(handler.processors) > 0
        
    def test_create_with_custom_processors(self):
        """Test creating handler with custom processors."""
        layer_registry = {"test": Mock()}
        
        # Create custom processor
        custom_proc = Mock()
        custom_proc.can_handle = AsyncMock(return_value=True)
        
        handler = CommandHandlerFactory.create(
            layer_registry,
            custom_processors=[custom_proc]
        )
        
        # Should include custom processor
        assert custom_proc in handler.processors
        # Should still have batch processor at end
        assert isinstance(handler.processors[-1], BatchCommandProcessor)