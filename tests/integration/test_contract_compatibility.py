"""
Integration tests for contract compatibility across Esper system.

This module validates that all contract models work together properly
and maintain compatibility across different service boundaries.
"""

import json

from esper.contracts.assets import Blueprint
from esper.contracts.assets import Seed
from esper.contracts.assets import TrainingSession
from esper.contracts.enums import BlueprintState
from esper.contracts.enums import SeedState
from esper.contracts.messages import BlueprintSubmitted
from esper.contracts.messages import OonaMessage
from esper.contracts.messages import TopicNames
from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import HealthSignal
from esper.contracts.operational import SystemStatePacket


class TestCrossContractCompatibility:
    """Test cross-contract compatibility and integration scenarios."""

    def test_assets_to_messages_integration(self):
        """Test that asset models integrate properly with message contracts."""
        # Create a blueprint
        blueprint = Blueprint(
            name="integration-test-bp",
            description="Blueprint for integration testing",
            architecture={"type": "Linear", "units": 128},
            hyperparameters={"lr": 0.001, "dropout": 0.1},
            created_by="integration-test",
        )

        # Create blueprint submission message
        blueprint_submitted = BlueprintSubmitted(
            blueprint_id=str(blueprint.blueprint_id),
            submitted_by=blueprint.created_by,
        )

        # Create Oona message envelope
        oona_message = OonaMessage(
            sender_id="Tamiyo-Controller",
            trace_id="test-trace-123",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload=blueprint_submitted.model_dump(),
        )

        # Verify serialization compatibility
        message_json = oona_message.model_dump_json()
        reconstructed_message = OonaMessage.model_validate_json(message_json)

        assert reconstructed_message.sender_id == "Tamiyo-Controller"
        assert reconstructed_message.topic == TopicNames.COMPILATION_BLUEPRINT_SUBMITTED
        assert reconstructed_message.payload["blueprint_id"] == str(
            blueprint.blueprint_id
        )
        assert reconstructed_message.payload["submitted_by"] == blueprint.created_by

    def test_assets_to_operational_integration(self):
        """Test that asset models integrate with operational contracts."""
        # Create training session with multiple assets
        seeds = [
            Seed(layer_id=i, position=j, state=SeedState.TRAINING)
            for i in range(3)
            for j in range(2)
        ]

        blueprints = [
            Blueprint(
                name=f"bp-{i}",
                description=f"Blueprint {i}",
                architecture={"layers": i + 1, "units": 64 * (i + 1)},
                performance_metrics={"accuracy": 0.9 + (i * 0.02)},
                created_by="test-system",
            )
            for i in range(2)
        ]

        session = TrainingSession(
            name="operational-integration-test",
            description="Testing operational integration",
            training_model_config={"type": "TestNet", "layers": 5},
            training_config={"epochs": 100, "batch_size": 32},
            seeds=seeds,
            blueprints=blueprints,
            status="running",
        )

        # Create health signals from seeds
        for seed in seeds[:2]:  # Just test a couple of seeds
            health_signal = HealthSignal(
                layer_id=seed.layer_id,
                seed_id=seed.position,  # Use position as seed_id for testing
                chunk_id=0,
                epoch=1,
                activation_variance=0.1 + (seed.layer_id * 0.05),
                dead_neuron_ratio=0.02,
                avg_correlation=0.8,
                is_ready_for_transition=False,
            )

            # Verify integration
            assert health_signal.layer_id == seed.layer_id
            assert health_signal.seed_id == seed.position
            assert health_signal.activation_variance > 0

        # Create system state from the training session
        active_seeds = session.get_active_seed_count()
        system_state = SystemStatePacket(
            epoch=1,
            total_seeds=len(session.seeds),
            active_seeds=active_seeds,
            training_loss=0.25,
            validation_loss=0.3,
            system_load=0.65,
            memory_usage=0.45,
        )

        # Verify integration
        assert system_state.total_seeds == 6
        assert system_state.active_seeds == 6  # All seeds are TRAINING
        assert system_state.system_health() == "Normal"

    def test_full_workflow_integration(self):
        """Test a complete workflow involving all contract types."""
        # 1. Create initial assets
        blueprint = Blueprint(
            name="workflow-test-bp",
            description="Blueprint for full workflow test",
            architecture={
                "type": "Sequential",
                "layers": [
                    {"type": "Linear", "in_features": 784, "out_features": 128},
                    {"type": "ReLU"},
                    {"type": "Linear", "in_features": 128, "out_features": 10},
                ],
            },
            hyperparameters={"lr": 0.001, "momentum": 0.9},
            created_by="workflow-test",
        )

        seed = Seed(
            layer_id=1,
            position=0,
            state=SeedState.DORMANT,
            blueprint_id=str(blueprint.blueprint_id),
        )

        # 2. Create training session
        session = TrainingSession(
            name="workflow-integration-session",
            description="Full workflow integration test",
            training_model_config={"type": "MLPClassifier", "input_size": 784},
            training_config={"epochs": 50, "batch_size": 64},
            seeds=[seed],
            blueprints=[blueprint],
        )

        # 3. Simulate blueprint submission
        blueprint_submitted = BlueprintSubmitted(
            blueprint_id=str(blueprint.blueprint_id),
            submitted_by=blueprint.created_by,
        )

        submission_message = OonaMessage(
            sender_id="Tamiyo-Controller",
            trace_id="workflow-trace-456",
            topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            payload=blueprint_submitted.model_dump(),
        )

        # 4. Update blueprint state after compilation
        blueprint.state = BlueprintState.CHARACTERIZED
        blueprint.performance_metrics = {"compile_time": 23.5, "size_mb": 8.2}

        # 5. Activate seed after blueprint is ready
        seed.state = SeedState.TRAINING

        # 6. Create health signals from active seed
        health_signal = HealthSignal(
            layer_id=seed.layer_id,
            seed_id=seed.position,  # Use position as seed_id for testing
            chunk_id=0,
            epoch=5,
            activation_variance=0.08,
            dead_neuron_ratio=0.01,
            avg_correlation=0.9,
            is_ready_for_transition=True,
        )

        # 7. Update session status
        session.status = "running"

        # 8. Create system state snapshot
        system_state = SystemStatePacket(
            epoch=5,
            total_seeds=1,
            active_seeds=session.get_active_seed_count(),
            training_loss=0.08,
            validation_loss=0.12,
            system_load=0.45,
            memory_usage=0.35,
        )

        # 9. Create adaptation decision for testing
        adaptation_decision = AdaptationDecision(
            layer_name="layer_1",
            adaptation_type="optimize_parameters",
            confidence=0.85,
            urgency=0.6,
            metadata={"reason": "workflow_test"},
        )

        # Verify the complete workflow
        assert blueprint.state == BlueprintState.CHARACTERIZED
        assert blueprint.is_ready_for_deployment()
        assert seed.is_active()
        assert session.get_active_seed_count() == 1
        assert submission_message.topic == TopicNames.COMPILATION_BLUEPRINT_SUBMITTED
        assert health_signal.layer_id == seed.layer_id
        assert system_state.total_seeds == 1
        assert system_state.active_seeds == 1
        assert system_state.system_health() == "Normal"
        assert abs(adaptation_decision.confidence - 0.85) < 0.001

        # Verify all contracts serialize/deserialize properly together
        full_state = {
            "blueprint": json.loads(blueprint.model_dump_json()),
            "seed": json.loads(seed.model_dump_json()),
            "session": json.loads(session.model_dump_json()),
            "blueprint_submitted": json.loads(blueprint_submitted.model_dump_json()),
            "submission_message": json.loads(submission_message.model_dump_json()),
            "health_signal": json.loads(health_signal.model_dump_json()),
            "system_state": json.loads(system_state.model_dump_json()),
            "adaptation_decision": json.loads(adaptation_decision.model_dump_json()),
        }

        # Ensure all objects can be reconstructed
        reconstructed_blueprint = Blueprint.model_validate(full_state["blueprint"])
        reconstructed_seed = Seed.model_validate(full_state["seed"])
        reconstructed_session = TrainingSession.model_validate(full_state["session"])
        reconstructed_message = OonaMessage.model_validate(
            full_state["submission_message"]
        )

        assert reconstructed_blueprint.blueprint_id == blueprint.blueprint_id
        assert reconstructed_seed.seed_id == seed.seed_id
        assert reconstructed_session.session_id == session.session_id
        assert reconstructed_message.trace_id == "workflow-trace-456"

    def test_contract_version_compatibility(self):
        """Test that contracts maintain backward compatibility."""
        # Create objects with minimal required fields
        minimal_blueprint = Blueprint(
            name="minimal", description="minimal", architecture={}, created_by="test"
        )

        minimal_seed = Seed(layer_id=1, position=0)

        minimal_session = TrainingSession(
            name="minimal",
            description="minimal",
            training_model_config={},
            training_config={},
        )

        # Serialize and ensure they can be reconstructed
        blueprint_data = minimal_blueprint.model_dump()
        seed_data = minimal_seed.model_dump()
        session_data = minimal_session.model_dump()

        # Verify required fields are present
        assert "blueprint_id" in blueprint_data
        assert "created_at" in blueprint_data
        assert "state" in blueprint_data

        assert "seed_id" in seed_data
        assert "created_at" in seed_data
        assert "state" in seed_data

        assert "session_id" in session_data
        assert "created_at" in session_data
        assert "status" in session_data

        # Ensure reconstruction works
        Blueprint.model_validate(blueprint_data)
        Seed.model_validate(seed_data)
        TrainingSession.model_validate(session_data)

    def test_message_bus_integration(self):
        """Test integration through the message bus system."""
        # Create a blueprint
        blueprint = Blueprint(
            name="message-bus-test",
            description="Testing message bus integration",
            architecture={"type": "Transformer", "layers": 12},
            created_by="message-test",
        )

        # Create various message types
        blueprint_submitted = BlueprintSubmitted(
            blueprint_id=str(blueprint.blueprint_id),
            submitted_by=blueprint.created_by,
        )

        # Create messages for different topics
        messages = [
            OonaMessage(
                sender_id="Tamiyo-Controller",
                trace_id="msg-trace-1",
                topic=TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
                payload=blueprint_submitted.model_dump(),
            ),
            OonaMessage(
                sender_id="Tezzeret-Worker-1",
                trace_id="msg-trace-1",
                topic=TopicNames.COMPILATION_KERNEL_READY,
                payload={
                    "blueprint_id": str(blueprint.blueprint_id),
                    "status": "ready",
                },
            ),
            OonaMessage(
                sender_id="Urabrask-Evaluator",
                trace_id="msg-trace-1",
                topic=TopicNames.VALIDATION_KERNEL_CHARACTERIZED,
                payload={
                    "blueprint_id": str(blueprint.blueprint_id),
                    "metrics": {"accuracy": 0.95},
                },
            ),
        ]

        # Verify all messages can be serialized and reconstructed
        for message in messages:
            json_data = message.model_dump_json()
            reconstructed = OonaMessage.model_validate_json(json_data)

            assert reconstructed.trace_id == "msg-trace-1"
            assert "blueprint_id" in reconstructed.payload
            assert reconstructed.payload["blueprint_id"] == str(blueprint.blueprint_id)

        # Verify topic consistency
        topics = [msg.topic for msg in messages]
        expected_topics = [
            TopicNames.COMPILATION_BLUEPRINT_SUBMITTED,
            TopicNames.COMPILATION_KERNEL_READY,
            TopicNames.VALIDATION_KERNEL_CHARACTERIZED,
        ]
        assert topics == expected_topics
