"""
Tests for enum definitions and validation.
"""

import time

from esper.contracts.enums import (
    BlueprintState,
    BlueprintStatus,
    ComponentType,
    KernelStatus,
    SeedState,
    SystemHealth,
)


class TestSeedState:
    """Test cases for SeedState enum."""

    def test_seed_state_values(self):
        """Test SeedState enum values."""
        assert SeedState.DORMANT.value == "dormant"
        assert SeedState.GERMINATED.value == "germinated"
        assert SeedState.TRAINING.value == "training"
        assert SeedState.GRAFTING.value == "grafting"
        assert SeedState.FOSSILIZED.value == "fossilized"
        assert SeedState.CULLED.value == "culled"

    def test_seed_state_iteration(self):
        """Test SeedState enum iteration."""
        states = list(SeedState)
        expected_states = [
            SeedState.DORMANT,
            SeedState.GERMINATED,
            SeedState.TRAINING,
            SeedState.GRAFTING,
            SeedState.FOSSILIZED,
            SeedState.CULLED,
        ]
        assert states == expected_states

    def test_seed_state_serialization_performance(self):
        """Test SeedState serialization performance."""
        states = [SeedState.DORMANT, SeedState.TRAINING, SeedState.GERMINATED]

        start_time = time.perf_counter()
        for _ in range(10000):
            for state in states:
                # Test value access and string conversion
                value = state.value
                str_repr = str(state)
                assert isinstance(value, str)
                assert isinstance(str_repr, str)
        elapsed = time.perf_counter() - start_time

        # Should be very fast for enum operations
        assert elapsed < 0.1, f"Enum operations took {elapsed:.3f}s, expected <0.1s"


class TestBlueprintState:
    """Test cases for BlueprintState enum."""

    def test_blueprint_state_values(self):
        """Test BlueprintState enum values."""
        assert BlueprintState.PROPOSED.value == "proposed"
        assert BlueprintState.COMPILING.value == "compiling"
        assert BlueprintState.VALIDATING.value == "validating"
        assert BlueprintState.CHARACTERIZED.value == "characterized"
        assert BlueprintState.DEPLOYED.value == "deployed"
        assert BlueprintState.FAILED.value == "failed"
        assert BlueprintState.ARCHIVED.value == "archived"

    def test_blueprint_state_progression(self):
        """Test typical blueprint state progression."""
        states = [
            BlueprintState.PROPOSED,
            BlueprintState.COMPILING,
            BlueprintState.VALIDATING,
            BlueprintState.CHARACTERIZED,
            BlueprintState.DEPLOYED,
        ]

        # Test that states can be compared
        for state in states:
            assert isinstance(state.value, str)
            assert len(state.value) > 0

    def test_blueprint_state_serialization(self):
        """Test BlueprintState serialization."""
        for state in BlueprintState:
            # Test that each state can be serialized and maintains its value
            value = state.value
            assert isinstance(value, str)
            assert BlueprintState(value) == state


class TestBlueprintStatus:
    """Test cases for BlueprintStatus enum."""

    def test_blueprint_status_values(self):
        """Test BlueprintStatus enum values."""
        assert BlueprintStatus.UNVALIDATED.value == "unvalidated"
        assert BlueprintStatus.COMPILING.value == "compiling"
        assert BlueprintStatus.VALIDATED.value == "validated"
        assert BlueprintStatus.INVALID.value == "invalid"

    def test_blueprint_status_usage(self):
        """Test BlueprintStatus usage scenarios."""
        # Test common status checks
        status = BlueprintStatus.VALIDATED
        assert status == BlueprintStatus.VALIDATED
        assert status != BlueprintStatus.INVALID

        # Test that all statuses are strings
        for status in BlueprintStatus:
            assert isinstance(status.value, str)


class TestKernelStatus:
    """Test cases for KernelStatus enum."""

    def test_kernel_status_values(self):
        """Test KernelStatus enum values."""
        assert KernelStatus.VALIDATED.value == "validated"
        assert KernelStatus.INVALID.value == "invalid"
        assert KernelStatus.TESTING.value == "testing"
        assert KernelStatus.DEPLOYED.value == "deployed"

    def test_kernel_status_progression(self):
        """Test kernel status progression."""
        statuses = [
            KernelStatus.TESTING,
            KernelStatus.VALIDATED,
            KernelStatus.DEPLOYED,
        ]

        for status in statuses:
            assert isinstance(status.value, str)
            assert len(status.value) > 0


class TestSystemHealth:
    """Test cases for SystemHealth enum."""

    def test_system_health_values(self):
        """Test SystemHealth enum values."""
        assert SystemHealth.HEALTHY.value == "healthy"
        assert SystemHealth.DEGRADED.value == "degraded"
        assert SystemHealth.CRITICAL.value == "critical"
        assert SystemHealth.OFFLINE.value == "offline"

    def test_system_health_severity_ordering(self):
        """Test SystemHealth severity implications."""
        health_states = [
            SystemHealth.HEALTHY,
            SystemHealth.DEGRADED,
            SystemHealth.CRITICAL,
            SystemHealth.OFFLINE,
        ]

        # Each state should have a string value
        for health in health_states:
            assert isinstance(health.value, str)
            assert len(health.value) > 0

    def test_system_health_usage_scenarios(self):
        """Test common SystemHealth usage scenarios."""
        # Test that we can check health states
        current_health = SystemHealth.HEALTHY

        if current_health == SystemHealth.HEALTHY:
            is_healthy = True
        elif current_health == SystemHealth.DEGRADED:
            is_healthy = False
        elif current_health in [SystemHealth.CRITICAL, SystemHealth.OFFLINE]:
            is_healthy = False
        else:
            is_healthy = False

        assert is_healthy  # Should be healthy in test
        assert current_health != SystemHealth.CRITICAL
        assert current_health == SystemHealth.HEALTHY


class TestComponentType:
    """Test cases for ComponentType enum."""

    def test_component_type_values(self):
        """Test ComponentType enum values."""
        assert ComponentType.TAMIYO.value == "tamiyo"
        assert ComponentType.KARN.value == "karn"
        assert ComponentType.KASMINA.value == "kasmina"
        assert ComponentType.TEZZERET.value == "tezzeret"
        assert ComponentType.URABRASK.value == "urabrask"
        assert ComponentType.TOLARIA.value == "tolaria"
        assert ComponentType.URZA.value == "urza"
        assert ComponentType.OONA.value == "oona"
        assert ComponentType.NISSA.value == "nissa"
        assert ComponentType.SIMIC.value == "simic"
        assert ComponentType.EMRAKUL.value == "emrakul"

    def test_component_type_completeness(self):
        """Test that all expected components are present."""
        expected_components = {
            "tamiyo",  # Strategic Controller
            "karn",  # Generative Architect
            "kasmina",  # Execution Layer
            "tezzeret",  # Compilation Forge
            "urabrask",  # Evaluation Engine
            "tolaria",  # Training Orchestrator
            "urza",  # Central Library
            "oona",  # Message Bus
            "nissa",  # Observability
            "simic",  # Policy Training Environment
            "emrakul",  # Architectural Sculptor
        }

        actual_components = {comp.value for comp in ComponentType}
        assert actual_components == expected_components

    def test_component_type_usage(self):
        """Test ComponentType usage patterns."""
        # Test that component types can be used for identification
        for component in ComponentType:
            assert isinstance(component.value, str)
            assert len(component.value) > 0
            assert component.value.islower()  # Should be lowercase

        # Test specific component checks
        assert ComponentType.TAMIYO.value == "tamiyo"
        assert ComponentType.TOLARIA.value == "tolaria"


class TestEnumPerformance:
    """Performance tests for enum operations."""

    def test_enum_access_performance(self):
        """Test enum access performance across all enums."""
        enums_to_test = [
            (SeedState, list(SeedState)),
            (BlueprintState, list(BlueprintState)),
            (BlueprintStatus, list(BlueprintStatus)),
            (KernelStatus, list(KernelStatus)),
            (SystemHealth, list(SystemHealth)),
            (ComponentType, list(ComponentType)),
        ]

        for enum_class, enum_values in enums_to_test:
            start_time = time.perf_counter()

            # Test value access performance
            for _ in range(5000):
                for enum_val in enum_values:
                    _ = enum_val.value
                    _ = str(enum_val)
                    _ = repr(enum_val)

            elapsed = time.perf_counter() - start_time
            assert elapsed < 0.1, (
                f"{enum_class.__name__} access took {elapsed:.3f}s, expected <0.1s"
            )

    def test_enum_comparison_performance(self):
        """Test enum comparison performance."""
        start_time = time.perf_counter()

        for _ in range(10000):
            # Test various enum comparisons
            assert SeedState.DORMANT != SeedState.TRAINING  # Different values
            state = SeedState.TRAINING
            assert state == SeedState.TRAINING  # Same values
            assert BlueprintStatus.VALIDATED != BlueprintStatus.INVALID  # Different
            assert SystemHealth.HEALTHY != SystemHealth.CRITICAL

        elapsed = time.perf_counter() - start_time
        assert elapsed < 0.1, f"Enum comparisons took {elapsed:.3f}s, expected <0.1s"

    def test_enum_serialization_roundtrip_performance(self):
        """Test enum serialization round-trip performance."""
        test_values = [
            SeedState.TRAINING,
            BlueprintStatus.VALIDATED,
            SystemHealth.HEALTHY,
            ComponentType.TAMIYO,
        ]

        start_time = time.perf_counter()

        for _ in range(5000):
            for enum_val in test_values:
                # Serialize to value and back
                serialized = enum_val.value
                reconstructed = type(enum_val)(serialized)
                assert reconstructed == enum_val

        elapsed = time.perf_counter() - start_time
        assert elapsed < 0.1, (
            f"Enum serialization round-trip took {elapsed:.3f}s, expected <0.1s"
        )
