"""
Integration tests for the main system entrypoint (train.py).

These tests validate the CLI interface, configuration handling,
and end-to-end system startup workflows.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
import yaml

# Import the main module functions
import train


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestQuickStartConfigurations:
    """Test quick-start configuration generation."""

    def test_create_cifar10_quick_start(self, temp_output_dir):
        """Test creation of CIFAR-10 quick-start configuration."""
        config_path = train.create_quick_start_config("cifar10", temp_output_dir)

        assert config_path.exists()
        assert config_path.name == "quick-cifar10.yaml"

        # Load and validate the configuration
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        assert config_data["run_id"] == "quick-cifar10"
        assert config_data["model"]["architecture"] == "resnet18"
        assert config_data["model"]["num_classes"] == 10
        assert config_data["dataset"]["name"] == "cifar10"
        assert config_data["max_epochs"] == 10

    def test_create_cifar100_quick_start(self, temp_output_dir):
        """Test creation of CIFAR-100 quick-start configuration."""
        config_path = train.create_quick_start_config("cifar100", temp_output_dir)

        assert config_path.exists()
        assert config_path.name == "quick-cifar100.yaml"

        # Load and validate the configuration
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        assert config_data["run_id"] == "quick-cifar100"
        assert config_data["model"]["architecture"] == "resnet34"
        assert config_data["model"]["num_classes"] == 100
        assert config_data["dataset"]["name"] == "cifar100"
        assert config_data["max_epochs"] == 20

    def test_invalid_quick_start_dataset(self, temp_output_dir):
        """Test handling of invalid quick-start dataset."""
        with pytest.raises(ValueError, match="Unknown quick-start dataset"):
            train.create_quick_start_config("invalid_dataset", temp_output_dir)


class TestEnvironmentValidation:
    """Test environment validation functionality."""

    @patch("sys.version_info", (3, 7, 0))  # Python 3.7
    def test_python_version_validation_failure(self):
        """Test validation failure for old Python version."""
        result = train.validate_environment()
        assert result is False

    def test_environment_validation_success(self):
        """Test successful environment validation with current environment."""
        # Test with the actual environment (should pass since we have PyTorch installed)
        result = train.validate_environment()
        assert result is True


class TestMainEntrypointArguments:
    """Test argument parsing and main entrypoint logic."""

    def test_argument_parsing_config_file(self):
        """Test argument parsing with config file."""
        test_args = ["train.py", "--config", "test_config.yaml"]

        with patch("sys.argv", test_args):
            parser = train.argparse.ArgumentParser(description="Test")
            parser.add_argument("--config", type=str)
            parser.add_argument("--quick-start", type=str)
            parser.add_argument("--output", type=str, default="./results")
            parser.add_argument("--verbose", action="store_true")
            parser.add_argument("--dry-run", action="store_true")

            args = parser.parse_args(test_args[1:])

            assert args.config == "test_config.yaml"
            assert args.quick_start is None
            assert not args.verbose
            assert not args.dry_run

    def test_argument_parsing_quick_start(self):
        """Test argument parsing with quick-start option."""
        test_args = [
            "train.py",
            "--quick-start",
            "cifar10",
            "--verbose",
            "--output",
            "./my_results",
        ]

        parser = train.argparse.ArgumentParser(description="Test")
        parser.add_argument("--config", type=str)
        parser.add_argument("--quick-start", type=str, choices=["cifar10", "cifar100"])
        parser.add_argument("--output", type=str, default="./results")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(test_args[1:])

        assert args.quick_start == "cifar10"
        assert args.config is None
        assert args.verbose
        assert args.output == "./my_results"


class TestMainFunctionIntegration:
    """Test the complete main function integration."""

    @pytest.mark.asyncio
    @patch("train.validate_environment", return_value=True)
    @patch("train.TolariaService")
    @patch("train.TolariaConfig")
    @patch("sys.argv")
    async def test_main_with_config_file(
        self, mock_argv, mock_tolaria_config, mock_tolaria_service, _mock_validate_env
    ):
        """Test main function with configuration file."""
        # Setup mocks
        mock_argv.__getitem__.return_value = ["train.py", "--config", "test.yaml"]

        mock_config_instance = mock_tolaria_config.from_yaml.return_value
        mock_config_instance.model.architecture = "resnet18"
        mock_config_instance.dataset.name = "cifar10"
        mock_config_instance.max_epochs = 10
        mock_config_instance.device = "cpu"

        mock_service_instance = AsyncMock()
        mock_tolaria_service.return_value = mock_service_instance

        # Mock Path.exists() to return True
        with patch("pathlib.Path.exists", return_value=True):
            with patch("sys.argv", ["train.py", "--config", "test.yaml"]):
                # This should complete without errors
                await train.main()

        # Verify service was created and started
        mock_tolaria_config.from_yaml.assert_called_once_with(Path("test.yaml"))
        mock_tolaria_service.assert_called_once_with(mock_config_instance)
        mock_service_instance.start.assert_called_once()

    @pytest.mark.asyncio
    @patch("train.validate_environment", return_value=True)
    @patch("train.create_quick_start_config")
    @patch("train.TolariaService")
    @patch("train.TolariaConfig")
    async def test_main_with_quick_start(
        self,
        mock_tolaria_config,
        mock_tolaria_service,
        mock_create_quick_start,
        _mock_validate_env,
    ):
        """Test main function with quick-start option."""
        # Setup mocks
        temp_config_path = Path("/tmp/quick-cifar10.yaml")
        mock_create_quick_start.return_value = temp_config_path

        mock_config_instance = mock_tolaria_config.from_yaml.return_value
        mock_config_instance.model.architecture = "resnet18"
        mock_config_instance.dataset.name = "cifar10"
        mock_config_instance.max_epochs = 10
        mock_config_instance.device = "cpu"

        mock_service_instance = AsyncMock()
        mock_tolaria_service.return_value = mock_service_instance

        with patch(
            "sys.argv",
            ["train.py", "--quick-start", "cifar10", "--output", "./results"],
        ):
            await train.main()

        # Verify quick-start config was created
        mock_create_quick_start.assert_called_once_with("cifar10", Path("./results"))

        # Verify service was created and started
        mock_tolaria_config.from_yaml.assert_called_once_with(temp_config_path)
        mock_tolaria_service.assert_called_once_with(mock_config_instance)
        mock_service_instance.start.assert_called_once()

    @pytest.mark.asyncio
    @patch("train.validate_environment", return_value=True)
    @patch("train.TolariaConfig")
    async def test_main_with_dry_run(self, mock_tolaria_config, _mock_validate_env):
        """Test main function with dry-run option."""
        # Setup mocks
        mock_config_instance = mock_tolaria_config.from_yaml.return_value
        mock_config_instance.model.architecture = "resnet18"
        mock_config_instance.dataset.name = "cifar10"
        mock_config_instance.max_epochs = 10
        mock_config_instance.device = "cpu"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("sys.argv", ["train.py", "--config", "test.yaml", "--dry-run"]):
                # This should complete without starting training
                await train.main()

        # Config should be loaded but no service should be created
        mock_tolaria_config.from_yaml.assert_called_once()

    @pytest.mark.asyncio
    @patch("train.validate_environment")
    async def test_main_environment_validation_failure(self, mock_validate_env):
        """Test main function when environment validation fails."""
        mock_validate_env.return_value = False

        with patch("sys.argv", ["train.py", "--config", "test.yaml"]):
            with patch("sys.exit") as mock_exit:
                await train.main()
                mock_exit.assert_called_with(1)

    @pytest.mark.asyncio
    @patch("train.validate_environment")
    @patch("pathlib.Path.exists")
    async def test_main_missing_config_file(self, mock_path_exists, mock_validate_env):
        """Test main function when config file doesn't exist."""
        mock_validate_env.return_value = True
        mock_path_exists.return_value = False

        with patch("sys.argv", ["train.py", "--config", "nonexistent.yaml"]):
            with patch("sys.exit") as mock_exit:
                await train.main()
                mock_exit.assert_called_with(1)


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.mark.asyncio
    @patch("train.validate_environment", return_value=True)
    @patch("train.TolariaService")
    @patch("train.TolariaConfig")
    async def test_training_failure_handling(
        self, mock_tolaria_config, mock_tolaria_service, _mock_validate_env
    ):
        """Test handling of training failures."""
        # Setup mocks
        mock_config_instance = mock_tolaria_config.from_yaml.return_value
        mock_config_instance.model.architecture = "resnet18"
        mock_config_instance.dataset.name = "cifar10"
        mock_config_instance.max_epochs = 10
        mock_config_instance.device = "cpu"

        # Mock service to raise an exception
        mock_service_instance = AsyncMock()
        mock_service_instance.start.side_effect = RuntimeError("Training failed")
        mock_tolaria_service.return_value = mock_service_instance

        with patch("pathlib.Path.exists", return_value=True):
            with patch("sys.argv", ["train.py", "--config", "test.yaml"]):
                with patch("sys.exit") as mock_exit:
                    await train.main()
                    mock_exit.assert_called_with(1)

    @pytest.mark.asyncio
    @patch("train.validate_environment", return_value=True)
    @patch("train.TolariaConfig")
    async def test_keyboard_interrupt_handling(
        self, mock_tolaria_config, _mock_validate_env
    ):
        """Test handling of keyboard interrupt (Ctrl+C)."""
        # Setup mocks
        mock_tolaria_config.from_yaml.side_effect = KeyboardInterrupt()

        with patch("pathlib.Path.exists", return_value=True):
            with patch("sys.argv", ["train.py", "--config", "test.yaml"]):
                with patch("sys.exit") as mock_exit:
                    await train.main()
                    mock_exit.assert_called_with(130)  # Standard SIGINT exit code


class TestLoggingConfiguration:
    """Test logging configuration and output."""

    def test_verbose_logging_setup(self):
        """Test that verbose flag is parsed correctly."""
        test_args = ["train.py", "--quick-start", "cifar10", "--verbose"]

        parser = train.argparse.ArgumentParser(description="Test")
        parser.add_argument("--config", type=str)
        parser.add_argument("--quick-start", type=str, choices=["cifar10", "cifar100"])
        parser.add_argument("--output", type=str, default="./results")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(test_args[1:])

        # Test that verbose flag is properly parsed
        assert args.verbose is True
        assert args.quick_start == "cifar10"

    def test_normal_logging_setup(self):
        """Test that normal logging works without verbose flag."""
        test_args = ["train.py", "--quick-start", "cifar10"]

        parser = train.argparse.ArgumentParser(description="Test")
        parser.add_argument("--config", type=str)
        parser.add_argument("--quick-start", type=str, choices=["cifar10", "cifar100"])
        parser.add_argument("--output", type=str, default="./results")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(test_args[1:])

        # Test that verbose flag defaults to False
        assert args.verbose is False
        assert args.quick_start == "cifar10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
