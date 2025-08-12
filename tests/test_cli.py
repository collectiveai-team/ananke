"""Tests for CLI functionality."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from ananke.cli import app


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestCLIInit:
    """Test cases for CLI init command."""

    def test_init_project(self, runner, temp_dir):
        """Test project initialization."""
        project_path = Path(temp_dir) / "test_project"

        result = runner.invoke(app, ["init", str(project_path)])

        assert result.exit_code == 0
        assert "Project initialized successfully" in result.stdout

        # Check that project structure was created
        assert project_path.exists()
        assert (project_path / "configs").exists()
        assert (project_path / "configs" / "data").exists()
        assert (project_path / "configs" / "model").exists()
        assert (project_path / "experiments").exists()
        assert (project_path / "data").exists()
        assert (project_path / "results").exists()

    def test_init_existing_project(self, runner, temp_dir):
        """Test initializing in existing directory."""
        project_path = Path(temp_dir) / "existing_project"
        project_path.mkdir()

        result = runner.invoke(app, ["init", str(project_path)])

        assert result.exit_code == 0
        # Should still work and create subdirectories
        assert (project_path / "configs").exists()

    @pytest.mark.skip(reason="Skipping --with-examples implementation for now")
    def test_init_with_examples(self, runner, temp_dir):
        """Test project initialization with examples."""
        project_path = Path(temp_dir) / "test_project_examples"

        result = runner.invoke(app, ["init", str(project_path), "--with-examples"])

        assert result.exit_code == 0
        assert "Project initialized successfully" in result.stdout

        # Check that example files were created
        assert (project_path / "configs" / "data" / "example_data.yaml").exists()
        assert (project_path / "configs" / "models" / "example_model.yaml").exists()


class TestCLIList:
    """Test cases for CLI list command."""

    def test_list_configs_empty_directory(self, runner, temp_dir):
        """Test listing configs in empty directory."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = runner.invoke(app, ["list-configs"])
        finally:
            os.chdir(original_cwd)

        # Should handle empty directory gracefully
        assert result.exit_code == 0

    def test_list_data_configs(self, runner, temp_dir):
        """Test listing data configurations."""
        # Create a mock data config file
        configs_dir = Path(temp_dir) / "configs" / "data"
        configs_dir.mkdir(parents=True)

        config_file = configs_dir / "test_data.yaml"
        config_file.write_text("""
name: test_data
features: [feature_1, feature_2]
target_feature: target
feature_window_hours: 24
target_window_hours: 6
""")

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = runner.invoke(app, ["list-configs"])
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "test_data" in result.stdout

    def test_list_model_configs(self, runner, temp_dir):
        """Test listing model configurations."""
        # Create a mock model config file
        configs_dir = Path(temp_dir) / "configs" / "model"
        configs_dir.mkdir(parents=True)

        config_file = configs_dir / "test_model.yaml"
        config_file.write_text("""
name: test_model
model_class: sklearn.ensemble.RandomForestRegressor
model_params:
  n_estimators: 100
  random_state: 42
type: sklearn
""")

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            result = runner.invoke(app, ["list-configs"])
        finally:
            os.chdir(original_cwd)

        assert result.exit_code == 0
        assert "test_model" in result.stdout


class TestCLIValidate:
    """Test cases for CLI validate command."""

    def test_validate_data_config_valid(self, runner, temp_dir):
        """Test validating a valid data configuration."""
        config_file = Path(temp_dir) / "valid_data.yaml"
        config_file.write_text("""
name: valid_data
features: [feature_1, feature_2]
target_feature: target
feature_window_hours: 24
target_window_hours: 6
train_stride_hours: 6
test_stride_hours: 6
future_feature_cols: [feature_1]
past_feature_cols: [feature_1, feature_2]
target_cols: [target]
preprocessing:
  scaler:
    type: standard
    params: {}
  imputation:
    enabled: false
  smoothing:
    enabled: false
""")

        result = runner.invoke(app, ["validate", "data-config", str(config_file)])

        assert result.exit_code == 0
        assert "configuration is valid" in result.stdout

    def test_validate_data_config_invalid(self, runner, temp_dir):
        """Test validating an invalid data configuration."""
        config_file = Path(temp_dir) / "invalid_data.yaml"
        config_file.write_text("""
name: invalid_data
# Missing required fields
features: [feature_1]
""")

        result = runner.invoke(app, ["validate", "data-config", str(config_file)])

        assert result.exit_code != 0
        assert "Validation failed" in result.stdout

    def test_validate_model_config_valid(self, runner, temp_dir):
        """Test validating a valid model configuration."""
        config_file = Path(temp_dir) / "valid_model.yaml"
        config_file.write_text("""
name: valid_model
model_class: sklearn.ensemble.RandomForestRegressor
model_params:
  n_estimators: 100
  random_state: 42
type: sklearn
""")

        result = runner.invoke(app, ["validate", "model-config", str(config_file)])

        assert result.exit_code == 0
        assert "configuration is valid" in result.stdout


class TestCLIRun:
    """Test cases for CLI run command."""

    @pytest.mark.skip(reason="Skipping CLI run-experiment argument handling for now")
    @patch("ananke.core.runners.sklearn_runner.SklearnRunner")
    @patch("ananke.core.configs.data.config.DataConfig")
    @patch("ananke.core.configs.model.config.ModelConfig")
    def test_run_experiment(
        self, mock_model_config, mock_data_config, mock_runner, runner, temp_dir
    ):
        """Test running an experiment."""
        # Setup mocks
        mock_data_config.from_yaml.return_value = Mock()
        mock_model_config.from_yaml.return_value = Mock()

        mock_runner_instance = Mock()
        mock_runner_instance.run.return_value = {
            "status": "completed",
            "test_mse": 0.1,
            "runtime_seconds": 10.0,
        }
        mock_runner.return_value = mock_runner_instance

        # Create config files
        data_config_file = Path(temp_dir) / "data.yaml"
        model_config_file = Path(temp_dir) / "model.yaml"

        data_config_file.write_text("name: test_data")
        model_config_file.write_text("name: test_model")

        result = runner.invoke(
            app,
            [
                "run-experiment",
                str(data_config_file),
            ],
        )

        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.stdout
        mock_runner_instance.run.assert_called_once()

    def test_run_experiment_missing_config(self, runner, temp_dir):
        """Test running experiment with missing configuration."""
        result = runner.invoke(
            app,
            [
                "run",
                "--data-config",
                "nonexistent.yaml",
                "--model-config",
                "nonexistent.yaml",
            ],
        )

        assert result.exit_code != 0


class TestCLIVersion:
    """Test cases for CLI version command."""

    def test_version(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "timeseries_jarvis" in result.stdout
        assert "version" in result.stdout.lower()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.mark.skip(reason="Skipping CLI integration workflow for now")
    def test_full_workflow(self, runner, temp_dir):
        """Test complete CLI workflow."""
        project_path = Path(temp_dir) / "full_workflow_project"

        # 1. Initialize project
        result = runner.invoke(app, ["init", str(project_path), "--with-examples"])
        assert result.exit_code == 0

        # 2. List configurations
        result = runner.invoke(app, ["list", "data-configs"], cwd=str(project_path))
        assert result.exit_code == 0

        result = runner.invoke(app, ["list", "model-configs"], cwd=str(project_path))
        assert result.exit_code == 0

        # 3. Validate configurations
        data_config_path = project_path / "configs" / "data" / "example_data.yaml"
        if data_config_path.exists():
            result = runner.invoke(
                app, ["validate", "data-config", str(data_config_path)]
            )
            assert result.exit_code == 0

        model_config_path = project_path / "configs" / "models" / "example_model.yaml"
        if model_config_path.exists():
            result = runner.invoke(
                app, ["validate", "model-config", str(model_config_path)]
            )
            assert result.exit_code == 0

    def test_error_handling(self, runner, temp_dir):
        """Test CLI error handling."""
        # Test with invalid command
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

        # Test with invalid options
        result = runner.invoke(app, ["init"])  # Missing required argument
        assert result.exit_code != 0

        # Test with invalid file paths
        result = runner.invoke(
            app, ["validate", "data-config", "/nonexistent/path.yaml"]
        )
        assert result.exit_code != 0


class TestCLIHelp:
    """Test cases for CLI help functionality."""

    def test_main_help(self, runner):
        """Test main help command."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "timeseries_jarvis" in result.stdout
        assert "init" in result.stdout
        assert "run" in result.stdout
        assert "validate" in result.stdout

    def test_subcommand_help(self, runner):
        """Test subcommand help."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "Initialize a new TSJ project" in result.stdout

        result = runner.invoke(app, ["run-experiment", "--help"])

        assert result.exit_code == 0
        assert "Run a timeseries experiment" in result.stdout

    def test_validate_help(self, runner):
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0
        assert "Validate configuration files" in result.stdout
