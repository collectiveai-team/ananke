"""Tests for benchmarking modules."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.runners.sklearn_runner import SklearnRunConfig, SklearnRunner
from ananke.experiments.base import BaseExperiment


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data_configs():
    """Create sample data configurations."""
    return [
        DataConfig(
            name="config_1",
            features=["feature_1", "feature_2"],
            target_feature="target",
            feature_window_hours=24,
            target_window_hours=6,
            train_stride_hours=6,
            test_stride_hours=6,
            future_feature_cols=["feature_1"],
            past_feature_cols=["feature_1", "feature_2"],
            target_cols=["target"],
            preprocessing={
                "scaler": {"type": "standard", "params": {}},
                "imputation": {"enabled": False},
                "smoothing": {"enabled": False},
            },
        ),
        DataConfig(
            name="config_2",
            features=["feature_1", "feature_2", "feature_3"],
            target_feature="target",
            feature_window_hours=48,
            target_window_hours=12,
            train_stride_hours=12,
            test_stride_hours=12,
            future_feature_cols=["feature_1", "feature_2"],
            past_feature_cols=["feature_1", "feature_2", "feature_3"],
            target_cols=["target"],
            preprocessing={
                "scaler": {"type": "minmax", "params": {}},
                "imputation": {"enabled": True, "method": "mean"},
                "smoothing": {"enabled": True, "method": "ewm", "params": {"span": 3}},
            },
        ),
    ]


@pytest.fixture
def sample_model_configs():
    """Create sample model configurations."""
    return [
        ModelConfig(
            name="random_forest",
            model_class="sklearn.ensemble.RandomForestRegressor",
            model_params={
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42,
            },
            type="sklearn",
        ),
        ModelConfig(
            name="linear_regression",
            model_class="sklearn.linear_model.LinearRegression",
            model_params={},
            type="sklearn",
        ),
    ]


class ConcreteExperiment(BaseExperiment):
    """Concrete implementation of BaseExperiment for testing."""

    def __init__(self, temp_dir, data_configs, model_configs):
        super().__init__(
            name="test_experiment",
            description="Test experiment for unit testing",
            output_dir=temp_dir,
            mlflow_experiment_name="test_experiment",
        )
        self._data_configs = data_configs
        self._model_configs = model_configs

    def setup_data_configs(self):
        return self._data_configs

    def setup_model_configs(self):
        return self._model_configs

    def create_runner(self, data_config, model_config):
        run_config = SklearnRunConfig(
            name=f"{data_config.name}_{model_config.name}",
            output_dir=self.output_dir / "runs",
            log_to_mlflow=False,  # Disable MLflow for testing
        )
        return SklearnRunner(run_config, data_config, model_config)


class TestBaseExperiment:
    """Test cases for BaseExperiment."""

    def test_init(self, temp_dir):
        """Test BaseExperiment initialization."""
        experiment = ConcreteExperiment(temp_dir, [], [])

        assert experiment.name == "test_experiment"
        assert experiment.description == "Test experiment for unit testing"
        assert str(experiment.output_dir) == temp_dir
        assert experiment.mlflow_experiment_name == "test_experiment"
        assert experiment.results == {}

    def test_abstract_methods(self):
        """Test that BaseExperiment abstract methods must be implemented."""
        with pytest.raises(TypeError):
            BaseExperiment(
                name="test",
                description="test",
                output_dir="test",
                mlflow_experiment_name="test",
            )

    @patch("ananke.experiments.base.mlflow")
    def test_setup_mlflow(
        self, mock_mlflow, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test MLflow setup."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        experiment._setup_mlflow()

        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    def test_setup_output_directory(
        self, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test output directory setup."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        experiment._setup_output_directory()

        # Check that directories are created
        assert (experiment.output_dir / "runs").exists()
        assert (experiment.output_dir / "results").exists()
        assert (experiment.output_dir / "logs").exists()

    def test_generate_run_combinations(
        self, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test run combination generation."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        combinations = experiment._generate_run_combinations()

        # Should have 2 data configs × 2 model configs = 4 combinations
        assert len(combinations) == 4

        # Check combination structure
        for combo in combinations:
            assert "data_config" in combo
            assert "model_config" in combo
            assert "run_name" in combo
            assert isinstance(combo["data_config"], DataConfig)
            assert isinstance(combo["model_config"], ModelConfig)

    @patch("ananke.experiments.base.mlflow")
    def test_run_single_experiment(
        self, mock_mlflow, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test running a single experiment."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Mock the runner
        mock_runner = Mock()
        mock_runner.run.return_value = {
            "status": "completed",
            "train_mse": 0.1,
            "test_mse": 0.15,
            "runtime_seconds": 10.5,
        }

        with patch.object(experiment, "create_runner", return_value=mock_runner):
            result = experiment._run_single_experiment(
                sample_data_configs[0], sample_model_configs[0], "test_run"
            )

        assert result["status"] == "completed"
        assert "train_mse" in result
        assert "test_mse" in result
        assert "runtime_seconds" in result
        mock_runner.run.assert_called_once()

    @patch("ananke.experiments.base.mlflow")
    def test_run_single_experiment_with_error(
        self, mock_mlflow, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test handling errors in single experiment."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Mock the runner to raise an exception
        mock_runner = Mock()
        mock_runner.run.side_effect = Exception("Test error")

        with patch.object(experiment, "create_runner", return_value=mock_runner):
            result = experiment._run_single_experiment(
                sample_data_configs[0], sample_model_configs[0], "test_run"
            )

        assert "error" in result
        assert result["error"] == "Test error"

    @patch("ananke.experiments.base.mlflow")
    def test_run_full_experiment(
        self, mock_mlflow, temp_dir, sample_data_configs, sample_model_configs
    ):
        """Test running full experiment."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Mock successful runs
        mock_results = {
            "status": "completed",
            "train_mse": 0.1,
            "test_mse": 0.15,
            "runtime_seconds": 10.5,
        }

        with patch.object(
            experiment, "_run_single_experiment", return_value=mock_results
        ):
            results = experiment.run()

        # Should have results for all combinations
        assert len(results) == 4

        # Check result structure
        for run_name, result in results.items():
            assert isinstance(run_name, str)
            assert "status" in result
            assert result["status"] == "completed"

    def test_get_best_run(self, temp_dir, sample_data_configs, sample_model_configs):
        """Test getting best run from results."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Set up mock results
        experiment.results = {
            "run_1": {"test_mse": 0.2, "test_r2": 0.8},
            "run_2": {"test_mse": 0.1, "test_r2": 0.9},
            "run_3": {"test_mse": 0.3, "test_r2": 0.7},
        }

        # Test getting best by MSE (ascending)
        best_run = experiment.get_best_run("test_mse", ascending=True)
        assert best_run["run_name"] == "run_2"
        assert best_run["results"]["test_mse"] == 0.1

        # Test getting best by R2 (descending)
        best_run = experiment.get_best_run("test_r2", ascending=False)
        assert best_run["run_name"] == "run_2"
        assert best_run["results"]["test_r2"] == 0.9

        # Test with non-existent metric
        best_run = experiment.get_best_run("non_existent_metric")
        assert best_run is None

    def test_compare_runs(self, temp_dir, sample_data_configs, sample_model_configs):
        """Test comparing runs."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Set up mock results
        experiment.results = {
            "run_1": {"test_mse": 0.2, "test_r2": 0.8, "runtime_seconds": 10},
            "run_2": {"test_mse": 0.1, "test_r2": 0.9, "runtime_seconds": 15},
            "run_3": {"test_mse": 0.3, "test_r2": 0.7, "runtime_seconds": 8},
        }

        comparison_df = experiment.compare_runs(
            ["test_mse", "test_r2", "runtime_seconds"]
        )

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert "run_name" in comparison_df.columns
        assert "test_mse" in comparison_df.columns
        assert "test_r2" in comparison_df.columns
        assert "runtime_seconds" in comparison_df.columns

        # Check sorting (should be by test_mse ascending by default)
        assert comparison_df.iloc[0]["run_name"] == "run_2"
        assert comparison_df.iloc[0]["test_mse"] == 0.1

    def test_save_results(self, temp_dir, sample_data_configs, sample_model_configs):
        """Test saving results to file."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Set up mock results
        experiment.results = {
            "run_1": {"test_mse": 0.2, "test_r2": 0.8},
            "run_2": {"test_mse": 0.1, "test_r2": 0.9},
        }

        # Save results
        experiment.save_results()

        # Check that results file was created
        results_file = experiment.output_dir / "results" / "experiment_results.json"
        assert results_file.exists()

        # Check that comparison CSV was created
        comparison_file = experiment.output_dir / "results" / "run_comparison.csv"
        assert comparison_file.exists()

    def test_cleanup(self, temp_dir, sample_data_configs, sample_model_configs):
        """Test experiment cleanup."""
        experiment = ConcreteExperiment(
            temp_dir, sample_data_configs, sample_model_configs
        )

        # Create some temporary files
        experiment._setup_output_directory()
        temp_file = experiment.output_dir / "temp_file.txt"
        temp_file.write_text("temporary content")

        # Cleanup should not remove the output directory by default
        experiment.cleanup()

        # Directory should still exist
        assert experiment.output_dir.exists()


class TestExperimentIntegration:
    """Integration tests for experiment functionality."""

    @patch("ananke.experiments.base.mlflow")
    def test_end_to_end_experiment(self, mock_mlflow, temp_dir):
        """Test complete end-to-end experiment workflow."""
        # Create simple configs
        data_configs = [
            DataConfig(
                name="simple_data",
                features=["feature_1"],
                target_feature="target",
                feature_window_hours=24,
                target_window_hours=6,
                train_stride_hours=6,
                test_stride_hours=6,
                future_feature_cols=[],
                past_feature_cols=["feature_1"],
                target_cols=["target"],
                preprocessing={
                    "scaler": {"type": "none", "params": {}},
                    "imputation": {"enabled": False},
                    "smoothing": {"enabled": False},
                },
            )
        ]

        model_configs = [
            ModelConfig(
                name="simple_lr",
                model_class="sklearn.linear_model.LinearRegression",
                model_params={},
                type="sklearn",
            )
        ]

        experiment = ConcreteExperiment(temp_dir, data_configs, model_configs)

        # Mock the runner to return successful results
        mock_runner = Mock()
        mock_runner.run.return_value = {
            "status": "completed",
            "train_mse": 0.1,
            "test_mse": 0.15,
            "train_r2": 0.9,
            "test_r2": 0.85,
            "runtime_seconds": 5.0,
        }

        with patch.object(experiment, "create_runner", return_value=mock_runner):
            # Run experiment
            results = experiment.run()

            # Verify results
            assert len(results) == 1
            run_name = list(results.keys())[0]
            assert "simple_data_simple_lr" in run_name

            result = results[run_name]
            assert result["status"] == "completed"
            assert "test_mse" in result
            assert "test_r2" in result

            # Test analysis methods
            best_run = experiment.get_best_run("test_mse")
            assert best_run is not None
            assert best_run["run_name"] == run_name

            comparison_df = experiment.compare_runs(["test_mse", "test_r2"])
            assert len(comparison_df) == 1

            # Save and verify files
            experiment.save_results()
            assert (Path(temp_dir) / "results" / "experiment_results.json").exists()
            assert (Path(temp_dir) / "results" / "run_comparison.csv").exists()

    def test_experiment_with_failures(self, temp_dir):
        """Test experiment handling partial failures."""
        data_configs = [
            DataConfig(
                name="data_1",
                features=["feature_1"],
                target_feature="target",
                feature_window_hours=24,
                target_window_hours=6,
                train_stride_hours=6,
                test_stride_hours=6,
                future_feature_cols=[],
                past_feature_cols=["feature_1"],
                target_cols=["target"],
                preprocessing={
                    "scaler": {"type": "none", "params": {}},
                    "imputation": {"enabled": False},
                    "smoothing": {"enabled": False},
                },
            )
        ]

        model_configs = [
            ModelConfig(
                name="model_1",
                model_class="sklearn.linear_model.LinearRegression",
                model_params={},
                type="sklearn",
            ),
            ModelConfig(
                name="model_2",
                model_class="sklearn.ensemble.RandomForestRegressor",
                model_params={"n_estimators": 5},
                type="sklearn",
            ),
        ]

        experiment = ConcreteExperiment(temp_dir, data_configs, model_configs)

        # Mock one success and one failure
        def mock_run_side_effect(*args):
            if "model_1" in args[2]:  # run_name
                return {"status": "completed", "test_mse": 0.1}
            else:
                raise Exception("Model 2 failed")

        with patch.object(
            experiment, "_run_single_experiment", side_effect=mock_run_side_effect
        ):
            results = experiment.run()

        # Should have results for both runs (one success, one error)
        assert len(results) == 2

        # Check that one succeeded and one failed
        success_count = sum(
            1 for r in results.values() if r.get("status") == "completed"
        )
        error_count = sum(1 for r in results.values() if "error" in r)

        assert success_count == 1
        assert error_count == 1
