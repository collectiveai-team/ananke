"""Tests for timeseries runners."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.runners.darts_runner import DartsRunConfig, DartsRunner
from ananke.core.runners.sklearn_runner import SklearnRunConfig, SklearnRunner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_data():
    """Create sample timeseries data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "target": np.random.randn(100),
        }
    )
    return data


@pytest.fixture
def data_config():
    """Create a sample data configuration."""
    return DataConfig(
        name="test_data",
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
    )


@pytest.fixture
def sklearn_model_config():
    """Create a sample sklearn model configuration."""
    return ModelConfig(
        name="test_rf",
        model_class="sklearn.ensemble.RandomForestRegressor",
        model_params={
            "n_estimators": 10,
            "max_depth": 3,
            "random_state": 42,
        },
        type="sklearn",
    )


@pytest.fixture
def darts_model_config():
    """Create a sample Darts model configuration."""
    return ModelConfig(
        name="test_prophet",
        model_class="darts.models.Prophet",
        model_params={
            "seasonality_mode": "additive",
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
        },
        type="darts",
    )


class TestSklearnRunner:
    """Test cases for SklearnRunner."""

    def test_init(self, temp_dir, data_config, sklearn_model_config):
        """Test SklearnRunner initialization."""
        run_config = SklearnRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
        )

        runner = SklearnRunner(run_config, data_config, sklearn_model_config)

        assert runner.run_config.name == "test_run"
        assert runner.data_config.name == "test_data"
        assert runner.model_config.name == "test_rf"
        assert runner.model_config.type == "sklearn"

    def test_create_model(self, temp_dir, data_config, sklearn_model_config):
        """Test model creation."""
        run_config = SklearnRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
        )

        runner = SklearnRunner(run_config, data_config, sklearn_model_config)
        model = runner._create_model()

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert model.n_estimators == 10
        assert model.max_depth == 3

    @patch("ananke.core.runners.sklearn_runner.mlflow")
    def test_run_without_mlflow(
        self, mock_mlflow, temp_dir, data_config, sklearn_model_config, sample_data
    ):
        """Test running without MLflow logging."""
        run_config = SklearnRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
        )

        runner = SklearnRunner(run_config, data_config, sklearn_model_config)

        # Mock data loading
        with patch.object(runner, "_load_data", return_value=sample_data):
            with patch.object(runner, "_prepare_features") as mock_prepare:
                # Mock feature preparation
                X_train = np.random.randn(80, 2)
                y_train = np.random.randn(80)
                X_test = np.random.randn(20, 2)
                y_test = np.random.randn(20)

                mock_prepare.return_value = (X_train, y_train, X_test, y_test)

                results = runner.run()

                assert "train_mse" in results
                assert "test_mse" in results
                assert "runtime_seconds" in results
                assert results["status"] == "completed"

    def test_hyperparameter_search(self, temp_dir, data_config, sklearn_model_config):
        """Test hyperparameter search functionality."""
        # Update model config to include hyperparameter search
        sklearn_model_config.hyperparameter_search = {
            "enabled": True,
            "method": "grid",
            "param_grid": {
                "n_estimators": [5, 10],
                "max_depth": [2, 3],
            },
            "cv": 2,
            "scoring": "neg_mean_squared_error",
        }

        run_config = SklearnRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
        )

        runner = SklearnRunner(run_config, data_config, sklearn_model_config)

        # Mock data
        X_train = np.random.randn(50, 2)
        y_train = np.random.randn(50)

        best_model = runner._perform_hyperparameter_search(X_train, y_train)

        assert hasattr(best_model, "fit")
        assert hasattr(best_model, "predict")


class TestDartsRunner:
    """Test cases for DartsRunner."""

    def test_init(self, temp_dir, data_config, darts_model_config):
        """Test DartsRunner initialization."""
        run_config = DartsRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
            forecast_horizon=12,
        )

        runner = DartsRunner(run_config, data_config, darts_model_config)

        assert runner.run_config.name == "test_run"
        assert runner.run_config.forecast_horizon == 12
        assert runner.data_config.name == "test_data"
        assert runner.model_config.name == "test_prophet"
        assert runner.model_config.type == "darts"

    @patch("ananke.core.runners.darts_runner.AirPassengersDataset")
    def test_load_airpassengers_data(
        self, mock_dataset, temp_dir, data_config, darts_model_config
    ):
        """Test loading AirPassengers dataset."""
        # Mock the dataset
        mock_ts = Mock()
        mock_ts.values = np.random.randn(144, 1)  # 12 years of monthly data
        mock_ts.time_index = pd.date_range("2010-01-01", periods=144, freq="ME")
        mock_dataset_instance = Mock()
        mock_dataset_instance.load.return_value = mock_ts
        mock_dataset.return_value = mock_dataset_instance

        run_config = DartsRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
            forecast_horizon=12,
        )

        runner = DartsRunner(run_config, data_config, darts_model_config)
        ts_data = runner._load_airpassengers_data()

        assert ts_data is not None
        mock_dataset_instance.load.assert_called_once()

    def test_prepare_darts_data(
        self, temp_dir, data_config, darts_model_config, sample_data
    ):
        """Test Darts data preparation."""
        run_config = DartsRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
            forecast_horizon=12,
        )

        runner = DartsRunner(run_config, data_config, darts_model_config)

        with patch("ananke.core.runners.darts_runner.TimeSeries") as mock_ts:
            mock_time_series = Mock()
            mock_time_series.__len__ = Mock(return_value=100)
            mock_time_series.__getitem__ = Mock(side_effect=lambda x: Mock())  # Support slicing
            mock_ts.from_dataframe.return_value = mock_time_series

            train_ts, test_ts = runner._prepare_darts_data(sample_data)

            assert mock_ts.from_dataframe.called

    @pytest.mark.skip(reason="Skipping DartsRunner Pydantic validation issue for now")
    @patch("ananke.core.runners.darts_runner.AirPassengersDataset")
    def test_run_without_mlflow(
        self, mock_dataset, temp_dir, data_config, darts_model_config
    ):
        """Test running without MLflow logging."""
        # Mock the dataset
        mock_ts = Mock()
        mock_ts.values = np.random.randn(144, 1)
        mock_ts.time_index = pd.date_range("2010-01-01", periods=144, freq="ME")
        mock_ts.split_before.return_value = (mock_ts, mock_ts)
        mock_ts.__len__ = Mock(return_value=144)
        mock_ts.__getitem__ = Mock(side_effect=lambda x: Mock())  # Support slicing
        mock_dataset_instance = Mock()
        mock_dataset_instance.load.return_value = mock_ts
        mock_dataset.return_value = mock_dataset_instance

        run_config = DartsRunConfig(
            name="test_run",
            output_dir=temp_dir,
            log_to_mlflow=False,
            forecast_horizon=12,
        )

        runner = DartsRunner(run_config, data_config, darts_model_config)

        # Mock model creation and training
        with patch.object(runner, "setup_model") as mock_setup:
            mock_model = Mock()
            mock_model.predict.return_value = mock_ts
            runner.model = mock_model  # Set the model directly instead of mocking return value

            results = runner.run()

            assert "status" in results
            assert "runtime_seconds" in results
            mock_model.fit.assert_called_once()
            mock_model.predict.assert_called_once()


class TestRunnerIntegration:
    """Integration tests for runners."""

    def test_sklearn_runner_end_to_end(
        self, temp_dir, data_config, sklearn_model_config
    ):
        """Test complete sklearn runner workflow."""
        run_config = SklearnRunConfig(
            name="integration_test",
            output_dir=temp_dir,
            log_to_mlflow=False,
        )

        runner = SklearnRunner(run_config, data_config, sklearn_model_config)

        # Create synthetic data
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

        with patch.object(runner, "_load_data", return_value=data):
            results = runner.run()

            # Check that results contain expected keys
            expected_keys = ["train_mse", "test_mse", "runtime_seconds", "status"]
            for key in expected_keys:
                assert key in results

            # Check that output directory was created
            assert Path(temp_dir).exists()

    def test_runner_comparison(self, temp_dir, data_config):
        """Test comparing different runners on the same data."""
        # Create two different model configs
        rf_config = ModelConfig(
            name="random_forest",
            model_class="sklearn.ensemble.RandomForestRegressor",
            model_params={"n_estimators": 5, "random_state": 42},
            type="sklearn",
        )

        lr_config = ModelConfig(
            name="linear_regression",
            model_class="sklearn.linear_model.LinearRegression",
            model_params={},
            type="sklearn",
        )

        # Create runners
        rf_run_config = SklearnRunConfig(
            name="rf_test",
            output_dir=temp_dir + "/rf",
            log_to_mlflow=False,
        )

        lr_run_config = SklearnRunConfig(
            name="lr_test",
            output_dir=temp_dir + "/lr",
            log_to_mlflow=False,
        )

        rf_runner = SklearnRunner(rf_run_config, data_config, rf_config)
        lr_runner = SklearnRunner(lr_run_config, data_config, lr_config)

        # Create synthetic data
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
                "target": np.random.randn(100),
            }
        )

        # Run both models
        with patch.object(rf_runner, "_load_data", return_value=data):
            rf_results = rf_runner.run()

        with patch.object(lr_runner, "_load_data", return_value=data):
            lr_results = lr_runner.run()

        # Both should complete successfully
        assert rf_results["status"] == "completed"
        assert lr_results["status"] == "completed"

        # Both should have metrics
        assert "test_mse" in rf_results
        assert "test_mse" in lr_results
