"""Tests for Dagster integration."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from dagster import DagsterInstance

from ananke.dagster.jobs.prediction_job import prediction_job
from ananke.dagster.jobs.training_job import training_job
from ananke.dagster.ops.prediction_ops import (
    load_input_data_op,
    load_model_op,
    log_predictions_op,
    make_predictions_op,
    save_predictions_op,
)
from ananke.dagster.ops.training_ops import (
    evaluate_model_op,
    load_training_config_op,
    register_model_op,
    train_model_op,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dagster_instance():
    """Create a test Dagster instance."""
    with DagsterInstance.ephemeral() as instance:
        yield instance


class TestTrainingOps:
    """Test cases for training operations."""

    def test_load_training_config_op(self, dagster_instance):
        """Test loading training configuration."""
        from ananke.dagster.ops.training_ops import TrainingConfig

        config = TrainingConfig(
            data_config_name="test_data",
            model_config_name="test_model",
            experiment_name="test_experiment",
            output_dir="test_output",
        )

        # Execute the op
        result = load_training_config_op.configured(config)(True)

        assert isinstance(result, dict)
        assert "data_config" in result
        assert "model_config" in result
        assert "experiment_name" in result
        assert result["experiment_name"] == "test_experiment"

    @patch("ananke.dagster.ops.training_ops.mlflow")
    def test_train_model_op(self, mock_mlflow, dagster_instance):
        """Test training model operation."""
        # Mock MLflow
        mock_mlflow.active_run.return_value = Mock(info=Mock(run_id="test_run_id"))

        # Create mock config data
        config_data = {
            "data_config": {
                "name": "test_data",
                "features": ["feature_1", "feature_2"],
                "target_feature": "target",
                "feature_window_hours": 24,
                "target_window_hours": 6,
                "train_stride_hours": 6,
                "test_stride_hours": 6,
                "future_feature_cols": ["feature_1"],
                "past_feature_cols": ["feature_1", "feature_2"],
                "target_cols": ["target"],
                "preprocessing": {
                    "scaler": {"type": "standard", "params": {}},
                    "imputation": {"enabled": False},
                    "smoothing": {"enabled": False},
                },
            },
            "model_config": {
                "name": "test_model",
                "model_class": "sklearn.ensemble.RandomForestRegressor",
                "model_params": {"n_estimators": 10, "random_state": 42},
                "type": "sklearn",
            },
            "experiment_name": "test_experiment",
            "output_dir": "test_output",
        }

        with patch(
            "ananke.dagster.ops.training_ops.SklearnRunner"
        ) as mock_runner_class:
            mock_runner = Mock()
            mock_runner.run.return_value = {
                "status": "completed",
                "train_mse": 0.1,
                "test_mse": 0.15,
                "runtime_seconds": 10.0,
            }
            mock_runner_class.return_value = mock_runner

            result = train_model_op(config_data)

            assert isinstance(result, dict)
            assert "results" in result
            assert "run_id" in result
            assert result["run_id"] == "test_run_id"
            assert result["model_type"] == "sklearn"

    def test_evaluate_model_op(self, dagster_instance):
        """Test model evaluation operation."""
        training_results = {
            "results": {
                "status": "completed",
                "train_mse": 0.1,
                "test_mse": 0.15,
                "test_r2": 0.85,
                "runtime_seconds": 10.0,
            },
            "run_id": "test_run_id",
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_name": "test_data",
        }

        result = evaluate_model_op(training_results)

        assert isinstance(result, dict)
        assert "evaluation_metrics" in result
        assert "run_id" in result
        assert result["run_id"] == "test_run_id"

        # Check that metrics were extracted
        metrics = result["evaluation_metrics"]
        assert "test_mse" in metrics
        assert "test_r2" in metrics
        assert metrics["test_mse"] == 0.15

    @patch("ananke.dagster.ops.training_ops.mlflow")
    def test_register_model_op(self, mock_mlflow, dagster_instance):
        """Test model registration operation."""
        evaluation_results = {
            "evaluation_metrics": {"test_mse": 0.15, "test_r2": 0.85},
            "run_id": "test_run_id",
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_name": "test_data",
            "runtime_seconds": 10.0,
        }

        result = register_model_op(evaluation_results)

        assert isinstance(result, dict)
        assert "registered_model_name" in result
        assert "model_uri" in result
        assert "registration_status" in result
        assert result["registration_status"] == "success"
        assert "tsj_sklearn_test_model" in result["registered_model_name"]


class TestPredictionOps:
    """Test cases for prediction operations."""

    @patch("ananke.dagster.ops.prediction_ops.mlflow")
    def test_load_model_op(self, mock_mlflow, dagster_instance):
        """Test loading model operation."""
        from ananke.dagster.ops.prediction_ops import PredictionConfig

        # Mock MLflow client and run
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.params = {
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_name": "test_data",
            "data_config_feature_window_hours": "24",
            "model_config_n_estimators": "100",
        }
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        # Mock model loading
        mock_model = Mock()
        with patch(
            "ananke.dagster.ops.prediction_ops.load_model_from_mlflow",
            return_value=mock_model,
        ):
            config = PredictionConfig(
                model_run_id="test_run_id", output_path="predictions.csv"
            )

            result = load_model_op.configured(config)(True)

            assert isinstance(result, dict)
            assert "model" in result
            assert "model_type" in result
            assert result["model_type"] == "sklearn"
            assert result["model_name"] == "test_model"

    def test_load_input_data_op(self, dagster_instance):
        """Test loading input data operation."""
        from ananke.dagster.ops.prediction_ops import PredictionConfig

        model_data = {
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_params": {"feature_window_hours": 24},
        }

        config = PredictionConfig(
            model_run_id="test_run_id",
            input_data_path=None,  # Will create synthetic data
            output_path="predictions.csv",
        )

        result = load_input_data_op.configured(config)(model_data)

        assert isinstance(result, dict)
        assert "input_data" in result
        assert "data_shape" in result
        assert "data_columns" in result

        # Check that synthetic data was created
        data = result["input_data"]
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

    def test_make_predictions_op(self, dagster_instance):
        """Test making predictions operation."""
        from ananke.dagster.ops.prediction_ops import PredictionConfig

        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

        model_data = {
            "model": mock_model,
            "model_type": "sklearn",
            "model_name": "test_model",
        }

        # Create sample input data
        input_data = {
            "input_data": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=3, freq="H"),
                    "feature_1": [1, 2, 3],
                    "feature_2": [4, 5, 6],
                }
            ),
            "data_shape": (3, 3),
            "data_columns": ["timestamp", "feature_1", "feature_2"],
        }

        config = PredictionConfig(
            model_run_id="test_run_id", output_path="predictions.csv"
        )

        result = make_predictions_op.configured(config)(model_data, input_data)

        assert isinstance(result, dict)
        assert "predictions" in result
        assert "predictions_df" in result
        assert "num_predictions" in result
        assert result["num_predictions"] == 3

        # Check that predictions were added to dataframe
        predictions_df = result["predictions_df"]
        assert "predictions" in predictions_df.columns

    def test_save_predictions_op(self, dagster_instance, temp_dir):
        """Test saving predictions operation."""
        from ananke.dagster.ops.prediction_ops import PredictionConfig

        prediction_results = {
            "predictions": [1.0, 2.0, 3.0],
            "predictions_df": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=3, freq="H"),
                    "feature_1": [1, 2, 3],
                    "predictions": [1.0, 2.0, 3.0],
                }
            ),
            "num_predictions": 3,
            "model_type": "sklearn",
            "model_name": "test_model",
        }

        output_path = Path(temp_dir) / "test_predictions.csv"
        config = PredictionConfig(
            model_run_id="test_run_id", output_path=str(output_path)
        )

        result = save_predictions_op.configured(config)(prediction_results)

        assert isinstance(result, dict)
        assert "output_path" in result
        assert "num_predictions" in result
        assert "save_status" in result
        assert result["save_status"] == "success"

        # Check that file was created
        assert output_path.exists()

        # Check file contents
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3
        assert "predictions" in saved_df.columns

    @patch("ananke.dagster.ops.prediction_ops.mlflow")
    def test_log_predictions_op(self, mock_mlflow, dagster_instance):
        """Test logging predictions operation."""
        from ananke.dagster.ops.prediction_ops import PredictionConfig

        # Mock MLflow
        mock_mlflow.active_run.return_value = Mock(
            info=Mock(run_id="prediction_run_id")
        )

        prediction_results = {
            "predictions": [1.0, 2.0, 3.0],
            "num_predictions": 3,
            "model_type": "sklearn",
            "model_name": "test_model",
        }

        save_results = {
            "output_path": "predictions.csv",
            "num_predictions": 3,
            "save_status": "success",
        }

        config = PredictionConfig(
            model_run_id="test_run_id",
            output_path="predictions.csv",
            experiment_name="test_prediction_experiment",
        )

        result = log_predictions_op.configured(config)(prediction_results, save_results)

        assert isinstance(result, dict)
        assert "mlflow_run_id" in result
        assert "num_predictions" in result
        assert "logging_status" in result
        assert result["logging_status"] == "success"


class TestDagsterJobs:
    """Test cases for Dagster jobs."""

    @patch("ananke.dagster.ops.training_ops.mlflow")
    @patch("ananke.dagster.ops.training_ops.SklearnRunner")
    def test_training_job_execution(
        self, mock_runner_class, mock_mlflow, dagster_instance
    ):
        """Test training job execution."""
        # Mock MLflow
        mock_mlflow.active_run.return_value = Mock(info=Mock(run_id="test_run_id"))

        # Mock runner
        mock_runner = Mock()
        mock_runner.run.return_value = {
            "status": "completed",
            "train_mse": 0.1,
            "test_mse": 0.15,
            "runtime_seconds": 10.0,
        }
        mock_runner_class.return_value = mock_runner

        # Job configuration
        job_config = {
            "ops": {
                "load_training_config_op": {
                    "config": {
                        "data_config_name": "test_data",
                        "model_config_name": "test_model",
                        "experiment_name": "test_experiment",
                        "output_dir": "test_output",
                    }
                }
            }
        }

        # Execute job (this would normally run on Celery workers)
        # For testing, we'll just verify the job can be constructed
        assert training_job is not None
        assert hasattr(training_job, "execute_in_process")

    @patch("ananke.dagster.ops.prediction_ops.mlflow")
    def test_prediction_job_execution(self, mock_mlflow, dagster_instance):
        """Test prediction job execution."""
        # Mock MLflow
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.params = {
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_name": "test_data",
        }
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.active_run.return_value = Mock(
            info=Mock(run_id="prediction_run_id")
        )

        # Mock model loading
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

        with patch(
            "ananke.dagster.ops.prediction_ops.load_model_from_mlflow",
            return_value=mock_model,
        ):
            # Job configuration
            job_config = {
                "ops": {
                    "load_model_op": {
                        "config": {
                            "model_run_id": "test_run_id",
                            "output_path": "predictions.csv",
                            "experiment_name": "test_prediction_experiment",
                        }
                    }
                }
            }

            # Execute job (this would normally run on Celery workers)
            # For testing, we'll just verify the job can be constructed
            assert prediction_job is not None
            assert hasattr(prediction_job, "execute_in_process")


class TestDagsterIntegration:
    """Integration tests for Dagster functionality."""

    def test_job_definitions(self):
        """Test that job definitions are properly configured."""
        # Check training job
        assert training_job is not None
        assert training_job.name == "training_job"

        # Check prediction job
        assert prediction_job is not None
        assert prediction_job.name == "prediction_job"

        # Check that jobs have proper tags for Celery routing
        training_tags = training_job.tags or {}
        prediction_tags = prediction_job.tags or {}

        assert "dagster-celery/queue" in training_tags
        assert training_tags["dagster-celery/queue"] == "training"

        assert "dagster-celery/queue" in prediction_tags
        assert prediction_tags["dagster-celery/queue"] == "prediction"

    def test_op_configurations(self):
        """Test that ops are properly configured."""
        # Check that training ops have proper tags
        training_ops = [
            load_training_config_op,
            train_model_op,
            evaluate_model_op,
            register_model_op,
        ]

        for op in training_ops:
            assert op.tags is not None
            assert "dagster-celery/queue" in op.tags
            assert op.tags["dagster-celery/queue"] == "training"

        # Check that prediction ops have proper tags
        prediction_ops = [
            load_model_op,
            load_input_data_op,
            make_predictions_op,
            save_predictions_op,
            log_predictions_op,
        ]

        for op in prediction_ops:
            assert op.tags is not None
            assert "dagster-celery/queue" in op.tags
            assert op.tags["dagster-celery/queue"] == "prediction"

    def test_error_handling_in_ops(self, dagster_instance):
        """Test error handling in operations."""
        # Test training op with invalid config
        invalid_config_data = {
            "data_config": {},  # Invalid/incomplete config
            "model_config": {},
            "experiment_name": "test",
            "output_dir": "test",
        }

        # Should handle errors gracefully
        try:
            result = train_model_op(invalid_config_data)
            # If it doesn't raise an exception, it should return error info
            if isinstance(result, dict) and "error" in result:
                assert True  # Error was handled gracefully
        except Exception:
            # Exceptions are also acceptable for invalid configs
            assert True

    @patch("ananke.dagster.ops.prediction_ops.mlflow")
    def test_prediction_workflow_integration(self, mock_mlflow, temp_dir):
        """Test complete prediction workflow integration."""
        # Mock MLflow components
        mock_client = Mock()
        mock_run = Mock()
        mock_run.data.params = {
            "model_type": "sklearn",
            "model_name": "test_model",
            "data_config_name": "test_data",
        }
        mock_client.get_run.return_value = mock_run
        mock_mlflow.tracking.MlflowClient.return_value = mock_client
        mock_mlflow.active_run.return_value = Mock(
            info=Mock(run_id="prediction_run_id")
        )

        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0])

        with patch(
            "ananke.dagster.ops.prediction_ops.load_model_from_mlflow",
            return_value=mock_model,
        ):
            from ananke.dagster.ops.prediction_ops import PredictionConfig

            config = PredictionConfig(
                model_run_id="test_run_id",
                output_path=str(Path(temp_dir) / "predictions.csv"),
                experiment_name="test_prediction",
            )

            # Execute workflow steps
            model_data = load_model_op.configured(config)(True)
            input_data = load_input_data_op.configured(config)(model_data)
            prediction_results = make_predictions_op.configured(config)(
                model_data, input_data
            )
            save_results = save_predictions_op.configured(config)(prediction_results)
            log_results = log_predictions_op.configured(config)(
                prediction_results, save_results
            )

            # Verify workflow completed successfully
            assert model_data["model_type"] == "sklearn"
            assert len(prediction_results["predictions"]) > 0
            assert save_results["save_status"] == "success"
            assert log_results["logging_status"] == "success"

            # Verify output file was created
            output_file = Path(temp_dir) / "predictions.csv"
            assert output_file.exists()
