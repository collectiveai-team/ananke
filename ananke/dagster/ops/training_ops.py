"""Dagster operations for model training workflows."""

import logging
from typing import Any

import mlflow
from dagster import Config, In, OpExecutionContext, Out, op
from pydantic import Field

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.runners.darts_runner import DartsRunConfig, DartsRunner
from ananke.core.runners.sklearn_runner import SklearnRunConfig, SklearnRunner
from ananke.core.utils.mlflow_utils import log_data_config, log_model_config, start_run

logger = logging.getLogger(__name__)


class TrainingConfig(Config):
    """Configuration for training operations."""

    data_config_name: str = Field(description="Name of the data configuration")
    model_config_name: str = Field(description="Name of the model configuration")
    experiment_name: str = Field(
        default="tsj_training", description="MLflow experiment name"
    )
    run_name: str | None = Field(default=None, description="MLflow run name")
    output_dir: str = Field(
        default="results", description="Output directory for results"
    )


@op(
    out=Out(dict[str, Any], description="Training configuration and metadata"),
    tags={"dagster-celery/queue": "training"},
    description="Load and validate training configurations",
)
def load_training_config_op(
    context: OpExecutionContext, config: TrainingConfig
) -> dict[str, Any]:
    """Load and validate training configurations."""
    context.log.info(
        f"Loading training config: {config.data_config_name}, {config.model_config_name}"
    )

    # In a real implementation, you would load these from files
    # For now, we'll create example configurations

    # Example data config
    data_config = DataConfig(
        name=config.data_config_name,
        features=["feature_1", "feature_2", "feature_3"],
        target_feature="target",
        feature_window_hours=24,
        target_window_hours=6,
        train_stride_hours=6,
        test_stride_hours=6,
        future_feature_cols=["feature_1", "feature_2"],
        past_feature_cols=["feature_1", "feature_2", "feature_3"],
        target_cols=["target"],
        preprocessing={
            "scaler": {"type": "standard", "params": {}},
            "imputation": {"enabled": True, "method": "forward_fill"},
            "smoothing": {"enabled": False},
        },
    )

    # Example model config
    model_config = ModelConfig(
        name=config.model_config_name,
        model_class="sklearn.ensemble.RandomForestRegressor",
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
        type="sklearn",
    )

    return {
        "data_config": data_config.dict(),
        "model_config": model_config.dict(),
        "experiment_name": config.experiment_name,
        "run_name": config.run_name,
        "output_dir": config.output_dir,
    }


@op(
    ins={"config_data": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Training results and model metadata"),
    tags={"dagster-celery/queue": "training"},
    description="Train a timeseries model",
)
def train_model_op(
    context: OpExecutionContext, config_data: dict[str, Any]
) -> dict[str, Any]:
    """Train a timeseries model."""
    context.log.info("Starting model training...")

    # Reconstruct configurations
    data_config = DataConfig(**config_data["data_config"])
    model_config = ModelConfig(**config_data["model_config"])

    # Create run configuration
    run_name = config_data.get("run_name") or f"{data_config.name}_{model_config.name}"

    # Create appropriate runner based on model type
    if model_config.type == "sklearn":
        run_config = SklearnRunConfig(
            name=run_name,
            output_dir=config_data["output_dir"],
            log_to_mlflow=True,
        )
        runner = SklearnRunner(run_config, data_config, model_config)
    elif model_config.type == "darts":
        run_config = DartsRunConfig(
            name=run_name,
            output_dir=config_data["output_dir"],
            log_to_mlflow=True,
            forecast_horizon=12,
        )
        runner = DartsRunner(run_config, data_config, model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_config.type}")

    # Start MLflow run
    with start_run(
        run_name=run_name,
        experiment_name=config_data["experiment_name"],
        tags={
            "model_type": model_config.type,
            "data_config": data_config.name,
            "model_config": model_config.name,
            "dagster_job": "training",
        },
    ):
        # Log configurations
        log_data_config(data_config.dict())
        log_model_config(model_config.dict())

        # Run training
        results = runner.run()

        # Get MLflow run info
        run_info = mlflow.active_run()
        run_id = run_info.info.run_id if run_info else None

        context.log.info(f"Training completed. MLflow run ID: {run_id}")

        return {
            "results": results,
            "run_id": run_id,
            "model_type": model_config.type,
            "model_name": model_config.name,
            "data_config_name": data_config.name,
        }


@op(
    ins={"training_results": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Evaluation results and metrics"),
    tags={"dagster-celery/queue": "training"},
    description="Evaluate trained model performance",
)
def evaluate_model_op(
    context: OpExecutionContext, training_results: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate trained model performance."""
    context.log.info("Evaluating model performance...")

    results = training_results["results"]
    run_id = training_results["run_id"]

    # Extract key metrics
    evaluation_metrics = {}

    # Look for common metrics
    metric_keys = [
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "train_mse",
        "train_rmse",
        "train_mae",
        "train_r2",
        "val_mse",
        "val_rmse",
        "val_mae",
        "val_r2",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    for key in metric_keys:
        if key in results:
            evaluation_metrics[key] = results[key]

    # Log evaluation summary
    context.log.info("Model evaluation completed:")
    for metric, value in evaluation_metrics.items():
        context.log.info(f"  {metric}: {value}")

    return {
        "evaluation_metrics": evaluation_metrics,
        "run_id": run_id,
        "model_type": training_results["model_type"],
        "model_name": training_results["model_name"],
        "data_config_name": training_results["data_config_name"],
        "runtime_seconds": results.get("runtime_seconds"),
    }


@op(
    ins={"evaluation_results": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Model registration results"),
    tags={"dagster-celery/queue": "training"},
    description="Register trained model in MLflow",
)
def register_model_op(
    context: OpExecutionContext, evaluation_results: dict[str, Any]
) -> dict[str, Any]:
    """Register trained model in MLflow."""
    context.log.info("Registering model in MLflow...")

    run_id = evaluation_results["run_id"]
    model_name = evaluation_results["model_name"]
    model_type = evaluation_results["model_type"]

    try:
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run_id}/model"
        registered_model_name = f"tsj_{model_type}_{model_name}"

        # In a real implementation, you would register the model
        # For now, we'll just log the registration intent
        context.log.info(f"Would register model: {registered_model_name}")
        context.log.info(f"Model URI: {model_uri}")

        # Log model registration with MLflow
        if mlflow.active_run():
            mlflow.log_param("registered_model_name", registered_model_name)
            mlflow.log_param("model_uri", model_uri)

        return {
            "registered_model_name": registered_model_name,
            "model_uri": model_uri,
            "run_id": run_id,
            "registration_status": "success",
        }

    except Exception as e:
        context.log.error(f"Model registration failed: {str(e)}")
        return {
            "run_id": run_id,
            "registration_status": "failed",
            "error": str(e),
        }
