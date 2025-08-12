"""Dagster operations for model prediction workflows."""

import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from dagster import Config, In, OpExecutionContext, Out, op
from pydantic import Field

from ananke.core.utils.mlflow_utils import load_model_from_mlflow, start_run

logger = logging.getLogger(__name__)


class PredictionConfig(Config):
    """Configuration for prediction operations."""

    model_run_id: str = Field(description="MLflow run ID of the trained model")
    input_data_path: str | None = Field(
        default=None, description="Path to input data for predictions"
    )
    output_path: str = Field(
        default="predictions.csv", description="Output path for predictions"
    )
    experiment_name: str = Field(
        default="tsj_prediction", description="MLflow experiment name"
    )
    run_name: str | None = Field(default=None, description="MLflow run name")


@op(
    out=Out(dict[str, Any], description="Model metadata and configuration"),
    tags={"dagster-celery/queue": "prediction"},
    description="Load trained model and its configuration from MLflow",
)
def load_model_op(
    context: OpExecutionContext, config: PredictionConfig
) -> dict[str, Any]:
    """Load trained model and its configuration from MLflow."""
    context.log.info(f"Loading model from MLflow run: {config.model_run_id}")

    try:
        # Get MLflow client
        client = mlflow.tracking.MlflowClient()

        # Get run information
        run = client.get_run(config.model_run_id)
        run_data = run.data

        # Extract model metadata
        model_type = run_data.params.get("model_type", "unknown")
        model_name = run_data.params.get("model_name", "unknown")
        data_config_name = run_data.params.get("data_config_name", "unknown")

        # Load data configuration from MLflow parameters
        data_config_params = {}
        for key, value in run_data.params.items():
            if key.startswith("data_config_"):
                param_name = key.replace("data_config_", "")
                # Convert string values to appropriate types
                if value.lower() in ["true", "false"]:
                    data_config_params[param_name] = value.lower() == "true"
                elif value.isdigit():
                    data_config_params[param_name] = int(value)
                else:
                    try:
                        data_config_params[param_name] = float(value)
                    except ValueError:
                        data_config_params[param_name] = value

        # Load model configuration from MLflow parameters
        model_config_params = {}
        for key, value in run_data.params.items():
            if key.startswith("model_config_"):
                param_name = key.replace("model_config_", "")
                # Convert string values to appropriate types
                if value.lower() in ["true", "false"]:
                    model_config_params[param_name] = value.lower() == "true"
                elif value.isdigit():
                    model_config_params[param_name] = int(value)
                else:
                    try:
                        model_config_params[param_name] = float(value)
                    except ValueError:
                        model_config_params[param_name] = value

        # Load the actual model
        model_uri = f"runs:/{config.model_run_id}/model"
        model = load_model_from_mlflow(model_uri)

        context.log.info(
            f"Successfully loaded model: {model_name} (type: {model_type})"
        )

        return {
            "model": model,
            "model_type": model_type,
            "model_name": model_name,
            "data_config_name": data_config_name,
            "data_config_params": data_config_params,
            "model_config_params": model_config_params,
            "model_run_id": config.model_run_id,
            "model_uri": model_uri,
        }

    except Exception as e:
        context.log.error(f"Failed to load model: {str(e)}")
        raise


@op(
    ins={"model_data": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Input data for predictions"),
    tags={"dagster-celery/queue": "prediction"},
    description="Load and prepare input data for predictions",
)
def load_input_data_op(
    context: OpExecutionContext, config: PredictionConfig, model_data: dict[str, Any]
) -> dict[str, Any]:
    """Load and prepare input data for predictions."""
    context.log.info("Loading input data for predictions...")

    # In a real implementation, you would load data from the specified path
    # For now, we'll create example data
    if config.input_data_path:
        context.log.info(f"Loading data from: {config.input_data_path}")
        # data = pd.read_csv(config.input_data_path)
        # For demo purposes, create synthetic data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
                "feature_1": range(100),
                "feature_2": [x * 2 for x in range(100)],
                "feature_3": [x * 0.5 for x in range(100)],
            }
        )
    else:
        context.log.info("No input data path provided, creating synthetic data")
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
                "feature_1": range(100),
                "feature_2": [x * 2 for x in range(100)],
                "feature_3": [x * 0.5 for x in range(100)],
            }
        )

    context.log.info(f"Loaded data with shape: {data.shape}")

    return {
        "input_data": data,
        "data_shape": data.shape,
        "data_columns": list(data.columns),
    }


@op(
    ins={"model_data": In(dict[str, Any]), "input_data": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Prediction results"),
    tags={"dagster-celery/queue": "prediction"},
    description="Generate predictions using the loaded model",
)
def make_predictions_op(
    context: OpExecutionContext,
    config: PredictionConfig,
    model_data: dict[str, Any],
    input_data: dict[str, Any],
) -> dict[str, Any]:
    """Generate predictions using the loaded model."""
    context.log.info("Generating predictions...")

    model = model_data["model"]
    data = input_data["input_data"]
    model_type = model_data["model_type"]

    try:
        # Generate predictions based on model type
        if model_type == "sklearn":
            # For sklearn models, use predict method
            feature_cols = [col for col in data.columns if col != "timestamp"]
            X = data[feature_cols]
            predictions = model.predict(X)

        elif model_type == "darts":
            # For Darts models, predictions would be handled differently
            # This is a simplified example
            predictions = [0.5] * len(data)  # Placeholder

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create predictions DataFrame
        predictions_df = data.copy()
        predictions_df["predictions"] = predictions

        context.log.info(f"Generated {len(predictions)} predictions")

        return {
            "predictions": predictions,
            "predictions_df": predictions_df,
            "num_predictions": len(predictions),
            "model_type": model_type,
            "model_name": model_data["model_name"],
        }

    except Exception as e:
        context.log.error(f"Prediction generation failed: {str(e)}")
        raise


@op(
    ins={"prediction_results": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Saved prediction results"),
    tags={"dagster-celery/queue": "prediction"},
    description="Save predictions to output file",
)
def save_predictions_op(
    context: OpExecutionContext,
    config: PredictionConfig,
    prediction_results: dict[str, Any],
) -> dict[str, Any]:
    """Save predictions to output file."""
    context.log.info(f"Saving predictions to: {config.output_path}")

    try:
        predictions_df = prediction_results["predictions_df"]

        # Ensure output directory exists
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        predictions_df.to_csv(output_path, index=False)

        context.log.info(f"Successfully saved {len(predictions_df)} predictions")

        return {
            "output_path": str(output_path),
            "num_predictions": len(predictions_df),
            "file_size_bytes": output_path.stat().st_size,
            "save_status": "success",
        }

    except Exception as e:
        context.log.error(f"Failed to save predictions: {str(e)}")
        return {
            "save_status": "failed",
            "error": str(e),
        }


@op(
    ins={"prediction_results": In(dict[str, Any]), "save_results": In(dict[str, Any])},
    out=Out(dict[str, Any], description="Prediction job summary"),
    tags={"dagster-celery/queue": "prediction"},
    description="Log prediction results to MLflow",
)
def log_predictions_op(
    context: OpExecutionContext,
    config: PredictionConfig,
    prediction_results: dict[str, Any],
    save_results: dict[str, Any],
) -> dict[str, Any]:
    """Log prediction results to MLflow."""
    context.log.info("Logging prediction results to MLflow...")

    run_name = config.run_name or f"prediction_{config.model_run_id[:8]}"

    with start_run(
        run_name=run_name,
        experiment_name=config.experiment_name,
        tags={
            "job_type": "prediction",
            "model_run_id": config.model_run_id,
            "model_type": prediction_results["model_type"],
            "model_name": prediction_results["model_name"],
        },
    ):
        # Log prediction metrics
        mlflow.log_param("model_run_id", config.model_run_id)
        mlflow.log_param("num_predictions", prediction_results["num_predictions"])
        mlflow.log_param("output_path", save_results.get("output_path", ""))

        # Log prediction statistics
        predictions = prediction_results["predictions"]
        mlflow.log_metric("prediction_mean", float(pd.Series(predictions).mean()))
        mlflow.log_metric("prediction_std", float(pd.Series(predictions).std()))
        mlflow.log_metric("prediction_min", float(pd.Series(predictions).min()))
        mlflow.log_metric("prediction_max", float(pd.Series(predictions).max()))

        # Log predictions file as artifact
        if save_results.get("save_status") == "success":
            mlflow.log_artifact(save_results["output_path"], "predictions")

        # Get MLflow run info
        run_info = mlflow.active_run()
        run_id = run_info.info.run_id if run_info else None

        context.log.info(f"Logged prediction results to MLflow run: {run_id}")

        return {
            "mlflow_run_id": run_id,
            "num_predictions": prediction_results["num_predictions"],
            "output_path": save_results.get("output_path"),
            "logging_status": "success",
        }
