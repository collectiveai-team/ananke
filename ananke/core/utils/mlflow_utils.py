"""
MLflow utilities for tracking experiments, models, and results.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "default",
) -> str:
    """
    Set up MLflow tracking.

    Args:
        tracking_uri: URI for MLflow tracking server. If None, uses local filesystem.
        experiment_name: Name of the experiment to use.

    Returns:
        experiment_id: ID of the created or existing experiment.
    """
    # Set tracking URI
    if tracking_uri:
        # Handle file URIs properly
        if tracking_uri.startswith("/"):
            # This is an absolute path without the file: prefix
            os.makedirs(tracking_uri, exist_ok=True)
            tracking_uri = f"file:{tracking_uri}"
        elif not tracking_uri.startswith(
            ("file:", "http:", "https:", "postgresql:", "mysql:")
        ):
            # This is a relative path without a protocol prefix
            os.makedirs(tracking_uri, exist_ok=True)
            tracking_uri = f"file:{tracking_uri}"

        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Default to mlruns directory if no URI specified
        default_tracking_dir = "mlruns"
        os.makedirs(default_tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{default_tracking_dir}")

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION", None)
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=artifact_location
        )

    return experiment_id


def start_run(
    run_name: str | None = None,
    experiment_name: str = "default",
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
    run_id: str | None = None,
) -> mlflow.ActiveRun:
    """
    Start an MLflow run, handling existing runs properly.

    Args:
        run_name: Name for the run.
        experiment_name: Name of the experiment to use.
        tracking_uri: URI for MLflow tracking server.
        tags: Dictionary of tags to add to the run.
        nested: Whether this run is nested within another run.
        run_id: Specific run ID to resume (if provided)

    Returns:
        mlflow.ActiveRun: The active MLflow run.
    """
    # End any active runs first unless nested is True
    if not nested and mlflow.active_run():
        mlflow.end_run()

    # Setup MLflow tracking
    experiment_id = setup_mlflow(tracking_uri, experiment_name)

    # Start a new run or resume an existing one
    if run_id:
        try:
            return mlflow.start_run(
                run_id=run_id,
                experiment_id=experiment_id,
                nested=nested,
            )
        except mlflow.exceptions.MlflowException as e:
            print(f"Warning: Could not resume run {run_id}: {e}")
            print("Starting a new run instead.")
            # Fall through to start a new run

    # Start a new run
    return mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags=tags,
        nested=nested,
    )


def log_eda_stats(
    df: pd.DataFrame,
    prefix: str = "",
    include_nulls: bool = True,
    include_unique_counts: bool = True,
    include_descriptive_stats: bool = True,
) -> None:
    """
    Log exploratory data analysis statistics to MLflow.

    Args:
        df: DataFrame to analyze.
        prefix: Prefix to add to metric names.
        include_nulls: Whether to log null counts.
        include_unique_counts: Whether to log unique value counts.
        include_descriptive_stats: Whether to log descriptive statistics.
    """
    # Add separator to prefix if it's not empty
    if prefix and not prefix.endswith("."):
        prefix = f"{prefix}."

    # Log basic dataframe info
    mlflow.log_param(f"{prefix}shape", str(df.shape))
    mlflow.log_param(f"{prefix}columns", str(list(df.columns)))

    # Log null counts
    if include_nulls:
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            mlflow.log_metric(f"{prefix}nulls.{col}", count)
        mlflow.log_metric(f"{prefix}nulls.total", null_counts.sum())

    # Log unique counts
    if include_unique_counts:
        for col in df.select_dtypes(include=["object", "category"]).columns:
            mlflow.log_metric(f"{prefix}unique_values.{col}", df[col].nunique())

    # Log descriptive statistics
    if include_descriptive_stats:
        for col in df.select_dtypes(include=["number"]).columns:
            mlflow.log_metric(f"{prefix}mean.{col}", df[col].mean())
            mlflow.log_metric(f"{prefix}median.{col}", df[col].median())
            mlflow.log_metric(f"{prefix}std.{col}", df[col].std())
            mlflow.log_metric(f"{prefix}min.{col}", df[col].min())
            mlflow.log_metric(f"{prefix}max.{col}", df[col].max())


def log_pandas_df(
    df: pd.DataFrame,
    artifact_path: str,
    file_format: str = "csv",
) -> None:
    """
    Log a pandas DataFrame as an artifact.

    Args:
        df: DataFrame to log.
        artifact_path: Path within the run's artifact directory.
        file_format: Format to save the DataFrame (csv or parquet).
    """
    # Ensure artifact_path is not empty
    if not artifact_path:
        artifact_path = "dataframes"

    # Create a temporary file path
    temp_dir = Path("temp_artifacts")
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / f"data.{file_format}"

    try:
        # Save DataFrame to file
        if file_format.lower() == "csv":
            df.to_csv(file_path, index=False)
        elif file_format.lower() == "parquet":
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Log the artifact
        mlflow.log_artifact(str(file_path), artifact_path)
    finally:
        # Clean up temporary file
        if file_path.exists():
            file_path.unlink()


def log_figure(
    fig: Any, artifact_path: str, filename: str = "figure", close_figure: bool = True
) -> None:
    """
    Log a matplotlib or plotly figure as an artifact in MLflow.

    Args:
        fig: Figure to log (matplotlib.Figure, plt module, or plotly.graph_objects.Figure).
        artifact_path: Path where the figure will be saved in the MLflow run.
        filename: Name of the file (without extension).
        close_figure: Whether to close the figure after logging (matplotlib only).
    """
    # Ensure artifact_path is not empty
    if not artifact_path:
        artifact_path = "figures"

    # Create temporary directory
    temp_dir = Path("temp_artifacts")
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / f"{filename}.png"

    try:
        if fig == plt:  # If plt module is passed directly
            # Save the current figure
            plt.savefig(temp_path, bbox_inches="tight", dpi=300)
            if close_figure:
                plt.close()
        elif hasattr(fig, "savefig"):  # matplotlib Figure object
            fig.savefig(temp_path, bbox_inches="tight", dpi=300)
            if close_figure:
                plt.close(fig)
        elif hasattr(fig, "write_image"):  # plotly Figure object
            fig.write_image(str(temp_path))
        else:
            raise ValueError(f"Unsupported figure type: {type(fig)}")

        # Log the artifact
        mlflow.log_artifact(str(temp_path), artifact_path)
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


def log_model_config(config: dict[str, Any], prefix: str = "model") -> None:
    """
    Log model configuration parameters to MLflow.

    Args:
        config: Configuration dictionary to log
        prefix: Prefix for parameter names
    """
    for key, value in config.items():
        param_name = f"{prefix}_{key}" if prefix else key

        # Convert complex types to strings
        if isinstance(value, (dict, list)):
            mlflow.log_param(param_name, str(value))
        else:
            mlflow.log_param(param_name, value)


def log_data_config(config: dict[str, Any], prefix: str = "data") -> None:
    """
    Log data configuration parameters to MLflow.

    Args:
        config: Configuration dictionary to log
        prefix: Prefix for parameter names
    """
    for key, value in config.items():
        param_name = f"{prefix}_{key}" if prefix else key

        # Convert complex types to strings
        if isinstance(value, (dict, list)):
            mlflow.log_param(param_name, str(value))
        else:
            mlflow.log_param(param_name, value)


def get_best_run(
    experiment_name: str, metric_name: str, ascending: bool = False
) -> mlflow.entities.Run | None:
    """
    Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric_name: Name of the metric to optimize
        ascending: Whether to sort in ascending order (True for minimization)

    Returns:
        Best run or None if no runs found
    """
    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{metric_name} IS NOT NULL",
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        return runs[0] if runs else None
    except Exception as e:
        print(f"Error getting best run: {e}")
        return None


def load_model_from_run(run_id: str, model_name: str = "model") -> Any:
    """
    Load a model from an MLflow run.

    Args:
        run_id: ID of the run containing the model
        model_name: Name of the model artifact

    Returns:
        Loaded model
    """
    try:
        model_uri = f"runs:/{run_id}/{model_name}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model from run {run_id}: {e}")
        return None


def load_model_from_mlflow(
    model_name: str, mlflow_tracking_uri: str = "localhost:5000"
):
    # Get the model from MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # Get the latest production model from MLflow
    logger.info(
        "No run_id provided. Searching for the latest model in 'Production' stage."
    )
    latest_versions = client.get_latest_versions(name=model_name)
    if not latest_versions:
        raise ValueError(f"No production models found for '{model_name}' in MLflow")
    run_id = latest_versions[0].run_id
    logger.info(f"Using latest production model from run: {run_id}")

    # Load model and data configurations from the MLflow run
    try:
        # Download and load data_config.json
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, "data_config.json", tmpdir)
            with open(local_path) as f:
                data_config_dict = json.load(f)
        data_config = DataConfig(**data_config_dict)
        logger.info(f"Successfully loaded data_config.json from run {run_id}.")
        # Download and load model_config.json
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(run_id, "model_config.json", tmpdir)
            with open(local_path) as f:
                model_config_dict = json.load(f)
        model_config = ModelConfig(**model_config_dict)
        logger.info(f"Successfully loaded model_config.json from run {run_id}.")

    except Exception as e:
        logger.error(f"Failed to load configurations from run {run_id}: {e}")
        raise

    return run_id, data_config, model_config
