"""Base runner framework for timeseries experiments."""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.data.meta.dataset import TimeSeriesData, TimeSeriesDataset

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    name: str
    output_dir: Path
    seed: int = 42
    log_level: str = "INFO"
    save_model: bool = True
    save_results: bool = True
    log_to_mlflow: bool = True
    save_predictions: bool = False
    additional_params: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


class BaseRunner(ABC):
    """Base class for all benchmark runners."""

    def __init__(
        self, run_config: RunConfig, data_config: DataConfig, model_config: ModelConfig
    ):
        """
        Initialize the runner with configuration.

        Args:
            run_config: The run configuration
            data_config: The data configuration
            model_config: The model configuration
        """
        self.run_config = run_config
        self.data_config = data_config
        self.model_config = model_config
        self.results: dict[str, Any] = {}
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.time_series_dataset: TimeSeriesDataset | None = None

        # Set up output directory
        self.output_dir = Path(run_config.output_dir) / run_config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self._setup_logging()

        # Set random seeds
        self.set_random_seeds(run_config.seed)

    def _setup_logging(self) -> None:
        """Set up logging for the runner."""
        log_file = self.output_dir / f"{self.run_config.name}.log"

        # Configure logging
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

        logging.basicConfig(
            level=getattr(logging, self.run_config.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

    @staticmethod
    def set_random_seeds(seed: int) -> None:
        """
        Set random seeds for reproducibility.

        Args:
            seed: Random seed to use
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run(self, run_name: str | None = None) -> dict[str, Any]:
        """
        Run the benchmark.

        Returns:
            Dictionary of results
        """
        self.start_time = time.time()
        logger.info(f"Starting benchmark run: {self.run_config.name}")

        try:
            # Use MLflow context if enabled
            context = (
                self._get_mlflow_context(run_name)
                if self.run_config.log_to_mlflow
                else nullcontext()
            )

            with context:
                # Log configurations to MLflow if enabled
                if self.run_config.log_to_mlflow:
                    self._log_configs_to_mlflow()

                # Prepare data
                logger.info("Preparing data...")
                self.time_series_dataset = self.prepare_data()

                # Setup model
                logger.info("Setting up model...")
                self.setup_model()

                # Train model
                logger.info("Training model...")
                self.train()

                # Make predictions and evaluate
                logger.info("Making predictions and evaluating...")
                self._evaluate_all_splits()

                # Save results
                if self.run_config.save_results:
                    self.save_results()

                self.cleanup()

        except Exception as e:
            logger.error(f"Error during benchmark run: {str(e)}", exc_info=True)
            self.results["error"] = str(e)
            self.results["status"] = "failed"

        self.end_time = time.time()
        self.results["runtime_seconds"] = self.end_time - self.start_time
        
        # Set status if not already set (e.g., by error handling)
        if "status" not in self.results:
            self.results["status"] = "completed"
            
        logger.info(
            f"Benchmark run completed in {self.results['runtime_seconds']:.2f} seconds"
        )
        return self.results

    def _get_mlflow_context(self, run_name: str | None):
        """Get MLflow context for experiment tracking."""
        experiment_name = self.run_config.additional_params.get(
            "experiment_name", "timeseries_experiments"
        )

        # Set or create experiment
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

        return mlflow.start_run(run_name=run_name or self.run_config.name)

    def _log_configs_to_mlflow(self) -> None:
        """Log configurations to MLflow."""
        # Log run config
        for key, value in self.run_config.__dict__.items():
            if key != "additional_params":
                mlflow.log_param(f"run_{key}", value)

        # Log additional params
        for key, value in self.run_config.additional_params.items():
            mlflow.log_param(f"run_{key}", value)

        # Log data config
        for key, value in self.data_config.__dict__.items():
            if not isinstance(value, (dict, list)):
                mlflow.log_param(f"data_{key}", value)

        # Log model config
        mlflow.log_param("model_name", self.model_config.name)
        mlflow.log_param("model_type", self.model_config.type)
        mlflow.log_param(
            "model_class",
            self.model_config.serialize_model_class(
                self.model_config.model_class, None
            ),
        )

        for key, value in self.model_config.model_params.items():
            mlflow.log_param(f"model_{key}", value)

    def _evaluate_all_splits(self) -> None:
        """Evaluate model on all data splits."""
        if not self.time_series_dataset:
            raise ValueError("Dataset not prepared. Call prepare_data() first.")

        # Evaluate on training data
        if self.time_series_dataset.train_data:
            train_predictions = self.predict(self.time_series_dataset.train_data)
            train_targets = self._to_array(
                [data.target for data in self.time_series_dataset.train_data]
            )
            train_metrics = self.evaluate(
                train_targets,
                train_predictions,
                target_names=self.data_config.target_cols,
                prefix="train_",
            )
            self.results.update(train_metrics)

        # Evaluate on validation data
        if self.time_series_dataset.val_data:
            val_predictions = self.predict(self.time_series_dataset.val_data)
            val_targets = self._to_array(
                [data.target for data in self.time_series_dataset.val_data]
            )
            val_metrics = self.evaluate(
                val_targets,
                val_predictions,
                target_names=self.data_config.target_cols,
                prefix="val_",
            )
            self.results.update(val_metrics)

        # Evaluate on test data
        if self.time_series_dataset.test_data:
            test_predictions = self.predict(self.time_series_dataset.test_data)
            test_targets = self._to_array(
                [data.target for data in self.time_series_dataset.test_data]
            )
            test_metrics = self.evaluate(
                test_targets,
                test_predictions,
                target_names=self.data_config.target_cols,
                prefix="test_",
            )
            self.results.update(test_metrics)

        # Log metrics to MLflow
        if self.run_config.log_to_mlflow:
            for metric_name, metric_value in self.results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

    @abstractmethod
    def prepare_data(self) -> TimeSeriesDataset:
        """Prepare data for the benchmark."""
        pass

    @abstractmethod
    def setup_model(self) -> None:
        """Set up the model for the benchmark."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, data: list[TimeSeriesData]) -> np.ndarray:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def _to_array(self, data: list[Any]) -> np.ndarray:
        """Convert data to array."""
        pass

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        target_names: list[str] | None = None,
        prefix: str = "",
        event_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate the model and store results."""
        if len(y_pred) == 0:
            return {}

        # Reshape if needed
        if len(y_true.shape) == 3 and y_true.shape[2] == 1:
            y_true = y_true.squeeze(axis=2)
        if len(y_pred.shape) == 3 and y_pred.shape[2] == 1:
            y_pred = y_pred.squeeze(axis=2)

        # Import evaluation function (to be implemented)
        try:
            from ananke.core.evaluation.metrics import evaluate_predictions

            return evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                target_names=target_names,
                prefix=prefix,
                event_threshold=event_threshold or self.data_config.event_threshold,
            )
        except ImportError:
            logger.warning("Evaluation metrics not available. Returning empty metrics.")
            return {}

    def save_results(self) -> None:
        """Save benchmark results to disk."""
        results_file = (
            self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        import json

        # Convert non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, (np.ndarray, pd.Series)):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return str(obj)

        with open(results_file, "w") as f:
            json.dump(self.results, f, default=json_serialize, indent=2)

        logger.info(f"Results saved to {results_file}")

    def cleanup(self) -> None:
        """Clean up resources after the benchmark."""
        pass
