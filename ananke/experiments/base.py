"""Base experiment class for timeseries benchmarking."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.runners.base_runner import BaseRunner
from ananke.core.utils.mlflow_utils import setup_mlflow, start_run

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Base class for timeseries experiments."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        output_dir: str | Path = "experiments",
        mlflow_experiment_name: str | None = None,
        mlflow_tracking_uri: str | None = None,
    ):
        """
        Initialize base experiment.

        Args:
            name: Name of the experiment
            description: Description of the experiment
            output_dir: Directory to save experiment outputs
            mlflow_experiment_name: MLflow experiment name
            mlflow_tracking_uri: MLflow tracking URI
        """
        self.name = name
        self.description = description
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # MLflow setup
        self.mlflow_experiment_name = mlflow_experiment_name or f"tsj_{name}"
        self.mlflow_tracking_uri = mlflow_tracking_uri

        # Results storage
        self.results: dict[str, Any] = {}
        self.runners: list[BaseRunner] = []

    @abstractmethod
    def setup_data_configs(self) -> list[DataConfig]:
        """Setup data configurations for the experiment."""
        pass

    @abstractmethod
    def setup_model_configs(self) -> list[ModelConfig]:
        """Setup model configurations for the experiment."""
        pass

    @abstractmethod
    def create_runner(
        self, data_config: DataConfig, model_config: ModelConfig
    ) -> BaseRunner:
        """Create a runner for the given configurations."""
        pass

    def run(self) -> dict[str, Any]:
        """Run the complete experiment."""
        logger.info(f"Starting experiment: {self.name}")

        # Setup MLflow
        self._setup_mlflow()

        # Setup output directory
        self._setup_output_directory()

        # Get all run combinations
        combinations = self._generate_run_combinations()

        logger.info(f"Running {len(combinations)} experiment combinations")

        experiment_results = {}

        # Run all combinations
        for combo in combinations:
            data_config = combo["data_config"]
            model_config = combo["model_config"]
            run_name = combo["run_name"]
            logger.info(f"Running: {run_name}")

            try:
                results = self._run_single_experiment(
                    data_config, model_config, run_name
                )
                experiment_results[run_name] = results

                if "error" not in results:
                    logger.info(f"Completed: {run_name}")
                else:
                    logger.error(f"Failed: {run_name}")
            except Exception as e:
                logger.error(f"Exception in run {run_name}: {str(e)}")
                experiment_results[run_name] = {"error": str(e)}

        self.results = experiment_results

        # Save summary results
        self.save_experiment_summary()
        self.save_results()

        logger.info(f"Experiment completed: {self.name}")
        return experiment_results

    def save_experiment_summary(self) -> None:
        """Save experiment summary to file."""
        summary_file = self.output_dir / f"{self.name}_summary.json"

        # Create summary
        summary = {
            "experiment_name": self.name,
            "description": self.description,
            "total_runs": len(self.results),
            "successful_runs": len(
                [r for r in self.results.values() if "error" not in r]
            ),
            "failed_runs": len([r for r in self.results.values() if "error" in r]),
            "results": self.results,
        }

        # Convert non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            if isinstance(obj, Path):
                return str(obj)
            return str(obj)

        with open(summary_file, "w") as f:
            json.dump(summary, f, default=json_serialize, indent=2)

        logger.info(f"Experiment summary saved to {summary_file}")

    def get_best_run(
        self, metric: str, ascending: bool = False
    ) -> dict[str, Any] | None:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: Whether to sort in ascending order

        Returns:
            Best run results or None
        """
        valid_runs = {
            name: results
            for name, results in self.results.items()
            if "error" not in results and metric in results
        }

        if not valid_runs:
            return None

        best_run_name = (
            min(valid_runs.keys(), key=lambda x: valid_runs[x][metric])
            if ascending
            else max(valid_runs.keys(), key=lambda x: valid_runs[x][metric])
        )

        return {"run_name": best_run_name, "results": valid_runs[best_run_name]}

    def compare_runs(self, metrics: list[str]) -> pd.DataFrame:
        """
        Compare runs across specified metrics.

        Args:
            metrics: List of metrics to compare

        Returns:
            DataFrame with comparison results sorted by first metric
        """
        comparison_data = []

        for run_name, results in self.results.items():
            if "error" not in results:
                row = {"run_name": run_name}
                for metric in metrics:
                    row[metric] = results.get(metric, None)
                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by the first metric in ascending order (best performance first)
        if len(df) > 0 and len(metrics) > 0 and metrics[0] in df.columns:
            df = df.sort_values(by=metrics[0], ascending=True).reset_index(drop=True)

        return df

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking (private method for testing)."""
        if self.mlflow_tracking_uri:
            setup_mlflow(self.mlflow_tracking_uri, self.mlflow_experiment_name)
        else:
            mlflow.set_experiment(self.mlflow_experiment_name)

    def _setup_output_directory(self) -> None:
        """Setup output directory structure (private method for testing)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "runs").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

    def _generate_run_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations of data and model configs (private method for testing)."""
        data_configs = self.setup_data_configs()
        model_configs = self.setup_model_configs()

        combinations = []
        for data_config in data_configs:
            for model_config in model_configs:
                run_name = f"{data_config.name}_{model_config.name}"
                combinations.append(
                    {
                        "data_config": data_config,
                        "model_config": model_config,
                        "run_name": run_name,
                    }
                )

        return combinations

    def _run_single_experiment(
        self, data_config: DataConfig, model_config: ModelConfig, run_name: str
    ) -> dict[str, Any]:
        """Run a single experiment configuration (private method for testing)."""
        try:
            # Create runner
            runner = self.create_runner(data_config, model_config)
            self.runners.append(runner)

            # Run experiment
            with start_run(
                run_name=run_name,
                experiment_name=self.mlflow_experiment_name,
                tracking_uri=self.mlflow_tracking_uri,
                tags={
                    "experiment": self.name,
                    "data_config": data_config.name,
                    "model_config": model_config.name,
                    "model_type": model_config.type,
                },
            ):
                # Log experiment metadata
                mlflow.log_param("experiment_name", self.name)
                if self.description:
                    mlflow.log_param("experiment_description", self.description)

                # Run the experiment
                results = runner.run(run_name)
                return results

        except Exception as e:
            logger.error(f"Error in run {run_name}: {str(e)}")
            return {"error": str(e)}

    def save_results(self) -> None:
        """Save experiment results to files."""
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # Save detailed results as JSON
        results_file = results_dir / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, default=str, indent=2)

        # Save comparison CSV if we have results
        if self.results:
            comparison_data = []
            for run_name, results in self.results.items():
                if "error" not in results:
                    row = {"run_name": run_name}
                    row.update(results)
                    comparison_data.append(row)

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_file = results_dir / "run_comparison.csv"
                comparison_df.to_csv(comparison_file, index=False)

    def cleanup(self) -> None:
        """Clean up experiment resources."""
        for runner in self.runners:
            if hasattr(runner, "cleanup"):
                runner.cleanup()

        logger.info(f"Experiment cleanup completed: {self.name}")
