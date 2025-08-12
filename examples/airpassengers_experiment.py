"""AirPassengers dataset experiment example using TSJ."""

import logging

from darts.models import AutoARIMA, ExponentialSmoothing, Prophet

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.runners.darts_runner import DartsRunConfig, DartsRunner
from ananke.experiments.base import BaseExperiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirPassengersExperiment(BaseExperiment):
    """AirPassengers forecasting experiment."""

    def __init__(self):
        super().__init__(
            name="airpassengers_forecasting",
            description="Forecasting experiment on AirPassengers dataset using multiple models",
            output_dir="results/airpassengers",
            mlflow_experiment_name="airpassengers_forecasting",
        )

    def setup_data_configs(self) -> list[DataConfig]:
        """Setup data configurations for AirPassengers."""
        return [
            DataConfig(
                name="airpassengers_monthly",
                features=["passengers"],
                target_feature="passengers",
                feature_window_hours=12,  # 12 months lookback
                target_window_hours=12,  # 12 months forecast
                train_stride_hours=1,  # Monthly stride
                test_stride_hours=1,
                future_feature_cols=[],
                past_feature_cols=["passengers"],
                target_cols=["passengers"],
                preprocessing={
                    "scaler": {"type": "none", "params": {}},
                    "imputation": {"enabled": False},
                    "smoothing": {"enabled": False},
                },
            )
        ]

    def setup_model_configs(self) -> list[ModelConfig]:
        """Setup model configurations for forecasting."""
        return [
            ModelConfig(
                name="prophet",
                model_class=Prophet,
                model_params={
                    "seasonality_mode": "multiplicative",
                    "yearly_seasonality": True,
                    "weekly_seasonality": False,
                    "daily_seasonality": False,
                },
                type="darts",
            ),
            ModelConfig(
                name="auto_arima",
                model_class=AutoARIMA,
                model_params={
                    "seasonal": True,
                    "season_length": 12,  # Monthly seasonality
                },
                type="darts",
            ),
            ModelConfig(
                name="exponential_smoothing",
                model_class=ExponentialSmoothing,
                model_params={
                    "seasonal_periods": 12,
                },
                type="darts",
            ),
        ]

    def create_runner(
        self, data_config: DataConfig, model_config: ModelConfig
    ) -> DartsRunner:
        """Create a Darts runner for the configurations."""
        run_config = DartsRunConfig(
            name=f"{data_config.name}_{model_config.name}",
            output_dir=self.output_dir / "runs",
            log_to_mlflow=True,
            forecast_horizon=12,  # Forecast 12 months ahead
        )

        return DartsRunner(run_config, data_config, model_config)


def main():
    """Run the AirPassengers experiment."""
    # Create and run experiment
    experiment = AirPassengersExperiment()
    results = experiment.run()

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)

    for run_name, result in results.items():
        print(f"\nRun: {run_name}")
        if "error" in result:
            print(f"  Status: FAILED - {result['error']}")
        else:
            print("  Status: SUCCESS")
            print(f"  Runtime: {result.get('runtime_seconds', 'N/A'):.2f}s")

            # Print key metrics if available
            for metric in ["test_mse", "test_mae", "test_r2"]:
                if metric in result:
                    print(f"  {metric}: {result[metric]:.4f}")

    # Find best model
    best_run = experiment.get_best_run("test_mse", ascending=True)
    if best_run:
        print(f"\nBest Model: {best_run['run_name']}")
        print(f"Best MSE: {best_run['results']['test_mse']:.4f}")

    # Create comparison table
    comparison_df = experiment.compare_runs(["test_mse", "test_mae", "test_r2"])
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))

    # Cleanup
    experiment.cleanup()


if __name__ == "__main__":
    main()
