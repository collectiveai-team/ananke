"""Runner configuration management."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ananke.core.runners.darts_runner import DartsRunConfig
from ananke.core.runners.sklearn_runner import SklearnRunConfig


class RunnerConfig(BaseModel):
    """Configuration for runners."""

    name: str = Field(description="Name of the runner configuration")
    runner_type: str = Field(description="Type of runner (sklearn, darts, pytorch)")
    run_config: dict[str, Any] = Field(
        default_factory=dict, description="Runner-specific configuration"
    )

    class Config:
        """Pydantic configuration."""

        extra = "allow"

    @classmethod
    def from_yaml(cls, file_path: str) -> "RunnerConfig":
        """Load configuration from YAML file."""
        with open(file_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def create_run_config(self, output_dir: str, name: str | None = None) -> Any:
        """Create appropriate run configuration based on runner type."""
        run_name = name or self.name

        if self.runner_type == "sklearn":
            return SklearnRunConfig(
                name=run_name, output_dir=output_dir, **self.run_config
            )
        elif self.runner_type == "darts":
            return DartsRunConfig(
                name=run_name, output_dir=output_dir, **self.run_config
            )
        else:
            raise ValueError(f"Unsupported runner type: {self.runner_type}")


def get_default_runner_configs() -> list[RunnerConfig]:
    """Get default runner configurations."""
    return [
        RunnerConfig(
            name="sklearn_default",
            runner_type="sklearn",
            run_config={
                "log_to_mlflow": True,
                "save_model": True,
                "save_predictions": True,
                "save_feature_importance": True,
            },
        ),
        RunnerConfig(
            name="darts_default",
            runner_type="darts",
            run_config={
                "log_to_mlflow": True,
                "save_model": True,
                "save_predictions": True,
                "forecast_horizon": 12,
                "backtest_config": {
                    "start": 0.7,
                    "forecast_horizon": 12,
                    "stride": 1,
                    "retrain": True,
                },
            },
        ),
        RunnerConfig(
            name="sklearn_hyperparameter_search",
            runner_type="sklearn",
            run_config={
                "log_to_mlflow": True,
                "save_model": True,
                "save_predictions": True,
                "save_feature_importance": True,
                "hyperparameter_search": {
                    "enabled": True,
                    "method": "grid",
                    "cv": 5,
                    "scoring": "neg_mean_squared_error",
                },
            },
        ),
        RunnerConfig(
            name="darts_backtesting",
            runner_type="darts",
            run_config={
                "log_to_mlflow": True,
                "save_model": True,
                "save_predictions": True,
                "forecast_horizon": 24,
                "backtest_config": {
                    "start": 0.6,
                    "forecast_horizon": 24,
                    "stride": 6,
                    "retrain": False,
                    "overlap_end": False,
                },
            },
        ),
    ]


def create_example_runner_config(file_path: str) -> None:
    """Create an example runner configuration file."""
    config = RunnerConfig(
        name="example_sklearn_runner",
        runner_type="sklearn",
        run_config={
            "log_to_mlflow": True,
            "save_model": True,
            "save_predictions": True,
            "save_feature_importance": True,
            "hyperparameter_search": {
                "enabled": False,
                "method": "grid",
                "cv": 5,
                "scoring": "neg_mean_squared_error",
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15],
                },
            },
        },
    )

    config.to_yaml(file_path)


def list_runner_configs(config_dir: str) -> list[str]:
    """List all runner configuration files in a directory."""
    config_path = Path(config_dir)
    if not config_path.exists():
        return []

    yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
    return [f.name for f in yaml_files]


def validate_runner_config(file_path: str) -> bool:
    """Validate a runner configuration file."""
    try:
        config = RunnerConfig.from_yaml(file_path)

        # Additional validation
        if config.runner_type not in ["sklearn", "darts", "pytorch"]:
            raise ValueError(f"Invalid runner type: {config.runner_type}")

        # Validate run_config based on runner type
        if config.runner_type == "sklearn":
            # Validate sklearn-specific config
            if "hyperparameter_search" in config.run_config:
                hp_config = config.run_config["hyperparameter_search"]
                if hp_config.get("enabled", False):
                    required_keys = ["method", "cv", "scoring"]
                    for key in required_keys:
                        if key not in hp_config:
                            raise ValueError(
                                f"Missing required hyperparameter search config: {key}"
                            )

        elif config.runner_type == "darts":
            # Validate darts-specific config
            if "backtest_config" in config.run_config:
                bt_config = config.run_config["backtest_config"]
                required_keys = ["start", "forecast_horizon"]
                for key in required_keys:
                    if key not in bt_config:
                        raise ValueError(f"Missing required backtest config: {key}")

        return True

    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False
