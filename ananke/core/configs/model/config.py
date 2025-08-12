"""Generalizable model configuration for timeseries experiments."""

import glob
import importlib
import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, field_serializer, field_validator


class ModelConfig(BaseModel):
    """Configuration for timeseries models."""

    model_class: type[Any]
    model_params: dict[str, Any]
    type: Literal["pytorch", "darts", "sklearn", "custom"]
    name: str
    fit_on_predict_required: bool = False

    # Training configuration
    training_params: dict[str, Any] | None = None
    early_stopping: dict[str, Any] | None = None

    # Validation configuration
    validation_strategy: str | None = (
        "holdout"  # holdout, cross_validation, time_series_split
    )
    validation_params: dict[str, Any] | None = None

    # Hyperparameter search configuration
    hyperparameter_search: dict[str, Any] | None = None

    @field_validator("model_class", mode="before")
    @classmethod
    def resolve_model_class(cls, v):
        """Resolve model class from string if needed."""
        if isinstance(v, str):
            return resolve_class_from_string(v)
        return v

    @field_serializer("model_class")
    def serialize_model_class(self, value: type[Any], _info) -> str:
        """Convert model class type to its fully qualified string name for serialization."""
        return f"{value.__module__}.{value.__name__}"

    class Config:
        arbitrary_types_allowed = True


def resolve_class_from_string(class_string: str) -> type[Any]:
    """Resolve a class from its fully qualified string name.

    Args:
        class_string: Fully qualified class name (e.g., 'sklearn.linear_model.LinearRegression')

    Returns:
        The resolved class type

    Raises:
        ImportError: If the module or class cannot be imported
        AttributeError: If the class doesn't exist in the module
    """
    try:
        module_name, class_name = class_string.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import class '{class_string}': {e}")


def get_model_configurations(
    config_dir: str, config_name: str | None = None
) -> list[ModelConfig]:
    """Get model configurations from YAML files.

    Args:
        config_dir: Directory containing configuration files
        config_name: Optional name of a specific configuration to load
                   If None, all configurations are loaded

    Returns:
        List of ModelConfig objects
    """
    # Create directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Created model configuration directory: {config_dir}")

    # Find all YAML configuration files
    all_config_files = glob.glob(os.path.join(config_dir, "*.yaml"))

    if not all_config_files:
        print("No model configuration files found. Using default configuration.")
        return [_get_default_config()]

    configs = []

    # If specific config requested, filter for it
    if config_name is not None:
        target_file = os.path.join(config_dir, f"{config_name}.yaml")
        if os.path.exists(target_file):
            all_config_files = [target_file]
        else:
            print(f"Warning: Configuration '{config_name}' not found.")
            return []

    for config_file in all_config_files:
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            # Create ModelConfig object
            config = ModelConfig(**config_data)

            configs.append(config)
            print(f"Loaded model configuration from {os.path.basename(config_file)}")

        except Exception as e:
            print(f"Error loading configuration from {config_file}: {str(e)}")

    return configs


def _get_default_config() -> ModelConfig:
    """Get a default model configuration."""
    return ModelConfig(
        name="linear_regression",
        model_class="sklearn.linear_model.LinearRegression",
        model_params={},
        type="sklearn",
        fit_on_predict_required=False,
        training_params={},
        validation_strategy="holdout",
        validation_params={"test_size": 0.2, "random_state": 42},
    )


def create_example_configs(config_dir: str) -> None:
    """Create example model configuration YAML files."""

    # Skip if files already exist
    if glob.glob(os.path.join(config_dir, "*.yaml")):
        print(f"Example configurations already exist in {config_dir}")
        return

    # Linear Regression example
    linear_config = {
        "name": "linear_regression",
        "model_class": "sklearn.linear_model.LinearRegression",
        "model_params": {},
        "type": "sklearn",
        "fit_on_predict_required": False,
        "training_params": {},
        "validation_strategy": "holdout",
        "validation_params": {"test_size": 0.2, "random_state": 42},
    }

    # Random Forest example
    rf_config = {
        "name": "random_forest",
        "model_class": "sklearn.ensemble.RandomForestRegressor",
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
        "type": "sklearn",
        "fit_on_predict_required": False,
        "training_params": {},
        "validation_strategy": "holdout",
        "validation_params": {"test_size": 0.2, "random_state": 42},
    }

    # LSTM example (PyTorch Lightning)
    lstm_config = {
        "name": "lstm",
        "model_class": "ananke.core.models.torch.lstm.LSTMForecaster",
        "model_params": {
            "input_size": 10,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "output_size": 1,
        },
        "type": "pytorch",
        "fit_on_predict_required": False,
        "training_params": {
            "max_epochs": 100,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 10,
            "mode": "min",
        },
        "validation_strategy": "holdout",
        "validation_params": {"test_size": 0.2},
    }

    configs = [
        ("linear_regression.yaml", linear_config),
        ("random_forest.yaml", rf_config),
        ("lstm.yaml", lstm_config),
    ]

    for filename, config in configs:
        try:
            filepath = os.path.join(config_dir, filename)
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"Created example model configuration at {filepath}")
        except Exception as e:
            print(f"Error creating example configuration {filename}: {str(e)}")


# Common model class mappings for convenience
COMMON_MODELS = {
    # Sklearn models
    "linear_regression": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "lasso": "sklearn.linear_model.Lasso",
    "random_forest": "sklearn.ensemble.RandomForestRegressor",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "svr": "sklearn.svm.SVR",
    # Darts models (if available)
    "prophet": "darts.models.Prophet",
    "arima": "darts.models.AutoARIMA",
    "ets": "darts.models.AutoETS",
    "nlinear": "darts.models.NLinearModel",
    "rnn": "darts.models.RNNModel",
    "tft": "darts.models.TFTModel",
    "lightgbm": "darts.models.LightGBMModel",
}
