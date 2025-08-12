"""Generalizable data configuration for timeseries experiments."""

import glob
import os
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for timeseries data processing and windowing."""

    name: str
    features: list[str]
    target_feature: str
    stride: int = 1
    feature_window_hours: float
    target_window_hours: float
    train_stride_hours: float
    test_stride_hours: float
    incremental_features: bool = False

    # Optional filtering parameters
    min_feature_window_hours: float | None = None
    max_feature_window_hours: float | None = None

    # Feature configuration
    features_to_forecast: list[str] | None = None
    feature_forecaster_config: dict[str, Any] | None = None
    future_feature_cols: list[str] = Field(default_factory=list)
    past_feature_cols: list[str] = Field(default_factory=list)
    target_cols: list[str] = Field(default_factory=list)

    # Time range configuration
    train_start: str | None = None
    train_end: str | None = None
    test_start: str | None = None
    test_end: str | None = None

    # Event and filtering configuration
    event_threshold: float | None = None
    filter_months: list[int] | None = None
    val_ratio: float = 0.0

    # Preprocessing configuration
    preprocessing: dict[str, Any] = Field(default_factory=dict)


def get_data_configurations(
    config_dir: str, config_name: str | None = None
) -> list[DataConfig]:
    """Get data preprocessing configurations from YAML files.

    Args:
        config_dir: Directory containing configuration files
        config_name: Optional name of a specific configuration to load
                   If None, all configurations are loaded

    Returns:
        List of DataConfig objects
    """
    # Create directory if it doesn't exist
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        print(f"Created data configuration directory: {config_dir}")

    # Find all YAML configuration files
    all_config_files = glob.glob(os.path.join(config_dir, "*.yaml"))

    if not all_config_files:
        print("No data configuration files found. Using default configuration.")
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

            # Create DataConfig object
            config = DataConfig(**config_data)

            # Apply default preprocessing if not specified
            config = _apply_default_preprocessing(config)

            configs.append(config)
            print(f"Loaded data configuration from {os.path.basename(config_file)}")

        except Exception as e:
            print(f"Error loading configuration from {config_file}: {str(e)}")

    return configs


def _get_default_config() -> DataConfig:
    """Get a default data configuration."""
    return DataConfig(
        name="default",
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        target_feature="target",
        stride=1,
        feature_window_hours=24,
        target_window_hours=6,
        train_stride_hours=6,
        test_stride_hours=6,
        min_feature_window_hours=24,
        future_feature_cols=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        past_feature_cols=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        target_cols=["target"],
        preprocessing={
            "scaler": {"type": "standard", "params": {}},
            "imputation": {
                "enabled": False,
                "method": "forward_fill",
                "params": {},
            },
            "smoothing": {
                "enabled": False,
                "method": "ewm",
                "params": {"alpha": 0.3},
            },
        },
    )


def _apply_default_preprocessing(config: DataConfig) -> DataConfig:
    """Apply default preprocessing configurations if missing."""
    if not config.preprocessing:
        config.preprocessing = {}

    preproc = config.preprocessing

    # Add default preprocessing configurations if missing
    if "scaler" not in preproc:
        preproc["scaler"] = {"type": "standard", "params": {}}

    if "imputation" not in preproc:
        preproc["imputation"] = {
            "enabled": False,
            "method": "forward_fill",
            "params": {},
        }

    if "smoothing" not in preproc:
        preproc["smoothing"] = {
            "enabled": False,
            "method": "ewm",
            "params": {"alpha": 0.3},
        }

    return config


def create_example_config(config_dir: str) -> None:
    """Create an example data configuration YAML file."""
    example_path = os.path.join(config_dir, "default.yaml")

    # Skip if file already exists
    if os.path.exists(example_path):
        print(f"Example configuration already exists at {example_path}")
        return

    example_config = {
        "name": "default",
        "features": [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
        "target_feature": "target",
        "stride": 1,
        "feature_window_hours": 24,
        "target_window_hours": 6,
        "train_stride_hours": 6,
        "test_stride_hours": 6,
        "future_feature_cols": [
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        "past_feature_cols": [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
        "target_cols": ["target"],
        "preprocessing": {
            "scaler": {
                "type": "standard",  # standard, minmax, robust, none
                "params": {},
            },
            "imputation": {
                "enabled": True,
                "method": "forward_fill",  # forward_fill, backward_fill, interpolate
                "params": {},
            },
            "smoothing": {
                "enabled": True,
                "method": "ewm",  # ewm, rolling_mean, savgol
                "params": {
                    "alpha": 0.3  # for ewm
                    # "window": 5,  # for rolling_mean
                    # "window_length": 11, "polyorder": 3  # for savgol
                },
            },
        },
        "data_split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "augmentation": {
            "enabled": False,
            "methods": [],  # Could include "jitter", "scaling", "time_warp", etc.
        },
    }

    try:
        with open(example_path, "w") as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)
        print(f"Created example data configuration at {example_path}")
    except Exception as e:
        print(f"Error creating example configuration: {str(e)}")
