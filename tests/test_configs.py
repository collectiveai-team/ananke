"""Tests for configuration modules."""

import tempfile
from pathlib import Path

import yaml

from ananke.core.configs.data.config import (
    DataConfig,
    create_example_config,
    get_data_configurations,
)
from ananke.core.configs.model.config import (
    ModelConfig,
    create_example_configs,
    get_model_configurations,
)


class TestDataConfig:
    """Test DataConfig functionality."""

    def test_data_config_creation(self):
        """Test creating a DataConfig object."""
        config = DataConfig(
            name="test_config",
            features=["feature1", "feature2"],
            target_feature="target",
            feature_window_hours=24,
            target_window_hours=6,
            train_stride_hours=6,
            test_stride_hours=6,
            future_feature_cols=["feature1"],
            past_feature_cols=["feature1", "feature2"],
            target_cols=["target"],
        )

        assert config.name == "test_config"
        assert len(config.features) == 2
        assert config.target_feature == "target"

    def test_get_data_configurations(self):
        """Test loading data configurations from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create example config
            create_example_config(temp_dir)

            # Load configurations
            configs = get_data_configurations(temp_dir)

            assert len(configs) == 1
            assert configs[0].name == "default"

    def test_create_example_config(self):
        """Test creating example configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_example_config(temp_dir)

            config_file = Path(temp_dir) / "default.yaml"
            assert config_file.exists()

            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            assert "name" in config_data
            assert "features" in config_data


class TestModelConfig:
    """Test ModelConfig functionality."""

    def test_model_config_creation(self):
        """Test creating a ModelConfig object."""
        config = ModelConfig(
            name="test_model",
            model_class="sklearn.linear_model.LinearRegression",
            model_params={},
            type="sklearn",
        )

        assert config.name == "test_model"
        assert config.type == "sklearn"

    def test_get_model_configurations(self):
        """Test loading model configurations from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create example configs
            create_example_configs(temp_dir)

            # Load configurations
            configs = get_model_configurations(temp_dir)

            assert len(configs) >= 1
            assert any(config.type == "sklearn" for config in configs)

    def test_create_example_configs(self):
        """Test creating example configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_example_configs(temp_dir)

            # Check that files were created
            config_files = list(Path(temp_dir).glob("*.yaml"))
            assert len(config_files) >= 1
