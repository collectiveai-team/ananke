# Ananke (ties to predicting the future)

A comprehensive timeseries experimentation framework extracted from generalizable components of production ML systems.

## Features

- **Unified Configuration System**: YAML-based configuration for data processing and model parameters
- **Multiple ML Frameworks**: Support for scikit-learn, PyTorch, Darts, and custom models
- **Experiment Tracking**: Built-in MLflow integration for experiment management
- **Evaluation Framework**: Comprehensive metrics and visualization for timeseries models
- **CLI Interface**: Command-line tools for project initialization and experiment management
- **Modular Architecture**: Clean separation of concerns with extensible components

## Installation

```bash
# Install from source
cd ananke
uv pip install -e .

# Install with all dependencies
uv pip install -e ".[all]"
```

## Quick Start

### 1. Initialize a New Project

```bash
ananke init my_timeseries_project
cd my_timeseries_project
```

This creates a project structure with example configurations and an experiment script.

### 2. Configure Your Data and Models

Edit the configuration files in the `configs/` directory:

- `configs/data/default.yaml`: Data processing configuration
- `configs/model/*.yaml`: Model configurations

### 3. Run an Experiment

```bash
python experiments/example_experiment.py
```

Or use the CLI:

```bash
ananke run-experiment experiments/example_experiment.py
```

## Architecture

### Core Components

- **`ananke.core.data`**: Data structures and processing utilities
- **`ananke.core.runners`**: Base runner framework and implementations
- **`ananke.core.evaluation`**: Metrics and visualization tools
- **`ananke.core.utils`**: MLflow utilities and helper functions

### Configuration System

- **`ananke.core.configs.data`**: Data configuration management
- **`ananke.core.configs.model`**: Model configuration management

### CLI Interface

- **`ananke.cli`**: Command-line interface for project management

## Usage Examples

### Basic Experiment

```python
from pathlib import Path
from ananke.core.configs.data.config import get_data_configurations
from ananke.core.configs.model.config import get_model_configurations
from ananke.core.runners.sklearn_runner import SklearnRunner, SklearnRunConfig

# Load configurations
data_configs = get_data_configurations("configs/data")
model_configs = get_model_configurations("configs/model")

# Create run configuration
run_config = SklearnRunConfig(
    name="my_experiment",
    output_dir=Path("results"),
    log_to_mlflow=True,
)

# Run experiment
for data_config in data_configs:
    for model_config in model_configs:
        if model_config.type == "sklearn":
            runner = SklearnRunner(run_config, data_config, model_config)
            results = runner.run()
            print(f"Results: {results}")
```

### Custom Runner Implementation

```python
from ananke.core.runners.base_runner import BaseRunner
from ananke.core.data.meta.dataset import TimeSeriesDataset

class CustomRunner(BaseRunner):
    def prepare_data(self) -> TimeSeriesDataset:
        # Implement your data preparation logic
        pass

    def setup_model(self) -> None:
        # Implement your model setup logic
        pass

    def train(self) -> None:
        # Implement your training logic
        pass

    def predict(self, data):
        # Implement your prediction logic
        pass

    def _to_array(self, data):
        # Implement data conversion logic
        pass
```

### MLflow Integration

```python
from ananke.core.utils.mlflow_utils import setup_mlflow, start_run, log_figure
import matplotlib.pyplot as plt

# Setup MLflow
experiment_id = setup_mlflow(
    tracking_uri="file:./mlruns",
    experiment_name="my_experiments"
)

# Start a run
with start_run(run_name="test_run"):
    # Your experiment code here

    # Log a figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    log_figure(fig, "plots", "my_plot")
```

## Configuration

### Data Configuration

```yaml
name: "my_data_config"
features:
  - "feature_1"
  - "feature_2"
  - "feature_3"
target_feature: "target"
feature_window_hours: 24
target_window_hours: 6
train_stride_hours: 6
test_stride_hours: 6
future_feature_cols:
  - "feature_1"
  - "feature_2"
past_feature_cols:
  - "feature_1"
  - "feature_2"
  - "feature_3"
target_cols:
  - "target"
preprocessing:
  scaler:
    type: "standard"
    params: {}
  imputation:
    enabled: true
    method: "forward_fill"
  smoothing:
    enabled: false
    method: "ewm"
    params:
      alpha: 0.3
```

### Model Configuration

```yaml
name: "random_forest"
model_class: "sklearn.ensemble.RandomForestRegressor"
model_params:
  n_estimators: 100
  max_depth: 10
  random_state: 42
type: "sklearn"
training_params: {}
validation_strategy: "holdout"
validation_params:
  test_size: 0.2
  random_state: 42
```

## CLI Commands

```bash
# Initialize a new project
ananke init my_project

# List available configurations
ananke list-configs --config-dir configs

# Validate a configuration file
ananke validate-config configs/data/default.yaml

# Run an experiment
ananke run-experiment experiments/my_experiment.py

# Show version
ananke version
```

## Supported Model Types

### Scikit-learn Models
- Linear models (LinearRegression, Ridge, Lasso)
- Ensemble methods (RandomForest, GradientBoosting)
- Support Vector Machines
- And any other scikit-learn compatible model

### PyTorch Models
- Custom neural networks
- LSTM/GRU models
- Transformer architectures
- PyTorch Lightning integration

### Darts Models
- Prophet
- ARIMA/AutoARIMA
- Exponential Smoothing
- Neural network models (NLinear, TFT, etc.)

## Evaluation Metrics

Ananke provides comprehensive evaluation capabilities:

- **Regression Metrics**: MSE, RMSE, MAE, R², MAPE
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Time Series Specific**: Event detection, threshold crossing analysis
- **Visualization**: Confusion matrices, prediction plots, feature importance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This package was extracted from generalizable components developed for production timeseries ML systems, with a focus on reproducibility and modularity.
