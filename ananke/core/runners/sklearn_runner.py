"""Scikit-learn runner for timeseries experiments."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.data.meta.dataset import TimeSeriesData, TimeSeriesDataset
from ananke.core.runners.base_runner import BaseRunner, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class SklearnRunConfig(RunConfig):
    """Configuration for scikit-learn benchmark runs."""

    param_grid: dict[str, list[Any]] | None = None
    cv_folds: int = 5
    search_type: str = "grid"  # "grid" or "random"
    n_iter: int = 10  # For random search
    scoring: str | None = None
    refit: bool = True


class SklearnRunner(BaseRunner):
    """Benchmark runner for scikit-learn models."""

    def __init__(
        self,
        run_config: SklearnRunConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
    ):
        """
        Initialize the scikit-learn runner.

        Args:
            run_config: The run configuration
            data_config: The data configuration
            model_config: The model configuration
        """
        super().__init__(
            run_config=run_config, data_config=data_config, model_config=model_config
        )
        self.sklearn_config = run_config
        self.model: BaseEstimator | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

    def prepare_data(self) -> TimeSeriesDataset:
        """Prepare data for scikit-learn models."""
        logger.info("Preparing data for sklearn model...")

        # This is a placeholder - in practice, you'd implement your data loading logic
        # For now, we'll create dummy data to demonstrate the structure

        # Create dummy timeseries data
        n_samples = 1000
        n_features = len(self.data_config.features)
        n_targets = len(self.data_config.target_cols)

        # Generate synthetic data for demonstration
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples, n_targets)

        # Split data
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)

        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]
        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        # Store for later use
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Create TimeSeriesData objects (convert numpy arrays to DataFrames)
        feature_cols = self.data_config.features
        target_col = self.data_config.target_feature

        train_series_df = pd.DataFrame(X_train, columns=feature_cols)
        train_target_df = pd.DataFrame(y_train.reshape(-1, 1), columns=[target_col])
        train_data = [TimeSeriesData(series=train_series_df, target=train_target_df)]

        val_data = None
        if X_val is not None:
            val_series_df = pd.DataFrame(X_val, columns=feature_cols)
            val_target_df = pd.DataFrame(y_val.reshape(-1, 1), columns=[target_col])
            val_data = [TimeSeriesData(series=val_series_df, target=val_target_df)]

        test_series_df = pd.DataFrame(X_test, columns=feature_cols)
        test_target_df = pd.DataFrame(y_test.reshape(-1, 1), columns=[target_col])
        test_data = [TimeSeriesData(series=test_series_df, target=test_target_df)]

        return TimeSeriesDataset(
            train_data=train_data, val_data=val_data, test_data=test_data
        )

    def setup_model(self) -> None:
        """Set up the scikit-learn model."""
        logger.info(f"Setting up sklearn model: {self.model_config.name}")

        # Get model class and instantiate
        model_class = self.model_config.model_class
        model_params = self.model_config.model_params.copy()

        # Handle hyperparameter search
        if self.sklearn_config.param_grid:
            base_model = model_class(**model_params)

            if self.sklearn_config.search_type == "grid":
                self.model = GridSearchCV(
                    base_model,
                    param_grid=self.sklearn_config.param_grid,
                    cv=self.sklearn_config.cv_folds,
                    scoring=self.sklearn_config.scoring,
                    refit=self.sklearn_config.refit,
                    n_jobs=-1,
                    verbose=1,
                )
            elif self.sklearn_config.search_type == "random":
                self.model = RandomizedSearchCV(
                    base_model,
                    param_distributions=self.sklearn_config.param_grid,
                    n_iter=self.sklearn_config.n_iter,
                    cv=self.sklearn_config.cv_folds,
                    scoring=self.sklearn_config.scoring,
                    refit=self.sklearn_config.refit,
                    n_jobs=-1,
                    verbose=1,
                )
            else:
                raise ValueError(
                    f"Unknown search type: {self.sklearn_config.search_type}"
                )
        else:
            self.model = model_class(**model_params)

    def train(self) -> None:
        """Train the scikit-learn model."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")

        logger.info("Training sklearn model...")

        # Reshape targets if needed
        y_train = self.y_train
        if len(y_train.shape) > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()

        # Train the model
        self.model.fit(self.X_train, y_train)

        # Log best parameters if hyperparameter search was used
        if hasattr(self.model, "best_params_"):
            logger.info(f"Best parameters: {self.model.best_params_}")
            self.results["best_params"] = self.model.best_params_

        if hasattr(self.model, "best_score_"):
            logger.info(f"Best CV score: {self.model.best_score_}")
            self.results["best_cv_score"] = self.model.best_score_

    def predict(self, data: list[TimeSeriesData]) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features from TimeSeriesData
        X = np.vstack([ts_data.series for ts_data in data])

        # Make predictions
        predictions = self.model.predict(X)

        # Ensure predictions have the right shape
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)

        return predictions

    def _to_array(self, data: list[Any]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data[0], np.ndarray):
            return np.vstack(data)
        elif isinstance(data[0], pd.DataFrame):
            return np.vstack([df.values for df in data])
        else:
            return np.array(data)

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        import joblib

        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        import joblib

        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance if available."""
        if self.model is None:
            return None

        # Get the actual model (handle GridSearchCV/RandomizedSearchCV)
        model = (
            self.model.best_estimator_
            if hasattr(self.model, "best_estimator_")
            else self.model
        )

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = self.data_config.features
            return dict(zip(feature_names, importance, strict=False))
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # Take first output for multi-output
            feature_names = self.data_config.features
            return dict(zip(feature_names, np.abs(coef), strict=False))

        return None

    def _create_model(self) -> BaseEstimator:
        """Create and return the scikit-learn model (private method for testing)."""
        model_class = self.model_config.model_class
        model_params = self.model_config.model_params.copy()
        return model_class(**model_params)

    def _load_data(self) -> pd.DataFrame:
        """Load and return data (private method for testing)."""
        # This is a placeholder implementation for testing
        # In practice, this would load real data based on data_config
        n_samples = 100

        # Generate synthetic data
        dates = pd.date_range("2023-01-01", periods=n_samples, freq="h")
        data = {
            "timestamp": dates,
            **{
                feature: np.random.randn(n_samples)
                for feature in self.data_config.features
            },
            self.data_config.target_feature: np.random.randn(n_samples),
        }

        return pd.DataFrame(data)

    def _perform_hyperparameter_search(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> BaseEstimator:
        """Perform hyperparameter search (private method for testing)."""
        if not self.model_config.hyperparameter_search:
            raise ValueError("Hyperparameter search not configured")

        search_config = self.model_config.hyperparameter_search
        model_class = self.model_config.model_class
        base_model = model_class(**self.model_config.model_params)

        if search_config.get("method") == "grid":
            from sklearn.model_selection import GridSearchCV

            search_model = GridSearchCV(
                base_model,
                param_grid=search_config.get("param_grid", {}),
                cv=search_config.get("cv", 5),
                scoring=search_config.get("scoring"),
                n_jobs=-1,
            )
        else:
            from sklearn.model_selection import RandomizedSearchCV

            search_model = RandomizedSearchCV(
                base_model,
                param_distributions=search_config.get("param_grid", {}),
                n_iter=search_config.get("n_iter", 10),
                cv=search_config.get("cv", 5),
                scoring=search_config.get("scoring"),
                n_jobs=-1,
            )

        search_model.fit(X_train, y_train)
        return search_model.best_estimator_

    def _prepare_features(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for training and testing (private method for testing)."""
        # Extract features and target
        feature_cols = self.data_config.features
        target_col = self.data_config.target_feature

        X = data[feature_cols].values
        y = data[target_col].values

        # Split into train/test (80/20 split)
        split_idx = int(0.8 * len(X))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        return X_train, y_train, X_test, y_test
