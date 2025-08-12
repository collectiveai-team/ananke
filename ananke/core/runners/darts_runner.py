"""Darts runner for timeseries experiments."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.datasets import AirPassengersDataset

from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.data.meta.dataset import TimeSeriesData, TimeSeriesDataset
from ananke.core.runners.base_runner import BaseRunner, RunConfig

logger = logging.getLogger(__name__)


@dataclass
class DartsRunConfig(RunConfig):
    """Configuration for Darts benchmark runs."""

    search_type: str = "random"  # "grid" or "random"
    n_iter: int = 10  # For random search
    scoring: str | None = None
    refit: bool = True
    forecast_horizon: int = 12  # Number of steps to forecast


class DartsRunner(BaseRunner):
    """Benchmark runner for Darts models."""

    def __init__(
        self,
        run_config: DartsRunConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
    ):
        """
        Initialize the Darts runner.

        Args:
            run_config: The run configuration
            data_config: The data configuration
            model_config: The model configuration
        """
        super().__init__(
            run_config=run_config, data_config=data_config, model_config=model_config
        )
        self.darts_config = run_config
        self.model: Any | None = None
        self.train_series: TimeSeries | None = None
        self.val_series: TimeSeries | None = None
        self.test_series: TimeSeries | None = None

    def prepare_data(self) -> TimeSeriesDataset:
        """Prepare data for Darts models."""
        logger.info("Preparing data for Darts model...")

        # For demonstration, we'll use AirPassengers dataset
        # In practice, you'd load your own data here
        series = AirPassengersDataset().load()

        # Split the data
        train_size = int(0.7 * len(series))
        val_size = int(0.15 * len(series))

        self.train_series = series[:train_size]
        self.val_series = series[train_size : train_size + val_size]
        self.test_series = series[train_size + val_size :]

        # Create TimeSeriesData objects for compatibility
        train_data = [
            TimeSeriesData(
                series=self.train_series,
                target=self.train_series,  # For Darts, series and target are the same
            )
        ]

        val_data = (
            [
                TimeSeriesData(
                    series=self.val_series,
                    target=self.val_series,
                )
            ]
            if self.val_series is not None
            else None
        )

        test_data = [
            TimeSeriesData(
                series=self.test_series,
                target=self.test_series,
            )
        ]

        return TimeSeriesDataset(
            train_data=train_data, val_data=val_data, test_data=test_data
        )

    def setup_model(self) -> None:
        """Set up the Darts model."""
        logger.info(f"Setting up Darts model: {self.model_config.name}")

        # Get model class and instantiate
        model_class = self.model_config.model_class
        model_params = self.model_config.model_params.copy()

        # Create the model
        self.model = model_class(**model_params)

    def train(self) -> None:
        """Train the Darts model."""
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        if self.train_series is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")

        logger.info("Training Darts model...")

        # Train the model
        if hasattr(self.model, "fit"):
            # For models that need explicit fitting
            if self.val_series is not None:
                self.model.fit(self.train_series, val_series=self.val_series)
            else:
                self.model.fit(self.train_series)

        logger.info("Darts model training completed")

    def predict(self, data: list[TimeSeriesData]) -> np.ndarray:
        """Make predictions using the trained Darts model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = []

        for ts_data in data:
            # Get the series to predict from
            series = ts_data.series

            # Make prediction
            if hasattr(self.model, "predict"):
                pred = self.model.predict(
                    n=self.darts_config.forecast_horizon, series=series
                )
            else:
                # For models that don't have predict method
                pred = self.model.forecast(self.darts_config.forecast_horizon)

            # Convert to numpy array
            pred_values = pred.values()
            predictions.append(pred_values)

        # Stack predictions
        if predictions:
            return np.vstack(predictions)
        else:
            return np.array([])

    def _to_array(self, data: list[Any]) -> np.ndarray:
        """Convert data to numpy array."""
        if not data:
            return np.array([])

        arrays = []
        for item in data:
            if isinstance(item, TimeSeries):
                arrays.append(item.values())
            elif isinstance(item, np.ndarray):
                arrays.append(item)
            elif isinstance(item, pd.DataFrame):
                arrays.append(item.values)
            else:
                arrays.append(np.array(item))

        return np.vstack(arrays) if arrays else np.array([])

    def get_series_for_fit(
        self, data: list[TimeSeriesData]
    ) -> tuple[list[TimeSeries], list[TimeSeries], list[TimeSeries]]:
        """
        Extract series, past covariates, and future covariates from the dataset.

        Args:
            data: List of TimeSeriesData objects

        Returns:
            Tuple of (series, past_covariates, future_covariates)
        """
        if not data:
            return [], [], []

        series_list = []
        past_covariates_list = []
        future_covariates_list = []

        for item in data:
            # Handle series
            if isinstance(item.series, TimeSeries):
                series_list.append(item.series)
            else:
                # Convert to TimeSeries if needed
                series_list.append(TimeSeries.from_values(item.series))

            # Handle covariates
            if item.past_covariates is not None:
                if isinstance(item.past_covariates, TimeSeries):
                    past_covariates_list.append(item.past_covariates)
                else:
                    past_covariates_list.append(
                        TimeSeries.from_values(item.past_covariates)
                    )
            else:
                past_covariates_list.append(None)

            if item.future_covariates is not None:
                if isinstance(item.future_covariates, TimeSeries):
                    future_covariates_list.append(item.future_covariates)
                else:
                    future_covariates_list.append(
                        TimeSeries.from_values(item.future_covariates)
                    )
            else:
                future_covariates_list.append(None)

        return series_list, past_covariates_list, future_covariates_list

    def save_model(self, path: str) -> None:
        """Save the trained Darts model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        if hasattr(self.model, "save"):
            self.model.save(path)
        else:
            # Fallback to pickle
            import pickle

            with open(path, "wb") as f:
                pickle.dump(self.model, f)

        logger.info(f"Darts model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained Darts model."""
        try:
            # Try to load using Darts method first
            from darts.models import load_model

            self.model = load_model(path)
        except:
            # Fallback to pickle
            import pickle

            with open(path, "rb") as f:
                self.model = pickle.load(f)

        logger.info(f"Darts model loaded from {path}")

    def backtest(
        self,
        series: TimeSeries,
        start: float = 0.7,
        forecast_horizon: int | None = None,
    ) -> TimeSeries:
        """
        Perform backtesting on the series.

        Args:
            series: The time series to backtest on
            start: The point to start backtesting (as fraction of series length)
            forecast_horizon: Number of steps to forecast

        Returns:
            Backtesting predictions as TimeSeries
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        forecast_horizon = forecast_horizon or self.darts_config.forecast_horizon

        if hasattr(self.model, "backtest"):
            return self.model.backtest(
                series=series, start=start, forecast_horizon=forecast_horizon
            )
        else:
            logger.warning("Model does not support backtesting")
            return None
