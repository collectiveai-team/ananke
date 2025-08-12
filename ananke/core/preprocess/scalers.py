"""Scaling preprocessors for timeseries data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from ananke.core.preprocess.base import BasePreprocessor


class StandardScaler(BasePreprocessor):
    """Standard scaler for timeseries data."""

    def __init__(self, columns: list[str] | None = None, with_mean: bool = True, with_std: bool = True):
        """
        Initialize standard scaler.

        Args:
            columns: List of columns to scale. If None, scales all numeric columns.
            with_mean: Whether to center the data
            with_std: Whether to scale to unit variance
        """
        super().__init__(columns=columns, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.scaler = SklearnStandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "StandardScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            if self.columns is not None:
                # Only fit on specified columns
                self.scaler.fit(data[self.columns].values)
                self.fitted_columns = self.columns
            else:
                # Fit on all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.scaler.fit(data[numeric_cols].values)
                self.fitted_columns = numeric_cols
            self.original_columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.fitted_columns = None
            self.original_columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Transform only the fitted columns
                scaled_values = self.scaler.transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = scaled_values
            return result
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Inverse transform only the fitted columns
                original_values = self.scaler.inverse_transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = original_values
            return result
        else:
            return self.scaler.inverse_transform(data)


class MinMaxScaler(BasePreprocessor):
    """Min-max scaler for timeseries data."""

    def __init__(self, columns: list[str] | None = None, feature_range: tuple = (0, 1)):
        """
        Initialize min-max scaler.

        Args:
            columns: List of columns to scale. If None, scales all numeric columns.
            feature_range: Desired range of transformed data
        """
        super().__init__(columns=columns, feature_range=feature_range)
        self.columns = columns
        self.scaler = SklearnMinMaxScaler(feature_range=feature_range)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MinMaxScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            if self.columns is not None:
                # Only fit on specified columns
                self.scaler.fit(data[self.columns].values)
                self.fitted_columns = self.columns
            else:
                # Fit on all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.scaler.fit(data[numeric_cols].values)
                self.fitted_columns = numeric_cols
            self.original_columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.fitted_columns = None
            self.original_columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Transform only the fitted columns
                scaled_values = self.scaler.transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = scaled_values
            return result
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Inverse transform only the fitted columns
                original_values = self.scaler.inverse_transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = original_values
            return result
        else:
            return self.scaler.inverse_transform(data)


class RobustScaler(BasePreprocessor):
    """Robust scaler for timeseries data."""

    def __init__(
        self,
        columns: list[str] | None = None,
        quantile_range: tuple = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        """
        Initialize robust scaler.

        Args:
            columns: List of columns to scale. If None, scales all numeric columns.
            quantile_range: Quantile range used to calculate scale
            with_centering: Whether to center the data at the median
            with_scaling: Whether to scale the data to interquartile range
        """
        super().__init__(
            columns=columns,
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
        self.columns = columns
        self.scaler = SklearnRobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

    def fit(self, data: pd.DataFrame | np.ndarray) -> "RobustScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            if self.columns is not None:
                # Only fit on specified columns
                self.scaler.fit(data[self.columns].values)
                self.fitted_columns = self.columns
            else:
                # Fit on all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.scaler.fit(data[numeric_cols].values)
                self.fitted_columns = numeric_cols
            self.original_columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.fitted_columns = None
            self.original_columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Transform only the fitted columns
                scaled_values = self.scaler.transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = scaled_values
            return result
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Inverse transform only the fitted columns
                original_values = self.scaler.inverse_transform(data[self.fitted_columns].values)
                result[self.fitted_columns] = original_values
            return result
        else:
            return self.scaler.inverse_transform(data)
