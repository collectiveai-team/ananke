"""Scaling preprocessors for timeseries data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from ananke.core.preprocess.base import BasePreprocessor


class StandardScaler(BasePreprocessor):
    """Standard scaler for timeseries data."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Initialize standard scaler.

        Args:
            with_mean: Whether to center the data
            with_std: Whether to scale to unit variance
        """
        super().__init__(with_mean=with_mean, with_std=with_std)
        self.scaler = SklearnStandardScaler(with_mean=with_mean, with_std=with_std)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "StandardScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            self.scaler.fit(data.values)
            self.columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            scaled_data = self.scaler.transform(data.values)
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            original_data = self.scaler.inverse_transform(data.values)
            return pd.DataFrame(original_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.inverse_transform(data)


class MinMaxScaler(BasePreprocessor):
    """Min-max scaler for timeseries data."""

    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Initialize min-max scaler.

        Args:
            feature_range: Desired range of transformed data
        """
        super().__init__(feature_range=feature_range)
        self.scaler = SklearnMinMaxScaler(feature_range=feature_range)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MinMaxScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            self.scaler.fit(data.values)
            self.columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            scaled_data = self.scaler.transform(data.values)
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            original_data = self.scaler.inverse_transform(data.values)
            return pd.DataFrame(original_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.inverse_transform(data)


class RobustScaler(BasePreprocessor):
    """Robust scaler for timeseries data."""

    def __init__(
        self,
        quantile_range: tuple = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        """
        Initialize robust scaler.

        Args:
            quantile_range: Quantile range used to calculate scale
            with_centering: Whether to center the data at the median
            with_scaling: Whether to scale the data to interquartile range
        """
        super().__init__(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )
        self.scaler = SklearnRobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling,
        )

    def fit(self, data: pd.DataFrame | np.ndarray) -> "RobustScaler":
        """Fit the scaler to the data."""
        if isinstance(data, pd.DataFrame):
            self.scaler.fit(data.values)
            self.columns = data.columns
            self.index = data.index
        else:
            self.scaler.fit(data)
            self.columns = None
            self.index = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            scaled_data = self.scaler.transform(data.values)
            return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the scaled data."""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted")

        if isinstance(data, pd.DataFrame):
            original_data = self.scaler.inverse_transform(data.values)
            return pd.DataFrame(original_data, columns=data.columns, index=data.index)
        else:
            return self.scaler.inverse_transform(data)
