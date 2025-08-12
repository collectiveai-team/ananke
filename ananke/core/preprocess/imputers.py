"""Imputation preprocessors for timeseries data."""

import numpy as np
import pandas as pd

from ananke.core.preprocess.base import BasePreprocessor


class ForwardFillImputer(BasePreprocessor):
    """Forward fill imputer for timeseries data."""

    def __init__(self, limit: int | None = None):
        """
        Initialize forward fill imputer.

        Args:
            limit: Maximum number of consecutive NaN values to forward fill
        """
        super().__init__(limit=limit)
        self.limit = limit

    def fit(self, data: pd.DataFrame | np.ndarray) -> "ForwardFillImputer":
        """Fit the imputer (no-op for forward fill)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using forward fill."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            return data.fillna(method="ffill", limit=self.limit)
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            filled = df.fillna(method="ffill", limit=self.limit)
            return filled.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no-op for imputation)."""
        return data


class BackwardFillImputer(BasePreprocessor):
    """Backward fill imputer for timeseries data."""

    def __init__(self, limit: int | None = None):
        """
        Initialize backward fill imputer.

        Args:
            limit: Maximum number of consecutive NaN values to backward fill
        """
        super().__init__(limit=limit)
        self.limit = limit

    def fit(self, data: pd.DataFrame | np.ndarray) -> "BackwardFillImputer":
        """Fit the imputer (no-op for backward fill)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using backward fill."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            return data.fillna(method="bfill", limit=self.limit)
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            filled = df.fillna(method="bfill", limit=self.limit)
            return filled.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no-op for imputation)."""
        return data


class InterpolateImputer(BasePreprocessor):
    """Interpolation imputer for timeseries data."""

    def __init__(self, method: str = "linear", limit: int | None = None):
        """
        Initialize interpolation imputer.

        Args:
            method: Interpolation method ('linear', 'polynomial', 'spline', etc.)
            limit: Maximum number of consecutive NaN values to interpolate
        """
        super().__init__(method=method, limit=limit)
        self.method = method
        self.limit = limit

    def fit(self, data: pd.DataFrame | np.ndarray) -> "InterpolateImputer":
        """Fit the imputer (no-op for interpolation)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using interpolation."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            return data.interpolate(method=self.method, limit=self.limit)
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            interpolated = df.interpolate(method=self.method, limit=self.limit)
            return interpolated.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no-op for imputation)."""
        return data


class MeanImputer(BasePreprocessor):
    """Mean imputer for timeseries data."""

    def __init__(self):
        """Initialize mean imputer."""
        super().__init__()
        self.means = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MeanImputer":
        """Fit the imputer by computing means."""
        if isinstance(data, pd.DataFrame):
            self.means = data.mean()
            self.columns = data.columns
        else:
            self.means = np.nanmean(data, axis=0)
            self.columns = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using mean imputation."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            return data.fillna(self.means)
        else:
            # For numpy arrays
            result = data.copy()
            for i, mean_val in enumerate(self.means):
                mask = np.isnan(result[:, i])
                result[mask, i] = mean_val
            return result

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no-op for imputation)."""
        return data


class MedianImputer(BasePreprocessor):
    """Median imputer for timeseries data."""

    def __init__(self):
        """Initialize median imputer."""
        super().__init__()
        self.medians = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MedianImputer":
        """Fit the imputer by computing medians."""
        if isinstance(data, pd.DataFrame):
            self.medians = data.median()
            self.columns = data.columns
        else:
            self.medians = np.nanmedian(data, axis=0)
            self.columns = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using median imputation."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            return data.fillna(self.medians)
        else:
            # For numpy arrays
            result = data.copy()
            for i, median_val in enumerate(self.medians):
                mask = np.isnan(result[:, i])
                result[mask, i] = median_val
            return result

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no-op for imputation)."""
        return data
