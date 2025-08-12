"""Imputation preprocessors for timeseries data."""

import numpy as np
import pandas as pd

from ananke.core.preprocess.base import BasePreprocessor


class ForwardFillImputer(BasePreprocessor):
    """Forward fill imputer for timeseries data."""

    def __init__(self, columns: list[str] | None = None, limit: int | None = None):
        """
        Initialize forward fill imputer.

        Args:
            columns: List of columns to impute. If None, imputes all columns.
            limit: Maximum number of consecutive NaN values to forward fill
        """
        super().__init__(columns=columns, limit=limit)
        self.columns = columns
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
            result = data.copy()
            if self.columns is not None:
                # Impute only specified columns
                result[self.columns] = result[self.columns].fillna(method="ffill", limit=self.limit)
            else:
                # Impute all columns
                result = result.fillna(method="ffill", limit=self.limit)
            return result
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

    def __init__(self, columns: list[str] | None = None, limit: int | None = None):
        """
        Initialize backward fill imputer.

        Args:
            columns: List of columns to impute. If None, imputes all columns.
            limit: Maximum number of consecutive NaN values to backward fill
        """
        super().__init__(columns=columns, limit=limit)
        self.columns = columns
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
            result = data.copy()
            if self.columns is not None:
                # Impute only specified columns
                result[self.columns] = result[self.columns].fillna(method="bfill", limit=self.limit)
            else:
                # Impute all columns
                result = result.fillna(method="bfill", limit=self.limit)
            return result
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

    def __init__(self, columns: list[str] | None = None, method: str = "linear", limit: int | None = None):
        """
        Initialize interpolation imputer.

        Args:
            columns: List of columns to impute. If None, imputes all columns.
            method: Interpolation method ('linear', 'polynomial', 'spline', etc.)
            limit: Maximum number of consecutive NaN values to interpolate
        """
        super().__init__(columns=columns, method=method, limit=limit)
        self.columns = columns
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
            result = data.copy()
            if self.columns is not None:
                # Impute only specified columns
                result[self.columns] = result[self.columns].interpolate(method=self.method, limit=self.limit)
            else:
                # Impute all columns
                result = result.interpolate(method=self.method, limit=self.limit)
            return result
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

    def __init__(self, columns: list[str] | None = None):
        """Initialize mean imputer.

        Args:
            columns: List of columns to impute. If None, imputes all numeric columns.
        """
        super().__init__(columns=columns)
        self.columns = columns
        self.means = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MeanImputer":
        """Fit the imputer by computing means."""
        if isinstance(data, pd.DataFrame):
            if self.columns is not None:
                # Only compute means for specified columns
                self.means = data[self.columns].mean()
                self.fitted_columns = self.columns
            else:
                # Compute means for all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.means = data[numeric_cols].mean()
                self.fitted_columns = numeric_cols
            self.original_columns = data.columns
        else:
            self.means = np.nanmean(data, axis=0)
            self.fitted_columns = None
            self.original_columns = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using mean imputation."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Impute only the fitted columns
                result[self.fitted_columns] = result[self.fitted_columns].fillna(self.means)
            return result
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

    def __init__(self, columns: list[str] | None = None):
        """Initialize median imputer.

        Args:
            columns: List of columns to impute. If None, imputes all numeric columns.
        """
        super().__init__(columns=columns)
        self.columns = columns
        self.medians = None

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MedianImputer":
        """Fit the imputer by computing medians."""
        if isinstance(data, pd.DataFrame):
            if self.columns is not None:
                # Only compute medians for specified columns
                self.medians = data[self.columns].median()
                self.fitted_columns = self.columns
            else:
                # Compute medians for all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                self.medians = data[numeric_cols].median()
                self.fitted_columns = numeric_cols
            self.original_columns = data.columns
        else:
            self.medians = np.nanmedian(data, axis=0)
            self.fitted_columns = None
            self.original_columns = None

        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using median imputation."""
        if not self.is_fitted:
            raise ValueError("Imputer is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.fitted_columns is not None:
                # Impute only the fitted columns
                result[self.fitted_columns] = result[self.fitted_columns].fillna(self.medians)
            return result
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
