"""Smoothing preprocessors for timeseries data."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ananke.core.preprocess.base import BasePreprocessor


class EWMSmoother(BasePreprocessor):
    """Exponentially weighted moving average smoother."""

    def __init__(self, columns: list[str] | None = None, alpha: float | None = None, span: int | None = None, adjust: bool = True):
        """
        Initialize EWM smoother.

        Args:
            columns: List of columns to smooth. If None, smooths all numeric columns.
            alpha: Smoothing factor (0 < alpha <= 1). Mutually exclusive with span.
            span: Span for the exponentially weighted window. Mutually exclusive with alpha.
            adjust: Whether to adjust for bias
        """
        if alpha is not None and span is not None:
            raise ValueError("Cannot specify both alpha and span")
        if alpha is None and span is None:
            alpha = 0.3  # Default value

        super().__init__(columns=columns, alpha=alpha, span=span, adjust=adjust)
        self.columns = columns
        self.alpha = alpha
        self.span = span
        self.adjust = adjust

    def fit(self, data: pd.DataFrame | np.ndarray) -> "EWMSmoother":
        """Fit the smoother (no-op for EWM)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using EWM smoothing."""
        if not self.is_fitted:
            raise ValueError("Smoother is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.columns is not None:
                # Smooth only specified columns
                if self.alpha is not None:
                    result[self.columns] = result[self.columns].ewm(alpha=self.alpha, adjust=self.adjust).mean()
                else:
                    result[self.columns] = result[self.columns].ewm(span=self.span, adjust=self.adjust).mean()
            else:
                # Smooth all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if self.alpha is not None:
                    result[numeric_cols] = result[numeric_cols].ewm(alpha=self.alpha, adjust=self.adjust).mean()
                else:
                    result[numeric_cols] = result[numeric_cols].ewm(span=self.span, adjust=self.adjust).mean()
            return result
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            smoothed = df.ewm(alpha=self.alpha, adjust=self.adjust).mean()
            return smoothed.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no exact inverse for EWM)."""
        # EWM smoothing doesn't have an exact inverse
        return data


class RollingMeanSmoother(BasePreprocessor):
    """Rolling mean smoother."""

    def __init__(
        self, columns: list[str] | None = None, window: int = 5, center: bool = True, min_periods: int | None = None
    ):
        """
        Initialize rolling mean smoother.

        Args:
            columns: List of columns to smooth. If None, smooths all numeric columns.
            window: Size of the rolling window
            center: Whether to center the window
            min_periods: Minimum number of observations required
        """
        super().__init__(columns=columns, window=window, center=center, min_periods=min_periods)
        self.columns = columns
        self.window = window
        self.center = center
        self.min_periods = min_periods

    def fit(self, data: pd.DataFrame | np.ndarray) -> "RollingMeanSmoother":
        """Fit the smoother (no-op for rolling mean)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using rolling mean smoothing."""
        if not self.is_fitted:
            raise ValueError("Smoother is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.columns is not None:
                # Smooth only specified columns
                result[self.columns] = result[self.columns].rolling(
                    window=self.window, center=self.center, min_periods=self.min_periods
                ).mean()
            else:
                # Smooth all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                result[numeric_cols] = result[numeric_cols].rolling(
                    window=self.window, center=self.center, min_periods=self.min_periods
                ).mean()
            return result
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            smoothed = df.rolling(
                window=self.window, center=self.center, min_periods=self.min_periods
            ).mean()
            return smoothed.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no exact inverse for rolling mean)."""
        # Rolling mean smoothing doesn't have an exact inverse
        return data


class SavgolSmoother(BasePreprocessor):
    """Savitzky-Golay filter smoother."""

    def __init__(
        self, columns: list[str] | None = None, window_length: int = 11, polyorder: int = 3, mode: str = "interp"
    ):
        """
        Initialize Savitzky-Golay smoother.

        Args:
            columns: List of columns to smooth. If None, smooths all numeric columns.
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used to fit the samples
            mode: How to handle boundaries ('interp', 'mirror', 'constant', etc.)
        """
        super().__init__(columns=columns, window_length=window_length, polyorder=polyorder, mode=mode)
        self.columns = columns
        self.window_length = window_length
        self.polyorder = polyorder
        self.mode = mode

        # Ensure window_length is odd
        if window_length % 2 == 0:
            self.window_length = window_length + 1

    def fit(self, data: pd.DataFrame | np.ndarray) -> "SavgolSmoother":
        """Fit the smoother (no-op for Savgol)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using Savitzky-Golay smoothing."""
        if not self.is_fitted:
            raise ValueError("Smoother is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            cols_to_smooth = self.columns if self.columns is not None else data.select_dtypes(include=[np.number]).columns.tolist()

            for col in cols_to_smooth:
                if len(data) >= self.window_length:
                    result[col] = savgol_filter(
                        data[col].values,
                        self.window_length,
                        self.polyorder,
                        mode=self.mode,
                    )
            return result
        else:
            if len(data) < self.window_length:
                return data

            if data.ndim == 1:
                return savgol_filter(
                    data, self.window_length, self.polyorder, mode=self.mode
                )
            else:
                smoothed = np.zeros_like(data)
                for i in range(data.shape[1]):
                    smoothed[:, i] = savgol_filter(
                        data[:, i], self.window_length, self.polyorder, mode=self.mode
                    )
                return smoothed

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no exact inverse for Savgol)."""
        # Savitzky-Golay smoothing doesn't have an exact inverse
        return data


class MedianSmoother(BasePreprocessor):
    """Median filter smoother."""

    def __init__(self, columns: list[str] | None = None, window: int = 5, center: bool = True):
        """
        Initialize median smoother.

        Args:
            columns: List of columns to smooth. If None, smooths all numeric columns.
            window: Size of the rolling window
            center: Whether to center the window
        """
        super().__init__(columns=columns, window=window, center=center)
        self.columns = columns
        self.window = window
        self.center = center

    def fit(self, data: pd.DataFrame | np.ndarray) -> "MedianSmoother":
        """Fit the smoother (no-op for median)."""
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Transform the data using median smoothing."""
        if not self.is_fitted:
            raise ValueError("Smoother is not fitted")

        if isinstance(data, pd.DataFrame):
            result = data.copy()
            if self.columns is not None:
                # Smooth only specified columns
                result[self.columns] = result[self.columns].rolling(window=self.window, center=self.center).median()
            else:
                # Smooth all numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                result[numeric_cols] = result[numeric_cols].rolling(window=self.window, center=self.center).median()
            return result
        else:
            # For numpy arrays, convert to DataFrame temporarily
            df = pd.DataFrame(data)
            smoothed = df.rolling(window=self.window, center=self.center).median()
            return smoothed.values

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform (no exact inverse for median)."""
        # Median smoothing doesn't have an exact inverse
        return data
