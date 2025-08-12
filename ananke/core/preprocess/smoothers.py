"""Smoothing preprocessors for timeseries data."""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ananke.core.preprocess.base import BasePreprocessor


class EWMSmoother(BasePreprocessor):
    """Exponentially weighted moving average smoother."""

    def __init__(self, alpha: float = 0.3, adjust: bool = True):
        """
        Initialize EWM smoother.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
            adjust: Whether to adjust for bias
        """
        super().__init__(alpha=alpha, adjust=adjust)
        self.alpha = alpha
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
            return data.ewm(alpha=self.alpha, adjust=self.adjust).mean()
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
        self, window: int = 5, center: bool = True, min_periods: int | None = None
    ):
        """
        Initialize rolling mean smoother.

        Args:
            window: Size of the rolling window
            center: Whether to center the window
            min_periods: Minimum number of observations required
        """
        super().__init__(window=window, center=center, min_periods=min_periods)
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
            return data.rolling(
                window=self.window, center=self.center, min_periods=self.min_periods
            ).mean()
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
        self, window_length: int = 11, polyorder: int = 3, mode: str = "interp"
    ):
        """
        Initialize Savitzky-Golay smoother.

        Args:
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used to fit the samples
            mode: How to handle boundaries ('interp', 'mirror', 'constant', etc.)
        """
        super().__init__(window_length=window_length, polyorder=polyorder, mode=mode)
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
            smoothed_data = data.copy()
            for col in data.columns:
                if len(data) >= self.window_length:
                    smoothed_data[col] = savgol_filter(
                        data[col].values,
                        self.window_length,
                        self.polyorder,
                        mode=self.mode,
                    )
            return smoothed_data
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

    def __init__(self, window: int = 5, center: bool = True):
        """
        Initialize median smoother.

        Args:
            window: Size of the rolling window
            center: Whether to center the window
        """
        super().__init__(window=window, center=center)
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
            return data.rolling(window=self.window, center=self.center).median()
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
