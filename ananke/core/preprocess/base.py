"""Base preprocessing interface for timeseries data."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BasePreprocessor(ABC):
    """Base class for all preprocessing operations."""

    def __init__(self, **params):
        """Initialize preprocessor with parameters."""
        self.params = params
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame | np.ndarray) -> "BasePreprocessor":
        """Fit the preprocessor to the data."""
        pass

    @abstractmethod
    def transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Transform the data using fitted preprocessor."""
        pass

    def fit_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Fit and transform the data in one step."""
        return self.fit(data).transform(data)

    @abstractmethod
    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform the data."""
        pass

    def get_params(self) -> dict[str, Any]:
        """Get preprocessor parameters."""
        return self.params.copy()

    def set_params(self, **params) -> "BasePreprocessor":
        """Set preprocessor parameters."""
        self.params.update(params)
        return self


class PreprocessingPipeline:
    """Pipeline for chaining multiple preprocessing steps."""

    def __init__(self, steps: list[tuple[str, BasePreprocessor]]):
        """
        Initialize preprocessing pipeline.

        Args:
            steps: List of (name, preprocessor) tuples
        """
        self.steps = steps
        self.named_steps = {name: preprocessor for name, preprocessor in steps}

    def fit(self, data: pd.DataFrame | np.ndarray) -> "PreprocessingPipeline":
        """Fit all preprocessing steps."""
        current_data = data

        for name, preprocessor in self.steps:
            preprocessor.fit(current_data)
            current_data = preprocessor.transform(current_data)

        return self

    def transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Transform data through all preprocessing steps."""
        current_data = data

        for name, preprocessor in self.steps:
            if not preprocessor.is_fitted:
                raise ValueError(f"Preprocessor '{name}' is not fitted")
            current_data = preprocessor.transform(current_data)

        return current_data

    def fit_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Fit and transform data in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Inverse transform data through all preprocessing steps in reverse order."""
        current_data = data

        # Apply inverse transforms in reverse order
        for name, preprocessor in reversed(self.steps):
            current_data = preprocessor.inverse_transform(current_data)

        return current_data

    def get_step(self, name: str) -> BasePreprocessor:
        """Get a preprocessing step by name."""
        return self.named_steps[name]
