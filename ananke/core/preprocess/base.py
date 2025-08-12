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

    def __init__(self, steps: list[tuple[str, BasePreprocessor]] | list[BasePreprocessor]):
        """
        Initialize preprocessing pipeline.

        Args:
            steps: List of (name, preprocessor) tuples or list of preprocessor objects
        """
        # Handle both formats: list of tuples or list of preprocessors
        if steps and isinstance(steps[0], tuple):
            # Format: [(name, preprocessor), ...]
            self.steps = steps
            self.named_steps = dict(steps)
        else:
            # Format: [preprocessor, ...]
            # Generate names automatically
            self.steps = [(f"step_{i}", preprocessor) for i, preprocessor in enumerate(steps)]
            self.named_steps = dict(self.steps)

    def fit(self, data: pd.DataFrame | np.ndarray) -> "PreprocessingPipeline":
        """Fit all preprocessing steps."""
        current_data = data

        for _, preprocessor in self.steps:
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
        for _, preprocessor in reversed(self.steps):
            current_data = preprocessor.inverse_transform(current_data)

        return current_data

    def get_step(self, name: str) -> BasePreprocessor:
        """Get a preprocessing step by name."""
        return self.named_steps[name]

    def __repr__(self) -> str:
        """Return string representation of the pipeline."""
        step_names = [f"{name}: {preprocessor.__class__.__name__}" for name, preprocessor in self.steps]
        return f"PreprocessingPipeline(steps=[{', '.join(step_names)}])"
