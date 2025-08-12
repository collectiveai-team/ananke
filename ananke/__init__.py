"""
Timeseries Jarvis (TSJ) - A comprehensive timeseries experimentation framework.

This package provides a unified framework for timeseries machine learning experiments,
including data processing, model training, evaluation, and experiment tracking.
"""

__version__ = "0.1.0"
__author__ = "Collective AI"
__email__ = "info@collective.ai"

# Core imports for easy access
from ananke.core.configs.data.config import DataConfig
from ananke.core.configs.model.config import ModelConfig
from ananke.core.data.meta.dataset import TimeSeriesData, TimeSeriesDataset
from ananke.core.runners.base_runner import BaseRunner, RunConfig

__all__ = [
    "TimeSeriesData",
    "TimeSeriesDataset",
    "BaseRunner",
    "RunConfig",
    "DataConfig",
    "ModelConfig",
]
