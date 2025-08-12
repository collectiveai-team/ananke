"""Core timeseries data structures."""

from typing import Any, TypeAlias

import pandas as pd
from darts import TimeSeries
from pydantic import BaseModel
from torch import Tensor

SeriesType: TypeAlias = "pd.DataFrame | Tensor | TimeSeries"


class TimeSeriesData(BaseModel):
    """Container for a single timeseries sample with features and targets."""

    series: SeriesType
    target: SeriesType
    past_covariates: Any | None = None
    future_covariates: Any | None = None
    series_time_index: Any | None = None
    target_time_index: Any | None = None

    class Config:
        arbitrary_types_allowed = True


class TimeSeriesDataset(BaseModel):
    """Container for train/validation/test timeseries datasets."""

    train_data: list[TimeSeriesData]
    val_data: list[TimeSeriesData] | None = None
    test_data: list[TimeSeriesData]

    class Config:
        arbitrary_types_allowed = True
