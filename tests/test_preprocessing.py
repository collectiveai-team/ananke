"""Tests for preprocessing modules."""

import numpy as np
import pandas as pd
import pytest

from ananke.core.preprocess.base import BasePreprocessor, PreprocessingPipeline
from ananke.core.preprocess.imputers import (
    BackwardFillImputer,
    ForwardFillImputer,
    InterpolateImputer,
    MeanImputer,
    MedianImputer,
)
from ananke.core.preprocess.scalers import MinMaxScaler, RobustScaler, StandardScaler
from ananke.core.preprocess.smoothers import (
    EWMSmoother,
    MedianSmoother,
    RollingMeanSmoother,
    SavgolSmoother,
)


@pytest.fixture
def sample_data():
    """Create sample data with missing values and noise."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.random.randn(100) * 10 + 50,
            "feature_1": np.random.randn(100) * 5 + 20,
            "feature_2": np.random.randn(100) * 2 + 10,
        }
    )

    # Introduce some missing values
    data.loc[10:15, "value"] = np.nan
    data.loc[30:32, "feature_1"] = np.nan
    data.loc[50, "feature_2"] = np.nan

    return data


@pytest.fixture
def clean_data():
    """Create clean sample data without missing values."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="H")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "value": np.random.randn(100) * 10 + 50,
            "feature_1": np.random.randn(100) * 5 + 20,
            "feature_2": np.random.randn(100) * 2 + 10,
        }
    )
    return data


class TestBasePreprocessor:
    """Test cases for BasePreprocessor."""

    def test_abstract_methods(self):
        """Test that BasePreprocessor cannot be instantiated."""
        with pytest.raises(TypeError):
            BasePreprocessor()

    def test_concrete_implementation(self, sample_data):
        """Test concrete implementation of BasePreprocessor."""

        class ConcretePreprocessor(BasePreprocessor):
            def fit(self, data: pd.DataFrame) -> "ConcretePreprocessor":
                return self

            def transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data.copy()

            def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
                return data.copy()

        preprocessor = ConcretePreprocessor()

        # Test fit_transform
        result = preprocessor.fit_transform(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data.shape

        # Test individual methods
        fitted = preprocessor.fit(sample_data)
        assert fitted is preprocessor

        transformed = preprocessor.transform(sample_data)
        assert isinstance(transformed, pd.DataFrame)

        inverse = preprocessor.inverse_transform(transformed)
        assert isinstance(inverse, pd.DataFrame)


class TestScalers:
    """Test cases for scaling preprocessors."""

    def test_standard_scaler(self, clean_data):
        """Test StandardScaler functionality."""
        scaler = StandardScaler(columns=["value", "feature_1"])

        # Fit and transform
        scaled_data = scaler.fit_transform(clean_data)

        # Check that specified columns are scaled
        assert abs(scaled_data["value"].mean()) < 1e-10  # Should be ~0
        assert abs(scaled_data["value"].std() - 1.0) < 1e-10  # Should be ~1

        # Check that unspecified columns are unchanged
        pd.testing.assert_series_equal(
            scaled_data["timestamp"], clean_data["timestamp"]
        )
        pd.testing.assert_series_equal(
            scaled_data["feature_2"], clean_data["feature_2"]
        )

        # Test inverse transform
        inverse_data = scaler.inverse_transform(scaled_data)
        pd.testing.assert_frame_equal(inverse_data, clean_data, check_dtype=False)

    def test_minmax_scaler(self, clean_data):
        """Test MinMaxScaler functionality."""
        scaler = MinMaxScaler(columns=["value"], feature_range=(0, 1))

        scaled_data = scaler.fit_transform(clean_data)

        # Check that values are in [0, 1] range
        assert scaled_data["value"].min() >= 0
        assert scaled_data["value"].max() <= 1

        # Test inverse transform
        inverse_data = scaler.inverse_transform(scaled_data)
        pd.testing.assert_series_equal(
            inverse_data["value"], clean_data["value"], check_dtype=False
        )

    def test_robust_scaler(self, clean_data):
        """Test RobustScaler functionality."""
        scaler = RobustScaler(columns=["value", "feature_1"])

        scaled_data = scaler.fit_transform(clean_data)

        # Check that median is approximately 0
        assert abs(scaled_data["value"].median()) < 1e-10

        # Test inverse transform
        inverse_data = scaler.inverse_transform(scaled_data)
        pd.testing.assert_frame_equal(inverse_data, clean_data, check_dtype=False)

    def test_scaler_with_missing_values(self, sample_data):
        """Test scalers handle missing values correctly."""
        scaler = StandardScaler(columns=["value"])

        # Should handle missing values gracefully
        scaled_data = scaler.fit_transform(sample_data)

        # Missing values should remain missing
        assert scaled_data["value"].isna().sum() == sample_data["value"].isna().sum()


class TestImputers:
    """Test cases for imputation preprocessors."""

    def test_forward_fill_imputer(self, sample_data):
        """Test ForwardFillImputer functionality."""
        imputer = ForwardFillImputer(columns=["value"])

        imputed_data = imputer.fit_transform(sample_data)

        # Should have fewer missing values
        assert imputed_data["value"].isna().sum() <= sample_data["value"].isna().sum()

        # First few values might still be NaN if they start with NaN
        # But middle NaNs should be filled
        assert not imputed_data["value"].iloc[20:].isna().any()

    def test_backward_fill_imputer(self, sample_data):
        """Test BackwardFillImputer functionality."""
        imputer = BackwardFillImputer(columns=["feature_1"])

        imputed_data = imputer.fit_transform(sample_data)

        # Should have fewer or equal missing values
        assert (
            imputed_data["feature_1"].isna().sum()
            <= sample_data["feature_1"].isna().sum()
        )

    def test_interpolate_imputer(self, sample_data):
        """Test InterpolateImputer functionality."""
        imputer = InterpolateImputer(columns=["value"], method="linear")

        imputed_data = imputer.fit_transform(sample_data)

        # Should have no missing values in the middle (linear interpolation)
        # Only edge cases might remain
        middle_section = imputed_data["value"].iloc[5:-5]
        assert not middle_section.isna().any()

    def test_mean_imputer(self, sample_data):
        """Test MeanImputer functionality."""
        imputer = MeanImputer(columns=["feature_1"])

        # Calculate expected mean (excluding NaN)
        expected_mean = sample_data["feature_1"].mean()

        imputed_data = imputer.fit_transform(sample_data)

        # Should have no missing values
        assert not imputed_data["feature_1"].isna().any()

        # Check that NaN values were replaced with mean
        original_nan_mask = sample_data["feature_1"].isna()
        filled_values = imputed_data.loc[original_nan_mask, "feature_1"]
        assert all(abs(val - expected_mean) < 1e-10 for val in filled_values)

    def test_median_imputer(self, sample_data):
        """Test MedianImputer functionality."""
        imputer = MedianImputer(columns=["feature_2"])

        expected_median = sample_data["feature_2"].median()

        imputed_data = imputer.fit_transform(sample_data)

        # Should have no missing values
        assert not imputed_data["feature_2"].isna().any()

        # Check that NaN values were replaced with median
        original_nan_mask = sample_data["feature_2"].isna()
        if original_nan_mask.any():
            filled_values = imputed_data.loc[original_nan_mask, "feature_2"]
            assert all(abs(val - expected_median) < 1e-10 for val in filled_values)


class TestSmoothers:
    """Test cases for smoothing preprocessors."""

    def test_ewm_smoother(self, clean_data):
        """Test EWMSmoother functionality."""
        smoother = EWMSmoother(columns=["value"], span=5)

        smoothed_data = smoother.fit_transform(clean_data)

        # Smoothed data should have same shape
        assert smoothed_data.shape == clean_data.shape

        # Smoothed values should be different from original
        assert not smoothed_data["value"].equals(clean_data["value"])

        # Should have less variance (smoother)
        assert smoothed_data["value"].var() <= clean_data["value"].var()

    def test_rolling_mean_smoother(self, clean_data):
        """Test RollingMeanSmoother functionality."""
        smoother = RollingMeanSmoother(columns=["value"], window=5)

        smoothed_data = smoother.fit_transform(clean_data)

        # Should have same shape
        assert smoothed_data.shape == clean_data.shape

        # Should be smoother (less variance)
        # Skip first few values which might be NaN due to rolling window
        valid_original = clean_data["value"].iloc[5:]
        valid_smoothed = smoothed_data["value"].iloc[5:]

        assert valid_smoothed.var() <= valid_original.var()

    def test_savgol_smoother(self, clean_data):
        """Test SavgolSmoother functionality."""
        smoother = SavgolSmoother(columns=["value"], window_length=11, polyorder=2)

        smoothed_data = smoother.fit_transform(clean_data)

        # Should have same shape
        assert smoothed_data.shape == clean_data.shape

        # Should be smoother
        assert smoothed_data["value"].var() <= clean_data["value"].var()

    def test_median_smoother(self, clean_data):
        """Test MedianSmoother functionality."""
        smoother = MedianSmoother(columns=["value"], window=5)

        smoothed_data = smoother.fit_transform(clean_data)

        # Should have same shape
        assert smoothed_data.shape == clean_data.shape

        # Should be smoother
        valid_smoothed = smoothed_data["value"].dropna()
        valid_original = clean_data["value"].iloc[: len(valid_smoothed)]

        assert valid_smoothed.var() <= valid_original.var()


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline."""

    def test_empty_pipeline(self, sample_data):
        """Test empty pipeline."""
        pipeline = PreprocessingPipeline([])

        result = pipeline.fit_transform(sample_data)
        pd.testing.assert_frame_equal(result, sample_data)

    def test_single_step_pipeline(self, clean_data):
        """Test pipeline with single step."""
        scaler = StandardScaler(columns=["value"])
        pipeline = PreprocessingPipeline([scaler])

        result = pipeline.fit_transform(clean_data)

        # Should be equivalent to direct scaler usage
        expected = scaler.fit_transform(clean_data)
        pd.testing.assert_frame_equal(result, expected)

    def test_multi_step_pipeline(self, sample_data):
        """Test pipeline with multiple steps."""
        # Create pipeline: impute -> scale -> smooth
        imputer = MeanImputer(columns=["value", "feature_1"])
        scaler = StandardScaler(columns=["value", "feature_1"])
        smoother = EWMSmoother(columns=["value"], span=3)

        pipeline = PreprocessingPipeline([imputer, scaler, smoother])

        result = pipeline.fit_transform(sample_data)

        # Should have no missing values (imputed)
        assert not result[["value", "feature_1"]].isna().any().any()

        # Should be scaled (approximately mean 0 for value after smoothing)
        # Note: smoothing might change the mean slightly
        assert abs(result["value"].mean()) < 0.5

        # Should be smoother than original
        original_var = sample_data["value"].var()
        result_var = result["value"].var()
        # After imputation and scaling, variance should be different
        assert result_var != original_var

    def test_pipeline_inverse_transform(self, clean_data):
        """Test pipeline inverse transform."""
        scaler1 = StandardScaler(columns=["value"])
        scaler2 = MinMaxScaler(columns=["feature_1"])

        pipeline = PreprocessingPipeline([scaler1, scaler2])

        # Transform and inverse transform
        transformed = pipeline.fit_transform(clean_data)
        inverse = pipeline.inverse_transform(transformed)

        # Should recover original data (approximately)
        pd.testing.assert_frame_equal(
            inverse, clean_data, check_dtype=False, atol=1e-10
        )

    def test_pipeline_fit_transform_separate(self, clean_data, sample_data):
        """Test separate fit and transform calls."""
        scaler = StandardScaler(columns=["value"])
        pipeline = PreprocessingPipeline([scaler])

        # Fit on clean data
        pipeline.fit(clean_data)

        # Transform sample data (with missing values)
        result = pipeline.transform(sample_data)

        # Should use statistics from clean data
        # Missing values should remain missing
        assert result["value"].isna().sum() == sample_data["value"].isna().sum()

    def test_pipeline_step_access(self, clean_data):
        """Test accessing individual pipeline steps."""
        scaler = StandardScaler(columns=["value"])
        imputer = MeanImputer(columns=["feature_1"])

        pipeline = PreprocessingPipeline([scaler, imputer])

        # Should be able to access steps
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0] is scaler
        assert pipeline.steps[1] is imputer

        # Test string representation
        repr_str = repr(pipeline)
        assert "PreprocessingPipeline" in repr_str
        assert "StandardScaler" in repr_str
        assert "MeanImputer" in repr_str


class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""

    def test_full_preprocessing_workflow(self, sample_data):
        """Test complete preprocessing workflow."""
        # Create comprehensive pipeline
        steps = [
            # 1. Impute missing values
            ForwardFillImputer(columns=["value"]),
            MeanImputer(columns=["feature_1", "feature_2"]),
            # 2. Scale features
            StandardScaler(columns=["value", "feature_1"]),
            MinMaxScaler(columns=["feature_2"]),
            # 3. Smooth noisy signals
            EWMSmoother(columns=["value"], span=3),
        ]

        pipeline = PreprocessingPipeline(steps)

        # Process data
        processed_data = pipeline.fit_transform(sample_data)

        # Verify results
        assert processed_data.shape == sample_data.shape

        # Should have no missing values
        assert (
            not processed_data[["value", "feature_1", "feature_2"]].isna().any().any()
        )

        # Value should be scaled and smoothed
        assert abs(processed_data["value"].mean()) < 0.5  # Approximately centered

        # Feature_2 should be in [0, 1] range
        assert processed_data["feature_2"].min() >= 0
        assert processed_data["feature_2"].max() <= 1

        # Timestamp should be unchanged
        pd.testing.assert_series_equal(
            processed_data["timestamp"], sample_data["timestamp"]
        )

    def test_preprocessing_robustness(self):
        """Test preprocessing with edge cases."""
        # Create challenging data
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="H"),
                "all_nan": [np.nan] * 10,
                "constant": [5.0] * 10,
                "single_value": [1.0] + [np.nan] * 9,
                "normal": np.random.randn(10),
            }
        )

        # Create robust pipeline
        pipeline = PreprocessingPipeline(
            [
                ForwardFillImputer(columns=["single_value"]),
                MeanImputer(columns=["all_nan"]),  # Will fill with NaN -> 0
                StandardScaler(columns=["normal"]),
                # Skip constant column (would cause division by zero in scaling)
            ]
        )

        # Should handle edge cases gracefully
        result = pipeline.fit_transform(data)

        assert result.shape == data.shape
        # Normal column should be scaled
        assert abs(result["normal"].mean()) < 1e-10
