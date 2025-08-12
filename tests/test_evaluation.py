"""Tests for evaluation metrics."""

import numpy as np
import pytest

from ananke.core.evaluation.metrics import compute_regression_metrics


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise
    return y_true, y_pred


class TestComputeRegressionMetrics:
    """Test cases for compute_regression_metrics function."""

    def test_basic_metrics_calculation(self, regression_data):
        """Test basic regression metrics calculation."""
        y_true, y_pred = regression_data

        metrics = compute_regression_metrics(y_true, y_pred)

        # Check that all expected metrics are present
        expected_keys = ["mse", "rmse", "mae", "r2", "mape", "max_error", "std_error"]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

        # Check basic properties
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["max_error"] >= 0
        assert metrics["rmse"] == np.sqrt(metrics["mse"])

    def test_with_prefix(self, regression_data):
        """Test metrics calculation with prefix."""
        y_true, y_pred = regression_data
        prefix = "test_"

        metrics = compute_regression_metrics(y_true, y_pred, prefix=prefix)

        # Check that all metrics have the prefix
        expected_keys = [
            "test_mse",
            "test_rmse",
            "test_mae",
            "test_r2",
            "test_mape",
            "test_max_error",
            "test_std_error",
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        metrics = compute_regression_metrics(y_true, y_pred)

        # Perfect predictions should have zero error metrics
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["max_error"] == 0.0
        assert metrics["std_error"] == 0.0
        assert abs(metrics["r2"] - 1.0) < 1e-10

    def test_multidimensional_arrays(self):
        """Test with multidimensional arrays."""
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = y_true + 0.1  # Add small error

        metrics = compute_regression_metrics(y_true, y_pred)

        # Should work with flattened arrays
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert metrics["mse"] > 0  # Should have some error

    def test_edge_cases(self):
        """Test edge cases for regression metrics."""
        # Single value
        y_true_single = np.array([1.0])
        y_pred_single = np.array([1.5])

        metrics = compute_regression_metrics(y_true_single, y_pred_single)
        assert metrics["mse"] == 0.25
        assert metrics["mae"] == 0.5
        assert metrics["max_error"] == 0.5

        # All zeros
        y_true_zeros = np.zeros(10)
        y_pred_zeros = np.zeros(10)

        metrics = compute_regression_metrics(y_true_zeros, y_pred_zeros)
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_mape_calculation(self):
        """Test MAPE calculation with different scenarios."""
        # Test with non-zero values
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        metrics = compute_regression_metrics(y_true, y_pred)
        assert metrics["mape"] >= 0
        assert isinstance(metrics["mape"], float)

        # Test with values close to zero (should use epsilon to avoid division by zero)
        y_true_small = np.array([0.001, 0.002, 0.003])
        y_pred_small = np.array([0.0011, 0.0019, 0.0032])

        metrics = compute_regression_metrics(y_true_small, y_pred_small)
        assert np.isfinite(metrics["mape"])  # Should not be inf or nan

    def test_return_type_and_structure(self, regression_data):
        """Test that the function returns the correct type and structure."""
        y_true, y_pred = regression_data

        metrics = compute_regression_metrics(y_true, y_pred)

        # Should return a dictionary
        assert isinstance(metrics, dict)

        # All values should be numeric
        for key, value in metrics.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert np.isfinite(value)  # Should not be inf or nan
