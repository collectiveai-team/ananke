"""Generalizable evaluation metrics for timeseries models."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    """Compute regression metrics."""
    metrics = {}

    # Flatten arrays if multidimensional
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Basic regression metrics
    metrics[f"{prefix}mse"] = mean_squared_error(y_true_flat, y_pred_flat)
    metrics[f"{prefix}rmse"] = np.sqrt(metrics[f"{prefix}mse"])
    metrics[f"{prefix}mae"] = mean_absolute_error(y_true_flat, y_pred_flat)
    metrics[f"{prefix}r2"] = r2_score(y_true_flat, y_pred_flat)

    # Additional metrics
    metrics[f"{prefix}mape"] = (
        np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
    )
    metrics[f"{prefix}max_error"] = np.max(np.abs(y_true_flat - y_pred_flat))
    metrics[f"{prefix}std_error"] = np.std(y_true_flat - y_pred_flat)

    return metrics


def plot_feature_target_prediction(
    feature: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    target_name: str | None = None,
) -> plt.Figure:
    """
    Plot feature, target, and prediction with proper time alignment.

    Args:
        feature: Array of feature values
        target: Array of target values
        prediction: Array of prediction values
        target_name: Name of the target variable for y-axis label

    Returns:
        matplotlib figure object
    """
    feature_len = len(feature)
    target_len = len(target)

    # Create time indices for each series
    feature_time = np.arange(0, feature_len)
    target_time = np.arange(feature_len, feature_len + target_len)

    fig = plt.figure(figsize=(12, 6))

    # Plot feature data
    plt.plot(feature_time, feature, label="Feature", color="blue")

    # Plot target and prediction as continuation
    if target is not None:
        plt.plot(target_time, target, label="Target", color="orange")
    if prediction is not None:
        plt.plot(target_time, prediction, label="Prediction", color="green")

    # Add a vertical line to separate feature and target/prediction
    plt.axvline(x=feature_len, color="gray", linestyle="--", alpha=0.7)

    plt.legend()
    plt.title("Time Series Forecast: Feature → Target/Prediction")
    plt.xlabel("Time Steps")
    if target_name:
        plt.ylabel(target_name)
    plt.grid(True, alpha=0.3)

    return fig
