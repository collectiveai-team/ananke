"""Generic Dagster prediction job for timeseries models."""

import logging

from dagster import job
from dagster_celery import celery_executor

from ananke.dagster.ops.prediction_ops import (
    load_input_data_op,
    load_model_op,
    log_predictions_op,
    make_predictions_op,
    save_predictions_op,
)

logger = logging.getLogger(__name__)


@job(
    executor_def=celery_executor.configured(
        {
            "broker": {"env": "CELERY_BROKER_URL"},
            "backend": {"env": "CELERY_RESULT_BACKEND"},
            "task_default_queue": "prediction",
        }
    ),
    tags={"dagster-celery/queue": "prediction"},
)
def prediction_job():
    """Generic prediction job for timeseries models."""
    # Load model and configuration
    model_data = load_model_op()

    # Load input data
    input_data = load_input_data_op(model_data)

    # Generate predictions
    prediction_results = make_predictions_op(model_data, input_data)

    # Save predictions
    save_results = save_predictions_op(prediction_results)

    # Log to MLflow
    log_predictions_op(prediction_results, save_results)
