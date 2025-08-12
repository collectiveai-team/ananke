"""Generic Dagster training job for timeseries models."""

import logging

from dagster import job
from dagster_celery import celery_executor

from ananke.dagster.ops.training_ops import (
    evaluate_model_op,
    load_training_config_op,
    register_model_op,
    train_model_op,
)

logger = logging.getLogger(__name__)


@job(
    executor_def=celery_executor.configured(
        {
            "broker": {"env": "CELERY_BROKER_URL"},
            "backend": {"env": "CELERY_RESULT_BACKEND"},
            "task_default_queue": "training",
        }
    ),
    tags={"dagster-celery/queue": "training"},
)
def training_job():
    """Generic training job for timeseries models."""
    # Load configuration
    config_data = load_training_config_op()

    # Train model
    training_results = train_model_op(config_data)

    # Evaluate model
    evaluation_results = evaluate_model_op(training_results)

    # Register model
    register_model_op(evaluation_results)
