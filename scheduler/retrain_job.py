"""
scheduler/retrain_job.py — APScheduler-based daily model retraining.

Runs nightly at the configured hour (default 2AM UTC):
  1. Fetches the last N days of resolved Jira issues.
  2. Retrains both models.
  3. Compares new F1-macro against the previous model.
  4. Hot-swaps models only if the new model is not significantly worse.
  5. Logs the outcome with timestamp and metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Minimum acceptable F1 degradation before we reject the new model
_MAX_F1_DEGRADATION = 0.05


# ---------------------------------------------------------------------------
# Retraining job
# ---------------------------------------------------------------------------


def retrain_job(app: "FastAPI") -> None:
    """Nightly retraining job executed by APScheduler.

    Fetches recent resolved issues from Jira, retrains the models, and
    hot-swaps them into ``app.state`` only if performance doesn't degrade.

    Performance guard: if new F1-macro < old F1-macro - 0.05, the old
    model is kept and a warning is logged.

    Args:
        app: The running FastAPI application (carries ``app.state``).
    """
    logger.info("Scheduled retrain job started.")

    # ---- Gather previous model performance ----
    old_metadata = getattr(app.state, "model_metadata", {}) or {}
    old_f1: float | None = old_metadata.get("classifier_f1_macro")
    old_version: str | None = old_metadata.get("model_version")

    # ---- Fetch training data ----
    fetcher = app.state.fetcher
    try:
        df = fetcher.fetch_resolved_issues(
            extra_jql=f"created >= -{settings.retrain_lookback_days}d"
        )
    except Exception as exc:
        logger.error("Retrain job: failed to fetch Jira data — %s", exc, exc_info=True)
        return

    if df.empty or len(df) < 50:
        logger.warning(
            "Retrain job: only %d resolved issues found (min 50). Skipping retrain.", len(df)
        )
        return

    logger.info("Retrain job: fetched %d resolved issues.", len(df))

    # ---- Train ----
    from models.predictor import BugPredictor
    from models.trainer import BugPriorityTrainer

    try:
        trainer = BugPriorityTrainer()
        result = trainer.train(df=df)
    except Exception as exc:
        logger.error("Retrain job: training failed — %s", exc, exc_info=True)
        return

    new_f1 = result.classifier_metrics.f1_macro
    new_version = result.model_version

    logger.info(
        "Retrain job: new model version=%s | F1-macro=%.4f (old F1=%s)",
        new_version,
        new_f1,
        f"{old_f1:.4f}" if old_f1 is not None else "N/A",
    )

    # ---- Performance guard ----
    if old_f1 is not None and new_f1 < (old_f1 - _MAX_F1_DEGRADATION):
        logger.warning(
            "Retrain job: new model F1 (%.4f) is more than %.2f below old model F1 (%.4f). "
            "Keeping old model (version=%s).",
            new_f1,
            _MAX_F1_DEGRADATION,
            old_f1,
            old_version,
        )
        return

    # ---- Hot-swap ----
    try:
        app.state.predictor = BugPredictor.from_artefacts()
        app.state.model_metadata = {
            "model_version": new_version,
            "trained_at": result.trained_at,
            "feature_count": result.feature_count,
            "classifier_f1_macro": new_f1,
        }
        # Clear ranked-queue cache so next request uses the new model
        from api.routes import _queue_cache
        _queue_cache.clear()

        logger.info(
            "Retrain job: models hot-swapped successfully. "
            "New version=%s | F1=%.4f | RMSE=%.4f",
            new_version,
            new_f1,
            result.regressor_metrics.rmse,
        )
    except Exception as exc:
        logger.error(
            "Retrain job: failed to hot-swap models — %s. Old model still active.", exc, exc_info=True
        )


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler(app: "FastAPI") -> BackgroundScheduler:
    """Build and configure the APScheduler BackgroundScheduler.

    The scheduler is configured with a daily CronTrigger that fires at the
    hour defined by ``settings.retrain_cron_hour`` (UTC).

    Args:
        app: The running FastAPI application passed through to the job.

    Returns:
        Configured (but not yet started) :class:`BackgroundScheduler`.
    """
    scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.add_job(
        func=retrain_job,
        trigger=CronTrigger(hour=settings.retrain_cron_hour, minute=0, timezone="UTC"),
        kwargs={"app": app},
        id="daily_retrain",
        name="Daily model retraining",
        replace_existing=True,
        misfire_grace_time=3600,  # Allow up to 1hr late start (e.g. if server was down)
        coalesce=True,            # If multiple misfires, run only once on recovery
    )

    logger.info(
        "Retrain job scheduled: daily at %02d:00 UTC.", settings.retrain_cron_hour
    )
    return scheduler
