"""
api/routes.py — All FastAPI route handlers.

Routes are organised as an APIRouter so main.py can include them cleanly.
The predictor, fetcher, and metadata are accessed via app.state (set at
startup in main.py) to avoid circular imports and support hot-swap on retrain.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.security import APIKeyHeader

from api.schemas import (
    BugInput,
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    PredictionResponse,
    RankedIssue,
    RankedQueueResponse,
    RetrainResponse,
)
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# Simple in-memory cache for the ranked queue (TTL-based)
# ---------------------------------------------------------------------------

_queue_cache: dict[str, tuple[float, RankedQueueResponse]] = {}


def _get_cached_queue(project: str) -> RankedQueueResponse | None:
    """Return a cached ranked-queue response if still within TTL.

    Args:
        project: Jira project key.

    Returns:
        Cached :class:`RankedQueueResponse` or ``None`` if expired/absent.
    """
    entry = _queue_cache.get(project)
    if entry is None:
        return None
    cached_at, response = entry
    if time.time() - cached_at > settings.cache_ttl_seconds:
        del _queue_cache[project]
        return None
    return response


def _set_cached_queue(project: str, response: RankedQueueResponse) -> None:
    """Store a ranked-queue response in the in-memory cache.

    Args:
        project: Jira project key used as cache key.
        response: Response object to cache.
    """
    _queue_cache[project] = (time.time(), response)


# ---------------------------------------------------------------------------
# API-key dependency for protected endpoints
# ---------------------------------------------------------------------------


API_KEY_HEADER = APIKeyHeader(name="X-Retrain-API-Key", auto_error=False)


def verify_retrain_key(api_key: str | None = Depends(API_KEY_HEADER)) -> None:
    """Raise 403 if the provided API key does not match the configured secret.

    Args:
        api_key: Value from the ``X-Retrain-API-Key`` request header.

    Raises:
        HTTPException: 403 if the key is missing or incorrect.
    """
    if api_key != settings.retrain_api_key:
        raise HTTPException(status_code=403, detail="Invalid or missing retrain API key.")


# ---------------------------------------------------------------------------
# Helper: raw DataFrame → schema row dict
# ---------------------------------------------------------------------------


def _issue_row_to_dict(row: pd.Series) -> dict:
    """Convert a single Jira DataFrame row to a dict for BugInput.

    Args:
        row: A pandas Series representing one Jira issue.

    Returns:
        Dict of fields compatible with the BugInput schema.
    """
    return {
        "summary": str(row.get("summary", "")),
        "description": str(row.get("description", "")),
        "priority_name": str(row.get("priority_name", "unknown")),
        "priority_ordinal": int(row.get("priority_ordinal", 0)),
        "issuetype": str(row.get("issuetype", "Bug")),
        "components": str(row.get("components", "")),
        "labels": str(row.get("labels", "")),
        "reporter": str(row.get("reporter", "unknown")),
        "comment_count": int(row.get("comment_count", 0)),
        "vote_count": int(row.get("vote_count", 0)),
        "watch_count": int(row.get("watch_count", 0)),
        "story_points": float(row.get("story_points", 0.0)),
        "age_days": float(row.get("age_days", 0.0)),
        "description_length": int(row.get("description_length", 0)),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict severity and impact for a single bug",
    tags=["Predictions"],
)
async def predict(request: Request, bug: BugInput) -> PredictionResponse:
    """Predict severity label and business impact for a single bug.

    Runs feature engineering → XGBoost inference → SHAP explanation pipeline.

    Args:
        request: FastAPI Request (used to access app.state).
        bug: Raw bug input fields.

    Returns:
        Severity classification + impact score + SHAP explanations.
    """
    predictor = request.app.state.predictor
    raw_df = pd.DataFrame([bug.model_dump()])

    try:
        results = predictor.predict(raw_df, explain=True, top_n_explanations=5)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    result = results[0]
    return PredictionResponse(
        severity_label=result.severity_label,
        severity_confidence=result.severity_confidence,
        severity_probabilities=result.severity_probabilities,
        impact_score=result.impact_score,
        severity_explanation=result.severity_explanation,
        impact_explanation=result.impact_explanation,
    )


@router.get(
    "/ranked-queue",
    response_model=RankedQueueResponse,
    summary="Get all open issues ranked by predicted priority",
    tags=["Queue"],
)
async def ranked_queue(
    request: Request,
    project: str = Query(..., description="Jira project key, e.g. MYPROJ"),
    limit: int = Query(default=50, ge=1, le=500, description="Max issues to return"),
) -> RankedQueueResponse:
    """Fetch all open Jira issues and rank them by predicted impact score.

    Results are cached for ``CACHE_TTL_SECONDS`` (default 5 minutes).

    Args:
        request: FastAPI Request.
        project: Jira project key to query.
        limit: Maximum number of ranked issues to return.

    Returns:
        Ranked queue of open issues, highest impact first.
    """
    cached = _get_cached_queue(project)
    if cached is not None:
        logger.debug("Serving ranked queue from cache for project %s", project)
        cached.cached = True
        return cached

    fetcher = request.app.state.fetcher
    predictor = request.app.state.predictor

    try:
        open_df = fetcher.fetch_open_issues(project_key=project, max_results=limit)
    except Exception as exc:
        logger.exception("Failed to fetch open issues: %s", exc)
        raise HTTPException(status_code=502, detail=f"Jira fetch error: {exc}") from exc

    if open_df.empty:
        return RankedQueueResponse(
            project=project,
            total_issues=0,
            issues=[],
            generated_at=datetime.now(tz=timezone.utc),
            cached=False,
        )

    try:
        results = predictor.predict(open_df, explain=False)
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # Build ranked list
    ranked_data = []
    for i, (_, row) in enumerate(open_df.iterrows()):
        pred = results[i]
        ranked_data.append({
            "jira_key": str(row.get("jira_key", f"UNKNOWN-{i}")),
            "summary": str(row.get("summary", "")),
            "impact_score": pred.impact_score,
            "predicted_severity": pred.severity_label,
            "severity_confidence": pred.severity_confidence,
        })

    # Sort by impact_score descending, assign ranks
    ranked_data.sort(key=lambda x: x["impact_score"], reverse=True)

    issues = [
        RankedIssue(
            jira_key=item["jira_key"],
            summary=item["summary"],
            predicted_severity=item["predicted_severity"],
            severity_confidence=item["severity_confidence"],
            impact_score=item["impact_score"],
            rank=rank + 1,
        )
        for rank, item in enumerate(ranked_data[:limit])
    ]

    response = RankedQueueResponse(
        project=project,
        total_issues=len(open_df),
        issues=issues,
        generated_at=datetime.now(tz=timezone.utc),
        cached=False,
    )

    _set_cached_queue(project, response)
    return response


@router.get(
    "/explain/{jira_key}",
    response_model=ExplainResponse,
    summary="Get detailed SHAP explanation for a specific Jira issue",
    tags=["Explanations"],
)
async def explain_issue(request: Request, jira_key: str) -> ExplainResponse:
    """Fetch a Jira issue by key and return a detailed SHAP explanation.

    Args:
        request: FastAPI Request.
        jira_key: Jira issue key, e.g. ``"MYPROJ-123"``.

    Returns:
        Full SHAP explanation with feature values and attributions.
    """
    fetcher = request.app.state.fetcher
    predictor = request.app.state.predictor

    try:
        issue_df = fetcher.fetch_issue_by_key(jira_key)
    except Exception as exc:
        logger.exception("Failed to fetch issue %s: %s", jira_key, exc)
        raise HTTPException(status_code=404, detail=f"Issue {jira_key} not found: {exc}") from exc

    if issue_df.empty:
        raise HTTPException(status_code=404, detail=f"Issue {jira_key} returned no data.")

    try:
        results = predictor.predict(issue_df, explain=True, top_n_explanations=5)
    except Exception as exc:
        logger.exception("Prediction failed for %s: %s", jira_key, exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    result = results[0]
    row = issue_df.iloc[0]

    # Surface a curated subset of raw feature values for transparency
    key_features = {
        "priority_name": row.get("priority_name"),
        "comment_count": row.get("comment_count"),
        "watch_count": row.get("watch_count"),
        "vote_count": row.get("vote_count"),
        "components": row.get("components"),
        "labels": row.get("labels"),
        "reporter": row.get("reporter"),
        "age_days": row.get("age_days"),
        "description_length": row.get("description_length"),
        "story_points": row.get("story_points"),
    }

    return ExplainResponse(
        jira_key=jira_key,
        summary=str(row.get("summary", "")),
        predicted_severity=result.severity_label,
        impact_score=result.impact_score,
        feature_values={k: v for k, v in key_features.items() if v is not None},
        severity_explanation=result.severity_explanation,
        impact_explanation=result.impact_explanation,
    )


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    summary="Trigger manual model retraining",
    tags=["Admin"],
    dependencies=[Depends(verify_retrain_key)],
)
async def retrain(request: Request) -> RetrainResponse:
    """Trigger a full model retrain using the last 90 days of resolved issues.

    Protected by the ``X-Retrain-API-Key`` header. On success, hot-swaps the
    in-memory predictor without restarting the API.

    Args:
        request: FastAPI Request.

    Returns:
        Retrain status, new model version, and F1-macro score.
    """
    fetcher = request.app.state.fetcher
    metadata = request.app.state.model_metadata

    from data.preprocessor import BugFeatureEngineer
    from models.predictor import BugPredictor
    from models.trainer import BugPriorityTrainer

    logger.info("Manual retrain triggered via API.")

    try:
        df = fetcher.fetch_resolved_issues(
            extra_jql=f"created >= -{settings.retrain_lookback_days}d"
        )
        if df.empty or len(df) < 50:
            return RetrainResponse(
                status="failed",
                message=f"Insufficient data: only {len(df)} resolved issues found (min 50 required).",
            )

        trainer = BugPriorityTrainer()
        result = trainer.train(df=df)

        # Hot-swap in-memory predictor
        request.app.state.predictor = BugPredictor.from_artefacts()
        request.app.state.model_metadata = {
            "model_version": result.model_version,
            "trained_at": result.trained_at,
            "feature_count": result.feature_count,
            "classifier_f1_macro": result.classifier_metrics.f1_macro,
        }
        # Bust ranked-queue cache so stale predictions aren't served
        _queue_cache.clear()

        logger.info("Manual retrain succeeded. Version: %s", result.model_version)
        return RetrainResponse(
            status="triggered",
            model_version=result.model_version,
            f1_macro=result.classifier_metrics.f1_macro,
            message="Retraining complete. Models hot-swapped successfully.",
        )

    except Exception as exc:
        logger.exception("Manual retrain failed: %s", exc)
        return RetrainResponse(
            status="failed",
            message=f"Retraining failed: {exc}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API and model health check",
    tags=["Health"],
)
async def health(request: Request) -> HealthResponse:
    """Return current model version, training timestamp, and system status.

    Args:
        request: FastAPI Request.

    Returns:
        Health status and model metadata.
    """
    metadata = request.app.state.model_metadata or {}
    predictor_loaded = request.app.state.predictor is not None

    return HealthResponse(
        status="ok" if predictor_loaded else "degraded",
        model_version=metadata.get("model_version"),
        trained_at=metadata.get("trained_at"),
        feature_count=metadata.get("feature_count"),
        classifier_f1_macro=metadata.get("classifier_f1_macro"),
    )
