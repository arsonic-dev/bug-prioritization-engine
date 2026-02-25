"""
api/schemas.py — Pydantic v2 request/response models.

All public API shapes are defined here. Using Pydantic v2 field validators
and model_config for strict typing and clear OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class FeatureExplanation(BaseModel):
    """A single SHAP feature attribution.

    Attributes:
        feature: Human-readable feature name.
        impact: Signed SHAP value string, e.g. ``"+12.340"``.
        direction: Plain-English direction string.
    """

    feature: str = Field(..., description="Human-readable feature name")
    impact: str = Field(..., description='Signed SHAP value, e.g. "+3.142"')
    direction: str = Field(..., description='e.g. "increases severity"')


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class BugInput(BaseModel):
    """Raw fields for a new or open Jira bug — used by POST /predict.

    All fields mirror the JiraFetcher output schema so the same
    preprocessing pipeline applies unchanged.
    """

    summary: str = Field(..., min_length=1, description="Bug summary / title")
    description: str = Field(default="", description="Full bug description")
    priority_name: str = Field(
        default="unknown",
        description="Jira native priority: Blocker | Critical | Major | Minor | Trivial",
    )
    priority_ordinal: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Ordinal encoding of priority (0–5)",
    )
    issuetype: str = Field(default="Bug", description="Jira issue type")
    components: str = Field(
        default="",
        description="Pipe-delimited component names, e.g. 'Auth|Payments'",
    )
    labels: str = Field(
        default="",
        description="Pipe-delimited label names, e.g. 'regression|blocker'",
    )
    reporter: str = Field(default="unknown", description="Reporter display name")
    comment_count: int = Field(default=0, ge=0, description="Number of comments")
    vote_count: int = Field(default=0, ge=0, description="Number of votes")
    watch_count: int = Field(default=0, ge=0, description="Number of watchers")
    story_points: float = Field(default=0.0, ge=0.0, description="Story point estimate")
    age_days: float = Field(default=0.0, ge=0.0, description="Days since issue was created")
    description_length: int = Field(
        default=0,
        ge=0,
        description="Character count of description (auto-computed if 0)",
    )

    @model_validator(mode="after")
    def auto_description_length(self) -> "BugInput":
        """Auto-populate description_length from description if not provided."""
        if self.description_length == 0 and self.description:
            self.description_length = len(self.description)
        return self

    @field_validator("priority_name")
    @classmethod
    def normalise_priority(cls, v: str) -> str:
        """Lowercase and strip whitespace from priority_name."""
        return v.strip().lower()

    model_config = {"json_schema_extra": {
        "example": {
            "summary": "Login page throws 500 on empty password",
            "description": "Steps to reproduce: 1. Navigate to /login 2. Submit empty password field.",
            "priority_name": "critical",
            "priority_ordinal": 4,
            "issuetype": "Bug",
            "components": "Auth|Frontend",
            "labels": "regression|customer-reported",
            "reporter": "Jane Smith",
            "comment_count": 3,
            "vote_count": 5,
            "watch_count": 12,
            "story_points": 3.0,
            "age_days": 2.0,
        }
    }}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Response for POST /api/v1/predict.

    Attributes:
        severity_label: Predicted severity class.
        severity_confidence: Probability of the predicted class (0–1).
        severity_probabilities: Full class distribution.
        impact_score: Predicted business impact score (0–100).
        severity_explanation: Top SHAP features for severity.
        impact_explanation: Top SHAP features for impact score.
    """

    severity_label: str = Field(..., description="Critical | High | Medium | Low")
    severity_confidence: float = Field(..., ge=0.0, le=1.0)
    severity_probabilities: dict[str, float] = Field(
        ..., description="Probability per severity class"
    )
    impact_score: float = Field(..., ge=0.0, le=100.0, description="Business impact score (0–100)")
    severity_explanation: list[FeatureExplanation] = Field(
        default_factory=list, description="Top SHAP features driving severity prediction"
    )
    impact_explanation: list[FeatureExplanation] = Field(
        default_factory=list, description="Top SHAP features driving impact score"
    )


class RankedIssue(BaseModel):
    """A single issue entry in the ranked queue response.

    Attributes:
        jira_key: Jira issue key, e.g. ``"MYPROJ-123"``.
        summary: Issue summary.
        predicted_severity: Predicted severity label.
        severity_confidence: Model confidence for predicted severity.
        impact_score: Predicted business impact (0–100).
        rank: 1-based rank in the queue (1 = highest priority).
    """

    jira_key: str
    summary: str
    predicted_severity: str
    severity_confidence: float = Field(..., ge=0.0, le=1.0)
    impact_score: float = Field(..., ge=0.0, le=100.0)
    rank: int = Field(..., ge=1)


class RankedQueueResponse(BaseModel):
    """Response for GET /api/v1/ranked-queue.

    Attributes:
        project: Jira project key.
        total_issues: Total number of open issues ranked.
        issues: Ranked list of issues (highest impact first).
        generated_at: UTC timestamp of when the ranking was computed.
        cached: Whether this response was served from cache.
    """

    project: str
    total_issues: int
    issues: list[RankedIssue]
    generated_at: datetime
    cached: bool = False


class ExplainResponse(BaseModel):
    """Response for GET /api/v1/explain/{jira_key}.

    Attributes:
        jira_key: The requested Jira issue key.
        summary: Issue summary text.
        predicted_severity: Predicted severity label.
        impact_score: Predicted impact score.
        feature_values: Raw feature values used for this prediction.
        severity_explanation: SHAP explanations for severity.
        impact_explanation: SHAP explanations for impact score.
    """

    jira_key: str
    summary: str
    predicted_severity: str
    impact_score: float
    feature_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Key feature values used in this prediction",
    )
    severity_explanation: list[FeatureExplanation]
    impact_explanation: list[FeatureExplanation]


class RetrainResponse(BaseModel):
    """Response for POST /api/v1/retrain.

    Attributes:
        status: ``"triggered"`` or ``"failed"``.
        model_version: Version string of the newly trained model (if successful).
        f1_macro: Classifier F1-macro on the test split.
        message: Human-readable status message.
    """

    status: str
    model_version: str | None = None
    f1_macro: float | None = None
    message: str


class HealthResponse(BaseModel):
    """Response for GET /api/v1/health.

    Attributes:
        status: ``"ok"`` or ``"degraded"``.
        model_version: Version string of the currently loaded model.
        trained_at: UTC timestamp of the last training run.
        feature_count: Number of features in the current model.
        classifier_f1_macro: Last known F1-macro score.
    """

    status: str
    model_version: str | None = None
    trained_at: str | None = None
    feature_count: int | None = None
    classifier_f1_macro: float | None = None


class ErrorResponse(BaseModel):
    """Standard error envelope returned by the global exception handler.

    Attributes:
        error: Short error type string.
        detail: Detailed error message.
        status_code: HTTP status code.
    """

    error: str
    detail: str
    status_code: int
