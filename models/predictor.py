"""
models/predictor.py — Inference engine.

BugPredictor wraps the trained models and feature engineer into a single
predict() call. It is designed to be loaded once at API startup and reused
across all requests (stateless per-call, stateful model weights).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from data.preprocessor import BugFeatureEngineer
from models.explainer import BugExplainer, FeatureExplanation
from models.trainer import BugPriorityTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PredictionOutput
# ---------------------------------------------------------------------------


class PredictionOutput:
    """Container for a single bug prediction result.

    Attributes:
        severity_label: Predicted severity class (Critical/High/Medium/Low).
        severity_confidence: Probability of the predicted class (0.0–1.0).
        severity_probabilities: Full class probability distribution.
        impact_score: Predicted business impact score (0–100).
        severity_explanation: Top SHAP features driving severity prediction.
        impact_explanation: Top SHAP features driving impact prediction.
    """

    def __init__(
        self,
        severity_label: str,
        severity_confidence: float,
        severity_probabilities: dict[str, float],
        impact_score: float,
        severity_explanation: list[FeatureExplanation],
        impact_explanation: list[FeatureExplanation],
    ) -> None:
        self.severity_label = severity_label
        self.severity_confidence = round(severity_confidence, 4)
        self.severity_probabilities = {k: round(v, 4) for k, v in severity_probabilities.items()}
        self.impact_score = round(float(np.clip(impact_score, 0, 100)), 2)
        self.severity_explanation = severity_explanation
        self.impact_explanation = impact_explanation

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON responses.

        Returns:
            Dict representation of the prediction output.
        """
        return {
            "severity_label": self.severity_label,
            "severity_confidence": self.severity_confidence,
            "severity_probabilities": self.severity_probabilities,
            "impact_score": self.impact_score,
            "severity_explanation": self.severity_explanation,
            "impact_explanation": self.impact_explanation,
        }


# ---------------------------------------------------------------------------
# BugPredictor
# ---------------------------------------------------------------------------


class BugPredictor:
    """Runs end-to-end inference: raw features → prediction + explanation.

    Designed to be instantiated once at API startup and shared across all
    requests. Thread-safe for read-only model inference.

    Args:
        classifier: Fitted XGBClassifier.
        regressor: Fitted XGBRegressor.
        engineer: Fitted BugFeatureEngineer.
        explainer: Fitted BugExplainer.

    Example::

        predictor = BugPredictor.from_artefacts()
        result = predictor.predict(raw_issue_df)
        print(result.severity_label, result.impact_score)
    """

    # Maps integer class index → label string
    LABEL_DECODING: dict[int, str] = BugPriorityTrainer.LABEL_DECODING

    def __init__(
        self,
        classifier: XGBClassifier,
        regressor: XGBRegressor,
        engineer: BugFeatureEngineer,
        explainer: BugExplainer,
    ) -> None:
        self._classifier = classifier
        self._regressor = regressor
        self._engineer = engineer
        self._explainer = explainer

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_df: pd.DataFrame,
        explain: bool = True,
        top_n_explanations: int = 5,
    ) -> list[PredictionOutput]:
        """Run inference on one or more raw issue rows.

        Args:
            raw_df: DataFrame of raw Jira issue fields (same schema as
                training data). One row per issue.
            explain: Whether to generate SHAP explanations (slightly slower).
            top_n_explanations: Number of top SHAP features per explanation.

        Returns:
            List of :class:`PredictionOutput` objects, one per input row.
        """
        X = self._engineer.transform(raw_df)

        # --- Severity classification
        class_probs: np.ndarray = self._classifier.predict_proba(X)
        class_indices: np.ndarray = class_probs.argmax(axis=1)

        # --- Impact regression
        impact_scores: np.ndarray = self._regressor.predict(X)

        outputs: list[PredictionOutput] = []

        for i in range(len(X)):
            row_X = X.iloc[[i]]
            predicted_idx = int(class_indices[i])
            severity_label = self.LABEL_DECODING[predicted_idx]
            severity_confidence = float(class_probs[i, predicted_idx])
            severity_probabilities = {
                self.LABEL_DECODING[j]: float(class_probs[i, j])
                for j in range(class_probs.shape[1])
            }
            impact_score = float(impact_scores[i])

            if explain:
                sev_exp = self._explainer.explain_severity(
                    row_X, predicted_idx, top_n_explanations
                )
                imp_exp = self._explainer.explain_impact(row_X, top_n_explanations)
            else:
                sev_exp, imp_exp = [], []

            outputs.append(
                PredictionOutput(
                    severity_label=severity_label,
                    severity_confidence=severity_confidence,
                    severity_probabilities=severity_probabilities,
                    impact_score=impact_score,
                    severity_explanation=sev_exp,
                    impact_explanation=imp_exp,
                )
            )

        logger.debug("predict() completed for %d rows", len(outputs))
        return outputs

    def predict_single(
        self,
        raw_record: dict[str, Any],
        explain: bool = True,
        top_n_explanations: int = 5,
    ) -> PredictionOutput:
        """Convenience wrapper to predict for a single issue dict.

        Args:
            raw_record: Dict of raw Jira issue fields.
            explain: Whether to generate SHAP explanations.
            top_n_explanations: Number of top SHAP features.

        Returns:
            Single :class:`PredictionOutput`.
        """
        df = pd.DataFrame([raw_record])
        return self.predict(df, explain=explain, top_n_explanations=top_n_explanations)[0]

    # ------------------------------------------------------------------
    # Class method: build from saved artefacts
    # ------------------------------------------------------------------

    @classmethod
    def from_artefacts(cls) -> "BugPredictor":
        """Load all saved artefacts from disk and return a ready predictor.

        Returns:
            Fully initialised :class:`BugPredictor`.

        Raises:
            FileNotFoundError: If models have not been trained and saved yet.
        """
        classifier, regressor, engineer, metadata = BugPriorityTrainer.load_artefacts()
        explainer = BugExplainer(
            classifier=classifier,
            regressor=regressor,
            feature_names=engineer.get_feature_names(),
        )
        logger.info(
            "BugPredictor loaded. Model version: %s, trained at: %s",
            metadata.get("model_version"),
            metadata.get("trained_at"),
        )
        return cls(
            classifier=classifier,
            regressor=regressor,
            engineer=engineer,
            explainer=explainer,
        )
