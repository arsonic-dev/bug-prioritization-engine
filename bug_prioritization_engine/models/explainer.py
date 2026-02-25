"""
models/explainer.py — SHAP-based explanation engine.

BugExplainer wraps SHAP TreeExplainer for both the severity classifier and
impact regressor, producing human-readable feature attributions for any
single prediction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier, XGBRegressor

from bug_prioritization_engine.models.trainer import BugPriorityTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FeatureExplanation = dict[str, str]  # {"feature", "impact", "direction"}


# ---------------------------------------------------------------------------
# BugExplainer
# ---------------------------------------------------------------------------


class BugExplainer:
    """Produces SHAP-based explanations for severity and impact predictions.

    Wraps :class:`shap.TreeExplainer` for both models so explanations are
    fast (no model re-evaluation per sample) and consistent with the trained
    XGBoost trees.

    Args:
        classifier: Trained XGBClassifier.
        regressor: Trained XGBRegressor.
        feature_names: Ordered list of feature column names (from
            ``BugFeatureEngineer.get_feature_names()``).

    Example::

        explainer = BugExplainer(classifier, regressor, feature_names)
        clf_explanation = explainer.explain_severity(X_row, predicted_class_idx=0)
        reg_explanation = explainer.explain_impact(X_row)
    """

    def __init__(
        self,
        classifier: XGBClassifier,
        regressor: XGBRegressor,
        feature_names: list[str],
    ) -> None:
        self._classifier = classifier
        self._regressor = regressor
        self._feature_names = feature_names

        logger.info("Initialising SHAP TreeExplainer for classifier...")
        self._clf_explainer = shap.TreeExplainer(classifier)

        logger.info("Initialising SHAP TreeExplainer for regressor...")
        self._reg_explainer = shap.TreeExplainer(regressor)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain_severity(
        self,
        X: pd.DataFrame,
        predicted_class_idx: int,
        top_n: int = 5,
    ) -> list[FeatureExplanation]:
        """Return top-N SHAP feature contributions for the predicted severity class.

        For multiclass XGBoost, SHAP returns a 3-D array
        ``(n_samples, n_features, n_classes)``. We slice the column for the
        predicted class so explanations reflect *why this class was chosen*.

        Args:
            X: Single-row (or batch) feature DataFrame aligned to training columns.
            predicted_class_idx: Integer index of the predicted severity class
                (0=Critical, 1=High, 2=Medium, 3=Low).
            top_n: Number of top features to return (default 5).

        Returns:
            List of feature explanation dicts, sorted by absolute SHAP magnitude
            (descending).
        """
        shap_values = self._clf_explainer.shap_values(X)
        # shap_values shape: (n_samples, n_features, n_classes)
        # Slice to the predicted class for the first (and typically only) row
        if isinstance(shap_values, list):
            # Older SHAP versions return a list of arrays (one per class)
            class_shap = shap_values[predicted_class_idx][0]
        else:
            class_shap = shap_values[0, :, predicted_class_idx]

        return self._format_explanations(
            shap_array=class_shap,
            top_n=top_n,
            direction_label="severity",
        )

    def explain_impact(
        self,
        X: pd.DataFrame,
        top_n: int = 5,
    ) -> list[FeatureExplanation]:
        """Return top-N SHAP feature contributions for the impact score prediction.

        Args:
            X: Single-row (or batch) feature DataFrame aligned to training columns.
            top_n: Number of top features to return (default 5).

        Returns:
            List of feature explanation dicts, sorted by absolute SHAP magnitude
            (descending).
        """
        shap_values = self._reg_explainer.shap_values(X)
        # Regression: shape is (n_samples, n_features)
        row_shap = shap_values[0] if shap_values.ndim == 2 else shap_values

        return self._format_explanations(
            shap_array=row_shap,
            top_n=top_n,
            direction_label="impact score",
        )

    def explain_full(
        self,
        X: pd.DataFrame,
        predicted_class_idx: int,
        top_n: int = 5,
    ) -> dict[str, list[FeatureExplanation]]:
        """Return SHAP explanations for both severity and impact in one call.

        Args:
            X: Single-row feature DataFrame.
            predicted_class_idx: Predicted severity class index.
            top_n: Number of top features per model.

        Returns:
            Dict with keys ``"severity"`` and ``"impact"``, each mapping to a
            list of feature explanation dicts.
        """
        return {
            "severity": self.explain_severity(X, predicted_class_idx, top_n),
            "impact": self.explain_impact(X, top_n),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_explanations(
        self,
        shap_array: np.ndarray,
        top_n: int,
        direction_label: str,
    ) -> list[FeatureExplanation]:
        """Convert a raw SHAP value array into human-readable dicts.

        Args:
            shap_array: 1-D array of SHAP values, one per feature.
            top_n: Number of top features to include.
            direction_label: Label used in the direction string
                (e.g. ``"severity"`` or ``"impact score"``).

        Returns:
            List of dicts with keys ``feature``, ``impact``, ``direction``.
        """
        if len(shap_array) != len(self._feature_names):
            logger.warning(
                "SHAP array length %d != feature_names length %d — truncating",
                len(shap_array), len(self._feature_names),
            )

        pairs = sorted(
            zip(self._feature_names, shap_array),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )

        results: list[FeatureExplanation] = []
        for feature, value in pairs[:top_n]:
            direction = (
                f"increases {direction_label}" if value > 0 else f"decreases {direction_label}"
            )
            results.append(
                {
                    "feature": self._humanise_feature_name(feature),
                    "impact": f"{value:+.3f}",
                    "direction": direction,
                }
            )

        return results

    @staticmethod
    def _humanise_feature_name(name: str) -> str:
        """Convert internal feature column names to readable labels.

        Examples:
            ``"comp_Payments"``       → ``"Component: Payments"``
            ``"label_regression"``    → ``"Label: regression"``
            ``"emb_12"``              → ``"Text embedding dim 12"``
            ``"tfidf_34"``            → ``"TF-IDF feature 34"``
            ``"reporter_bug_rate"``   → ``"Reporter bug rate"``

        Args:
            name: Raw feature column name.

        Returns:
            Human-readable label string.
        """
        if name.startswith("comp_"):
            return f"Component: {name[5:]}"
        if name.startswith("label_"):
            return f"Label: {name[6:]}"
        if name.startswith("emb_"):
            return f"Text embedding dim {name[4:]}"
        if name.startswith("tfidf_"):
            return f"TF-IDF feature {name[6:]}"
        # Replace underscores with spaces and title-case
        return name.replace("_", " ").capitalize()

    # ------------------------------------------------------------------
    # Class method: build from saved artefacts
    # ------------------------------------------------------------------

    @classmethod
    def from_artefacts(cls) -> "BugExplainer":
        """Construct a BugExplainer by loading saved artefacts from disk.

        Returns:
            Fully initialised :class:`BugExplainer`.

        Raises:
            FileNotFoundError: If artefacts have not been saved yet.
        """
        classifier, regressor, engineer, _ = BugPriorityTrainer.load_artefacts()
        return cls(
            classifier=classifier,
            regressor=regressor,
            feature_names=engineer.get_feature_names(),
        )
