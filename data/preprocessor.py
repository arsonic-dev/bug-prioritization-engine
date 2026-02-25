"""
data/preprocessor.py — Feature engineering pipeline for bug prioritization.

BugFeatureEngineer transforms raw Jira DataFrames into model-ready feature
matrices.  It follows the sklearn fit/transform contract so it can be
serialised with joblib and reused at inference time without re-fitting.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OrdinalEncoder,
)

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_ORDINAL: dict[str, int] = {
    "blocker": 5,
    "critical": 4,
    "major": 3,
    "minor": 2,
    "trivial": 1,
    "unknown": 0,
}

PRIORITY_TO_SEVERITY = {
    "highest": "Highest",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "lowest": "Lowest",
    "unknown": "Low",
}

SEVERITY_CLASSES = ["Highest", "High", "Medium", "Low", "Lowest"]

IMPACT_WEIGHT_SEVERITY = 0.4
IMPACT_WEIGHT_WATCH = 0.3
IMPACT_WEIGHT_COMPONENT_FAILURE = 0.2
IMPACT_WEIGHT_COMMENT = 0.1

EMBEDDING_DIM_REDUCED = 32
TFIDF_MAX_FEATURES = 100


# ---------------------------------------------------------------------------
# BugFeatureEngineer
# ---------------------------------------------------------------------------


class BugFeatureEngineer:
    """Transforms raw Jira DataFrames into ML-ready feature matrices.

    The class follows sklearn's fit/transform contract:

    * ``fit_transform(df)``  — call during training; fits all sub-transformers
      and computes historical statistics from the training corpus.
    * ``transform(df)``      — call at inference; applies already-fitted
      transformers (raises if called before ``fit_transform``).

    Serialise the fitted instance with :func:`joblib.dump` so inference-time
    feature engineering is byte-for-byte identical to training.

    Args:
        embedding_model_name: Sentence-transformers model identifier.

    Example::

        engineer = BugFeatureEngineer()
        X_train, y_severity, y_impact = engineer.fit_transform(train_df)
        joblib.dump(engineer, "models/artifacts/feature_engineer.joblib")

        engineer = joblib.load("models/artifacts/feature_engineer.joblib")
        X_new = engineer.transform(new_df)
    """

    def __init__(self, embedding_model_name: str | None = None) -> None:
        self._embedding_model_name = embedding_model_name or settings.embedding_model

        self._embedding_model: SentenceTransformer | None = None
        self._pca: PCA | None = None
        self._tfidf: TfidfVectorizer | None = None
        self._component_mlb: MultiLabelBinarizer | None = None
        self._label_mlb: MultiLabelBinarizer | None = None
        self._issuetype_encoder: OrdinalEncoder | None = None

        self._reporter_bug_rate: dict[str, float] = {}
        self._component_failure_rate: dict[str, float] = {}
        self._global_failure_rate: float = 0.0

        self._feature_columns: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Fit all transformers on ``df`` and return feature matrix + targets.

        Args:
            df: Raw DataFrame produced by JiraFetcher.

        Returns:
            Tuple of ``(X, y_severity, y_impact)`` where X is a float64
            feature DataFrame, y_severity is a string Series of severity
            labels, and y_impact is a float Series of impact scores 0-100.
        """
        logger.info("fit_transform called on %d rows", len(df))
        df = self._sanitise(df)

        y_severity = self._build_severity_labels(df)
        y_impact = self._build_impact_scores(df)

        self._fit_historical_stats(df)

        X = self._build_features(df, fitting=True)
        self._feature_columns = list(X.columns)
        self._is_fitted = True
        logger.info("fit_transform complete. Feature matrix shape: %s", X.shape)
        return X, y_severity, y_impact

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformers to new data.

        Args:
            df: Raw DataFrame (same schema as training data).

        Returns:
            Feature DataFrame aligned to training column order.

        Raises:
            RuntimeError: If called before ``fit_transform``.
        """
        if not self._is_fitted:
            raise RuntimeError("BugFeatureEngineer must be fit before calling transform().")

        logger.info("transform called on %d rows", len(df))
        df = self._sanitise(df)
        X = self._build_features(df, fitting=False)
        X = X.reindex(columns=self._feature_columns, fill_value=0.0)
        return X

    def get_feature_names(self) -> list[str]:
        """Return ordered feature column names from training.

        Returns:
            List of feature names, or empty list if not yet fitted.
        """
        return list(self._feature_columns)

    # ------------------------------------------------------------------
    # Internal: data sanitisation
    # ------------------------------------------------------------------

    def _sanitise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce types and fill nulls so downstream code is null-free.

        Args:
            df: Raw input DataFrame.

        Returns:
            Cleaned copy of the DataFrame.
        """
        df = df.copy()

        for col in ("summary", "description"):
            df[col] = df.get(col, "").fillna("").astype(str)

        for col in ("comment_count", "vote_count", "watch_count",
                    "priority_ordinal", "description_length", "age_days",
                    "time_to_resolve_hours", "story_points"):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["priority_name"] = df.get("priority_name", "unknown").fillna("unknown").str.lower()
        df["issuetype"] = df.get("issuetype", "Bug").fillna("Bug").astype(str)
        df["reporter"] = df.get("reporter", "unknown").fillna("unknown").astype(str)
        df["components"] = df.get("components", "").fillna("").astype(str)
        df["labels"] = df.get("labels", "").fillna("").astype(str)

        return df

    # ------------------------------------------------------------------
    # Internal: target variable construction
    # ------------------------------------------------------------------

    def _build_severity_labels(self, df: pd.DataFrame) -> pd.Series:
        """Map Jira priority names to 4-class severity labels.

        Args:
            df: Sanitised DataFrame.

        Returns:
            Series of strings in {Critical, High, Medium, Low}.
        """
        return df["priority_name"].map(
            lambda p: PRIORITY_TO_SEVERITY.get(p, "Low")
        ).rename("severity_label")

    def _build_impact_scores(self, df: pd.DataFrame) -> pd.Series:
        """Compute composite impact score scaled to 0-100.

        Args:
            df: Sanitised DataFrame.

        Returns:
            Series of float scores in [0, 100].
        """
        severity_norm = df["priority_ordinal"] / 5.0

        watch_max = df["watch_count"].max() or 1.0
        watch_norm = df["watch_count"] / watch_max

        comment_max = df["comment_count"].max() or 1.0
        comment_norm = df["comment_count"] / comment_max

        component_failure = df["components"].apply(
            lambda c: self._compute_component_failure_for_row(c, df)
        )
        cfr_max = component_failure.max() or 1.0
        component_failure_norm = component_failure / cfr_max

        raw = (
            IMPACT_WEIGHT_SEVERITY * severity_norm
            + IMPACT_WEIGHT_WATCH * watch_norm
            + IMPACT_WEIGHT_COMPONENT_FAILURE * component_failure_norm
            + IMPACT_WEIGHT_COMMENT * comment_norm
        )
        return (raw * 100).clip(0, 100).rename("impact_score")

    def _compute_component_failure_for_row(
        self, component_str: str, df: pd.DataFrame
    ) -> float:
        """Compute mean severity ordinal for a pipe-delimited component string.

        Args:
            component_str: Pipe-delimited component names.
            df: Full training DataFrame used to compute averages during fit.

        Returns:
            Mean severity ordinal across all matching components.
        """
        if not component_str:
            return self._global_failure_rate or float(df["priority_ordinal"].mean())

        components = [c.strip() for c in component_str.split("|") if c.strip()]
        rates = []
        for comp in components:
            if comp in self._component_failure_rate:
                rates.append(self._component_failure_rate[comp])
            else:
                mask = df["components"].str.contains(comp, regex=False, na=False)
                if mask.any():
                    rates.append(float(df.loc[mask, "priority_ordinal"].mean()))
        return float(np.mean(rates)) if rates else (
            self._global_failure_rate or float(df["priority_ordinal"].mean())
        )

    # ------------------------------------------------------------------
    # Internal: historical statistics
    # ------------------------------------------------------------------

    def _fit_historical_stats(self, df: pd.DataFrame) -> None:
        """Compute and store per-reporter and per-component statistics.

        Args:
            df: Sanitised training DataFrame.
        """
        self._global_failure_rate = float(df["priority_ordinal"].mean())

        critical_mask = df["priority_ordinal"] >= 4
        reporter_total = df.groupby("reporter").size()
        reporter_critical = df[critical_mask].groupby("reporter").size()
        self._reporter_bug_rate = (reporter_critical / reporter_total).fillna(0.0).to_dict()
        logger.debug("Computed reporter_bug_rate for %d reporters", len(self._reporter_bug_rate))

        comp_exploded = (
            df[["components", "priority_ordinal"]]
            .copy()
            .assign(component=df["components"].str.split("|"))
            .explode("component")
        )
        comp_exploded["component"] = comp_exploded["component"].str.strip()
        comp_exploded = comp_exploded[comp_exploded["component"] != ""]
        self._component_failure_rate = (
            comp_exploded.groupby("component")["priority_ordinal"].mean().to_dict()
        )
        logger.debug("Computed component_failure_rate for %d components", len(self._component_failure_rate))

    # ------------------------------------------------------------------
    # Internal: feature matrix construction
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Orchestrate all feature groups and concatenate into one DataFrame.

        Args:
            df: Sanitised DataFrame.
            fitting: If True, fit sub-transformers; otherwise apply them.

        Returns:
            Combined float64 feature DataFrame with no nulls.
        """
        parts: list[pd.DataFrame] = [
            self._text_features(df, fitting),
            self._categorical_features(df, fitting),
            self._numerical_features(df),
        ]
        X = pd.concat(parts, axis=1)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return X

    # ------------------------------------------------------------------
    # Feature group: text
    # ------------------------------------------------------------------

    def _text_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """Build sentence-embedding (PCA-reduced) + TF-IDF features.

        Args:
            df: Sanitised DataFrame.
            fitting: Fit sub-transformers if True.

        Returns:
            DataFrame with columns emb_0..emb_31 and tfidf_0..tfidf_99.
        """
        combined_text = (df["summary"] + " " + df["description"]).tolist()
        summary_text = df["summary"].tolist()

        if fitting:
            logger.info("Loading sentence-transformer model: %s", self._embedding_model_name)
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
            self._pca = PCA(n_components=EMBEDDING_DIM_REDUCED, random_state=42)

        if self._embedding_model is None or self._pca is None:
            raise RuntimeError("Embedding model not loaded — call fit_transform first.")

        logger.debug("Encoding %d texts with sentence-transformer", len(combined_text))
        raw_embeddings: np.ndarray = self._embedding_model.encode(
            combined_text,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if fitting:
            emb_reduced = self._pca.fit_transform(raw_embeddings)
        else:
            emb_reduced = self._pca.transform(raw_embeddings)

        emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM_REDUCED)]
        emb_df = pd.DataFrame(emb_reduced, columns=emb_cols, index=df.index)

        if fitting:
            self._tfidf = TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                min_df=2,
            )
            tfidf_matrix = self._tfidf.fit_transform(summary_text).toarray()
        else:
            if self._tfidf is None:
                raise RuntimeError("TF-IDF vectorizer not fitted.")
            tfidf_matrix = self._tfidf.transform(summary_text).toarray()

        tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf_cols, index=df.index)

        logger.debug("Text features: emb=%s, tfidf=%s", emb_df.shape, tfidf_df.shape)
        return pd.concat([emb_df, tfidf_df], axis=1)

    # ------------------------------------------------------------------
    # Feature group: categorical
    # ------------------------------------------------------------------

    def _categorical_features(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """One-hot encode issuetype, components, labels; ordinal-encode priority.

        Args:
            df: Sanitised DataFrame.
            fitting: Fit sub-transformers if True.

        Returns:
            DataFrame of encoded categorical features.
        """
        parts: list[pd.DataFrame] = []

        # Ordinal priority
        parts.append(df["priority_ordinal"].astype(float).rename("priority_ordinal_enc").to_frame())

        # Issue type
        issuetype_vals = df[["issuetype"]]
        if fitting:
            self._issuetype_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            encoded = self._issuetype_encoder.fit_transform(issuetype_vals)
        else:
            if self._issuetype_encoder is None:
                raise RuntimeError("issuetype encoder not fitted.")
            encoded = self._issuetype_encoder.transform(issuetype_vals)
        parts.append(pd.DataFrame(encoded, columns=["issuetype_enc"], index=df.index))

        # Components (multi-label binarizer)
        components_list = df["components"].apply(
            lambda s: [c.strip() for c in s.split("|") if c.strip()]
        )
        if fitting:
            self._component_mlb = MultiLabelBinarizer()
            comp_matrix = self._component_mlb.fit_transform(components_list)
        else:
            if self._component_mlb is None:
                raise RuntimeError("Component MLB not fitted.")
            comp_matrix = self._component_mlb.transform(components_list)

        comp_cols = [f"comp_{c}" for c in self._component_mlb.classes_]
        parts.append(
            pd.DataFrame(comp_matrix, columns=comp_cols, index=df.index, dtype=float)
        )

        # Labels (multi-label binarizer)
        labels_list = df["labels"].apply(
            lambda s: [lbl.strip() for lbl in s.split("|") if lbl.strip()]
        )
        if fitting:
            self._label_mlb = MultiLabelBinarizer()
            label_matrix = self._label_mlb.fit_transform(labels_list)
        else:
            if self._label_mlb is None:
                raise RuntimeError("Label MLB not fitted.")
            label_matrix = self._label_mlb.transform(labels_list)

        label_cols = [f"label_{lbl}" for lbl in self._label_mlb.classes_]
        parts.append(
            pd.DataFrame(label_matrix, columns=label_cols, index=df.index, dtype=float)
        )

        return pd.concat(parts, axis=1)

    # ------------------------------------------------------------------
    # Feature group: numerical
    # ------------------------------------------------------------------

    def _numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assemble numerical features using historical lookup tables.

        Args:
            df: Sanitised DataFrame.

        Returns:
            DataFrame with raw + lookup numerical feature columns.
        """
        reporter_bug_rate = df["reporter"].map(self._reporter_bug_rate).fillna(
            float(np.mean(list(self._reporter_bug_rate.values()))) if self._reporter_bug_rate else 0.0
        )
        component_failure_rate = df["components"].apply(self._lookup_component_failure)

        return pd.DataFrame(
            {
                "comment_count": df["comment_count"].astype(float),
                "watch_count": df["watch_count"].astype(float),
                "vote_count": df["vote_count"].astype(float),
                "description_length": df["description_length"].astype(float),
                "age_days": df["age_days"].astype(float),
                "story_points": df["story_points"].astype(float),
                "reporter_bug_rate": reporter_bug_rate,
                "component_failure_rate": component_failure_rate,
            },
            index=df.index,
        )

    def _lookup_component_failure(self, component_str: str) -> float:
        """Look up stored component failure rate for one or more components.

        Args:
            component_str: Pipe-delimited component names.

        Returns:
            Mean failure rate, or global mean if no match found.
        """
        if not component_str:
            return self._global_failure_rate

        components = [c.strip() for c in component_str.split("|") if c.strip()]
        rates = [
            self._component_failure_rate[c]
            for c in components
            if c in self._component_failure_rate
        ]
        return float(np.mean(rates)) if rates else self._global_failure_rate

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted engineer to disk via joblib.

        Args:
            path: Destination file path.
        """
        joblib.dump(self, path)
        logger.info("BugFeatureEngineer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BugFeatureEngineer":
        """Load a previously saved engineer from disk.

        Args:
            path: File path of the joblib dump.

        Returns:
            Fitted BugFeatureEngineer instance.
        """
        instance: BugFeatureEngineer = joblib.load(path)
        logger.info("BugFeatureEngineer loaded from %s", path)
        return instance
