"""
models/trainer.py — Training pipeline for bug severity classification and impact regression.

BugPriorityTrainer orchestrates:
  1. Stratified 80/20 train/test split
  2. Optuna hyperparameter tuning (XGBoostClassifier, 20 trials)
  3. Stratified 5-fold cross-validation with F1-macro, precision, recall per class
  4. XGBoostRegressor for impact score (RMSE, MAE, R²)
  5. Joblib serialisation of both models + feature engineer
  6. Optional MLflow logging (enabled via MLFLOW_TRACKING_URI env var)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from bug_prioritization_engine.config import settings
from bug_prioritization_engine.data.preprocessor import SEVERITY_CLASSES, BugFeatureEngineer

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClassifierMetrics:
    """Evaluation metrics for the severity classifier."""
    f1_macro: float
    f1_per_class: dict[str, float]
    precision_per_class: dict[str, float]
    recall_per_class: dict[str, float]
    best_params: dict[str, Any]
    cv_f1_mean: float
    cv_f1_std: float


@dataclass
class RegressorMetrics:
    """Evaluation metrics for the impact score regressor."""
    rmse: float
    mae: float
    r2: float
    best_params: dict[str, Any]


@dataclass
class TrainingResult:
    """Aggregated output from a full training run."""
    classifier_metrics: ClassifierMetrics
    regressor_metrics: RegressorMetrics
    model_version: str
    trained_at: str
    feature_count: int
    train_size: int
    test_size: int


# ---------------------------------------------------------------------------
# BugPriorityTrainer
# ---------------------------------------------------------------------------


class BugPriorityTrainer:
    """Trains XGBoost severity classifier and impact score regressor.

    Args:
        model_path: Directory where artefacts are saved.
        n_optuna_trials: Number of Optuna trials per model (default 20).
        cv_folds: Number of stratified CV folds (default 5).
        random_state: Global random seed for reproducibility.

    Example::

        trainer = BugPriorityTrainer()
        result = trainer.train(df=historical_df)
        print(result.classifier_metrics.f1_macro)
    """

    CLASSIFIER_FNAME = "severity_classifier.joblib"
    REGRESSOR_FNAME = "impact_regressor.joblib"
    ENGINEER_FNAME = "feature_engineer.joblib"
    METADATA_FNAME = "training_metadata.json"

    LABEL_ENCODING: dict[str, int] = {
        "Highest": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3,
        "Lowest": 4,
    }
    LABEL_DECODING: dict[int, str] = {v: k for k, v in LABEL_ENCODING.items()}

    def __init__(
        self,
        model_path: Path | None = None,
        n_optuna_trials: int = 20,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self._model_path = model_path or settings.model_path
        self._n_trials = n_optuna_trials
        self._cv_folds = cv_folds
        self._rs = random_state

        self._classifier: XGBClassifier | None = None
        self._regressor: XGBRegressor | None = None
        self._engineer: BugFeatureEngineer | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame | None = None,
        csv_path: str | None = None,
    ) -> TrainingResult:
        """Run the full training pipeline.

        Args:
            df: Pre-loaded raw Jira DataFrame.
            csv_path: Path to a CSV file of raw Jira data.

        Returns:
            TrainingResult with metrics and metadata.

        Raises:
            ValueError: If neither df nor csv_path is provided.
        """
        if df is None and csv_path is None:
            raise ValueError("Provide either df or csv_path.")

        if df is None:
            logger.info("Loading training data from CSV: %s", csv_path)
            df = pd.read_csv(csv_path)

        logger.info("Training started. Input shape: %s", df.shape)
        self._validate_min_samples(df)

        self._engineer = BugFeatureEngineer()
        X, y_severity_raw, y_impact = self._engineer.fit_transform(df)

        y_severity = y_severity_raw.map(self.LABEL_ENCODING).astype(int)

        X_train, X_test, ys_train, ys_test, yi_train, yi_test = train_test_split(
            X, y_severity, y_impact,
            test_size=0.2,
            random_state=self._rs,
            stratify=y_severity,
        )
        logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))

        clf_metrics = self._train_classifier(X_train, X_test, ys_train, ys_test)
        reg_metrics = self._train_regressor(X_train, X_test, yi_train, yi_test)

        model_version = str(int(time.time()))
        trained_at = pd.Timestamp.now(tz="UTC").isoformat()
        self._save_artefacts(model_version, trained_at, clf_metrics, reg_metrics, X.shape[1])

        result = TrainingResult(
            classifier_metrics=clf_metrics,
            regressor_metrics=reg_metrics,
            model_version=model_version,
            trained_at=trained_at,
            feature_count=X.shape[1],
            train_size=len(X_train),
            test_size=len(X_test),
        )

        self._log_summary(result)
        self._try_mlflow(result, X_train)
        return result

    # ------------------------------------------------------------------
    # Classifier
    # ------------------------------------------------------------------

    def _train_classifier(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> ClassifierMetrics:
        """Tune and train the XGBoost severity classifier with Optuna.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            y_train: Integer-encoded training severity labels.
            y_test: Integer-encoded test severity labels.

        Returns:
            ClassifierMetrics with per-class and macro metrics.
        """
        logger.info("Tuning severity classifier (%d Optuna trials)...", self._n_trials)

        class_counts = y_train.value_counts().to_dict()
        total = len(y_train)
        n_classes = len(self.LABEL_ENCODING)
        scale_pos_weight = {
            cls: total / (n_classes * count)
            for cls, count in class_counts.items()
        }

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            cv_scores = self._cv_score_classifier(X_train, y_train, params, scale_pos_weight)
            return float(np.mean(cv_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self._rs),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Best classifier params: %s | CV F1-macro: %.4f", best_params, study.best_value)

        self._classifier = self._build_classifier(best_params)
        self._classifier.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self._classifier.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        label_map = {str(v): k for k, v in self.LABEL_ENCODING.items()}

        def extract(metric: str) -> dict[str, float]:
            return {
                label_map.get(k, k): v[metric]
                for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }

        cv_scores = self._cv_score_classifier(X_train, y_train, best_params, scale_pos_weight)

        return ClassifierMetrics(
            f1_macro=float(report["macro avg"]["f1-score"]),
            f1_per_class=extract("f1-score"),
            precision_per_class=extract("precision"),
            recall_per_class=extract("recall"),
            best_params=best_params,
            cv_f1_mean=float(np.mean(cv_scores)),
            cv_f1_std=float(np.std(cv_scores)),
        )

    def _cv_score_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict[str, Any],
        scale_pos_weight: dict[int, float],
    ) -> list[float]:
        """Stratified k-fold CV, returns per-fold F1-macro scores.

        Args:
            X: Feature matrix.
            y: Integer severity labels.
            params: XGBoost hyperparameters.
            scale_pos_weight: Per-class weight mapping (unused directly —
                XGBoost multiclass handles imbalance via sample_weight if needed).

        Returns:
            List of F1-macro scores, one per fold.
        """
        skf = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=self._rs)
        scores: list[float] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            clf = self._build_classifier(params)
            clf.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
            y_pred = clf.predict(X.iloc[val_idx])
            score = f1_score(y.iloc[val_idx], y_pred, average="macro", zero_division=0)
            scores.append(float(score))
            logger.debug("CV fold %d F1-macro: %.4f", fold + 1, score)

        return scores

    def _build_classifier(self, params: dict[str, Any]) -> XGBClassifier:
        """Instantiate XGBClassifier with given hyperparameters.

        Args:
            params: Hyperparameters from Optuna.

        Returns:
            Configured but unfitted XGBClassifier.
        """
        return XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=len(self.LABEL_ENCODING),
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            random_state=self._rs,
            n_jobs=-1,
            early_stopping_rounds=None,
        )

    # ------------------------------------------------------------------
    # Regressor
    # ------------------------------------------------------------------

    def _train_regressor(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> RegressorMetrics:
        """Tune and train the XGBoost impact score regressor.

        Args:
            X_train: Training feature matrix.
            X_test: Test feature matrix.
            y_train: Training impact scores (0-100).
            y_test: Test impact scores (0-100).

        Returns:
            RegressorMetrics with RMSE, MAE, R².
        """
        logger.info("Tuning impact regressor (%d Optuna trials)...", self._n_trials)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
            }
            y_binned = pd.cut(y_train, bins=4, labels=False)
            skf = StratifiedKFold(n_splits=self._cv_folds, shuffle=True, random_state=self._rs)
            rmses: list[float] = []
            for train_idx, val_idx in skf.split(X_train, y_binned):
                reg = self._build_regressor(params)
                reg.fit(
                    X_train.iloc[train_idx], y_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    verbose=False,
                )
                preds = reg.predict(X_train.iloc[val_idx])
                rmses.append(float(np.sqrt(mean_squared_error(y_train.iloc[val_idx], preds))))
            return float(np.mean(rmses))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self._rs),
        )
        study.optimize(objective, n_trials=self._n_trials, show_progress_bar=False)

        best_params = study.best_params
        logger.info("Best regressor params: %s | CV RMSE: %.4f", best_params, study.best_value)

        self._regressor = self._build_regressor(best_params)
        self._regressor.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self._regressor.predict(X_test)
        return RegressorMetrics(
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            mae=float(mean_absolute_error(y_test, y_pred)),
            r2=float(r2_score(y_test, y_pred)),
            best_params=best_params,
        )

    def _build_regressor(self, params: dict[str, Any]) -> XGBRegressor:
        """Instantiate XGBRegressor with given hyperparameters.

        Args:
            params: Hyperparameters from Optuna.

        Returns:
            Configured but unfitted XGBRegressor.
        """
        return XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=self._rs,
            n_jobs=-1,
            early_stopping_rounds=20,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_artefacts(
        self,
        model_version: str,
        trained_at: str,
        clf_metrics: ClassifierMetrics,
        reg_metrics: RegressorMetrics,
        feature_count: int,
    ) -> None:
        """Persist models, feature engineer, and metadata to disk.

        Args:
            model_version: Unix-timestamp string used as version identifier.
            trained_at: ISO-8601 UTC timestamp.
            clf_metrics: Classifier evaluation metrics.
            reg_metrics: Regressor evaluation metrics.
            feature_count: Number of features in the trained model.
        """
        path = self._model_path
        joblib.dump(self._classifier, path / self.CLASSIFIER_FNAME)
        joblib.dump(self._regressor, path / self.REGRESSOR_FNAME)
        joblib.dump(self._engineer, path / self.ENGINEER_FNAME)

        metadata = {
            "model_version": model_version,
            "trained_at": trained_at,
            "feature_count": feature_count,
            "classifier": asdict(clf_metrics),
            "regressor": asdict(reg_metrics),
        }
        (path / self.METADATA_FNAME).write_text(json.dumps(metadata, indent=2))
        logger.info("Artefacts saved to %s (version=%s)", path, model_version)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _validate_min_samples(self, df: pd.DataFrame) -> None:
        """Raise if dataset is too small for reliable training.

        Args:
            df: Raw input DataFrame.

        Raises:
            ValueError: If the dataset has fewer than 50 rows.
        """
        if len(df) < 50:
            raise ValueError(
                f"Dataset has only {len(df)} rows. Minimum 50 required."
            )

    def _log_summary(self, result: TrainingResult) -> None:
        """Pretty-print training results to the logger.

        Args:
            result: Completed TrainingResult.
        """
        c = result.classifier_metrics
        r = result.regressor_metrics
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE — version %s", result.model_version)
        logger.info(
            "Features: %d | Train: %d | Test: %d",
            result.feature_count, result.train_size, result.test_size,
        )
        logger.info("--- Severity Classifier ---")
        logger.info("  F1-macro (test):  %.4f", c.f1_macro)
        logger.info("  CV F1-macro:      %.4f +/- %.4f", c.cv_f1_mean, c.cv_f1_std)
        for cls in SEVERITY_CLASSES:
            logger.info(
                "  %-10s  P=%.3f  R=%.3f  F1=%.3f",
                cls,
                c.precision_per_class.get(cls, 0.0),
                c.recall_per_class.get(cls, 0.0),
                c.f1_per_class.get(cls, 0.0),
            )
        logger.info("--- Impact Regressor ---")
        logger.info("  RMSE=%.4f  MAE=%.4f  R2=%.4f", r.rmse, r.mae, r.r2)
        logger.info("=" * 60)

    def _try_mlflow(self, result: TrainingResult, X_train: pd.DataFrame) -> None:
        """Optionally log to MLflow if MLFLOW_TRACKING_URI is set.

        Args:
            result: Completed TrainingResult.
            X_train: Training feature matrix (shape logged as metric).
        """
        import os
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            return
        try:
            import mlflow
            c = result.classifier_metrics
            r = result.regressor_metrics
            with mlflow.start_run(run_name=f"bug_priority_{result.model_version}"):
                mlflow.log_params(c.best_params)
                mlflow.log_metrics({
                    "clf_f1_macro": c.f1_macro,
                    "clf_cv_f1_mean": c.cv_f1_mean,
                    "clf_cv_f1_std": c.cv_f1_std,
                    "reg_rmse": r.rmse,
                    "reg_mae": r.mae,
                    "reg_r2": r.r2,
                    "feature_count": float(result.feature_count),
                })
                mlflow.log_artifact(str(self._model_path / self.METADATA_FNAME))
            logger.info("MLflow run logged successfully.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("MLflow logging skipped: %s", exc)

    # ------------------------------------------------------------------
    # Convenience loader
    # ------------------------------------------------------------------

    @classmethod
    def load_artefacts(
        cls, model_path: Path | None = None
    ) -> tuple[XGBClassifier, XGBRegressor, BugFeatureEngineer, dict]:
        """Load persisted models and metadata from disk.

        Args:
            model_path: Directory containing saved artefacts.

        Returns:
            Tuple of (classifier, regressor, feature_engineer, metadata_dict).

        Raises:
            FileNotFoundError: If any expected artefact file is missing.
        """
        path = model_path or settings.model_path
        classifier = joblib.load(path / cls.CLASSIFIER_FNAME)
        regressor = joblib.load(path / cls.REGRESSOR_FNAME)
        engineer = joblib.load(path / cls.ENGINEER_FNAME)
        metadata = json.loads((path / cls.METADATA_FNAME).read_text())
        logger.info(
            "Artefacts loaded from %s (version=%s)",
            path, metadata.get("model_version"),
        )
        return classifier, regressor, engineer, metadata
