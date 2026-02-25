"""
tests/test_preprocessor.py — Unit tests for BugFeatureEngineer.

Tests cover:
  - fit_transform produces correct shapes and column counts
  - transform (inference) aligns to training columns
  - Target variables (severity labels, impact scores) are valid
  - Historical lookup tables are populated correctly
  - Null / edge-case handling (empty text, missing components)
  - fit_transform -> save -> load -> transform round-trip (joblib)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("JIRA_URL", "https://test.atlassian.net")
os.environ.setdefault("JIRA_EMAIL", "test@example.com")
os.environ.setdefault("JIRA_API_TOKEN", "dummy-token")
os.environ.setdefault("JIRA_PROJECT_KEY", "TEST")
os.environ.setdefault("RETRAIN_API_KEY", "test-key")
os.environ.setdefault("MODEL_PATH", "/tmp/test_models")

from data.preprocessor import BugFeatureEngineer, SEVERITY_CLASSES  # noqa: E402


def _make_df(n: int = 60) -> pd.DataFrame:
    """Build a synthetic DataFrame that mimics JiraFetcher output."""
    rng = np.random.default_rng(42)
    priorities = ["blocker", "critical", "major", "minor", "trivial"]
    components_pool = ["Auth", "Payments", "Frontend", "Backend", ""]
    labels_pool = ["regression", "customer-reported", ""]
    priority_ordinal_map = {"blocker": 5, "critical": 4, "major": 3, "minor": 2, "trivial": 1}

    rows = []
    for i in range(n):
        priority = priorities[i % len(priorities)]
        rows.append({
            "jira_key": f"TEST-{i + 1}",
            "summary": f"Bug {i}: something is broken in the system" if i % 3 != 0 else "",
            "description": f"Detailed description for bug {i}. Steps: click button {i}." if i % 4 != 0 else "",
            "priority_name": priority,
            "priority_ordinal": priority_ordinal_map[priority],
            "issuetype": "Bug",
            "components": components_pool[i % len(components_pool)],
            "labels": labels_pool[i % len(labels_pool)],
            "reporter": f"user_{i % 10}",
            "comment_count": int(rng.integers(0, 20)),
            "vote_count": int(rng.integers(0, 10)),
            "watch_count": int(rng.integers(0, 30)),
            "story_points": float(rng.choice([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])),
            "age_days": float(rng.integers(0, 365)),
            "description_length": 0,
            "time_to_resolve_hours": float(rng.integers(1, 200)),
        })

    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return _make_df(60)


@pytest.fixture(scope="module")
def fitted_result(sample_df):
    eng = BugFeatureEngineer()
    X, y_sev, y_imp = eng.fit_transform(sample_df)
    return eng, X, y_sev, y_imp


# ---------------------------------------------------------------------------
# fit_transform output shapes
# ---------------------------------------------------------------------------

class TestFitTransformShapes:
    def test_row_count_matches(self, fitted_result, sample_df):
        _, X, _, _ = fitted_result
        assert len(X) == len(sample_df)

    def test_no_nulls(self, fitted_result):
        _, X, _, _ = fitted_result
        assert not X.isnull().any().any()

    def test_no_infinities(self, fitted_result):
        _, X, _, _ = fitted_result
        assert not np.isinf(X.values).any()

    def test_embedding_columns(self, fitted_result):
        _, X, _, _ = fitted_result
        assert len([c for c in X.columns if c.startswith("emb_")]) == 32

    def test_tfidf_columns_present(self, fitted_result):
        _, X, _, _ = fitted_result
        assert len([c for c in X.columns if c.startswith("tfidf_")]) > 0

    def test_numerical_columns_present(self, fitted_result):
        _, X, _, _ = fitted_result
        for col in ("comment_count", "watch_count", "vote_count", "reporter_bug_rate"):
            assert col in X.columns

    def test_feature_names_match_columns(self, fitted_result):
        eng, X, _, _ = fitted_result
        assert eng.get_feature_names() == list(X.columns)


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------

class TestTargets:
    def test_severity_labels_valid(self, fitted_result):
        _, _, y_sev, _ = fitted_result
        assert set(y_sev.unique()).issubset(set(SEVERITY_CLASSES))

    def test_severity_length(self, fitted_result, sample_df):
        _, _, y_sev, _ = fitted_result
        assert len(y_sev) == len(sample_df)

    def test_impact_in_range(self, fitted_result):
        _, _, _, y_imp = fitted_result
        assert (y_imp >= 0).all() and (y_imp <= 100).all()

    def test_blocker_is_critical(self, fitted_result, sample_df):
        _, _, y_sev, _ = fitted_result
        assert (y_sev[sample_df["priority_name"] == "blocker"] == "Critical").all()

    def test_trivial_is_low(self, fitted_result, sample_df):
        _, _, y_sev, _ = fitted_result
        assert (y_sev[sample_df["priority_name"] == "trivial"] == "Low").all()


# ---------------------------------------------------------------------------
# Historical stats
# ---------------------------------------------------------------------------

class TestHistoricalStats:
    def test_reporter_bug_rate_populated(self, fitted_result):
        eng, _, _, _ = fitted_result
        assert len(eng._reporter_bug_rate) > 0

    def test_reporter_rates_in_range(self, fitted_result):
        eng, _, _, _ = fitted_result
        for rate in eng._reporter_bug_rate.values():
            assert 0.0 <= rate <= 1.0

    def test_component_failure_rate_populated(self, fitted_result):
        eng, _, _, _ = fitted_result
        assert len(eng._component_failure_rate) > 0

    def test_global_failure_rate_positive(self, fitted_result):
        eng, _, _, _ = fitted_result
        assert eng._global_failure_rate > 0


# ---------------------------------------------------------------------------
# transform / inference mode
# ---------------------------------------------------------------------------

class TestTransform:
    def test_shape_matches_training(self, fitted_result, sample_df):
        eng, X_fit, _, _ = fitted_result
        X_inf = eng.transform(sample_df.iloc[:10])
        assert X_inf.shape[1] == X_fit.shape[1]
        assert len(X_inf) == 10

    def test_column_order_preserved(self, fitted_result, sample_df):
        eng, X_fit, _, _ = fitted_result
        X_inf = eng.transform(sample_df.iloc[:5])
        assert list(X_inf.columns) == list(X_fit.columns)

    def test_unknown_component_no_crash(self, fitted_result, sample_df):
        eng, X_fit, _, _ = fitted_result
        row = sample_df.iloc[[0]].copy()
        row["components"] = "UnseenComponent_ABC"
        X = eng.transform(row)
        assert not X.isnull().any().any()

    def test_raises_before_fit(self):
        eng = BugFeatureEngineer()
        with pytest.raises(RuntimeError, match="must be fit"):
            eng.transform(pd.DataFrame([{"summary": "test"}]))

    def test_empty_text_no_crash(self, fitted_result):
        eng, X_fit, _, _ = fitted_result
        row = pd.DataFrame([{
            "summary": "", "description": "", "priority_name": "major",
            "priority_ordinal": 3, "issuetype": "Bug", "components": "",
            "labels": "", "reporter": "unknown", "comment_count": 0,
            "vote_count": 0, "watch_count": 0, "story_points": 0.0,
            "age_days": 0.0, "description_length": 0,
        }])
        X = eng.transform(row)
        assert not X.isnull().any().any()
        assert X.shape[1] == X_fit.shape[1]


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_load_identical_output(self, fitted_result, sample_df):
        eng, X_original, _, _ = fitted_result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "eng.joblib")
            eng.save(path)
            loaded = BugFeatureEngineer.load(path)
            X_loaded = loaded.transform(sample_df)
        pd.testing.assert_frame_equal(
            X_original.reset_index(drop=True),
            X_loaded.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )

    def test_loaded_is_fitted(self, fitted_result):
        eng, _, _, _ = fitted_result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "eng.joblib")
            eng.save(path)
            loaded = BugFeatureEngineer.load(path)
        assert loaded._is_fitted

    def test_loaded_feature_names_match(self, fitted_result):
        eng, _, _, _ = fitted_result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "eng.joblib")
            eng.save(path)
            loaded = BugFeatureEngineer.load(path)
        assert loaded.get_feature_names() == eng.get_feature_names()
