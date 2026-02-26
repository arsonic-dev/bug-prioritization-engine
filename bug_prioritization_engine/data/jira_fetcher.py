"""
data/jira_fetcher.py — Jira REST API ingestion client.

Uses the `jira` Python SDK under the hood, adds pagination, exponential-backoff
retry logic, and returns a tidy pandas DataFrame ready for feature engineering.
"""

from __future__ import annotations
import requests
import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from jira import JIRA, JIRAError

from bug_prioritization_engine.config import settings

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
}

FIELDS_TO_FETCH = [
    "summary",
    "description",
    "priority",
    "issuetype",
    "status",
    "components",
    "labels",
    "fixVersions",
    "reporter",
    "assignee",
    "created",
    "resolutiondate",
    "comment",
    "votes",
    "watches",
    "customfield_10016",  # story points — may not exist on all Jira configs
]

_MAX_RESULTS_PER_PAGE = 100
_MAX_TOTAL_ISSUES = 10_000
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _exponential_backoff(attempt: int, base: float = _BACKOFF_BASE) -> None:
    """Sleep for base^attempt seconds (capped at 60s)."""
    wait = min(base**attempt, 60.0)
    logger.warning("Rate-limited or server error — retrying in %.1fs (attempt %d)", wait, attempt)
    time.sleep(wait)


def _safe_display_name(field: Any | None) -> str | None:
    """Extract displayName from a Jira user object, or return None."""
    if field is None:
        return None
    return getattr(field, "displayName", None)


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse a Jira ISO-8601 datetime string to a timezone-aware datetime."""
    if not dt_str:
        return None
    # Jira returns e.g. "2024-03-15T10:22:00.000+0000"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    logger.debug("Could not parse datetime string: %s", dt_str)
    return None


def _hours_between(start: datetime | None, end: datetime | None) -> float | None:
    """Return fractional hours between two datetimes, or None if either is missing."""
    if start is None or end is None:
        return None
    delta = end - start
    return delta.total_seconds() / 3600


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class JiraFetcher:
    """Fetches bug/issue data from Jira and returns a pandas DataFrame.

    Args:
        url: Jira instance base URL (defaults to ``settings.jira_url``).
        email: Authenticating user email (defaults to ``settings.jira_email``).
        api_token: Jira API token (defaults to ``settings.jira_api_token``).

    Example::

        fetcher = JiraFetcher()
        df = fetcher.fetch_resolved_issues(project_key="MYPROJ", max_results=500)
    """

    def __init__(
        self,
        url: str | None = None,
        email: str | None = None,
        api_token: str | None = None,
    ) -> None:
        self._url = url or settings.jira_url
        self._email = email or settings.jira_email
        self._api_token = api_token or settings.jira_api_token
        self._client: JIRA | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise the JIRA SDK client.  Called lazily on first use."""
        if self._client is not None:
            return
        logger.info("Connecting to Jira at %s as %s", self._url, self._email)
        try:
            self._client = JIRA(
                server=self._url,
                basic_auth=(self._email, self._api_token),
                options={"verify": True},
            )
            # Smoke-test: fetch current user
            myself = self._client.myself()
            logger.info("Connected to Jira successfully. Authenticated as: %s", myself.get("displayName"))
        except JIRAError as exc:
            self._handle_jira_error(exc, context="connect()")
            raise

    @property
    def client(self) -> JIRA:
        """Return the connected JIRA client, connecting if necessary."""
        if self._client is None:
            self.connect()
        return self._client  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_resolved_issues(
        self,
        project_key: str | None = None,
        max_results: int = _MAX_TOTAL_ISSUES,
        extra_jql: str = "",
    ) -> pd.DataFrame:
        """Fetch resolved issues from a Jira project.

        Args:
            project_key: Jira project key (e.g. ``"MYPROJ"``).  Defaults to
                ``settings.jira_project_key``.
            max_results: Upper bound on total issues to retrieve (hard-capped
                at 10,000 by this client).
            extra_jql: Optional JQL fragment appended with AND, e.g.
                ``"created >= -90d"`` for incremental loads.

        Returns:
            A :class:`pandas.DataFrame` with one row per issue and the columns
            documented in :pydata:`FIELDS_TO_FETCH` plus derived fields.
        """
        project_key = project_key or settings.jira_project_key
        max_results = min(max_results, _MAX_TOTAL_ISSUES)

        jql = f'project = "{project_key}" AND statusCategory = Done AND issuetype = Bug'
        if extra_jql:
            jql += f" AND {extra_jql}"
        jql += " ORDER BY created DESC"

        logger.info("Fetching resolved issues with JQL: %s", jql)

        raw_issues = self._paginate(jql=jql, total_limit=max_results)
        logger.info("Fetched %d issues from Jira", len(raw_issues))

        return self._to_dataframe(raw_issues)

    def fetch_open_issues(
        self,
        project_key: str | None = None,
        max_results: int = 500,
    ) -> pd.DataFrame:
        """Fetch open (unresolved) issues for the ranked-queue endpoint.

        Args:
            project_key: Jira project key.
            max_results: Max issues to retrieve.

        Returns:
            DataFrame of open issues.
        """
        project_key = project_key or settings.jira_project_key
        jql = (
            f'project = "{project_key}" AND statusCategory != Done '
            f'AND issuetype = Bug ORDER BY created DESC'
        )
        logger.info("Fetching open issues with JQL: %s", jql)
        raw_issues = self._paginate(jql=jql, total_limit=min(max_results, _MAX_TOTAL_ISSUES))
        logger.info("Fetched %d open issues from Jira", len(raw_issues))
        return self._to_dataframe(raw_issues)

    def fetch_issue_by_key(self, jira_key: str) -> pd.DataFrame:
        """Fetch a single issue by its key (e.g. ``"MYPROJ-123"``).

        Args:
            jira_key: The Jira issue key.

        Returns:
            A single-row DataFrame.

        Raises:
            JIRAError: If the issue does not exist or access is denied.
        """
        logger.info("Fetching single issue: %s", jira_key)
        issue = self._fetch_with_retry(lambda: self.client.issue(jira_key, fields=",".join(FIELDS_TO_FETCH)))
        return self._to_dataframe([issue])

    # ------------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------------

    def _paginate(self, jql: str, total_limit: int) -> list[Any]:
        """Paginate through Jira search results using REST API v3."""

        issues: list[Any] = []
        start_at = 0

        while len(issues) < total_limit:
            batch_size = min(_MAX_RESULTS_PER_PAGE, total_limit - len(issues))

            def _search(start=start_at, size=batch_size):
                url = f"{self._url}/rest/api/3/search/jql"

                payload = {
                    "jql": jql,
                    "startAt": start,
                    "maxResults": size,
                    "fields": FIELDS_TO_FETCH,
                }

                response = requests.post(
                    url,
                    json=payload,
                    auth=(self._email, self._api_token),
                    headers={"Accept": "application/json"},
                    timeout=30,
                )

                if response.status_code != 200:
                    raise JIRAError(
                        status_code=response.status_code,
                        text=response.text,
                        response=response,
                    )

                data = response.json()
                return data.get("issues", [])

            page = self._fetch_with_retry(_search)

            if not page:
                logger.debug("Empty page at startAt=%d — stopping pagination", start_at)
                break

            issues.extend(page)

            if len(page) < batch_size:
                break

            start_at += len(page)

        return issues

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Call ``fn`` with exponential-backoff retry on transient errors.

        Args:
            fn: Callable to invoke.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            Whatever ``fn`` returns on success.

        Raises:
            JIRAError: After all retries are exhausted or on non-retryable errors.
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except JIRAError as exc:
                self._handle_jira_error(exc, context=str(fn))
                if exc.status_code not in _RETRY_STATUSES or attempt == _MAX_RETRIES:
                    raise
                _exponential_backoff(attempt)
        return None  # unreachable — satisfies mypy

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_jira_error(exc: JIRAError, context: str = "") -> None:
        """Log a structured error message for a JIRAError.

        Args:
            exc: The caught :class:`jira.JIRAError`.
            context: Human-readable label for where the error occurred.
        """
        status = exc.status_code
        if status == 401:
            logger.error("[%s] Jira authentication failed (401). Check JIRA_EMAIL and JIRA_API_TOKEN.", context)
        elif status == 403:
            logger.error("[%s] Jira permission denied (403). Check project permissions.", context)
        elif status == 429:
            logger.warning("[%s] Jira rate limit hit (429).", context)
        elif status == 500:
            logger.error("[%s] Jira server error (500): %s", context, exc.text)
        else:
            logger.error("[%s] JIRAError status=%s: %s", context, status, exc.text)

    # ------------------------------------------------------------------
    # DataFrame construction
    # ------------------------------------------------------------------

    def _to_dataframe(self, issues: list[Any]) -> pd.DataFrame:
        """Convert a list of Jira issue objects to a cleaned DataFrame.

        Args:
            issues: List of Jira issue objects returned by the SDK.

        Returns:
            Cleaned :class:`pandas.DataFrame`.
        """
        rows = [self._extract_row(issue) for issue in issues]
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Derive time_to_resolve_hours
        df["time_to_resolve_hours"] = df.apply(
            lambda r: _hours_between(r["created_dt"], r["resolved_dt"]),
            axis=1,
        )

        # Derive age_days (relevant for open issues; will be NaN if resolved)
        now = datetime.now(tz=timezone.utc)
        df["age_days"] = df["created_dt"].apply(
            lambda dt: (now - dt).days if dt is not None else None
        )

        # Drop raw datetime columns — keep only derived numeric ones
        df.drop(columns=["created_dt", "resolved_dt"], inplace=True)

        logger.debug("DataFrame shape after _to_dataframe: %s", df.shape)
        return df

    def _extract_row(self, issue: Any) -> dict[str, Any]:
        """Extract a flat dictionary of fields from a single Jira issue object.

        Args:
            issue: A Jira issue object from the SDK.

        Returns:
            Dictionary mapping column names to values.
        """
        f = issue["fields"]  # shorthand

        # --- Text
        summary: str = getattr(f, "summary", "") or ""
        description: str = getattr(f, "description", "") or ""

        # --- Priority
        priority_obj = getattr(f, "priority", None)
        priority_name: str = getattr(priority_obj, "name", "Unknown") if priority_obj else "Unknown"
        priority_ordinal: int = PRIORITY_ORDINAL.get(priority_name.lower(), 0)

        # --- Issue type / status
        issuetype_obj = getattr(f, "issuetype", None)
        issuetype: str = getattr(issuetype_obj, "name", "Unknown") if issuetype_obj else "Unknown"

        status_obj = getattr(f, "status", None)
        status: str = getattr(status_obj, "name", "Unknown") if status_obj else "Unknown"

        # --- Multi-value fields → pipe-delimited strings (easier to parse later)
        components_raw = getattr(f, "components", []) or []
        components: str = "|".join(c.name for c in components_raw if hasattr(c, "name"))

        labels: list[str] = getattr(f, "labels", []) or []
        labels_str: str = "|".join(labels)

        fix_versions_raw = getattr(f, "fixVersions", []) or []
        fix_versions: str = "|".join(v.name for v in fix_versions_raw if hasattr(v, "name"))

        # --- People
        reporter: str | None = _safe_display_name(getattr(f, "reporter", None))
        assignee: str | None = _safe_display_name(getattr(f, "assignee", None))

        # --- Dates
        created_dt: datetime | None = _parse_datetime(getattr(f, "created", None))
        resolved_dt: datetime | None = _parse_datetime(getattr(f, "resolutiondate", None))

        # --- Engagement
        comment_obj = getattr(f, "comment", None)
        comment_count: int = getattr(comment_obj, "total", 0) if comment_obj else 0

        votes_obj = getattr(f, "votes", None)
        vote_count: int = getattr(votes_obj, "votes", 0) if votes_obj else 0

        watches_obj = getattr(f, "watches", None)
        watch_count: int = getattr(watches_obj, "watchCount", 0) if watches_obj else 0

        # --- Custom fields
        story_points: float | None = getattr(f, "customfield_10016", None)

        return {
            "jira_key": issue["key"],
            "summary": summary,
            "description": description,
            "priority_name": priority_name,
            "priority_ordinal": priority_ordinal,
            "issuetype": issuetype,
            "status": status,
            "components": components,
            "labels": labels_str,
            "fix_versions": fix_versions,
            "reporter": reporter,
            "assignee": assignee,
            "created_dt": created_dt,
            "resolved_dt": resolved_dt,
            "comment_count": comment_count,
            "vote_count": vote_count,
            "watch_count": watch_count,
            "story_points": story_points,
            "description_length": len(description),
        }
