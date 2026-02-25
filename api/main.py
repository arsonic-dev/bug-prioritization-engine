"""
api/main.py — FastAPI application entry point.

Responsibilities:
  - CORS middleware configuration
  - Startup: load models into app.state for zero-latency request handling
  - Shutdown: graceful scheduler teardown
  - Global exception handler with structured JSON error responses
  - Router inclusion
"""

from __future__ import annotations

import json
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import settings

# ---------------------------------------------------------------------------
# Logging setup — JSON format
# ---------------------------------------------------------------------------

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    },
    "root": {
        "level": settings.log_level,
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup + shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan: load models on startup, teardown on shutdown.

    On startup:
      - Load trained XGBoost models + feature engineer from disk.
      - Initialise Jira fetcher.
      - Start APScheduler retraining job.

    On shutdown:
      - Gracefully stop the background scheduler.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the running application.
    """
    # ---- STARTUP ----
    logger.info("Starting Bug Prioritization Engine...")

    # Jira fetcher (lazy connect — no network call until first use)
    from data.jira_fetcher import JiraFetcher
    app.state.fetcher = JiraFetcher()
    logger.info("JiraFetcher initialised.")

    # Load models (predictor = classifier + regressor + engineer + explainer)
    from models.predictor import BugPredictor
    metadata_path = settings.model_path / "training_metadata.json"

    if metadata_path.exists():
        try:
            app.state.predictor = BugPredictor.from_artefacts()
            app.state.model_metadata = json.loads(metadata_path.read_text())
            logger.info(
                "Models loaded. Version: %s",
                app.state.model_metadata.get("model_version"),
            )
        except Exception as exc:
            logger.warning(
                "Could not load models (not yet trained?): %s — starting in degraded mode.", exc
            )
            app.state.predictor = None
            app.state.model_metadata = {}
    else:
        logger.warning(
            "No trained model found at %s — API running in degraded mode. "
            "Call POST /api/v1/retrain to train an initial model.",
            settings.model_path,
        )
        app.state.predictor = None
        app.state.model_metadata = {}

    # Start background scheduler
    from scheduler.retrain_job import build_scheduler
    scheduler = build_scheduler(app)
    scheduler.start()
    app.state.scheduler = scheduler
    logger.info("APScheduler started.")

    yield  # ---- APPLICATION RUNNING ----

    # ---- SHUTDOWN ----
    logger.info("Shutting down Bug Prioritization Engine...")
    if hasattr(app.state, "scheduler") and app.state.scheduler.running:
        app.state.scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title="Bug Prioritization Engine",
        description=(
            "ML-powered REST API that predicts bug severity and business impact "
            "from Jira data, ranks open issues, and explains predictions with SHAP."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production: list specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Global exception handler ----
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a structured JSON error for any unhandled exception.

        Args:
            request: The incoming request.
            exc: The unhandled exception.

        Returns:
            JSONResponse with error details and HTTP 500 status.
        """
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "detail": str(exc),
                "status_code": 500,
            },
        )

    # ---- Routers ----
    app.include_router(router)

    return app


# Singleton app instance
app = create_app()


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        log_config=None,
    )
