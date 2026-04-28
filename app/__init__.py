"""
app/__init__.py
────────────────
FastAPI application factory.

Calling ``create_app()`` wires together:
- Logging configuration
- CORS middleware
- Global exception handlers
- API router mounting

Keeping construction in a factory function (rather than a module-level
``app = FastAPI()``) makes the application easy to instantiate with
different settings in tests.
"""

import logging
import traceback

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import configure_logging, get_logger

log = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    Returns
    -------
    FastAPI
        Ready-to-serve ASGI application.
    """
    settings = get_settings()

    # Configure logging before anything else so early errors are captured
    configure_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-driven system to automate end-to-end support ticket processing. "
            "Tickets are ingested, classified, prioritized, analyzed, and responded "
            "to automatically — all logged and returned as structured JSON."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ─────────────────────────────────────────────────

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        """
        Catch Pydantic ``ValidationError`` raised outside of FastAPI's
        automatic request-body parsing (e.g. inside a service layer).
        """
        log.warning("Pydantic ValidationError on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "body": None},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for any exception not handled by a more
        specific handler.  Logs the full traceback server-side and
        returns a generic 500 so internal details are not leaked.
        """
        log.error(
            "Unhandled exception on %s %s:\n%s",
            request.method,
            request.url.path,
            traceback.format_exc(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    # ── Startup / shutdown events ──────────────────────────────────────────

    @app.on_event("startup")
    async def on_startup() -> None:
        log.info(
            "🚀 %s v%s starting in %s mode — model: %s",
            settings.app_name,
            settings.app_version,
            settings.app_env,
            settings.active_model,
        )

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        log.info("Application shutting down.")

    # ── Router ─────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app
