"""
main.py
────────
Application entry point.

Run directly:
    python main.py

Or via uvicorn (recommended for production):
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

Or with auto-reload during development:
    uvicorn main:app --reload
"""

import uvicorn
from app import create_app

# Build the ASGI application.
# ``create_app()`` configures logging, middleware, routes, and
# exception handlers — see app/__init__.py for details.
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Auto-restart on code changes (dev only)
        log_level="info",
        access_log=True,
    )
