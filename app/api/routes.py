"""
app/api/routes.py
──────────────────
FastAPI router with all ticket-processing endpoints.

Endpoints
---------
POST   /api/v1/tickets/process       Process a ticket through the AI pipeline
GET    /api/v1/tickets/{ticket_id}   Retrieve a processed ticket by ID
GET    /api/v1/health                System health check

The router uses dependency injection for the ``WorkflowOrchestrator`` so
routes are independently testable without touching the LLM provider API.
"""

import time
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.models.ticket import ProcessedTicket, TicketRequest, WorkflowStatus
from app.services.workflow import WorkflowOrchestrator
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# ── In-memory ticket store (dev/demo only) ─────────────────────────────────────
# In production, replace with a database (PostgreSQL, DynamoDB, etc.)
_ticket_store: dict[str, ProcessedTicket] = {}


# ── Dependencies ───────────────────────────────────────────────────────────────

def get_orchestrator() -> WorkflowOrchestrator:
    """
    FastAPI dependency that provides a ``WorkflowOrchestrator`` instance.

    Override this in tests:
        app.dependency_overrides[get_orchestrator] = lambda: MockOrchestrator()
    """
    return WorkflowOrchestrator()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/tickets/process",
    response_model=ProcessedTicket,
    status_code=status.HTTP_200_OK,
    summary="Process a support ticket",
    description=(
        "Submit a support ticket and run it through the full AI workflow pipeline. "
        "Returns a structured result including classification, priority, root-cause "
        "analysis, a customer response draft, and a complete audit log."
    ),
    responses={
        200: {"description": "Ticket processed successfully (may still contain errors in body)."},
        422: {"description": "Request body failed Pydantic validation."},
        500: {"description": "Unexpected internal server error."},
    },
    tags=["Tickets"],
)
async def process_ticket(
    ticket: TicketRequest,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)],
    request: Request,
) -> ProcessedTicket:
    """
    Process a support ticket through the AI workflow engine.

    The endpoint **always returns HTTP 200** — workflow errors are reported
    inside the response body (``status="error"``, ``error="..."``), not as
    HTTP error codes.  This keeps retry logic simple for callers and ensures
    the full audit log is always returned.

    **Pipeline stages executed:**

    1. **INTAKE** — parse and validate ticket fields
    2. **CLASSIFY** — AI: category, subcategory, sentiment
    3. **PRIORITIZE** — AI: urgency/impact scores, SLA target
    4. **ANALYZE** — AI: root cause, affected systems, escalation flag
    5. **RESPOND** — AI: customer draft, action items, internal notes
    6. **COMPLETE** — package result, compute timing
    """
    client_ip = request.client.host if request.client else "unknown"
    log.info(
        "Ticket process request — company=%s plan=%s ip=%s",
        ticket.company,
        ticket.plan.value,
        client_ip,
    )

    processed = orchestrator.run(ticket)

    # Persist to in-memory store
    _ticket_store[processed.ticket_id] = processed

    log.info(
        "Ticket process response — ticket_id=%s status=%s time_ms=%d",
        processed.ticket_id,
        processed.status.value,
        processed.processing_time_ms,
    )
    return processed


@router.get(
    "/tickets/{ticket_id}",
    response_model=ProcessedTicket,
    summary="Retrieve a processed ticket",
    description="Fetch a previously processed ticket by its ID (in-memory store — dev only).",
    responses={
        200: {"description": "Ticket found."},
        404: {"description": "Ticket not found."},
    },
    tags=["Tickets"],
)
async def get_ticket(ticket_id: str) -> ProcessedTicket:
    """
    Return a previously processed ticket by ``ticket_id``.

    Ticket IDs are returned in the ``process`` response body.
    Example: ``TKT-A3F9C``
    """
    ticket_id = ticket_id.upper()
    if ticket_id not in _ticket_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{ticket_id}' not found. "
                   "Tickets are stored in-memory and lost on server restart.",
        )
    return _ticket_store[ticket_id]


@router.get(
    "/health",
    summary="Health check",
    description="Returns the application health status and configuration summary.",
    tags=["System"],
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """
    Lightweight health-check endpoint for load balancers and monitoring.

    Returns configuration metadata so it is easy to verify which model
    and environment are active without checking server logs.
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env,
        "model": settings.active_model,
        "tickets_in_store": len(_ticket_store),
    }
