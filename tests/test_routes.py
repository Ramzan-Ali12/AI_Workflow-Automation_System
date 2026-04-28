"""
tests/test_routes.py
─────────────────────
Integration tests for FastAPI endpoints.

Uses FastAPI's ``TestClient`` and dependency overrides to avoid hitting
the external LLM API.  The ``WorkflowOrchestrator`` is replaced with a
``FakeOrchestrator`` that returns pre-built responses synchronously.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.api.routes import get_orchestrator
from app.models.ticket import (
    PipelineLogEntry,
    PlanTier,
    PriorityLevel,
    ProcessedTicket,
    SentimentLabel,
    TicketAnalysis,
    TicketCategory,
    TicketClassification,
    TicketMetadata,
    TicketPriority,
    TicketRequest,
    TicketResponse,
    WorkflowResult,
    WorkflowStatus,
    LogLevel,
)


# ── Fake orchestrator ──────────────────────────────────────────────────────────

class FakeOrchestrator:
    """Returns a successful ``ProcessedTicket`` without calling the AI."""

    def run(self, ticket: TicketRequest) -> ProcessedTicket:
        result = WorkflowResult(
            classification=TicketClassification(
                category=TicketCategory.technical,
                subcategory="api",
                sentiment=SentimentLabel.neutral,
                confidence=0.9,
            ),
            priority=TicketPriority(
                level=PriorityLevel.medium,
                urgency_score=5,
                impact_score=5,
                business_impact="Moderate impact on a subset of users.",
                estimated_sla_hours=8,
            ),
            analysis=TicketAnalysis(
                root_cause_hypothesis="Likely a configuration issue.",
                affected_systems=["api"],
                similar_issue_pattern="misconfiguration",
                requires_escalation=False,
                escalation_reason=None,
            ),
            response=TicketResponse(
                draft="Thank you for contacting us. We will look into this shortly.",
                action_items=["Review configuration"],
                internal_notes="Standard priority — assign to tier-1.",
                suggested_kb_articles=["api-config-guide"],
            ),
            metadata=TicketMetadata(
                ticket_id="TKT-FAKE1",
                processing_timestamp=datetime.utcnow(),
                word_count=10,
                operator_tier=ticket.plan,
                model_used="gpt-4o-mini",
            ),
        )
        log_entries = [
            PipelineLogEntry(level=LogLevel.INFO, stage="INTAKE", message="Ticket received"),
            PipelineLogEntry(level=LogLevel.SUCCESS, stage="COMPLETE", message="Done"),
        ]
        return ProcessedTicket(
            ticket_id="TKT-FAKE1",
            status=WorkflowStatus.complete,
            processing_time_ms=42,
            pipeline_log=log_entries,
            result=result,
        )


class FakeErrorOrchestrator:
    """Returns a failed ``ProcessedTicket`` simulating an AI error."""

    def run(self, ticket: TicketRequest) -> ProcessedTicket:
        return ProcessedTicket(
            ticket_id="TKT-ERR01",
            status=WorkflowStatus.error,
            processing_time_ms=10,
            pipeline_log=[],
            result=None,
            error="Simulated AI connection error.",
        )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with the fake (success) orchestrator injected."""
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: FakeOrchestrator()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def error_client():
    """TestClient with the fake (error) orchestrator injected."""
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: FakeErrorOrchestrator()
    return TestClient(app, raise_server_exceptions=False)


VALID_PAYLOAD = {
    "company": "Test Corp",
    "email": "test@testcorp.com",
    "subject": "Something is broken in our system",
    "description": "This is a detailed description of the issue that is long enough to pass validation.",
    "plan": "business",
}


# ── Health check ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"

    def test_health_includes_model(self, client):
        data = client.get("/api/v1/health").json()
        assert "model" in data


# ── Process endpoint (success path) ───────────────────────────────────────────

class TestProcessTicketSuccess:

    def test_returns_200(self, client):
        resp = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_ticket_id(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert "ticket_id" in data
        assert data["ticket_id"].startswith("TKT-")

    def test_response_status_is_complete(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["status"] == "complete"

    def test_result_is_present(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["result"] is not None

    def test_classification_block_present(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert "classification" in data["result"]
        assert "category" in data["result"]["classification"]

    def test_priority_block_present(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert "priority" in data["result"]
        assert "level" in data["result"]["priority"]

    def test_pipeline_log_is_list(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert isinstance(data["pipeline_log"], list)
        assert len(data["pipeline_log"]) > 0

    def test_processing_time_is_non_negative(self, client):
        data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["processing_time_ms"] >= 0


# ── Process endpoint (error path) ─────────────────────────────────────────────

class TestProcessTicketError:

    def test_ai_error_still_returns_200(self, error_client):
        """Workflow errors are returned in the body, not as HTTP errors."""
        resp = error_client.post("/api/v1/tickets/process", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_ai_error_status_is_error(self, error_client):
        data = error_client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["status"] == "error"

    def test_ai_error_result_is_null(self, error_client):
        data = error_client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["result"] is None

    def test_ai_error_field_is_populated(self, error_client):
        data = error_client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        assert data["error"] is not None


# ── Validation errors ──────────────────────────────────────────────────────────

class TestRequestValidation:

    def test_missing_required_field_returns_422(self, client):
        payload = {**VALID_PAYLOAD}
        del payload["email"]
        resp = client.post("/api/v1/tickets/process", json=payload)
        assert resp.status_code == 422

    def test_invalid_email_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "email": "not-an-email"}
        resp = client.post("/api/v1/tickets/process", json=payload)
        assert resp.status_code == 422

    def test_invalid_plan_returns_422(self, client):
        payload = {**VALID_PAYLOAD, "plan": "diamond"}
        resp = client.post("/api/v1/tickets/process", json=payload)
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/api/v1/tickets/process", json={})
        assert resp.status_code == 422


# ── Retrieve endpoint ──────────────────────────────────────────────────────────

class TestGetTicket:

    def test_processed_ticket_is_retrievable(self, client):
        # First process a ticket to store it
        post_data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        ticket_id = post_data["ticket_id"]

        resp = client.get(f"/api/v1/tickets/{ticket_id}")
        assert resp.status_code == 200

    def test_unknown_ticket_returns_404(self, client):
        resp = client.get("/api/v1/tickets/TKT-XXXXX")
        assert resp.status_code == 404

    def test_ticket_id_is_case_insensitive(self, client):
        post_data = client.post("/api/v1/tickets/process", json=VALID_PAYLOAD).json()
        ticket_id = post_data["ticket_id"].lower()   # lowercase version

        resp = client.get(f"/api/v1/tickets/{ticket_id}")
        assert resp.status_code == 200
