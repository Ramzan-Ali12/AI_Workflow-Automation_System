"""
tests/test_workflow.py
───────────────────────
Unit tests for ``WorkflowOrchestrator``.

The external LLM API is never called in these tests — ``AIProcessor`` is
replaced with lightweight fakes via dependency injection.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.models.ticket import (
    LogLevel,
    PipelineLogEntry,
    PlanTier,
    PriorityLevel,
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
)
from app.services.ai_processor import AIConnectionError, AIResponseParseError
from app.services.workflow import WorkflowOrchestrator


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ticket() -> TicketRequest:
    return TicketRequest(
        company="Test Corp",
        email="test@testcorp.com",
        subject="Test issue description",
        description="This is a long enough description to pass validation checks in our system.",
        plan=PlanTier.enterprise,
    )


def _make_workflow_result(ticket_id: str = "TKT-TEST1") -> WorkflowResult:
    """Build a realistic ``WorkflowResult`` for use in test fakes."""
    return WorkflowResult(
        classification=TicketClassification(
            category=TicketCategory.technical,
            subcategory="api_integration",
            sentiment=SentimentLabel.frustrated,
            confidence=0.92,
        ),
        priority=TicketPriority(
            level=PriorityLevel.high,
            urgency_score=8,
            impact_score=7,
            business_impact="Multiple customers unable to complete transactions.",
            estimated_sla_hours=4,
        ),
        analysis=TicketAnalysis(
            root_cause_hypothesis="Likely a misconfigured timeout on the upstream gateway.",
            affected_systems=["payment_api", "checkout_service"],
            similar_issue_pattern="upstream_timeout",
            requires_escalation=True,
            escalation_reason="Revenue impact — escalate to infrastructure team.",
        ),
        response=TicketResponse(
            draft="Thank you for reaching out. We have escalated this issue to our infrastructure team and will update you within 4 hours.",
            action_items=["Check upstream timeout config", "Review error logs", "Notify affected customers"],
            internal_notes="Escalate immediately to infra on-call.",
            suggested_kb_articles=["api-timeout-troubleshooting", "payment-gateway-errors"],
        ),
        metadata=TicketMetadata(
            ticket_id=ticket_id,
            processing_timestamp=datetime.utcnow(),
            word_count=12,
            operator_tier=PlanTier.enterprise,
            model_used="gpt-4o-mini",
        ),
    )


# ── Happy-path tests ───────────────────────────────────────────────────────────

class TestWorkflowOrchestratorSuccess:

    def test_returns_complete_status_on_success(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        orchestrator = WorkflowOrchestrator(processor=mock_processor)
        result = orchestrator.run(sample_ticket)

        assert result.status == WorkflowStatus.complete

    def test_result_is_populated(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.result is not None
        assert result.result.classification.category == TicketCategory.technical
        assert result.result.priority.level == PriorityLevel.high

    def test_ticket_id_is_generated(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.ticket_id.startswith("TKT-")
        assert len(result.ticket_id) == 9  # TKT- + 5 hex chars

    def test_processing_time_is_positive(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.processing_time_ms >= 0

    def test_audit_log_is_populated(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert len(result.pipeline_log) > 0
        levels = {e.level for e in result.pipeline_log}
        assert LogLevel.INFO in levels

    def test_escalation_warning_logged(self, sample_ticket):
        """Escalation flag in AI result should appear in audit log as a WARN."""
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        warn_msgs = [e.message for e in result.pipeline_log if e.level == LogLevel.WARN]
        assert any("ESCALATION" in m for m in warn_msgs)

    def test_error_field_is_none_on_success(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.return_value = _make_workflow_result()

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.error is None


# ── Error-path tests ───────────────────────────────────────────────────────────

class TestWorkflowOrchestratorErrors:

    def test_connection_error_returns_error_status(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = AIConnectionError("Cannot reach API")

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.status == WorkflowStatus.error
        assert result.result is None

    def test_connection_error_populates_error_field(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = AIConnectionError("Connection refused")

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert "Connection refused" in (result.error or "")

    def test_parse_error_returns_error_status(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = AIResponseParseError("Malformed JSON")

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.status == WorkflowStatus.error

    def test_unhandled_exception_does_not_propagate(self, sample_ticket):
        """The orchestrator must never let an exception escape to the HTTP layer."""
        mock_processor = MagicMock()
        mock_processor.process.side_effect = RuntimeError("Unexpected crash")

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        assert result.status == WorkflowStatus.error

    def test_error_audit_log_contains_error_entry(self, sample_ticket):
        mock_processor = MagicMock()
        mock_processor.process.side_effect = AIConnectionError("Timeout")

        result = WorkflowOrchestrator(processor=mock_processor).run(sample_ticket)

        error_entries = [e for e in result.pipeline_log if e.level == LogLevel.ERROR]
        assert len(error_entries) >= 1
