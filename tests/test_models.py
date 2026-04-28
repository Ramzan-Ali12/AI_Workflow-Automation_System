"""
tests/test_models.py
─────────────────────
Unit tests for Pydantic models.

These tests verify field validation, enum coercion, and derived
properties without touching the API or the LLM SDK.
"""

import pytest
from pydantic import ValidationError

from app.models.ticket import (
    PlanTier,
    PriorityLevel,
    SentimentLabel,
    TicketClassification,
    TicketPriority,
    TicketRequest,
)


# ── TicketRequest ──────────────────────────────────────────────────────────────

class TestTicketRequest:

    def test_valid_ticket(self):
        ticket = TicketRequest(
            company="Acme Corp",
            email="user@acme.com",
            subject="Something is broken",
            description="A longer description that explains the issue in detail.",
            plan=PlanTier.enterprise,
        )
        assert ticket.company == "Acme Corp"
        assert ticket.plan == PlanTier.enterprise

    def test_whitespace_is_stripped(self):
        ticket = TicketRequest(
            company="  Acme Corp  ",
            email="user@acme.com",
            subject="  Subject with spaces  ",
            description="  Description with leading/trailing whitespace.  ",
        )
        assert ticket.company == "Acme Corp"
        assert ticket.subject == "Subject with spaces"
        assert ticket.description == "Description with leading/trailing whitespace."

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError, match="email"):
            TicketRequest(
                company="Acme",
                email="not-an-email",
                subject="Valid subject",
                description="Valid description with enough characters.",
            )

    def test_short_subject_raises(self):
        with pytest.raises(ValidationError, match="subject"):
            TicketRequest(
                company="Acme",
                email="user@acme.com",
                subject="Hi",           # too short (min_length=5)
                description="Valid description with enough characters.",
            )

    def test_short_description_raises(self):
        with pytest.raises(ValidationError, match="description"):
            TicketRequest(
                company="Acme",
                email="user@acme.com",
                subject="Valid subject",
                description="Too short",  # min_length=20
            )

    def test_default_plan_is_free(self):
        ticket = TicketRequest(
            company="Acme",
            email="user@acme.com",
            subject="Valid subject",
            description="A description long enough to pass validation.",
        )
        assert ticket.plan == PlanTier.free

    def test_plan_enum_coercion(self):
        ticket = TicketRequest(
            company="Acme",
            email="user@acme.com",
            subject="Valid subject",
            description="A description long enough to pass validation.",
            plan="business",           # string → enum
        )
        assert ticket.plan == PlanTier.business

    def test_invalid_plan_raises(self):
        with pytest.raises(ValidationError, match="plan"):
            TicketRequest(
                company="Acme",
                email="user@acme.com",
                subject="Valid subject",
                description="A description long enough to pass validation.",
                plan="diamond",        # not a valid tier
            )


# ── TicketClassification ───────────────────────────────────────────────────────

class TestTicketClassification:

    def test_valid_classification(self):
        clf = TicketClassification(
            category="technical",
            subcategory="payment_gateway",
            sentiment="frustrated",
            confidence=0.95,
        )
        assert clf.confidence == 0.95
        assert clf.sentiment == SentimentLabel.frustrated

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError, match="confidence"):
            TicketClassification(
                category="technical",
                subcategory="api",
                sentiment="neutral",
                confidence=1.5,        # > 1.0
            )


# ── TicketPriority ─────────────────────────────────────────────────────────────

class TestTicketPriority:

    def test_valid_priority(self):
        pri = TicketPriority(
            level="critical",
            urgency_score=9,
            impact_score=10,
            business_impact="Active revenue loss exceeding $10k/hour.",
            estimated_sla_hours=1,
        )
        assert pri.level == PriorityLevel.critical

    def test_urgency_score_bounds(self):
        with pytest.raises(ValidationError, match="urgency_score"):
            TicketPriority(
                level="high",
                urgency_score=11,      # max is 10
                impact_score=5,
                business_impact="Some impact.",
                estimated_sla_hours=4,
            )

    def test_negative_sla_raises(self):
        with pytest.raises(ValidationError, match="estimated_sla_hours"):
            TicketPriority(
                level="low",
                urgency_score=2,
                impact_score=2,
                business_impact="Minimal impact.",
                estimated_sla_hours=-1,
            )
