"""
app/models/ticket.py
─────────────────────
All Pydantic models used across the workflow pipeline.

Hierarchy:
  TicketRequest          ← inbound HTTP body
  ├── TicketClassification
  ├── TicketPriority
  ├── TicketAnalysis
  ├── TicketResponse
  └── TicketMetadata
  WorkflowResult         ← AI output (parsed from LLM JSON)
  PipelineLogEntry       ← one log line from the pipeline
  ProcessedTicket        ← final outbound HTTP response
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class PlanTier(str, Enum):
    free       = "free"
    starter    = "starter"
    business   = "business"
    enterprise = "enterprise"


class TicketCategory(str, Enum):
    technical       = "technical"
    billing         = "billing"
    account         = "account"
    feature_request = "feature_request"
    general         = "general"


class SentimentLabel(str, Enum):
    positive   = "positive"
    neutral    = "neutral"
    negative   = "negative"
    frustrated = "frustrated"


class PriorityLevel(str, Enum):
    critical = "critical"
    high     = "high"
    medium   = "medium"
    low      = "low"


class LogLevel(str, Enum):
    INFO    = "INFO"
    WARN    = "WARN"
    ERROR   = "ERROR"
    SUCCESS = "SUCCESS"
    SYS     = "SYS"


class WorkflowStatus(str, Enum):
    complete = "complete"
    error    = "error"


# ── Request Model ──────────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    """
    Inbound support ticket submitted by an operator or end-user.
    All string fields are stripped of surrounding whitespace automatically.
    """

    company: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Company or organisation name.",
        examples=["Acme Corp"],
    )
    email: EmailStr = Field(
        ...,
        description="Contact email address for the submitter.",
        examples=["sarah@acmecorp.com"],
    )
    subject: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="One-line summary of the issue.",
        examples=["Payment gateway returning 500 errors on checkout"],
    )
    description: str = Field(
        ...,
        min_length=20,
        max_length=5000,
        description="Full description of the issue, including any relevant context.",
        examples=["Since 9AM UTC our checkout has been failing..."],
    )
    plan: PlanTier = Field(
        default=PlanTier.free,
        description="Operator subscription tier.",
        examples=["enterprise"],
    )

    @field_validator("company", "subject", "description", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ── AI-Generated Sub-Models ────────────────────────────────────────────────────

class TicketClassification(BaseModel):
    """Output of the CLASSIFY pipeline stage."""

    category: TicketCategory = Field(
        ..., description="Top-level ticket category."
    )
    subcategory: str = Field(
        ..., description="Specific area within the category (e.g. 'payment_gateway')."
    )
    sentiment: SentimentLabel = Field(
        ..., description="Detected customer sentiment."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score (0–1)."
    )


class TicketPriority(BaseModel):
    """Output of the PRIORITIZE pipeline stage."""

    level: PriorityLevel = Field(..., description="Overall priority level.")
    urgency_score: int = Field(..., ge=1, le=10, description="Urgency score (1 = lowest, 10 = highest).")
    impact_score: int = Field(..., ge=1, le=10, description="Business impact score (1–10).")
    business_impact: str = Field(..., description="One-sentence business impact statement.")
    estimated_sla_hours: float = Field(
        ..., gt=0, description="Target resolution time in hours based on plan + priority."
    )


class TicketAnalysis(BaseModel):
    """Output of the ANALYZE pipeline stage."""

    root_cause_hypothesis: str = Field(
        ..., description="2–3 sentence hypothesis about the likely root cause."
    )
    affected_systems: list[str] = Field(
        default_factory=list,
        description="List of systems or services believed to be affected.",
    )
    similar_issue_pattern: str = Field(
        ..., description="Pattern label (e.g. 'upstream_timeout', 'misconfiguration')."
    )
    requires_escalation: bool = Field(
        ..., description="Whether this ticket should be escalated to a senior team."
    )
    escalation_reason: Optional[str] = Field(
        default=None, description="Reason for escalation, if applicable."
    )


class TicketResponse(BaseModel):
    """Output of the RESPOND pipeline stage."""

    draft: str = Field(
        ..., description="Professional, empathetic customer-facing response draft."
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Ordered list of concrete next actions for the support agent.",
    )
    internal_notes: str = Field(
        ..., description="Private notes for the support team (not sent to customer)."
    )
    suggested_kb_articles: list[str] = Field(
        default_factory=list,
        description="Suggested knowledge-base article topics to attach.",
    )


class TicketMetadata(BaseModel):
    """Metadata generated during pipeline execution."""

    ticket_id: str = Field(..., description="Unique ticket identifier.")
    processing_timestamp: datetime = Field(..., description="UTC timestamp of processing.")
    word_count: int = Field(..., description="Word count of the original description.")
    operator_tier: PlanTier = Field(..., description="Submitting operator's plan tier.")
    model_used: str = Field(..., description="LLM model that processed this ticket.")


# ── Composite Models ───────────────────────────────────────────────────────────

class WorkflowResult(BaseModel):
    """
    Complete structured output from the AI processing pipeline.
    This is what the LLM is asked to produce (as JSON) and what is
    stored + returned to the caller.
    """

    classification: TicketClassification
    priority: TicketPriority
    analysis: TicketAnalysis
    response: TicketResponse
    metadata: TicketMetadata


class PipelineLogEntry(BaseModel):
    """A single timestamped log entry from the workflow pipeline."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: LogLevel
    stage: str = Field(..., description="Pipeline stage that emitted this entry.")
    message: str

    def __str__(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        return f"[{ts}] [{self.level.value:7s}] [{self.stage:10s}] {self.message}"


class ProcessedTicket(BaseModel):
    """
    Final HTTP response returned by POST /api/v1/tickets/process.
    Includes the AI result, the full audit log, and timing information.
    """

    ticket_id: str = Field(
        default_factory=lambda: f"TKT-{uuid4().hex[:5].upper()}",
        description="Unique identifier for this processed ticket.",
    )
    status: WorkflowStatus
    processing_time_ms: int = Field(
        ..., description="Total wall-clock time for the pipeline in milliseconds."
    )
    pipeline_log: list[PipelineLogEntry] = Field(
        default_factory=list,
        description="Full ordered audit trail of pipeline execution.",
    )
    result: Optional[WorkflowResult] = Field(
        default=None,
        description="Structured AI output. Null if the workflow failed.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'.",
    )
