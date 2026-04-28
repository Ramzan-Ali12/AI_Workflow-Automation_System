"""
app/services/workflow.py
─────────────────────────
Pipeline orchestrator that runs each workflow stage in order,
captures a structured audit log, measures wall-clock time, and
returns a ``ProcessedTicket`` regardless of whether the run succeeded
or failed.

Pipeline stages
---------------
1. INTAKE      — validate & normalise the inbound ticket
2. CLASSIFY    — AI: category, subcategory, sentiment, confidence
3. PRIORITIZE  — AI: urgency/impact scores, SLA target
4. ANALYZE     — AI: root cause, affected systems, escalation flag
5. RESPOND     — AI: customer draft, action items, internal notes
6. COMPLETE    — package structured output, compute timing

All AI stages are performed in a **single API call** inside
``AIProcessor.process()``. The stages above are logical — each
corresponds to a section of the AI response that the orchestrator
validates and logs separately.

Usage
-----
    from app.services.workflow import WorkflowOrchestrator
    from app.models.ticket import TicketRequest

    orchestrator = WorkflowOrchestrator()
    ticket = TicketRequest(...)
    result = orchestrator.run(ticket)
    print(result.model_dump_json(indent=2))
"""

import time
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from app.models.ticket import (
    LogLevel,
    PipelineLogEntry,
    ProcessedTicket,
    TicketRequest,
    WorkflowResult,
    WorkflowStatus,
)
from app.services.ai_processor import (
    AIConnectionError,
    AIProcessorError,
    AIRateLimitError,
    AIResponseParseError,
    AIProcessor,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


class WorkflowOrchestrator:
    """
    Runs a ticket through the full six-stage AI processing pipeline.

    The orchestrator owns the audit log, timing, and error wrapping so
    that individual services stay focused on their single responsibility.

    Parameters
    ----------
    processor:
        Injectable ``AIProcessor`` instance. Defaults to a new one built
        from settings — override in tests with a mock.
    """

    def __init__(self, processor: AIProcessor | None = None) -> None:
        self._processor = processor or AIProcessor()

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, ticket: TicketRequest) -> ProcessedTicket:
        """
        Execute the full pipeline for *ticket* and return a ``ProcessedTicket``.

        This method **never raises** — errors are captured inside the
        returned object so the HTTP layer always receives a well-formed
        response.

        Parameters
        ----------
        ticket:
            Validated inbound ticket from the API route.

        Returns
        -------
        ProcessedTicket
            status="complete" on success, status="error" on any failure.
        """
        ticket_id   = f"TKT-{uuid4().hex[:5].upper()}"
        audit_log: list[PipelineLogEntry] = []
        started_at  = time.monotonic()

        def emit(level: LogLevel, stage: str, msg: str) -> None:
            """Append a log entry and echo it to the application logger."""
            entry = PipelineLogEntry(level=level, stage=stage, message=msg)
            audit_log.append(entry)
            # Mirror to the standard logger so ops can grep server logs
            log_fn = {
                LogLevel.INFO:    log.info,
                LogLevel.WARN:    log.warning,
                LogLevel.ERROR:   log.error,
                LogLevel.SUCCESS: log.info,
                LogLevel.SYS:     log.debug,
            }.get(level, log.info)
            log_fn("[%s] %s — %s", ticket_id, stage, msg)

        # ── STAGE 1: INTAKE ────────────────────────────────────────────────
        emit(LogLevel.SYS,  "INTAKE", f"Workflow started — ticket_id={ticket_id}")
        emit(LogLevel.INFO, "INTAKE", f"Operator: {ticket.company} | Plan: {ticket.plan.value}")
        emit(LogLevel.INFO, "INTAKE", f"Subject: {ticket.subject!r}")
        emit(LogLevel.INFO, "INTAKE", f"Description length: {len(ticket.description)} chars, "
                                       f"{len(ticket.description.split())} words")
        emit(LogLevel.INFO, "INTAKE", "Validation passed — all required fields present")

        try:
            # ── STAGE 2: CLASSIFY (+ 3, 4, 5 via single API call) ─────────
            emit(LogLevel.INFO, "CLASSIFY", "Invoking LLM pipeline (classify + prioritize + analyze + respond)…")

            result: WorkflowResult = self._processor.process(ticket, ticket_id)

            # Log classification output
            clf = result.classification
            emit(LogLevel.SUCCESS, "CLASSIFY",
                 f"Category={clf.category.value} | Subcategory={clf.subcategory} | "
                 f"Sentiment={clf.sentiment.value} | Confidence={clf.confidence:.0%}")

            # ── STAGE 3: PRIORITIZE ────────────────────────────────────────
            pri = result.priority
            emit(LogLevel.INFO, "PRIORITIZE",
                 f"Priority={pri.level.value.upper()} | Urgency={pri.urgency_score}/10 | "
                 f"Impact={pri.impact_score}/10 | SLA={pri.estimated_sla_hours}h")
            if pri.level.value in ("critical", "high"):
                emit(LogLevel.WARN, "PRIORITIZE",
                     f"High-priority ticket — SLA target: {pri.estimated_sla_hours}h")

            # ── STAGE 4: ANALYZE ───────────────────────────────────────────
            ana = result.analysis
            emit(LogLevel.INFO, "ANALYZE",
                 f"Pattern: {ana.similar_issue_pattern} | "
                 f"Systems: {', '.join(ana.affected_systems) or 'n/a'}")
            if ana.requires_escalation:
                emit(LogLevel.WARN, "ANALYZE",
                     f"ESCALATION REQUIRED — {ana.escalation_reason}")
            else:
                emit(LogLevel.INFO, "ANALYZE", "No escalation required")

            # ── STAGE 5: RESPOND ───────────────────────────────────────────
            resp = result.response
            emit(LogLevel.INFO, "RESPOND",
                 f"Response draft ready ({len(resp.draft)} chars) | "
                 f"Action items: {len(resp.action_items)} | "
                 f"KB suggestions: {len(resp.suggested_kb_articles)}")

            # ── STAGE 6: COMPLETE ──────────────────────────────────────────
            elapsed_ms = int((time.monotonic() - started_at) * 1000)
            emit(LogLevel.SUCCESS, "COMPLETE",
                 f"Ticket {ticket_id} fully processed in {elapsed_ms}ms")
            emit(LogLevel.SYS, "COMPLETE", "Workflow engine shutting down cleanly")

            return ProcessedTicket(
                ticket_id=ticket_id,
                status=WorkflowStatus.complete,
                processing_time_ms=elapsed_ms,
                pipeline_log=audit_log,
                result=result,
            )

        # ── Error handling — each type gets a distinct log message ─────────
        except AIRateLimitError as exc:
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "PRIORITIZE", emit,
                              "Rate limit hit — retry after a short delay")

        except AIConnectionError as exc:
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "CLASSIFY", emit,
                              "Cannot reach LLM API — check connectivity")

        except AIResponseParseError as exc:
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "CLASSIFY", emit,
                              "LLM returned malformed JSON — manual review required")

        except ValidationError as exc:
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "COMPLETE", emit,
                              f"Schema validation failed: {exc.error_count()} error(s)")

        except AIProcessorError as exc:
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "CLASSIFY", emit, "Unexpected AI processor error")

        except Exception as exc:  # noqa: BLE001 — catch-all for unexpected runtime errors
            log.exception("Unhandled exception in workflow for ticket_id=%s", ticket_id)
            return self._fail(ticket_id, audit_log, started_at, str(exc),
                              "COMPLETE", emit, "Unhandled internal error — see server logs")

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fail(
        ticket_id: str,
        audit_log: list[PipelineLogEntry],
        started_at: float,
        error_msg: str,
        stage: str,
        emit,
        summary: str,
    ) -> ProcessedTicket:
        """Build an error ``ProcessedTicket`` and append final log entries."""
        emit(LogLevel.ERROR, stage, summary)
        emit(LogLevel.SYS,   stage, "Workflow halted — manual review required")
        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        return ProcessedTicket(
            ticket_id=ticket_id,
            status=WorkflowStatus.error,
            processing_time_ms=elapsed_ms,
            pipeline_log=audit_log,
            result=None,
            error=error_msg,
        )
