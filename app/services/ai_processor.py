"""
app/services/ai_processor.py
──────────────────────────────
Thin, testable wrapper around the OpenAI Python SDK.

Responsibilities
----------------
- Build the structured prompt for ticket analysis
- Call the OpenAI API with retry-safe error handling
- Parse and validate the raw JSON response into a ``WorkflowResult``
- Raise typed exceptions so the caller can log and respond cleanly

Design notes
------------
All four AI stages (classify, prioritize, analyze, respond) are handled
in a **single API call**. This minimises latency, cost, and rate-limit
exposure while keeping the prompt auditable in one place.
"""

import json
from datetime import datetime

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config import get_settings
from app.models.ticket import (
    TicketAnalysis,
    TicketClassification,
    TicketMetadata,
    TicketPriority,
    TicketRequest,
    TicketResponse,
    WorkflowResult,
)
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Custom Exceptions ──────────────────────────────────────────────────────────

class AIProcessorError(Exception):
    """Base exception for all AI processor failures."""


class AIConnectionError(AIProcessorError):
    """Raised when the OpenAI API cannot be reached."""


class AIResponseParseError(AIProcessorError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


class AIRateLimitError(AIProcessorError):
    """Raised when the API returns a 429 rate-limit error."""


# ── Prompt Builder ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an AI-powered support ticket processing engine.
Your job is to analyse an inbound support ticket and return a single,
complete JSON object — nothing else. No markdown fences, no explanation,
no preamble.

The JSON must match this exact schema:

{
  "classification": {
    "category": "<technical|billing|account|feature_request|general>",
    "subcategory": "<specific area, e.g. payment_gateway, api_rate_limits>",
    "sentiment": "<positive|neutral|negative|frustrated>",
    "confidence": <float 0.0–1.0>
  },
  "priority": {
    "level": "<critical|high|medium|low>",
    "urgency_score": <int 1–10>,
    "impact_score": <int 1–10>,
    "business_impact": "<one sentence>",
    "estimated_sla_hours": <number>
  },
  "analysis": {
    "root_cause_hypothesis": "<2–3 sentences>",
    "affected_systems": ["<system_name>", "..."],
    "similar_issue_pattern": "<e.g. upstream_timeout, misconfiguration>",
    "requires_escalation": <true|false>,
    "escalation_reason": "<string or null>"
  },
  "response": {
    "draft": "<professional, empathetic 3–5 sentence customer reply>",
    "action_items": ["<action 1>", "<action 2>", "..."],
    "internal_notes": "<1–2 sentences for the support team>",
    "suggested_kb_articles": ["<topic 1>", "..."]
  }
}

Rules:
- Return ONLY the JSON object. No other text.
- Calibrate SLA hours to plan tier: free ≥ 72h, starter ≥ 24h,
  business ≥ 8h, enterprise ≥ 1h — then shorten based on priority.
- Set requires_escalation=true only for critical/high issues with
  significant revenue impact or data-security concerns.
"""


def _build_user_message(ticket: TicketRequest) -> str:
    """Format the ticket fields into a structured user message."""
    return (
        f"Company: {ticket.company}\n"
        f"Email: {ticket.email}\n"
        f"Plan: {ticket.plan.value}\n"
        f"Subject: {ticket.subject}\n"
        f"Description:\n{ticket.description}"
    )


# ── Processor Class ────────────────────────────────────────────────────────────

class AIProcessor:
    """
    Wraps the OpenAI API and converts a raw ticket into a
    fully-validated ``WorkflowResult``.

    Parameters
    ----------
    api_key:
        OpenAI API key. Defaults to the value in Settings.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.llm_api_key)
        self._model      = settings.active_model
        self._max_tokens = settings.max_tokens
        log.debug("AIProcessor initialised — model=%s", self._model)

    # ── Public API ─────────────────────────────────────────────────────────

    def process(self, ticket: TicketRequest, ticket_id: str) -> WorkflowResult:
        """
        Send the ticket to OpenAI and return a validated ``WorkflowResult``.

        Parameters
        ----------
        ticket:
            The validated inbound ticket.
        ticket_id:
            Pre-generated ticket identifier to embed in metadata.

        Returns
        -------
        WorkflowResult
            Fully-validated structured output.

        Raises
        ------
        AIConnectionError
            Network or timeout issues reaching the OpenAI API.
        AIRateLimitError
            The API returned HTTP 429.
        AIResponseParseError
            The model response could not be parsed into the expected schema.
        AIProcessorError
            Any other OpenAI API error.
        """
        log.info("Calling OpenAI API — ticket_id=%s model=%s", ticket_id, self._model)

        raw_text = self._call_api(ticket)
        log.debug("Raw LLM response (%d chars): %s", len(raw_text), raw_text[:200])

        payload = self._parse_json(raw_text, ticket_id)
        result  = self._build_result(payload, ticket, ticket_id)

        log.info(
            "AI processing complete — ticket_id=%s priority=%s category=%s",
            ticket_id,
            result.priority.level.value,
            result.classification.category.value,
        )
        return result

    # ── Private Helpers ────────────────────────────────────────────────────

    def _call_api(self, ticket: TicketRequest) -> str:
        """
        Execute the OpenAI Chat Completions API call and return the raw text response.
        All SDK exceptions are translated into typed ``AIProcessorError`` subclasses.
        """
        try:
            message = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(ticket)},
                ],
                response_format={"type": "json_object"},
            )
        except APITimeoutError as exc:
            raise AIConnectionError(f"OpenAI API timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise AIConnectionError(f"Cannot reach OpenAI API: {exc}") from exc
        except RateLimitError as exc:
            raise AIRateLimitError("OpenAI rate limit reached. Retry after a moment.") from exc
        except APIStatusError as exc:
            raise AIProcessorError(
                f"OpenAI API error {exc.status_code}: {exc.message}"
            ) from exc

        # Extract text from the first completion choice
        if not message.choices:
            raise AIProcessorError("OpenAI returned no completion choices.")
        content = message.choices[0].message.content
        if not content:
            raise AIProcessorError("OpenAI returned an empty response message.")

        return content

    def _parse_json(self, raw: str, ticket_id: str) -> dict:
        """
        Strip any accidental markdown fences and parse JSON.
        Raises ``AIResponseParseError`` with a descriptive message on failure.
        """
        # Remove markdown code fences if the model wrapped the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AIResponseParseError(
                f"Could not parse LLM response as JSON for ticket {ticket_id}: {exc}. "
                f"Raw snippet: {raw[:300]}"
            ) from exc

    def _build_result(
        self, payload: dict, ticket: TicketRequest, ticket_id: str
    ) -> WorkflowResult:
        """
        Validate the parsed JSON dict against Pydantic models and assemble
        the final ``WorkflowResult``, injecting server-side metadata.

        Pydantic will raise ``ValidationError`` if the LLM omitted a required
        field or returned an invalid type — that propagates as-is so the
        workflow layer can log it cleanly.
        """
        settings = get_settings()

        return WorkflowResult(
            classification=TicketClassification(**payload["classification"]),
            priority=TicketPriority(**payload["priority"]),
            analysis=TicketAnalysis(**payload["analysis"]),
            response=TicketResponse(**payload["response"]),
            metadata=TicketMetadata(
                ticket_id=ticket_id,
                processing_timestamp=datetime.utcnow(),
                word_count=len(ticket.description.split()),
                operator_tier=ticket.plan,
                model_used=settings.active_model,
            ),
        )
