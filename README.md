# AI Workflow Automation System

An AI-driven system that automates end-to-end support ticket processing using OpenAI Chat Completions, FastAPI, and structured Pydantic models. Tickets are ingested, classified, prioritized, analyzed, and responded to automatically — all logged and returned as structured JSON.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Application                     │
│                                                         │
│  POST /api/v1/tickets/process                           │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │            WorkflowOrchestrator                  │   │
│  │                                                  │   │
│  │  [1] INTAKE     → Validate & parse ticket        │   │
│  │  [2] CLASSIFY   → LLM: category + sentiment      │   │
│  │  [3] PRIORITIZE → LLM: urgency + SLA scoring     │   │
│  │  [4] ANALYZE    → LLM: root cause + systems      │   │
│  │  [5] RESPOND    → LLM: draft response + actions  │   │
│  │  [6] COMPLETE   → Package structured output      │   │
│  └─────────────────────────────────────────────────┘   │
│         │                                               │
│         ▼                                               │
│  Structured JSON Response + Full Audit Log              │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai_workflow_automation/
│
├── app/
│   ├── __init__.py
│   ├── config.py              # Environment config & settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── ticket.py          # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_processor.py    # OpenAI API integration
│   │   └── workflow.py        # Pipeline orchestration logic
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI route definitions
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Structured logging setup
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py         # Pydantic model tests
│   ├── test_workflow.py       # Workflow stage tests
│   └── test_routes.py         # API endpoint tests
│
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/ai-workflow-automation.git
cd ai-workflow-automation
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and add your OPENAI_API_KEY
```

### 3. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open interactive API docs

```
http://localhost:8000/docs        ← Swagger UI
http://localhost:8000/redoc       ← ReDoc
```

---

## API Reference

### `POST /api/v1/tickets/process`

Process a support ticket through the full AI pipeline.

**Request body:**

```json
{
  "company": "Acme Corp",
  "email": "sarah@acmecorp.com",
  "subject": "Payment gateway returning 500 errors",
  "description": "Since 9AM UTC our checkout has been failing...",
  "plan": "enterprise"
}
```

**Response:**

```json
{
  "ticket_id": "TKT-A3F9C",
  "status": "complete",
  "processing_time_ms": 2341,
  "pipeline_log": [...],
  "result": {
    "classification": {
      "category": "technical",
      "subcategory": "payment_gateway",
      "sentiment": "frustrated",
      "confidence": 0.97
    },
    "priority": {
      "level": "critical",
      "urgency_score": 9,
      "impact_score": 10,
      "business_impact": "Active revenue loss...",
      "estimated_sla_hours": 1
    },
    "analysis": {
      "root_cause_hypothesis": "...",
      "affected_systems": ["payment_api", "checkout_service"],
      "requires_escalation": true,
      "escalation_reason": "Revenue impact exceeds $10k"
    },
    "response": {
      "draft": "Thank you for reaching out...",
      "action_items": ["Check upstream timeout config", "..."],
      "internal_notes": "Escalate to infra team immediately.",
      "suggested_kb_articles": ["api-timeout-troubleshooting"]
    }
  }
}
```

### `GET /api/v1/health`

Returns system health status and configuration summary.

### `GET /api/v1/tickets/{ticket_id}`

Retrieve a previously processed ticket by ID (in-memory store in dev mode).

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key (preferred) |
| `OPENAI_MODEL` | ❌ | `gpt-4o-mini` | OpenAI model to use |
| `ANTHROPIC_API_KEY` | ❌ | — | Temporary fallback key (used only if `OPENAI_API_KEY` is not set) |
| `ANTHROPIC_MODEL` | ❌ | `claude-sonnet-4-20250514` | Legacy fallback model |
| `MAX_TOKENS` | ❌ | `1500` | Max tokens per LLM call |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |
| `APP_ENV` | ❌ | `development` | `development` or `production` |
| `CORS_ORIGINS` | ❌ | `*` | Allowed CORS origins |

---

## Design Decisions

- **Single LLM call** — all AI stages (classify, prioritize, analyze, respond) use one structured prompt to minimize latency and cost
- **Pydantic everywhere** — all inputs and outputs are validated models, never raw dicts
- **Workflow as a class** — `WorkflowOrchestrator` owns the pipeline, making stages easy to test, swap, or extend
- **Structured logging** — every pipeline stage emits a timestamped log entry with severity; the full audit trail is returned in the API response
- **Fail-fast error handling** — each stage raises typed exceptions that propagate cleanly to a FastAPI exception handler

---

## License

MIT
