# DB Explorer

A friendly web app for **non-technical users** to safely explore PostgreSQL databases with SQL, natural language, and an AI agent that can build downloadable Excel/Word reports.

Repo: https://github.com/hsk440/db-explorer

---

## What it does

Browse and query a PostgreSQL database through a Streamlit web UI:

| Tab | Purpose |
|---|---|
| **Schema Explorer** | Visually browse schemas, tables, columns, row counts, and preview data. |
| **SQL Query** | Write SELECT queries, see results as a table, download CSV / Excel. |
| **AI Data Assistant** | Ask questions in plain English. The AI writes SQL, runs it, and explains the results. For complex requests, it runs an agent loop with tools to build **real** Excel and Word reports with multiple sheets/sections. |
| **Quick Analytics** | Turn any query result into a Bar / Line / Scatter / Pie / Histogram chart. |
| **Activity Log** | View, filter, search, and download the app's detailed activity logs. |

### Safety first
- The PostgreSQL connection is opened **read-only** at the database level (`default_transaction_read_only=on` + `set_session(readonly=True)`). INSERT / UPDATE / DELETE / DROP are physically blocked by the database, not just the app.
- An additional client-side check blocks anything that doesn't start with `SELECT / WITH / EXPLAIN / SHOW / VALUES`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │
│  │ Schema Explorer  │  │ SQL Query        │  │ AI Assistant     │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘    │
│  ┌──────────────────┐  ┌──────────────────┐                          │
│  │ Quick Analytics  │  │ Activity Log     │                          │
│  └──────────────────┘  └──────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
           │                          │                         │
           ▼                          ▼                         ▼
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ PostgreSQL         │   │ LLM Client         │   │ Skills             │
│ (read-only conn)   │   │ (llm_client.py)    │   │ (skills.py)        │
│                    │   │ ┌────────────────┐ │   │ query_database     │
│ psycopg2-binary    │   │ │ LiteLLM        │ │   │ create_excel...    │
└────────────────────┘   │ │ router for     │ │   │ create_word...     │
                         │ │ providers      │ │   │ + artifact builders│
                         │ └────────────────┘ │   └────────────────────┘
                         │   ▼            ▼   │
                         │ Anthropic   Gemini │
                         │ (Claude)    (Google)│
                         └────────────────────┘
```

### File layout
```
ptcl-test/
├── app.py                 # Streamlit UI + DB helpers + AI orchestration (~1,270 lines)
├── llm_client.py          # Provider-agnostic LLM wrapper (Anthropic + Gemini)
├── skills.py              # Agent tools: query_database, create_excel_artifact, create_word_artifact
├── conftest.py            # pytest fixtures (Streamlit mock, DB mock, LiteLLM mock)
├── pytest.ini             # pytest config
├── requirements.txt       # Runtime dependencies
├── requirements-test.txt  # Dev/test dependencies
├── logs/                  # Daily log files (app_YYYY-MM-DD.log)
├── tests/                 # 207 tests across 10 files
└── README.md              # This file
```

---

## Key features

### 1. AI Data Assistant with three routing modes

The assistant's orchestrator (inside `ask_claude_for_sql`) returns one of three sentinels — Claude self-routes in a single call with no extra latency:

| Sentinel | What happens |
|---|---|
| Regular `SELECT ...` query | Fast path: execute → explain results |
| `NO_SQL_NEEDED` | Conversational chat (no DB call) |
| `USE_AGENT_MODE` | Multi-step agent with tools (see below) |

### 2. Agent loop with tool use (Claude-inspired)

For complex requests like *"give me tables organized by category in an Excel file"*, the app runs an agent loop:

1. Claude sees the question + database schema (prompt-cached)
2. Iteratively calls `query_database` to gather data
3. Calls `create_excel_artifact` or `create_word_artifact` to build real files
4. Returns a short summary; files appear as downloadable cards in chat

Tool-use features:
- **Self-healing SQL** — if a query fails (like the known `column "table_name" is ambiguous` error), the error is fed back to Claude who fixes and retries.
- **Result truncation** — query results >50 rows are compressed (first 20 + last 5 + summary stats) to save tokens.
- **Step cap** — 10 iterations max to prevent runaway loops.
- **Artifact persistence** — files stored in session state, downloadable across reruns.

### 3. Dual-provider support — Claude + Gemini

Switch providers, models, and tiers from the sidebar.

| Provider | Models | Tiers |
|---|---|---|
| **Anthropic (default)** | Claude Sonnet 4, Haiku, Opus | Standard, Priority |
| **Google** | Gemini 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite, 3.1 Pro Preview | Standard, Priority, Flex (50% off), Batch (50% off) |

**Smart routing** (default ON) automatically picks the right model per task type:
- Routing call → fast/cheap model (Haiku / Flash-Lite)
- Agent / multi-step → most capable model (Sonnet / 2.5 Pro)
- User override via dropdowns always wins

Implemented via **LiteLLM**, which:
- Gives a unified `litellm.completion()` API for both providers
- Translates Anthropic-style `cache_control` to Gemini's context caching
- Passes through Gemini's `service_tier` parameter
- Normalizes tool-call formats (we use OpenAI format internally)

### 4. Token optimization
- **Prompt caching** on the database schema (`cache_control: {"type": "ephemeral"}`) — first call pays full, subsequent calls in the 5-min window pay ~10%. Applies to both providers.
- **Smart result truncation** for SQL results fed back to the agent.
- **Lean chat history** — only the last 6 turns passed to the agent; assistant messages use a `summary` field when available.

### 5. Robust stop-reason handling
Handles every possible API response gracefully — normalized across providers:

| Stop reason | Behavior |
|---|---|
| `end_turn` / `stop` | Return text normally |
| `tool_use` / `tool_calls` | Dispatch tool, continue loop |
| `max_tokens` / `length` | **Recover**: text-only → auto-continue once; mid-tool-call → friendly "try narrower" message with preserved artifacts; simple functions → append truncation notice |
| `refusal` / `content_filter` / `SAFETY` | Friendly "model declined" message |
| Unknown | Treat as end_turn instead of crashing |

### 6. Exports (4 formats)
Every AI answer's data can be downloaded as:
- **CSV** (raw data)
- **Excel .xlsx** (raw data)
- **Word .docx** — full report with question, AI analysis, SQL, formatted data table
- **Text .txt** — plain-text version of the same report

Agent-built artifacts appear as their own download cards with title, summary, file size, and "⬇ Download" button.

### 7. Comprehensive logging
Every event logged to `logs/app_YYYY-MM-DD.log`:
- DB connections (host, user, timing)
- SQL queries (full query, row count, timing, errors)
- AI requests (provider, model, tier, question)
- Token usage (input, cached, output) per call
- Agent steps with stop reasons
- Tool calls with purpose and result status

The **Activity Log** tab lets you view, filter by level, search, and download logs right from the app.

---

## Setup

### Prerequisites
- Python 3.10+ (tested on 3.12)
- PostgreSQL credentials (host, port, database, user, password) — read-only privileges recommended
- Anthropic API key (default) or Google Gemini API key — for the AI Assistant tab

### Install

```bash
# 1. Clone or download
cd ptcl-test

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. (Optional) Install test dependencies
pip install -r requirements-test.txt
```

### Run

```bash
streamlit run app.py --server.headless true
```

Opens at **http://localhost:8501**.

### Configure in the sidebar
1. **Database Connection** — enter host/port/database/user/password, click *Connect (Read-Only)*
2. **AI Settings**
   - Provider: Anthropic (default) or Google
   - Model: *Auto (smart routing)* or a specific model
   - Tier: Standard / Priority / Flex / Batch
   - API Key: paste your Anthropic or Gemini key (label updates with the provider)

---

## How to use

### Example — simple question
> "How many users signed up last month?"

Fast path: Claude writes `SELECT COUNT(*) FROM users WHERE created_at >= ...`, runs it, explains the answer.

### Example — multi-step agent with Excel artifact
> "For each category, give me tables/views, fields, their description and counts in an excel file"

Agent path: Claude runs 2–3 queries, categorizes the results, builds a multi-sheet Excel file, and returns it as a download card in the chat.

### Example — conversational
> "What does this database contain?"

Conversational path: Claude summarizes the schema in plain English — no query run.

---

## Testing

207 tests across 10 files — all runnable with no real DB or API key.

```bash
# Run all tests (mocked, ~85 seconds)
pytest

# With coverage report
pytest --cov=app --cov=skills --cov=llm_client --cov-report=term-missing

# Single test file
pytest tests/test_stop_reasons.py -v
```

### Test files

| File | Focus | Tests |
|---|---|---|
| `test_pure_functions.py` | SQL safety check, connection helpers | 36 |
| `test_db_functions.py` | DB functions with mocked psycopg2 | 23 |
| `test_ai_functions.py` | Three simple AI functions (SQL / analyze / chat) | 19 |
| `test_agent_loop.py` | Multi-step agent with tool use | 12 |
| `test_ai_assistant_flow.py` | Routing logic, retry logic | 12 |
| `test_edge_cases.py` | API errors, fence stripping, data edge cases | 17 |
| `test_security.py` | SQL injection, read-only enforcement | 12 |
| `test_skills.py` | Tool definitions, dispatchers, artifact builders | 25 |
| `test_llm_client.py` | Provider abstraction, routing, stop-reason normalization | 36 |
| `test_stop_reasons.py` | Reproduces known bugs (max_tokens crash), recovery paths | 15 |
| **Total** | | **207** |

### Integration tests (optional, not run by default)
```bash
# Real PostgreSQL
DB_HOST=... DB_PORT=5432 DB_NAME=... DB_USER=... DB_PASSWORD=... pytest -m integration

# Real API
ANTHROPIC_API_KEY=sk-... pytest -m integration_api
```

---

## Bugs fixed during development

Documented here because the tests continue to guard against regression:

| Bug | Symptom | Root cause | Fix |
|---|---|---|---|
| A | Routing check `== "NO_SQL_NEEDED"` failed when AI appended explanation | Exact-match check | Use `startswith("NO_SQL_NEEDED")` |
| B | Markdown fences ``` ```SQL ``` / ``` ```postgresql ``` not stripped | `removeprefix("```sql")` is case-sensitive | Regex-based case-insensitive stripping |
| C | AI returning `(None, None)` caused silent failure | No `else` branch in router | Added fallback clause with logged warning |
| D | API returning empty `content[]` array raised `IndexError` | No guard on `content[0]` | Added length + None check |
| E | Query retries blocked by over-aggressive dangerous-keywords regex | Blacklisting any `CREATE/UPDATE/...` anywhere in SQL matched column names like `created_at` | Switched to whitelist: query must *start with* a safe keyword |
| F | Long responses ended prematurely ("Customer Service & Support: 15 (") | `max_tokens=1024` on `ask_claude_for_sql` silently truncated prose responses | Re-route prose through `ask_claude_no_sql` with 4096 tokens + explicit truncation detection |
| G | **`Unexpected stop reason: max_tokens`** agent crash | Loop only handled `end_turn` / `tool_use` | Full stop-reason coverage: auto-continue for text, graceful message for mid-tool-call, preserve already-built artifacts |

---

## Technology stack

| Layer | Library |
|---|---|
| Web UI | **Streamlit** |
| Database | **psycopg2-binary** (PostgreSQL driver) |
| Data / CSV | **pandas** |
| Charts | **plotly** |
| Excel export | **openpyxl** |
| Word export | **python-docx** |
| LLM provider abstraction | **LiteLLM** (unified interface for 100+ providers) |
| Tests | **pytest**, **pytest-cov**, **pytest-mock** |

---

## Known trade-offs

- **Agent loop is slow for simple questions** (15–30s vs 5–10s for fast path). Mitigated by the router: only complex multi-step requests go agentic.
- **Gemini prompt caching** is less precise than Anthropic's — LiteLLM auto-translates but only the first `cache_control` marker wins. Fine for our single-schema cache use case.
- **Tokenizers library (transitive dep of LiteLLM)** is a compiled Rust module that can only be initialized once per interpreter — conftest imports it at module load so tests don't re-import.
- **Client-side SQL safety check** is permissive by design (only validates the starting keyword). The real safety net is PostgreSQL's read-only connection mode.

---

## License

Personal/internal project — no license specified.

---

## Credits

Built collaboratively with Claude (Anthropic) through the Claude Code CLI.
