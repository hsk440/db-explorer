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

## System architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Schema Explorer  │  │ SQL Query        │  │ AI Assistant     │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│  ┌──────────────────┐  ┌──────────────────┐                                │
│  │ Quick Analytics  │  │ Activity Log     │                                │
│  └──────────────────┘  └──────────────────┘                                │
│  Sidebar: Provider / Model / Tier / API key / DB credentials               │
└────────────────────────────────────────────────────────────────────────────┘
           │                      │                          │
           ▼                      ▼                          ▼
┌────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐
│ PostgreSQL         │   │ llm_client.py       │   │ skills.py          │
│ read-only conn     │   │ ┌─────────────────┐ │   │ Tool definitions + │
│ psycopg2-binary    │   │ │ LiteLLM         │ │   │ dispatcher:        │
│                    │   │ │ unified router  │ │   │  • query_database  │
│ Used by:           │   │ └────┬──────┬─────┘ │   │  • create_excel    │
│  fetch_schemas     │   │      ▼      ▼       │   │  • create_word     │
│  fetch_tables      │   │ Anthropic  Google   │   │ + artifact builders│
│  fetch_columns     │   │  Claude   Gemini    │   │ + result truncation│
│  run_query         │   └─────────────────────┘   └────────────────────┘
└────────────────────┘
```

## AI Assistant — detailed architecture

The AI Assistant is where most of the complexity lives. Inspired by Claude's own architecture: **orchestrator**, **skills**, **token optimization**, **artifacts**.

```
 USER QUESTION (chat_input)
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR — ask_claude_for_sql()                         │
│  One LLM call that returns exactly one of three sentinels:   │
│                                                              │
│   (a) "USE_AGENT_MODE"   →  multi-step agent with tools      │
│   (b) "NO_SQL_NEEDED"    →  conversational reply, no DB      │
│   (c) "SELECT ..."       →  fast path: run + analyze         │
│                                                              │
│  Uses Haiku 4.5 / Flash-Lite (cheap + fast) for routing.     │
└──────────────────────────────────────────────────────────────┘
         │
    ┌────┼─────────────────────────────────────────────────┐
    ▼    ▼                                                 ▼
  [a]  [b]                                               [c]
 AGENT  CHAT                                          FAST PATH
  LOOP  PATH                                              │
    │    │                                                ▼
    │    │                                       ┌──────────────────┐
    │    │                                       │ run_query()      │
    │    │                                       │ (read-only)      │
    │    │                                       │  │               │
    │    │                                       │  ├─ fails?       │
    │    │                                       │  │  → AI fixes   │
    │    │                                       │  │    SQL (×2    │
    │    │                                       │  │    retries)   │
    │    │                                       │  ▼               │
    │    │                                       │ DataFrame        │
    │    │                                       └──────────────────┘
    │    │                                                │
    │    │                                                ▼
    │    │                                       ┌──────────────────┐
    │    │                                       │ ask_claude_to_   │
    │    │                                       │ analyze()        │
    │    │                                       │ Sonnet 4.6       │
    │    │                                       │ max_tokens=4096  │
    │    │                                       └──────────────────┘
    │    │                                                │
    │    ▼                                                │
    │ ┌────────────────────┐                              │
    │ │ ask_claude_no_sql()│                              │
    │ │ Sonnet 4.6         │                              │
    │ │ max_tokens=4096    │                              │
    │ └────────────────────┘                              │
    │            │                                        │
    ▼            │                                        │
┌──────────────────────────────────────────────────────┐  │
│ AGENT LOOP — run_agent_loop()                        │  │
│                                                      │  │
│  max_steps = 20      max_tokens = 8192               │  │
│  model     = Sonnet 4.6 / Gemini 3.1 Pro             │  │
│                                                      │  │
│  system prompt  ┌──────────────────────────┐         │  │
│    (cached with │ cache_control: ephemeral │         │  │
│     5-min TTL)  │  {schema_text}           │         │  │
│                 └──────────────────────────┘         │  │
│                                                      │  │
│  ┌──────────────────────────────────────────────┐    │  │
│  │ LITELLM COMPLETION (provider-agnostic)       │    │  │
│  │                                              │    │  │
│  │   tools = [query_database,                   │    │  │
│  │            create_excel_artifact,            │    │  │
│  │            create_word_artifact]             │    │  │
│  │                                              │    │  │
│  │   service_tier = mapped per provider:        │    │  │
│  │     Anthropic priority → "auto"              │    │  │
│  │     Anthropic standard → "standard_only"     │    │  │
│  │     Gemini priority  → "priority"            │    │  │
│  │     Gemini flex      → "flex"                │    │  │
│  │                                              │    │  │
│  │   Graceful fallback: retry without tier if   │    │  │
│  │   provider rejects the param.                │    │  │
│  └──────────────────────────────────────────────┘    │  │
│         │                                            │  │
│         ▼                                            │  │
│   normalized stop_reason:                            │  │
│     end_turn  →  return final text                   │  │
│     tool_use  →  dispatch → append results → loop    │  │
│     max_tok   →  recover: continue once (text)       │  │
│                            or graceful msg (tool)    │  │
│     refusal   →  friendly "model declined" message   │  │
│     unknown   →  treat as end_turn (don't crash)     │  │
│                                                      │  │
│  TOOL DISPATCH (skills.dispatch_tool)                │  │
│    ┌─────────────────────────────────────────────┐   │  │
│    │ query_database(sql, purpose)                │   │  │
│    │   → run_query() → DataFrame                 │   │  │
│    │   → truncate_query_result() saves tokens:   │   │  │
│    │       ≤ 50 rows: full data                  │   │  │
│    │       > 50 rows: head(20) + tail(5) + stats │   │  │
│    │       > 8 KB serialized: head(5) + summary  │   │  │
│    │   → JSON fed back to model (self-healing    │   │  │
│    │     on SQL errors: AI sees error and fixes) │   │  │
│    │                                             │   │  │
│    │ create_excel_artifact(filename, sheets)     │   │  │
│    │   → pd.ExcelWriter(openpyxl) per-sheet      │   │  │
│    │   → bytes stored in st.session_state        │   │  │
│    │   → rendered as download card in chat       │   │  │
│    │                                             │   │  │
│    │ create_word_artifact(filename, sections)    │   │  │
│    │   → python-docx: title, headings, tables    │   │  │
│    │   → supports markdown in body (bold, lists) │   │  │
│    │   → artifact stored + downloadable          │   │  │
│    └─────────────────────────────────────────────┘   │  │
└──────────────────────────────────────────────────────┘  │
         │                                                │
         ▼                                                ▼
┌────────────────────────────────────────────────────────────┐
│  RENDER IN CHAT                                            │
│  • markdown response text                                  │
│  • artifact cards (📊 / 📄) with download buttons          │
│  • "SQL query used" expander (with retry history)          │
│  • "Raw data" expander (with 4-format export: CSV / Excel /│
│    Word / Text)                                            │
│  • Persisted across reruns via st.session_state.artifacts  │
└────────────────────────────────────────────────────────────┘
```

### Smart routing by task

The orchestrator picks the cheapest-capable model per task (Claude is default — Gemini used only if the user switches the provider):

| Task | Anthropic | Gemini |
|---|---|---|
| Orchestrator (SQL routing) | **Haiku 4.5** | 2.5 Flash-Lite |
| Conversational (NO_SQL path) | Sonnet 4.6 | 2.5 Flash |
| Data analysis | Sonnet 4.6 | 2.5 Flash |
| Agent / multi-step tool use | **Sonnet 4.6** | **3.1 Pro Preview** |

Users can override via the sidebar (`Auto (smart routing)` or specific model name).

### File layout
```
ptcl-test/
├── app.py                 # Streamlit UI + DB helpers + AI orchestration (~1,280 lines)
├── llm_client.py          # Provider-agnostic LLM wrapper (Anthropic + Gemini via LiteLLM)
├── skills.py              # Agent tools: query_database, create_excel_artifact, create_word_artifact
├── conftest.py            # pytest fixtures (Streamlit mock, DB mock, LiteLLM mock)
├── pytest.ini             # pytest config
├── requirements.txt       # Runtime dependencies
├── requirements-test.txt  # Dev/test dependencies
├── logs/                  # Daily log files (app_YYYY-MM-DD.log)
├── tests/                 # 220 tests across 10 files
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

### 2. Agent loop with tool use

For complex requests like *"give me tables organized by category in an Excel file"*, the app runs an agent loop:

1. Claude sees the question + database schema (prompt-cached)
2. Iteratively calls `query_database` to gather data (up to 20 steps)
3. Calls `create_excel_artifact` or `create_word_artifact` to build real files
4. Returns a summary; files appear as downloadable cards in chat

Tool-use features:
- **Self-healing SQL** — if a query fails (like `column "table_name" is ambiguous`), the error is fed back to Claude who fixes and retries.
- **Result truncation** — query results >50 rows are compressed (first 20 + last 5 + summary stats) to save tokens.
- **Step cap** — **20 iterations max** (bumped from 10) to support multi-category tasks.
- **Partial-success reporting** — if steps are exhausted after some artifacts were built, the user sees a list of what was made and can ask for the rest.
- **Artifact persistence** — files stored in session state, downloadable across reruns.
- **Agent prompt discipline** — instructed to do ONE broad upfront query and categorize in-memory, instead of re-querying per category.

### 3. Dual-provider support — Claude + Gemini

Switch providers, models, and tiers from the sidebar. All model IDs verified against current vendor docs (April 2026):

**Anthropic models** ([docs](https://platform.claude.com/docs/en/about-claude/models/overview)):

| Model | API ID | Notes |
|---|---|---|
| Claude Opus 4.7 | `claude-opus-4-7` | Current flagship — best for agentic coding |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Best speed/intelligence balance |
| Claude Haiku 4.5 | `claude-haiku-4-5` | Cheapest + fastest |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | Still supported, pinned version |
| Claude Opus 4.6 | `claude-opus-4-6` | Prior generation |

**Gemini models** ([docs](https://ai.google.dev/gemini-api/docs/models)):

| Model | API ID | Notes |
|---|---|---|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | Current flagship |
| Gemini 3 Flash | `gemini-3-flash-preview` | New default Flash |
| Gemini 3.1 Flash-Lite Preview | `gemini-3.1-flash-lite-preview` | Cheapest |
| Gemini 2.5 Pro | `gemini-2.5-pro` | Legacy flagship |
| Gemini 2.5 Flash / Flash-Lite | `gemini-2.5-flash` / `-lite` | Legacy |

**Tiers** — mapped internally to the correct per-vendor wire values:

| UI label | Anthropic sends | Gemini sends |
|---|---|---|
| Standard | `service_tier=standard_only` | (omitted) |
| Priority | `service_tier=auto` | `service_tier=priority` |
| Flex | — | `service_tier=flex` |

Batch is **not** in the dropdown — it's a separate asynchronous API endpoint on both providers, not a per-request tier. Graceful fallback retries without `service_tier` if a provider rejects the param.

Implemented via **LiteLLM**, which:
- Gives a unified `litellm.completion()` API for both providers
- Translates Anthropic-style `cache_control` to Gemini's context caching
- Normalizes tool-call formats (we use OpenAI format internally)

### 4. Token optimization
- **Prompt caching** on the database schema (`cache_control: {"type": "ephemeral"}`) — first call pays full, subsequent calls in the 5-min window pay ~10%. Applies to both providers.
- **Smart result truncation** for SQL results fed back to the agent.
- **Lean chat history** — only the last 6 turns passed to the agent; assistant messages use a `summary` field when available.

### 5. Robust stop-reason handling
Handles every possible API response gracefully — normalized across providers:

| Stop reason (normalized) | Anthropic | Gemini | Behavior |
|---|---|---|---|
| `end_turn` | `end_turn` | `stop` / `STOP` | Return text/tool_calls normally |
| `tool_use` | `tool_use` | `tool_calls` | Dispatch tools, continue loop |
| `max_tokens` | `max_tokens` | `length` / `MAX_TOKENS` | **Recover**: auto-continue once for text; graceful message for mid-tool-call with preserved artifacts; simple functions append truncation notice |
| `refusal` | `refusal` | `SAFETY` / `BLOCKLIST` / `PROHIBITED_CONTENT` | Friendly "model declined" message |
| Unknown | any other | any other | Treat as end_turn (don't crash) |

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
   - Tier: Standard / Priority (Anthropic) · Standard / Priority / Flex (Gemini)
   - API Key: paste your Anthropic or Gemini key (label updates with the provider)

---

## How to use

### Example — simple question
> "How many users signed up last month?"

Fast path: Haiku/Flash-Lite writes `SELECT COUNT(*) FROM users WHERE created_at >= ...`, runs it, Sonnet/Flash explains the answer.

### Example — multi-step agent with Excel artifact
> "For each category, give me tables/views, fields, their description and counts in an excel file"

Agent path: Sonnet 4.6 / 3.1 Pro Preview runs one broad query, categorizes tables in memory, builds one Excel file per category, returns them as download cards.

### Example — conversational
> "What does this database contain?"

Conversational path: Sonnet/Flash summarizes the schema in plain English — no query run.

---

## Testing

**220 tests** across 10 files — all runnable with no real DB or API key.

```bash
# Run all tests (mocked, ~75 seconds)
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
| `test_agent_loop.py` | Multi-step agent with tool use | 14 |
| `test_ai_assistant_flow.py` | Routing logic, retry logic | 12 |
| `test_edge_cases.py` | API errors, fence stripping, data edge cases | 17 |
| `test_security.py` | SQL injection, read-only enforcement | 12 |
| `test_skills.py` | Tool definitions, dispatchers, artifact builders | 25 |
| `test_llm_client.py` | Provider abstraction, tier mapping, stop-reason normalization | 47 |
| `test_stop_reasons.py` | Reproduces known bugs (max_tokens crash), recovery paths | 15 |
| **Total** | | **220** |

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
| G | `Unexpected stop reason: max_tokens` agent crash | Loop only handled `end_turn` / `tool_use` | Full stop-reason coverage: auto-continue for text, graceful message for mid-tool-call, preserve already-built artifacts |
| H | Agent ran out of steps and hid successfully built artifacts behind "I reached my step limit" | Exit message didn't mention partial success | Bumped `AGENT_MAX_STEPS` 10 → 20; exit message now lists all built files and suggests asking for the rest |
| I | Outdated / deprecated model IDs (`claude-sonnet-4-20250514`, `claude-opus-4-20250514` — retire Jun 15 2026) and wrong tier values (`service_tier=priority` isn't a valid Anthropic **request** value) | Stale catalog, no per-provider tier mapping | Refreshed catalogs against vendor docs (Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5; Gemini 3.1 Pro Preview, 3 Flash, 3.1 Flash-Lite Preview); added `TIER_API_VALUE` mapping; removed Batch (separate API); graceful-fallback retry if LiteLLM rejects the param |

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
- **Batch tier not supported** — Batch is an async API endpoint (submit → poll → fetch). Adding it would require a different code path and UI. Live tiers only for now.
- **Model IDs are aliases, not dated snapshots** — `claude-sonnet-4-6` auto-upgrades to the latest 4.6 patch. Anthropic recommends dated snapshots for production stability. Users can pin via the manual model dropdown if needed.

---

## License

Personal/internal project — no license specified.

---

## Credits

Built collaboratively with Claude (Anthropic) through the Claude Code CLI.
