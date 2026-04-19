"""
Shared fixtures for DB Explorer test suite.

Key challenge: app.py calls st.set_page_config() and accesses st.session_state
at import time, which fails outside Streamlit runtime. We mock streamlit before
importing app.
"""

import sys
import types
import importlib
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Import litellm ONCE at module load. The `tokenizers` C extension it imports
# can only be initialized once per interpreter; re-importing via fixtures
# causes "PyO3 modules compiled for CPython 3.8 may only be initialized once".
import litellm  # noqa: F401


# ---------------------------------------------------------------------------
# Streamlit mock
# ---------------------------------------------------------------------------

class FakeSessionState(dict):
    """Dict-like object that mimics st.session_state."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def _make_tab_mock():
    """Create a mock that works as a context manager (for `with tab:`)."""
    tab = MagicMock()
    tab.__enter__ = MagicMock(return_value=tab)
    tab.__exit__ = MagicMock(return_value=False)
    return tab


def _make_streamlit_mock():
    """Build a mock streamlit module that won't crash on import."""
    mock_st = MagicMock()

    # session_state as a real dict so app.py's `for key in {...}.items()` works
    session_state = FakeSessionState({
        "connected": False,
        "host": "",
        "port": "5432",
        "dbname": "",
        "user": "",
        "password": "",
        "chat_history": [],
        "artifacts": {},
        "provider": "anthropic",
        "anthropic_api_key": "",
        "gemini_api_key": "",
        "model_label": "Auto (smart routing)",
        "tier": "standard",
        "smart_routing": True,
    })
    mock_st.session_state = session_state

    # tabs() must return exactly 5 context managers for the tuple unpack
    mock_st.tabs.return_value = [_make_tab_mock() for _ in range(5)]

    # form_submit_button returns False so sidebar form doesn't trigger connection
    mock_st.form_submit_button.return_value = False

    # columns() must return the right number of context managers
    def fake_columns(spec=2):
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_make_tab_mock() for _ in range(n)]
    mock_st.columns = fake_columns

    # buttons return False by default
    mock_st.button.return_value = False

    # chat_input returns None (no user input)
    mock_st.chat_input.return_value = None

    # stop() should do nothing (in real Streamlit it halts execution)
    mock_st.stop.return_value = None

    # number_input returns a sensible default
    mock_st.number_input.return_value = 500

    # cache_data should act as a transparent decorator
    def passthrough_cache(**kwargs):
        def decorator(func):
            return func
        return decorator

    mock_st.cache_data = passthrough_cache

    return mock_st


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Patch streamlit before any app import."""
    mock_st = _make_streamlit_mock()
    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st


@pytest.fixture
def app_module(mock_streamlit):
    """Import app.py with streamlit mocked. Returns the module."""
    # Remove cached import if present
    if "app" in sys.modules:
        del sys.modules["app"]

    # Add project root to path
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Mock psycopg2.connect during import so module-level code doesn't
    # try to connect (st.stop() is a no-op in our mock, so code past
    # the "not connected" guard runs and hits fetch_schemas etc.)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = MagicMock(
        fetchall=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=(0,)),
        fetchmany=MagicMock(return_value=[]),
        description=[],
    )
    with patch("psycopg2.connect", return_value=mock_conn):
        import app
    return app


@pytest.fixture
def session_state(mock_streamlit):
    """Direct access to the mocked session state dict."""
    return mock_streamlit.session_state


# ---------------------------------------------------------------------------
# Database mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cursor():
    """A mock psycopg2 cursor."""
    cursor = MagicMock()
    cursor.description = [("id",), ("name",)]
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = (0,)
    cursor.fetchmany.return_value = []
    return cursor


@pytest.fixture
def mock_connection(mock_cursor):
    """A mock psycopg2 connection that returns mock_cursor."""
    conn = MagicMock()
    conn.cursor.return_value = mock_cursor
    return conn


@pytest.fixture
def mock_psycopg2(app_module, mock_connection):
    """Patch psycopg2.connect in the app module to return mock_connection."""
    with patch.object(app_module.psycopg2, "connect", return_value=mock_connection) as mock_connect:
        yield mock_connect


# ---------------------------------------------------------------------------
# LiteLLM API mock
# ---------------------------------------------------------------------------

def make_api_response(
    text="SELECT 1",
    input_tokens=100,
    output_tokens=50,
    finish_reason="stop",
    tool_calls=None,
    cached_tokens=0,
):
    """Build a mock LiteLLM (OpenAI-format) response.

    - text: message.content
    - finish_reason: "stop" | "tool_calls" | "length" | "content_filter"
    - tool_calls: list of dicts like [{"id": "...", "name": "...", "input": {...}}]
      (we convert to the OpenAI shape)
    """
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = text

    # Convert our simplified tool_calls into OpenAI shape with .function.name, .function.arguments
    if tool_calls:
        import json as _json
        oai_tool_calls = []
        for i, tc in enumerate(tool_calls):
            tc_mock = MagicMock()
            tc_mock.id = tc.get("id", f"toolcall_{i}")
            tc_mock.function.name = tc["name"]
            args = tc.get("input", {})
            tc_mock.function.arguments = _json.dumps(args) if not isinstance(args, str) else args
            oai_tool_calls.append(tc_mock)
        message.tool_calls = oai_tool_calls
    else:
        message.tool_calls = None

    choice.message = message
    choice.finish_reason = finish_reason
    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    details = MagicMock()
    details.cached_tokens = cached_tokens
    usage.prompt_tokens_details = details
    response.usage = usage

    return response


@pytest.fixture
def mock_litellm():
    """Patch litellm.completion to return our mock responses."""
    import litellm
    with patch.object(litellm, "completion") as mock_completion:
        mock_completion.return_value = make_api_response()
        yield mock_completion


# Backwards-compatible alias — many existing tests still reference `mock_anthropic`.
# It now returns the same handle as `mock_litellm`, but tests that call
# `mock_anthropic.messages.create` need a small adapter shim.
class _LegacyMessagesShim:
    """Adapts `mock.messages.create(...)` calls onto `litellm.completion(...)`."""
    def __init__(self, litellm_mock):
        self._m = litellm_mock

    @property
    def messages(self):
        return self

    @property
    def create(self):
        return self._m.return_value

    # `messages.create` used as a callable or property getter
    def __call__(self, *args, **kwargs):
        return self._m(*args, **kwargs)


@pytest.fixture
def mock_anthropic(mock_litellm):
    """
    Backwards-compatible fixture name. Tests that use
    `mock_anthropic.messages.create.return_value = make_api_response(...)`
    continue to work — we simply forward to the underlying litellm mock.
    """
    # Expose the litellm mock but decorate it so .messages.create points at it
    shim = MagicMock()
    shim.messages = MagicMock()
    # Make .messages.create the same MagicMock as litellm.completion
    shim.messages.create = mock_litellm
    return shim


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA_TEXT = """public.users:
  id (integer)
  name (character varying)
  email (character varying)
  created_at (timestamp without time zone)

public.orders:
  id (integer)
  user_id (integer)
  amount (numeric)
  status (character varying)
  order_date (date)"""


@pytest.fixture
def sample_schema_text():
    return SAMPLE_SCHEMA_TEXT
