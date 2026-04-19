"""Tests for the agent loop (tool use) in app.py — LiteLLM / OpenAI format."""

import pytest
from unittest.mock import MagicMock, patch
import json

from conftest import make_api_response, SAMPLE_SCHEMA_TEXT


def _tool_use_response(tool_name, tool_input, tool_call_id="tool_1"):
    """Build a mock LiteLLM response that represents a tool call."""
    return make_api_response(
        text=None,
        finish_reason="tool_calls",
        tool_calls=[{"id": tool_call_id, "name": tool_name, "input": tool_input}],
    )


def _end_turn_response(text):
    """Build a mock LiteLLM response with finish_reason='stop'."""
    return make_api_response(text=text, finish_reason="stop")


def _base_conn_kwargs():
    return {"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"}


class TestAgentLoop:
    def test_immediate_end_turn(self, app_module, mock_anthropic):
        """Agent ends on first turn with no tool calls."""
        mock_anthropic.messages.create.return_value = _end_turn_response(
            "Here's the answer directly."
        )
        text, artifacts, err = app_module.run_agent_loop(
            "simple question", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert err is None
        assert text == "Here's the answer directly."
        assert artifacts == []

    def test_query_then_end(self, app_module, mock_anthropic, mock_psycopg2):
        """Agent calls query_database once then ends."""
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = [{"id": 1, "name": "Alice"}]
        cursor.description = [("id",), ("name",)]

        mock_anthropic.messages.create.side_effect = [
            _tool_use_response(
                "query_database",
                {"query": "SELECT * FROM users", "purpose": "fetch users"},
            ),
            _end_turn_response("Found 1 user: Alice"),
        ]

        text, artifacts, err = app_module.run_agent_loop(
            "who are the users", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert err is None
        assert "Alice" in text
        assert artifacts == []
        assert mock_anthropic.messages.create.call_count == 2

    def test_excel_artifact_created(self, app_module, mock_anthropic, mock_psycopg2):
        """Agent creates an Excel artifact."""
        mock_anthropic.messages.create.side_effect = [
            _tool_use_response(
                "create_excel_artifact",
                {
                    "filename": "tables.xlsx",
                    "title": "Tables",
                    "summary": "A list of tables",
                    "sheets": {"Sheet1": [{"name": "users"}, {"name": "orders"}]},
                },
            ),
            _end_turn_response("Excel file created. Download it below."),
        ]

        text, artifacts, err = app_module.run_agent_loop(
            "make me an excel", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert err is None
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "excel"
        assert artifacts[0]["filename"] == "tables.xlsx"
        assert artifacts[0]["bytes"][:2] == b"PK"

    def test_sql_error_returned_to_ai(self, app_module, mock_anthropic, mock_psycopg2):
        """When query fails, error goes back to agent; it can retry."""
        cursor = mock_psycopg2.return_value.cursor.return_value

        call_count = 0
        def execute_side_effect(sql):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("column ambiguous")

        cursor.execute.side_effect = execute_side_effect
        cursor.fetchmany.return_value = [{"id": 1}]
        cursor.description = [("id",)]

        mock_anthropic.messages.create.side_effect = [
            _tool_use_response(
                "query_database",
                {"query": "SELECT bad", "purpose": "test"},
                tool_call_id="t1",
            ),
            _tool_use_response(
                "query_database",
                {"query": "SELECT t.id FROM t", "purpose": "fixed"},
                tool_call_id="t2",
            ),
            _end_turn_response("Fixed and done."),
        ]

        text, artifacts, err = app_module.run_agent_loop(
            "test", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert err is None
        assert mock_anthropic.messages.create.call_count == 3

        # Verify the error went back to the agent via a tool-result message
        # (OpenAI format: role="tool", tool_call_id=..., content="...")
        second_call_messages = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_contents = [
            m["content"] for m in second_call_messages if m.get("role") == "tool"
        ]
        assert tool_result_contents, "No tool_result messages found in second call"
        assert any("ambiguous" in c.lower() for c in tool_result_contents)

    def test_max_steps_reached(self, app_module, mock_anthropic, mock_psycopg2):
        """If agent never ends, caps at AGENT_MAX_STEPS."""
        cursor = mock_psycopg2.return_value.cursor.return_value
        cursor.fetchmany.return_value = []
        cursor.description = []

        # Always return tool_use, never end_turn
        mock_anthropic.messages.create.return_value = _tool_use_response(
            "query_database",
            {"query": "SELECT 1", "purpose": "loop"},
        )

        text, artifacts, err = app_module.run_agent_loop(
            "infinite", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert "step limit" in text.lower()
        assert mock_anthropic.messages.create.call_count == app_module.AGENT_MAX_STEPS

    def test_api_error_returns(self, app_module, mock_anthropic):
        """API error propagates as err."""
        mock_anthropic.messages.create.side_effect = Exception("rate limit")
        text, artifacts, err = app_module.run_agent_loop(
            "test", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        assert err is not None
        assert "rate limit" in err

    def test_prompt_caching_enabled(self, app_module, mock_anthropic):
        """Schema text block has cache_control for prompt caching."""
        mock_anthropic.messages.create.return_value = _end_turn_response("done")
        app_module.run_agent_loop(
            "test", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        # The system prompt is passed through llm_client which wraps it as
        # {role: system, content: [list of blocks]} in the LiteLLM messages
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        sys_msg = [m for m in call_kwargs["messages"] if m.get("role") == "system"][0]
        # content is either a list of blocks (when agent uses caching) or a string
        content = sys_msg["content"]
        assert isinstance(content, list), "Expected list of blocks for prompt caching"
        schema_blocks = [b for b in content if "SCHEMA" in b.get("text", "")]
        assert len(schema_blocks) == 1
        assert schema_blocks[0].get("cache_control") == {"type": "ephemeral"}

    def test_tools_passed_to_api(self, app_module, mock_anthropic):
        """Agent passes the tool definitions to the API in OpenAI format."""
        mock_anthropic.messages.create.return_value = _end_turn_response("done")
        app_module.run_agent_loop(
            "test", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
        )
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        # LiteLLM OpenAI format: [{type: "function", function: {name, ...}}]
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "query_database" in tool_names
        assert "create_excel_artifact" in tool_names
        assert "create_word_artifact" in tool_names

    def test_chat_history_limited_to_6_turns(self, app_module, mock_anthropic):
        """Agent only includes last 6 chat history turns."""
        mock_anthropic.messages.create.return_value = _end_turn_response("ok")
        history = [
            {"role": "user", "content": f"Q{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"A{i}"}
            for i in range(20)
        ]
        app_module.run_agent_loop(
            "new question", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_base_conn_kwargs(),
            chat_history=history,
        )
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        # Non-system messages: 6 history + 1 new = 7
        non_system = [m for m in call_kwargs["messages"] if m.get("role") != "system"]
        assert len(non_system) == 7


class TestRoutingSentinels:
    def test_use_agent_mode_sentinel(self, app_module, mock_anthropic):
        """ask_claude_for_sql returns USE_AGENT_MODE for multi-step requests."""
        mock_anthropic.messages.create.return_value = make_api_response("USE_AGENT_MODE")
        sql, err = app_module.ask_claude_for_sql(
            "create an excel with all tables by category", SAMPLE_SCHEMA_TEXT, "key"
        )
        assert sql == "USE_AGENT_MODE"
        assert err is None

    def test_no_sql_needed_sentinel_still_works(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("NO_SQL_NEEDED")
        sql, _ = app_module.ask_claude_for_sql("thanks", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "NO_SQL_NEEDED"

    def test_regular_sql_still_works(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "SELECT * FROM users"
        )
        sql, _ = app_module.ask_claude_for_sql(
            "show users", SAMPLE_SCHEMA_TEXT, "key"
        )
        assert sql == "SELECT * FROM users"
