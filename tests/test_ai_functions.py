"""Tests for AI/Anthropic API functions."""

import pytest
from unittest.mock import MagicMock, patch
import sys
from conftest import make_api_response, SAMPLE_SCHEMA_TEXT


class TestAskClaudeForSql:
    def test_basic_sql_response(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "SELECT * FROM users LIMIT 500"
        )
        sql, err = app_module.ask_claude_for_sql("show all users", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT * FROM users LIMIT 500"
        assert err is None

    def test_strips_sql_fences(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "```sql\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_strips_plain_fences(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "```\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_uppercase_fence_stripped(self, app_module, mock_anthropic):
        """Fixed: uppercase ```SQL fence is now stripped."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "```SQL\nSELECT 1\n```"
        )
        sql, _ = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "SELECT 1"

    def test_no_sql_needed(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("NO_SQL_NEEDED")
        sql, _ = app_module.ask_claude_for_sql("thanks", SAMPLE_SCHEMA_TEXT, "key")
        assert sql == "NO_SQL_NEEDED"

    def test_no_sql_needed_with_trailing_text(self, app_module, mock_anthropic):
        """Documents bug: AI appends explanation after NO_SQL_NEEDED."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "NO_SQL_NEEDED\n\nThis is a telecom database."
        )
        sql, _ = app_module.ask_claude_for_sql("what is this db", SAMPLE_SCHEMA_TEXT, "key")
        # Returns the full string including the trailing text
        assert sql.startswith("NO_SQL_NEEDED")
        assert "telecom" in sql

    def test_prose_response(self, app_module, mock_anthropic):
        """AI returns text description instead of SQL."""
        mock_anthropic.messages.create.return_value = make_api_response(
            "Based on the database schema, this appears to be a comprehensive system"
        )
        sql, _ = app_module.ask_claude_for_sql("overview", SAMPLE_SCHEMA_TEXT, "key")
        assert sql.startswith("Based on")

    def test_api_error(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("rate limit exceeded")
        sql, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert sql is None
        assert "rate limit" in err

    def test_empty_response(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("")
        sql, err = app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        # Empty response now returns an error
        assert sql is None
        assert err is not None

    def test_model_parameter(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("SELECT 1")
        app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        # Default model for "sql" task is Opus (Haiku is excluded from this project).
        assert call_kwargs["model"] == "claude-opus-4-7"
        assert call_kwargs["max_tokens"] == 1024

    def test_chat_history_uses_summary(self, app_module, mock_anthropic):
        """Assistant messages in history should use 'summary' field if present."""
        mock_anthropic.messages.create.return_value = make_api_response("SELECT 1")
        history = [
            {"role": "user", "content": "show users"},
            {"role": "assistant", "content": "full long analysis...", "summary": "[Answered about: users]"},
        ]
        app_module.ask_claude_for_sql("next question", SAMPLE_SCHEMA_TEXT, "key", chat_history=history)
        messages = mock_anthropic.messages.create.call_args.kwargs["messages"]
        # The assistant message should use summary, not full content
        assistant_msg = [m for m in messages if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "[Answered about: users]"

    def test_no_history(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("SELECT 1")
        app_module.ask_claude_for_sql("test", SAMPLE_SCHEMA_TEXT, "key", chat_history=None)
        messages = mock_anthropic.messages.create.call_args.kwargs["messages"]
        # With LiteLLM the system prompt is in the messages list (role="system"),
        # plus the one user turn. So 2 messages total when no history.
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "test"


class TestAskClaudeToAnalyze:
    def test_basic(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "The data shows 42 users with an average order of $150."
        )
        analysis, err = app_module.ask_claude_to_analyze(
            "how many users", SAMPLE_SCHEMA_TEXT, "id,name\n1,Alice", "SELECT *", "key"
        )
        assert "42 users" in analysis
        assert err is None

    def test_error(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("timeout")
        _, err = app_module.ask_claude_to_analyze(
            "test", SAMPLE_SCHEMA_TEXT, "data", "SELECT 1", "key"
        )
        assert "timeout" in err

    def test_max_tokens_bumped(self, app_module, mock_anthropic):
        """Analyze was bumped from 2048 to 4096 to reduce truncation."""
        mock_anthropic.messages.create.return_value = make_api_response("analysis")
        app_module.ask_claude_to_analyze("test", SAMPLE_SCHEMA_TEXT, "data", "SELECT 1", "key")
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    def test_chat_history_uses_content(self, app_module, mock_anthropic):
        """analyze function uses 'content' for assistant messages, not 'summary'."""
        mock_anthropic.messages.create.return_value = make_api_response("analysis")
        history = [
            {"role": "user", "content": "prev question"},
            {"role": "assistant", "content": "full answer", "summary": "[short]"},
        ]
        app_module.ask_claude_to_analyze("q", SAMPLE_SCHEMA_TEXT, "d", "s", "key", chat_history=history)
        messages = mock_anthropic.messages.create.call_args.kwargs["messages"]
        assistant_msg = [m for m in messages if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "full answer"


class TestAskClaudeNoSql:
    def test_basic(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "This is a telecom database with customer service data."
        )
        response, err = app_module.ask_claude_no_sql("what is this", SAMPLE_SCHEMA_TEXT, "key")
        assert "telecom" in response
        assert err is None

    def test_error(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("auth failed")
        _, err = app_module.ask_claude_no_sql("test", SAMPLE_SCHEMA_TEXT, "key")
        assert "auth failed" in err

    def test_chat_history_uses_content(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response("response")
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "full", "summary": "[short]"},
        ]
        app_module.ask_claude_no_sql("q2", SAMPLE_SCHEMA_TEXT, "key", chat_history=history)
        messages = mock_anthropic.messages.create.call_args.kwargs["messages"]
        assistant_msg = [m for m in messages if m["role"] == "assistant"][0]
        assert assistant_msg["content"] == "full"
