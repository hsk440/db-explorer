"""Tests for stop-reason handling — reproduces the user's max_tokens crash
and verifies all recovery paths."""

import pytest
from unittest.mock import MagicMock
from conftest import make_api_response, SAMPLE_SCHEMA_TEXT


def _conn_kwargs():
    return {"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"}


# ===================================================================
# Bug reproduction: agent loop used to crash on max_tokens
# ===================================================================

class TestAgentMaxTokensRecovery:
    def test_max_tokens_with_text_auto_continues(self, app_module, mock_anthropic):
        """Agent hit max_tokens with text only -> auto-continue once."""
        mock_anthropic.messages.create.side_effect = [
            make_api_response(text="Partial analysis here...", finish_reason="length"),
            make_api_response(text="...continued to the end.", finish_reason="stop"),
        ]
        text, artifacts, err = app_module.run_agent_loop(
            "explain", SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        # Verify two API calls happened (original + continuation)
        assert mock_anthropic.messages.create.call_count == 2
        assert "continued" in text

    def test_max_tokens_only_continues_once(self, app_module, mock_anthropic):
        """If continuation also hits max_tokens, don't loop forever."""
        mock_anthropic.messages.create.return_value = make_api_response(
            text="Still more...", finish_reason="length",
        )
        text, artifacts, err = app_module.run_agent_loop(
            "explain", SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        # At most one continuation, so 2 total calls
        assert mock_anthropic.messages.create.call_count <= 2
        # Returns gracefully with truncation notice
        assert "truncat" in text.lower() or "Still more" in text

    def test_max_tokens_mid_tool_call_graceful_message(self, app_module, mock_anthropic, mock_psycopg2):
        """This is the user's reported bug.

        Agent is building a big Excel artifact, hits max_tokens mid-tool-call.
        Should return a graceful message, not 'Unexpected stop reason: max_tokens'.
        """
        mock_anthropic.messages.create.return_value = make_api_response(
            text=None,
            finish_reason="length",
            tool_calls=[{
                "id": "c1", "name": "create_excel_artifact",
                "input": {"filename": "huge.xlsx", "title": "Huge", "summary": "...", "sheets": {}},
            }],
        )
        text, artifacts, err = app_module.run_agent_loop(
            "for each category give me all the details in an excel file",
            SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        assert err is None, f"Should not return error; got: {err}"
        # Must NOT be the old crash message
        assert "Unexpected stop reason" not in (text or "")
        # Must contain actionable guidance
        lower = (text or "").lower()
        assert "narrow" in lower or "smaller" in lower or "fewer" in lower or "split" in lower

    def test_max_tokens_with_existing_artifacts_preserved(self, app_module, mock_anthropic, mock_psycopg2):
        """If some artifacts were built in earlier steps, don't lose them."""
        # Step 1: create artifact successfully
        # Step 2: hit max_tokens mid-tool-call
        mock_anthropic.messages.create.side_effect = [
            make_api_response(
                text=None,
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "c1", "name": "create_excel_artifact",
                    "input": {
                        "filename": "billing.xlsx", "title": "Billing",
                        "summary": "First one", "sheets": {"a": [{"x": 1}]},
                    },
                }],
            ),
            make_api_response(
                text=None,
                finish_reason="length",
                tool_calls=[{
                    "id": "c2", "name": "create_excel_artifact",
                    "input": {"filename": "b.xlsx", "title": "B", "summary": "", "sheets": {}},
                }],
            ),
        ]
        text, artifacts, err = app_module.run_agent_loop(
            "give me all categories as separate excel files",
            SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        # The first artifact was successfully built before max_tokens hit
        assert len(artifacts) >= 1
        assert artifacts[0]["filename"] == "billing.xlsx"


class TestAgentUnknownStopReason:
    def test_unknown_reason_treated_as_end_turn(self, app_module, mock_anthropic):
        """Unknown finish_reason should NOT crash — treat as end_turn."""
        mock_anthropic.messages.create.return_value = make_api_response(
            text="Some response", finish_reason="something_weird",
        )
        text, artifacts, err = app_module.run_agent_loop(
            "test", SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        assert "Some response" in text


class TestAgentRefusal:
    def test_content_filter_returns_friendly_message(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            text="",
            finish_reason="content_filter",
        )
        text, artifacts, err = app_module.run_agent_loop(
            "harmful question", SAMPLE_SCHEMA_TEXT, "k", conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        assert "declined" in text.lower() or "refus" in text.lower()


# ===================================================================
# Simple functions: max_tokens adds truncation notice
# ===================================================================

class TestSimpleFunctionsMaxTokens:
    def test_ask_no_sql_max_tokens_adds_notice(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            text="Partial answer", finish_reason="length",
        )
        text, err = app_module.ask_claude_no_sql("explain", SAMPLE_SCHEMA_TEXT, "k")
        assert err is None
        assert "Partial answer" in text
        assert "truncat" in text.lower()

    def test_ask_analyze_max_tokens_adds_notice(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            text="Analysis started", finish_reason="length",
        )
        text, err = app_module.ask_claude_to_analyze(
            "q", SAMPLE_SCHEMA_TEXT, "data", "SELECT 1", "k",
        )
        assert err is None
        assert "Analysis started" in text
        assert "truncat" in text.lower()

    def test_ask_sql_max_tokens_recovers_sentinel(self, app_module, mock_anthropic):
        """If max_tokens hit during routing but a sentinel is present, recover it."""
        mock_anthropic.messages.create.return_value = make_api_response(
            text="USE_AGENT_MODE\n\nexplanation that got cut off...",
            finish_reason="length",
        )
        sql, err = app_module.ask_claude_for_sql("big query", SAMPLE_SCHEMA_TEXT, "k")
        assert sql == "USE_AGENT_MODE"


# ===================================================================
# Refusal handling
# ===================================================================

class TestSimpleFunctionsRefusal:
    def test_ask_no_sql_refusal(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            text="", finish_reason="content_filter",
        )
        text, err = app_module.ask_claude_no_sql("bad", SAMPLE_SCHEMA_TEXT, "k")
        assert text is None
        assert err is not None
        assert "decline" in err.lower()


# ===================================================================
# Smart routing: Claude is the default
# ===================================================================

class TestSmartRoutingDefault:
    def test_default_provider_is_anthropic(self, session_state):
        """Session state initializes with Anthropic as default."""
        assert session_state["provider"] == "anthropic"

    def test_smart_routing_on_by_default(self, session_state):
        assert session_state["smart_routing"] is True

    def test_pick_model_uses_claude_when_anthropic_selected(self, app_module, session_state):
        session_state["provider"] = "anthropic"
        session_state["model_label"] = "Auto (smart routing)"
        session_state["smart_routing"] = True
        for task in ["sql", "conversational", "analyze", "agent"]:
            assert app_module.pick_model(task).startswith("claude"), f"task={task}"

    def test_pick_model_uses_gemini_when_gemini_selected(self, app_module, session_state):
        session_state["provider"] = "gemini"
        session_state["model_label"] = "Auto (smart routing)"
        session_state["smart_routing"] = True
        for task in ["sql", "conversational", "analyze", "agent"]:
            assert app_module.pick_model(task).startswith("gemini"), f"task={task}"

    def test_explicit_override_wins(self, app_module, session_state):
        """When user picks a specific model, smart routing is bypassed."""
        session_state["provider"] = "anthropic"
        session_state["model_label"] = "Claude Opus 4.7"
        session_state["smart_routing"] = False  # override engaged
        # Every task returns the picked model, not task-specific
        for task in ["sql", "agent"]:
            assert "opus" in app_module.pick_model(task).lower()
