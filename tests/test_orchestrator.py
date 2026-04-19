"""Tests for the Plan → Execute → Synthesize orchestrator."""

import pytest
from unittest.mock import MagicMock, patch

from conftest import make_api_response, SAMPLE_SCHEMA_TEXT


def _conn_kwargs():
    return {"host": "h", "port": 5432, "dbname": "d", "user": "u", "password": "p"}


def _excel_tool_response(tool_call_id, filename="test.xlsx", title="Test",
                         sheets=None):
    return make_api_response(
        text=None,
        finish_reason="tool_calls",
        tool_calls=[{
            "id": tool_call_id,
            "name": "create_excel_artifact",
            "input": {
                "filename": filename,
                "title": title,
                "summary": f"Data for {title}",
                "sheets": sheets or {"Data": [{"x": 1}]},
            },
        }],
    )


def _end_response(text):
    return make_api_response(text=text, finish_reason="stop")


# ===================================================================
# Planner
# ===================================================================

class TestPlanner:
    def test_planner_returns_empty_for_simple_question(self, app_module, mock_anthropic):
        """Planner can decline to split by returning empty subtasks."""
        mock_anthropic.messages.create.return_value = make_api_response(
            '{"subtasks": []}'
        )
        subtasks = app_module.plan_question(
            "how many users are there?", SAMPLE_SCHEMA_TEXT, "key",
        )
        assert subtasks == []

    def test_planner_splits_multi_area_question(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            '{"subtasks": ['
            '{"id": 1, "title": "Billing", "scope": "billing-related tables"},'
            '{"id": 2, "title": "Customer Support", "scope": "support tables"},'
            '{"id": 3, "title": "Network", "scope": "network ops tables"}'
            ']}'
        )
        subtasks = app_module.plan_question(
            "for each area, give me an excel with tables and fields",
            SAMPLE_SCHEMA_TEXT, "key",
        )
        assert len(subtasks) == 3
        titles = [s["title"] for s in subtasks]
        assert "Billing" in titles
        assert "Customer Support" in titles
        assert "Network" in titles

    def test_planner_strips_markdown_fences(self, app_module, mock_anthropic):
        """Planner output might come back wrapped in ```json fences."""
        mock_anthropic.messages.create.return_value = make_api_response(
            '```json\n{"subtasks": [{"id": 1, "title": "X", "scope": "y"}]}\n```'
        )
        subtasks = app_module.plan_question("q", SAMPLE_SCHEMA_TEXT, "key")
        assert len(subtasks) == 1

    def test_planner_caps_at_max_subtasks(self, app_module, mock_anthropic):
        many = ",".join(
            f'{{"id": {i}, "title": "T{i}", "scope": "s"}}'
            for i in range(25)
        )
        mock_anthropic.messages.create.return_value = make_api_response(
            f'{{"subtasks": [{many}]}}'
        )
        subtasks = app_module.plan_question("q", SAMPLE_SCHEMA_TEXT, "key")
        assert len(subtasks) == app_module.PLANNER_MAX_SUBTASKS

    def test_planner_invalid_json_falls_back_to_empty(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "This is not valid JSON"
        )
        subtasks = app_module.plan_question("q", SAMPLE_SCHEMA_TEXT, "key")
        assert subtasks == []

    def test_planner_api_error_falls_back_to_empty(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("api down")
        subtasks = app_module.plan_question("q", SAMPLE_SCHEMA_TEXT, "key")
        assert subtasks == []

    def test_planner_missing_fields_use_defaults(self, app_module, mock_anthropic):
        """Subtasks with missing id/title/scope get sensible defaults."""
        mock_anthropic.messages.create.return_value = make_api_response(
            '{"subtasks": [{}]}'
        )
        subtasks = app_module.plan_question("q", SAMPLE_SCHEMA_TEXT, "key")
        assert len(subtasks) == 1
        assert subtasks[0]["id"] == 1
        assert "Part" in subtasks[0]["title"]


# ===================================================================
# Synthesizer
# ===================================================================

class TestSynthesizer:
    def test_synthesize_with_artifacts(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.return_value = make_api_response(
            "I built 3 files covering Billing, Support, and Network. Download below."
        )
        artifacts = [
            {"title": "Billing", "filename": "billing.xlsx",
             "size_bytes": 5000, "summary": "Billing tables"},
            {"title": "Support", "filename": "support.xlsx",
             "size_bytes": 6000, "summary": "Support tables"},
            {"title": "Network", "filename": "network.xlsx",
             "size_bytes": 4000, "summary": "Network tables"},
        ]
        text = app_module.synthesize_response("original q", artifacts, "key")
        assert "3 files" in text or "3 file" in text

    def test_synthesize_empty_artifacts(self, app_module):
        """No artifacts → friendly fallback without API call."""
        text = app_module.synthesize_response("q", [], "key")
        assert "wasn't able" in text.lower() or "couldn't" in text.lower() or "no artifacts" in text.lower()

    def test_synthesize_api_error_falls_back(self, app_module, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("down")
        artifacts = [{"title": "X", "filename": "x.xlsx",
                      "size_bytes": 100, "summary": "s"}]
        text = app_module.synthesize_response("q", artifacts, "key")
        # Still returns a sensible fallback message, never crashes
        assert "1 file" in text


# ===================================================================
# Orchestrator: Plan → Execute → Synthesize
# ===================================================================

class TestOrchestrator:
    def test_no_split_falls_through_to_single_agent(self, app_module, mock_anthropic, mock_psycopg2):
        """Planner returns empty → we run the single-agent path."""
        # Sequence of API calls:
        # 1. Planner: returns {"subtasks": []}
        # 2. Agent loop: end_turn immediately with some text
        mock_anthropic.messages.create.side_effect = [
            make_api_response('{"subtasks": []}'),      # planner
            _end_response("Here's the answer directly."),  # agent
        ]
        text, artifacts, err = app_module.run_plan_execute_synthesize(
            "simple question", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        assert text == "Here's the answer directly."
        assert artifacts == []

    def test_multi_subtask_produces_one_artifact_per_task(
        self, app_module, mock_anthropic, mock_psycopg2
    ):
        """Planner splits into 2 subtasks; each sub-agent builds 1 artifact;
        synthesizer wraps it up."""
        mock_anthropic.messages.create.side_effect = [
            # 1. Planner
            make_api_response(
                '{"subtasks": ['
                '{"id": 1, "title": "A", "scope": "first area"},'
                '{"id": 2, "title": "B", "scope": "second area"}'
                ']}'
            ),
            # 2. Subtask 1: create_excel → end
            _excel_tool_response("c1", filename="a.xlsx", title="A"),
            _end_response("A done"),
            # 3. Subtask 2: create_excel → end
            _excel_tool_response("c2", filename="b.xlsx", title="B"),
            _end_response("B done"),
            # 4. Synthesizer
            make_api_response("Built 2 files covering A and B."),
        ]

        text, artifacts, err = app_module.run_plan_execute_synthesize(
            "for each area, build an excel",
            SAMPLE_SCHEMA_TEXT, "key", conn_kwargs=_conn_kwargs(),
        )
        assert err is None
        assert len(artifacts) == 2
        assert {a["filename"] for a in artifacts} == {"a.xlsx", "b.xlsx"}
        assert "2 files" in text

    def test_progress_callback_invoked(self, app_module, mock_anthropic, mock_psycopg2):
        """progress_cb gets called with current/total for the subtask loop."""
        progress_calls = []

        def track(msg, current=None, total=None):
            progress_calls.append((msg, current, total))

        mock_anthropic.messages.create.side_effect = [
            make_api_response(
                '{"subtasks": [{"id": 1, "title": "A", "scope": "s"}]}'
            ),
            _excel_tool_response("c1"),
            _end_response("done"),
            make_api_response("summary"),
        ]
        app_module.run_plan_execute_synthesize(
            "q", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_conn_kwargs(), progress_cb=track,
        )
        # Should have seen "Planning...", then "Building 1/1:...", then "Summarizing..."
        messages = [c[0] for c in progress_calls]
        assert any("Planning" in m for m in messages)
        assert any("Building 1/1" in m for m in messages)
        assert any("Summarizing" in m for m in messages)

    def test_artifact_callback_fires_per_artifact(self, app_module, mock_anthropic, mock_psycopg2):
        """artifact_cb receives each artifact as it's built (for live UI updates)."""
        live = []

        mock_anthropic.messages.create.side_effect = [
            make_api_response(
                '{"subtasks": ['
                '{"id": 1, "title": "A", "scope": "s"},'
                '{"id": 2, "title": "B", "scope": "s"}'
                ']}'
            ),
            _excel_tool_response("c1", filename="a.xlsx", title="A"),
            _end_response("A done"),
            _excel_tool_response("c2", filename="b.xlsx", title="B"),
            _end_response("B done"),
            make_api_response("summary"),
        ]
        app_module.run_plan_execute_synthesize(
            "q", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_conn_kwargs(),
            artifact_cb=live.append,
        )
        assert len(live) == 2
        assert live[0]["filename"] == "a.xlsx"
        assert live[1]["filename"] == "b.xlsx"

    def test_subtask_failure_continues_to_next(
        self, app_module, mock_anthropic, mock_psycopg2
    ):
        """If one subtask fails, we still build the others and note it in the summary."""
        mock_anthropic.messages.create.side_effect = [
            make_api_response(
                '{"subtasks": ['
                '{"id": 1, "title": "Good", "scope": "s"},'
                '{"id": 2, "title": "Bad", "scope": "s"}'
                ']}'
            ),
            # Subtask 1 succeeds
            _excel_tool_response("c1", filename="good.xlsx", title="Good"),
            _end_response("Good done"),
            # Subtask 2 throws API error
            Exception("API died on subtask 2"),
            # Synthesizer still runs
            make_api_response("Built 1 file."),
        ]
        text, artifacts, err = app_module.run_plan_execute_synthesize(
            "q", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_conn_kwargs(),
        )
        # err is None because orchestrator tolerates subtask-level failures
        assert err is None
        assert len(artifacts) == 1
        assert artifacts[0]["filename"] == "good.xlsx"
        # Failure is mentioned in the final message
        assert "failed" in text.lower() or "Bad" in text

    def test_smart_routing_applied_per_phase(self, app_module, mock_anthropic, mock_psycopg2):
        """Planner uses 'sql' model; agent uses 'agent' model; synthesizer uses 'conversational'."""
        mock_anthropic.messages.create.side_effect = [
            make_api_response('{"subtasks": []}'),       # planner
            _end_response("direct answer"),              # fallback agent
        ]
        app_module.run_plan_execute_synthesize(
            "q", SAMPLE_SCHEMA_TEXT, "key",
            conn_kwargs=_conn_kwargs(),
        )
        # Two calls: planner (first) and agent (second)
        calls = mock_anthropic.messages.create.call_args_list
        planner_model = calls[0].kwargs["model"]
        agent_model = calls[1].kwargs["model"]
        # Planner uses Opus for Anthropic (quality for task decomposition);
        # on Gemini falls back to Flash-Lite.
        assert "opus" in planner_model.lower() or "lite" in planner_model.lower()
        # Agent should use the more capable model (sonnet / pro)
        assert "sonnet" in agent_model.lower() or "pro" in agent_model.lower()


# ===================================================================
# Service tier unsupported cache
# ===================================================================

class TestTierCache:
    def test_first_rejection_caches_for_subsequent_calls(self, mock_litellm):
        import llm_client
        llm_client.reset_tier_cache()

        # First call raises "unsupported service_tier"
        # Second call succeeds (we expect no service_tier sent)
        # Third call also succeeds (cached, still no tier)
        mock_litellm.side_effect = [
            Exception("litellm.UnsupportedParamsError: anthropic does not support "
                      "parameters: ['service_tier']"),
            make_api_response("ok1"),
            make_api_response("ok2"),
        ]

        # First call: will see retry-without-tier internally
        llm_client.llm_complete(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
            api_key="k", tier="priority",
        )
        # Second call: should skip service_tier entirely (no retry, no warning)
        llm_client.llm_complete(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
            api_key="k", tier="priority",
        )

        # Total litellm.completion invocations:
        # Call 1: failed attempt + retry = 2 invocations
        # Call 2: single invocation (cache skips the param) = 1
        assert mock_litellm.call_count == 3

        # Second app-level call's kwargs should NOT include service_tier
        second_call_kwargs = mock_litellm.call_args_list[2].kwargs
        assert "service_tier" not in second_call_kwargs

        llm_client.reset_tier_cache()

    def test_different_models_tracked_separately(self, mock_litellm):
        """Cache is keyed on (provider, model); caching one shouldn't affect another."""
        import llm_client
        llm_client.reset_tier_cache()

        # Cache claude-sonnet-4-6 as unsupported
        mock_litellm.side_effect = [
            Exception("service_tier unsupported"),
            make_api_response("ok"),
            make_api_response("ok for a different model"),
        ]
        llm_client.llm_complete(
            model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
            api_key="k", tier="priority",
        )
        # Different model — should still try service_tier
        llm_client.llm_complete(
            model="claude-opus-4-7", messages=[{"role": "user", "content": "x"}],
            api_key="k", tier="priority",
        )
        # Third call should have sent service_tier (different model)
        third_kwargs = mock_litellm.call_args_list[2].kwargs
        assert third_kwargs.get("service_tier") == "auto"

        llm_client.reset_tier_cache()
