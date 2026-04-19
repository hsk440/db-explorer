"""Tests for the provider-agnostic LLM client (llm_client.py)."""

import pytest
from unittest.mock import MagicMock, patch

import llm_client
from conftest import make_api_response


class TestSmartRouting:
    def test_anthropic_default_provider(self):
        assert llm_client.smart_pick_model("anthropic", "sql").startswith("claude")
        assert llm_client.smart_pick_model("anthropic", "conversational").startswith("claude")
        assert llm_client.smart_pick_model("anthropic", "analyze").startswith("claude")
        assert llm_client.smart_pick_model("anthropic", "agent").startswith("claude")

    def test_gemini_fallback(self):
        assert llm_client.smart_pick_model("gemini", "sql").startswith("gemini")
        assert llm_client.smart_pick_model("gemini", "agent").startswith("gemini")

    def test_sql_picks_opus_for_anthropic(self):
        """Anthropic routing call / planner uses Opus (quality over cost)."""
        assert "opus" in llm_client.smart_pick_model("anthropic", "sql").lower()

    def test_sql_picks_flash_lite_for_gemini(self):
        assert "flash-lite" in llm_client.smart_pick_model("gemini", "sql").lower()

    def test_agent_picks_sonnet_for_anthropic(self):
        """Most capable model for agentic tool use."""
        assert "sonnet" in llm_client.smart_pick_model("anthropic", "agent").lower()

    def test_agent_picks_pro_for_gemini(self):
        assert "pro" in llm_client.smart_pick_model("gemini", "agent").lower()

    def test_unknown_task_falls_back(self):
        """Unknown task type defaults to conversational model."""
        result = llm_client.smart_pick_model("anthropic", "unknown-task")
        assert result.startswith("claude")


class TestProviderInference:
    def test_anthropic_model_current(self):
        assert llm_client.provider_of("claude-sonnet-4-6") == "anthropic"
        assert llm_client.provider_of("claude-opus-4-7") == "anthropic"
        assert llm_client.provider_of("claude-haiku-4-5") == "anthropic"

    def test_anthropic_model_legacy(self):
        """Legacy dated IDs still classified as anthropic."""
        assert llm_client.provider_of("claude-sonnet-4-20250514") == "anthropic"

    def test_gemini_with_prefix(self):
        assert llm_client.provider_of("gemini/gemini-2.5-pro") == "gemini"
        assert llm_client.provider_of("gemini/gemini-3.1-pro-preview") == "gemini"

    def test_gemini_bare_name(self):
        assert llm_client.provider_of("gemini-2.5-flash") == "gemini"


class TestStopReasonNormalization:
    def test_openai_stop(self):
        assert llm_client.normalize_stop_reason("stop") == llm_client.STOP_END_TURN

    def test_openai_tool_calls(self):
        assert llm_client.normalize_stop_reason("tool_calls") == llm_client.STOP_TOOL_USE

    def test_openai_length(self):
        assert llm_client.normalize_stop_reason("length") == llm_client.STOP_MAX_TOKENS

    def test_anthropic_end_turn(self):
        assert llm_client.normalize_stop_reason("end_turn") == llm_client.STOP_END_TURN

    def test_anthropic_tool_use(self):
        assert llm_client.normalize_stop_reason("tool_use") == llm_client.STOP_TOOL_USE

    def test_anthropic_max_tokens(self):
        assert llm_client.normalize_stop_reason("max_tokens") == llm_client.STOP_MAX_TOKENS

    def test_gemini_max_tokens(self):
        assert llm_client.normalize_stop_reason("MAX_TOKENS") == llm_client.STOP_MAX_TOKENS

    def test_gemini_safety(self):
        assert llm_client.normalize_stop_reason("SAFETY") == llm_client.STOP_REFUSAL

    def test_unknown(self):
        assert llm_client.normalize_stop_reason("weird") == llm_client.STOP_UNKNOWN

    def test_none(self):
        assert llm_client.normalize_stop_reason(None) == llm_client.STOP_UNKNOWN


class TestToolSchemaConversion:
    def test_anthropic_style_converted_to_openai(self):
        anthropic_tools = [
            {
                "name": "query_database",
                "description": "Run SQL",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
        converted = llm_client._convert_tools_to_openai(anthropic_tools)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "query_database"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_openai_style_passed_through(self):
        """If tools are already in OpenAI format, don't double-wrap."""
        openai_tools = [
            {"type": "function", "function": {"name": "foo", "description": "bar", "parameters": {}}}
        ]
        converted = llm_client._convert_tools_to_openai(openai_tools)
        assert converted == openai_tools


class TestLLMComplete:
    def test_basic_call(self, mock_litellm):
        """llm_complete returns a normalized LLMResponse."""
        mock_litellm.return_value = make_api_response("SELECT 1", finish_reason="stop")
        resp = llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "show"}],
            api_key="k",
        )
        assert resp.text == "SELECT 1"
        assert resp.stop_reason == llm_client.STOP_END_TURN
        assert resp.tool_calls == []
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50

    def test_tool_calls_parsed(self, mock_litellm):
        mock_litellm.return_value = make_api_response(
            text=None,
            finish_reason="tool_calls",
            tool_calls=[{"id": "c1", "name": "query_database", "input": {"query": "SELECT 1"}}],
        )
        resp = llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            api_key="k",
        )
        assert resp.stop_reason == llm_client.STOP_TOOL_USE
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "c1"
        assert resp.tool_calls[0].name == "query_database"
        assert resp.tool_calls[0].input == {"query": "SELECT 1"}

    def test_gemini_priority_maps_to_priority(self, mock_litellm):
        """Gemini 'priority' tier is passed verbatim."""
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="priority",
        )
        call_kwargs = mock_litellm.call_args.kwargs
        assert call_kwargs.get("service_tier") == "priority"

    def test_gemini_flex_maps_to_flex(self, mock_litellm):
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="flex",
        )
        assert mock_litellm.call_args.kwargs.get("service_tier") == "flex"

    def test_gemini_standard_tier_omits_param(self, mock_litellm):
        """Gemini 'standard' = default = don't send the param."""
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="standard",
        )
        assert "service_tier" not in mock_litellm.call_args.kwargs

    def test_anthropic_standard_maps_to_standard_only(self, mock_litellm):
        """Anthropic API requires 'standard_only' (not 'standard') to opt out of priority."""
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="standard",
        )
        assert mock_litellm.call_args.kwargs.get("service_tier") == "standard_only"

    def test_anthropic_priority_maps_to_auto(self, mock_litellm):
        """Anthropic's 'priority' user intent maps to 'auto' — prefer priority, fall back."""
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="priority",
        )
        assert mock_litellm.call_args.kwargs.get("service_tier") == "auto"

    def test_service_tier_unsupported_graceful_fallback(self, mock_litellm):
        """If LiteLLM rejects service_tier, we retry once without it."""
        # First call raises "unknown parameter service_tier"
        # Second call succeeds
        mock_litellm.side_effect = [
            Exception("litellm.BadRequestError: unexpected keyword argument 'service_tier'"),
            make_api_response("fallback ok"),
        ]
        resp = llm_client.llm_complete(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="priority",
        )
        assert resp.text == "fallback ok"
        # Called twice: first with service_tier, second without
        assert mock_litellm.call_count == 2
        # Second call must NOT have service_tier
        assert "service_tier" not in mock_litellm.call_args_list[1].kwargs

    def test_non_tier_errors_still_raise(self, mock_litellm):
        """Errors unrelated to service_tier must NOT trigger fallback."""
        mock_litellm.side_effect = Exception("network error: connection refused")
        with pytest.raises(Exception, match="network error"):
            llm_client.llm_complete(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "x"}],
                api_key="k",
                tier="priority",
            )
        assert mock_litellm.call_count == 1  # no retry

    def test_system_string_becomes_system_message(self, mock_litellm):
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            system="You are helpful.",
        )
        messages = mock_litellm.call_args.kwargs["messages"]
        sys_msgs = [m for m in messages if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "You are helpful."

    def test_system_list_passed_as_content_blocks(self, mock_litellm):
        """When system is a list (for prompt caching), it's passed as content."""
        mock_litellm.return_value = make_api_response("ok")
        system_blocks = [
            {"type": "text", "text": "Base"},
            {"type": "text", "text": "SCHEMA here", "cache_control": {"type": "ephemeral"}},
        ]
        llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            system=system_blocks,
        )
        messages = mock_litellm.call_args.kwargs["messages"]
        sys_msg = [m for m in messages if m["role"] == "system"][0]
        assert sys_msg["content"] == system_blocks  # list passed through

    def test_tools_converted_from_anthropic_to_openai(self, mock_litellm):
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        )
        tools_sent = mock_litellm.call_args.kwargs["tools"]
        assert tools_sent[0]["type"] == "function"
        assert tools_sent[0]["function"]["name"] == "t"

    def test_tool_call_with_invalid_json_args(self, mock_litellm):
        """Malformed JSON in tool call arguments shouldn't crash."""
        response = MagicMock()
        choice = MagicMock()
        message = MagicMock()
        message.content = ""
        tc = MagicMock()
        tc.id = "t1"
        tc.function.name = "query_database"
        tc.function.arguments = "{not valid json"
        message.tool_calls = [tc]
        choice.message = message
        choice.finish_reason = "tool_calls"
        response.choices = [choice]
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.prompt_tokens_details = None
        response.usage = usage
        mock_litellm.return_value = response

        resp = llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
        )
        # Should not crash; parse_error is recorded in input
        assert resp.tool_calls
        assert "_parse_error" in resp.tool_calls[0].input

    def test_cached_tokens_reported(self, mock_litellm):
        """LLMResponse exposes cached_input_tokens when LiteLLM returns them."""
        mock_litellm.return_value = make_api_response(
            "ok", input_tokens=1000, output_tokens=50, cached_tokens=800,
        )
        resp = llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
        )
        assert resp.cached_input_tokens == 800


class TestMessageHelpers:
    def test_build_assistant_tool_call_message(self):
        tc = llm_client.ToolCall(id="c1", name="foo", input={"x": 1})
        msg = llm_client.build_assistant_tool_call_message("thinking...", [tc])
        assert msg["role"] == "assistant"
        assert msg["content"] == "thinking..."
        assert msg["tool_calls"][0]["id"] == "c1"
        assert msg["tool_calls"][0]["function"]["name"] == "foo"
        import json
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"x": 1}

    def test_build_tool_result_message(self):
        msg = llm_client.build_tool_result_message("c1", '{"result": "ok"}')
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "c1"
        assert msg["content"] == '{"result": "ok"}'

    def test_build_tool_result_message_dict_content(self):
        """Non-string content is JSON-serialized."""
        msg = llm_client.build_tool_result_message("c1", {"result": "ok"})
        assert isinstance(msg["content"], str)
        assert "ok" in msg["content"]


class TestTierCatalog:
    def test_anthropic_tiers(self):
        assert llm_client.TIERS_BY_PROVIDER["anthropic"] == ["standard", "priority"]

    def test_gemini_tiers(self):
        assert "standard" in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "priority" in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "flex" in llm_client.TIERS_BY_PROVIDER["gemini"]

    def test_batch_removed_from_live_tiers(self):
        """Batch is a separate async API endpoint, not a per-request tier.
        It must not appear in the live-query tier dropdown."""
        assert "batch" not in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "batch" not in llm_client.TIERS_BY_PROVIDER["anthropic"]


class TestModelCatalogs:
    def test_anthropic_has_current_models(self):
        """Current generation Claude models are present."""
        values = set(llm_client.ANTHROPIC_MODELS.values())
        assert "claude-opus-4-7" in values
        assert "claude-sonnet-4-6" in values

    def test_anthropic_deprecated_removed(self):
        """Deprecated IDs (retire 2026-06-15) must not be offered."""
        values = set(llm_client.ANTHROPIC_MODELS.values())
        assert "claude-sonnet-4-20250514" not in values
        assert "claude-opus-4-20250514" not in values

    def test_anthropic_haiku_excluded(self):
        """Per project policy: Haiku is NOT offered — only Sonnet and Opus."""
        values = set(llm_client.ANTHROPIC_MODELS.values())
        assert "claude-haiku-4-5" not in values
        for label in llm_client.ANTHROPIC_MODELS.keys():
            assert "haiku" not in label.lower()

    def test_anthropic_smart_routing_no_haiku(self):
        """Smart routing for Anthropic never picks Haiku."""
        for task in ("sql", "conversational", "analyze", "agent"):
            picked = llm_client.smart_pick_model("anthropic", task)
            assert "haiku" not in picked.lower(), f"task={task} picked={picked}"

    def test_anthropic_planner_uses_opus(self):
        """The 'sql' task (orchestrator routing + planner) uses Opus for Anthropic."""
        assert "opus" in llm_client.smart_pick_model("anthropic", "sql").lower()

    def test_anthropic_labels_match_ids(self):
        """No mismatched labels (e.g. '4.5' pointing to '4-6' id)."""
        for label, model_id in llm_client.ANTHROPIC_MODELS.items():
            # Extract version like "4.7" from label; check it appears as "4-7" in id
            import re as _re
            m = _re.search(r"(\d+)\.(\d+)", label)
            if m:
                label_ver = f"{m.group(1)}-{m.group(2)}"
                assert label_ver in model_id, f"Label {label!r} and id {model_id!r} mismatch"

    def test_gemini_has_current_models(self):
        values = set(llm_client.GEMINI_MODELS.values())
        assert "gemini/gemini-3.1-pro-preview" in values
        assert "gemini/gemini-2.5-pro" in values
        assert "gemini/gemini-2.5-flash-lite" in values
