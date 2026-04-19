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

    def test_sql_picks_haiku_for_anthropic(self):
        """Cheap/fast model for the routing call."""
        assert "haiku" in llm_client.smart_pick_model("anthropic", "sql").lower()

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
    def test_anthropic_model(self):
        assert llm_client.provider_of("claude-sonnet-4-20250514") == "anthropic"

    def test_gemini_with_prefix(self):
        assert llm_client.provider_of("gemini/gemini-2.5-pro") == "gemini"

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

    def test_tier_passed_through(self, mock_litellm):
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="priority",
        )
        call_kwargs = mock_litellm.call_args.kwargs
        assert call_kwargs.get("service_tier") == "priority"

    def test_standard_tier_not_sent(self, mock_litellm):
        """Standard tier is the default — don't send it to save bandwidth."""
        mock_litellm.return_value = make_api_response("ok")
        llm_client.llm_complete(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "x"}],
            api_key="k",
            tier="standard",
        )
        call_kwargs = mock_litellm.call_args.kwargs
        assert "service_tier" not in call_kwargs

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
        assert "standard" in llm_client.TIERS_BY_PROVIDER["anthropic"]
        assert "priority" in llm_client.TIERS_BY_PROVIDER["anthropic"]

    def test_gemini_tiers(self):
        assert "standard" in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "priority" in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "flex" in llm_client.TIERS_BY_PROVIDER["gemini"]
        assert "batch" in llm_client.TIERS_BY_PROVIDER["gemini"]
