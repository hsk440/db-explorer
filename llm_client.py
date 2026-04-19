"""
Provider-agnostic LLM client wrapper built on LiteLLM.

Normalizes Anthropic + Gemini response shapes into a uniform `LLMResponse`
that the rest of the app can consume without caring which provider is active.

Key features:
- Supports Anthropic (Claude) and Google Gemini
- Tool/function calling via OpenAI-format (LiteLLM translates both ways)
- Prompt caching via Anthropic-style `cache_control` (auto-translated for Gemini)
- Tier selection: standard / priority / flex / batch
- Stop-reason normalization: end_turn | tool_use | max_tokens | refusal | unknown
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("db_explorer")


# ---------------------------------------------------------------------------
# Provider / model catalog
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS = {
    # Current generation (April 2026) — verified against
    # https://platform.claude.com/docs/en/about-claude/models/overview
    # NOTE: Haiku intentionally excluded per project policy — only Opus & Sonnet.
    "Claude Opus 4.7":   "claude-opus-4-7",              # current flagship
    "Claude Sonnet 4.6": "claude-sonnet-4-6",            # current best-value
    # Legacy (still available, not deprecated)
    "Claude Sonnet 4.5": "claude-sonnet-4-5-20250929",
    "Claude Opus 4.6":   "claude-opus-4-6",
}

GEMINI_MODELS = {
    # Gemini 3.x (current) — verified against https://ai.google.dev/gemini-api/docs/models
    "Gemini 3.1 Pro Preview":        "gemini/gemini-3.1-pro-preview",
    "Gemini 3 Flash":                "gemini/gemini-3-flash-preview",
    "Gemini 3.1 Flash-Lite Preview": "gemini/gemini-3.1-flash-lite-preview",
    # Gemini 2.5 (legacy but still supported)
    "Gemini 2.5 Pro":                "gemini/gemini-2.5-pro",
    "Gemini 2.5 Flash":              "gemini/gemini-2.5-flash",
    "Gemini 2.5 Flash-Lite":         "gemini/gemini-2.5-flash-lite",
}

# User-facing tier labels per provider. "Batch" is intentionally omitted —
# it is a separate asynchronous API endpoint, not a per-request service_tier.
TIERS_BY_PROVIDER = {
    "anthropic": ["standard", "priority"],
    "gemini":    ["standard", "priority", "flex"],
}

# Internal mapping to the exact value sent over the wire.
# Anthropic API service_tier accepts: "auto" | "standard_only" (request values).
# Gemini API service_tier accepts: "priority" | "flex" (omit for default/standard).
TIER_API_VALUE = {
    "anthropic": {
        "standard": "standard_only",   # never use priority capacity
        "priority": "auto",            # prefer priority, fall back to standard
    },
    "gemini": {
        "standard": None,              # omit (default)
        "priority": "priority",
        "flex":     "flex",
    },
}


# ---------------------------------------------------------------------------
# Smart routing picks — Claude preferred
# ---------------------------------------------------------------------------

SMART_ROUTING = {
    # Anthropic: Sonnet + Opus only (no Haiku per project policy).
    # The "sql" task drives both the routing call and the multi-step planner —
    # planning quality matters most, so we use Opus there despite the cost.
    "anthropic": {
        "sql":            "claude-opus-4-7",     # routing + planner (quality > cost)
        "conversational": "claude-sonnet-4-6",
        "analyze":        "claude-sonnet-4-6",
        "agent":          "claude-sonnet-4-6",   # best agentic capability
    },
    "gemini": {
        "sql":            "gemini/gemini-2.5-flash-lite",
        "conversational": "gemini/gemini-2.5-flash",
        "analyze":        "gemini/gemini-2.5-flash",
        "agent":          "gemini/gemini-3.1-pro-preview",  # most capable
    },
}


# Set of (provider, model) pairs known to reject the service_tier param.
# Populated on the first rejection; subsequent calls skip the param entirely
# to avoid re-triggering the warning log on every call.
_TIER_UNSUPPORTED: set[tuple[str, str]] = set()


def reset_tier_cache() -> None:
    """Clear the service_tier-unsupported cache. Mainly for tests."""
    _TIER_UNSUPPORTED.clear()


def smart_pick_model(provider: str, task: str) -> str:
    """Return the default model for a given provider + task type."""
    provider_map = SMART_ROUTING.get(provider, SMART_ROUTING["anthropic"])
    return provider_map.get(task, provider_map["conversational"])


def provider_of(model_id: str) -> str:
    """Infer provider from a LiteLLM model id."""
    if model_id.startswith("gemini/") or model_id.startswith("gemini-"):
        return "gemini"
    return "anthropic"


# ---------------------------------------------------------------------------
# Normalized response types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict

    def to_openai_assistant_format(self) -> dict:
        """Format for appending back to messages as assistant turn."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.input),
            },
        }


# Normalized stop reasons
STOP_END_TURN = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_UNKNOWN = "unknown"


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # one of the STOP_* constants above
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    raw: Any = None  # the underlying provider response, for debugging

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Stop-reason normalization
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    # LiteLLM / OpenAI values
    "stop": STOP_END_TURN,
    "tool_calls": STOP_TOOL_USE,
    "function_call": STOP_TOOL_USE,  # legacy
    "length": STOP_MAX_TOKENS,
    "content_filter": STOP_REFUSAL,
    # Anthropic-native values (LiteLLM sometimes passes through)
    "end_turn": STOP_END_TURN,
    "tool_use": STOP_TOOL_USE,
    "max_tokens": STOP_MAX_TOKENS,
    "refusal": STOP_REFUSAL,
    # Gemini values
    "STOP": STOP_END_TURN,
    "MAX_TOKENS": STOP_MAX_TOKENS,
    "SAFETY": STOP_REFUSAL,
    "BLOCKLIST": STOP_REFUSAL,
    "PROHIBITED_CONTENT": STOP_REFUSAL,
    "RECITATION": STOP_REFUSAL,
}


def normalize_stop_reason(raw: Optional[str]) -> str:
    if not raw:
        return STOP_UNKNOWN
    return _FINISH_REASON_MAP.get(str(raw), STOP_UNKNOWN)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def llm_complete(
    model: str,
    messages: list,
    api_key: str,
    system: Optional[Any] = None,
    max_tokens: int = 4096,
    tools: Optional[list] = None,
    tier: str = "standard",
) -> LLMResponse:
    """
    Single-turn completion call.

    Args:
        model: LiteLLM model id (e.g. "claude-sonnet-4-20250514" or "gemini/gemini-2.5-pro")
        messages: list of {role, content} or {role, tool_calls} dicts (OpenAI format)
        api_key: API key for the provider
        system: system prompt — str, or list of {type, text, cache_control?} blocks
        max_tokens: output token cap
        tools: list of OpenAI-format function tool definitions (converted from our
               Anthropic-style schema by _convert_tools_to_openai)
        tier: "standard" | "priority" | "flex" | "batch"
    """
    import litellm

    # Build messages with system prompt prepended
    full_messages = []
    if system is not None:
        if isinstance(system, str):
            full_messages.append({"role": "system", "content": system})
        else:
            # List of content blocks — pass through (LiteLLM supports this)
            full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    # Convert our Anthropic-style tool schema to OpenAI format for LiteLLM
    openai_tools = _convert_tools_to_openai(tools) if tools else None

    # Map our UI-facing tier (standard/priority/flex) to the actual wire value
    # expected by each provider's API. Gemini uses "priority"/"flex" directly,
    # Anthropic uses "auto"/"standard_only", and some tiers mean "omit the param".
    provider = provider_of(model)
    api_tier = TIER_API_VALUE.get(provider, {}).get(tier)

    # Skip service_tier entirely if we've already learned this provider/model
    # pair rejects it (avoids repeating the warning log on every single call).
    if api_tier is not None and (provider, model) in _TIER_UNSUPPORTED:
        api_tier = None

    # Build kwargs
    kwargs = {
        "model": model,
        "messages": full_messages,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }
    if openai_tools:
        kwargs["tools"] = openai_tools
    if api_tier is not None:
        kwargs["service_tier"] = api_tier

    t0 = time.time()
    try:
        response = litellm.completion(**kwargs)
    except Exception as e:
        # Graceful fallback: some LiteLLM / provider combinations don't accept
        # service_tier. Rather than surfacing a confusing error, drop the param,
        # remember it, and retry once.
        err_str = str(e).lower()
        if api_tier is not None and ("service_tier" in err_str or "unsupported" in err_str
                                    or "unknown" in err_str or "unexpected" in err_str
                                    or "bad request" in err_str):
            _TIER_UNSUPPORTED.add((provider, model))
            logger.warning(
                f"service_tier not supported by {provider}/{model} — caching and "
                f"dropping for future calls. Original error: {e}"
            )
            kwargs.pop("service_tier", None)
            response = litellm.completion(**kwargs)
        else:
            raise
    elapsed = time.time() - t0

    # Parse response
    try:
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"LLM response missing expected fields: {e}")
        raise

    # Text content
    text = (getattr(message, "content", None) or "").strip() if getattr(message, "content", None) else ""

    # Tool calls (OpenAI format -> our ToolCall)
    tool_calls = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    for tc in raw_tool_calls:
        try:
            args_raw = tc.function.arguments
            args_dict = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse tool call args: {e} | raw={args_raw!r}")
            args_dict = {"_parse_error": str(e), "_raw": str(args_raw)}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args_dict))

    stop_reason = normalize_stop_reason(finish_reason)
    # If we got tool calls but the finish_reason didn't match, normalize it
    if tool_calls and stop_reason == STOP_END_TURN:
        stop_reason = STOP_TOOL_USE

    # Token counts (LiteLLM standard fields)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    # Cache-hit tokens — LiteLLM exposes this under prompt_tokens_details.cached_tokens
    cached = 0
    if usage:
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            cached = getattr(details, "cached_tokens", 0) or 0

    logger.info(
        f"LLM CALL OK | {model} | tier={tier} | {elapsed:.2f}s | "
        f"tokens in={input_tokens} cached={cached} out={output_tokens} | "
        f"stop={stop_reason}"
    )

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        raw=response,
    )


# ---------------------------------------------------------------------------
# Tool schema conversion (our Anthropic-style -> OpenAI function format)
# ---------------------------------------------------------------------------

def _convert_tools_to_openai(tools: list) -> list:
    """
    Our skills.TOOL_DEFINITIONS uses Anthropic-style:
        {name, description, input_schema}
    LiteLLM wants OpenAI function-calling format:
        {type: "function", function: {name, description, parameters}}
    """
    converted = []
    for t in tools:
        if "function" in t:
            # Already OpenAI-format
            converted.append(t)
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


# ---------------------------------------------------------------------------
# Message construction helpers for the agent loop
# ---------------------------------------------------------------------------

def build_assistant_tool_call_message(text: str, tool_calls: list[ToolCall]) -> dict:
    """Build an assistant message representing tool calls (OpenAI format)."""
    msg = {"role": "assistant"}
    if text:
        msg["content"] = text
    else:
        msg["content"] = None
    if tool_calls:
        msg["tool_calls"] = [tc.to_openai_assistant_format() for tc in tool_calls]
    return msg


def build_tool_result_message(tool_call_id: str, content: str) -> dict:
    """Build a tool result message (OpenAI format uses role='tool')."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content if isinstance(content, str) else json.dumps(content, default=str),
    }
