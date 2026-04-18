# =============================================================================
# core/llm.py — Provider-Agnostic LLM Abstraction Layer
# =============================================================================
#
# PUBLIC INTERFACE — the only three functions the rest of JARVIS calls:
#
#   chat(messages, model, provider, max_tokens)
#       Conversation in, text response out.
#
#   tool_call(messages, tools, model, provider, max_tokens)
#       Conversation + tool schemas in, tool decision or text out.
#
#   structured(prompt, schema_hint, model, provider, max_tokens)
#       Prompt in, guaranteed parsed Python dict out.
#
# CONVENIENCE WRAPPERS for agents (always use local Ollama):
#
#   agent_chat(messages, max_tokens)
#   agent_tool_call(messages, tools, max_tokens)
#
# SWITCHING PROVIDERS:
#   Change ACTIVE_PROVIDER in config.py — nothing else needs to change.
#   "google"    → google.genai (current, works with UPI)
#   "anthropic" → anthropic SDK
#   "openai"    → openai SDK
#   "ollama"    → local Ollama
#
# GOOGLE SDK NOTE:
#   This file uses `google.genai` (the new SDK, package: google-genai).
#   The old `google.generativeai` package is deprecated and no longer
#   receives updates. Install with: pip install google-genai
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Any

# ── Google Gemini (new SDK) ───────────────────────────────────────────────────
try:
    from google import genai as google_genai
    from google.genai import types as google_types
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ── Anthropic Claude ──────────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── OpenAI ────────────────────────────────────────────────────────────────────
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Ollama (local) ────────────────────────────────────────────────────────────
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config import (
    ACTIVE_PROVIDER,
    ORCHESTRATOR_MODEL,
    AGENT_MODEL,
    GOOGLE_API_KEY,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
)


# =============================================================================
# LLMResponse — Unified Return Object
# =============================================================================
#
# Every call to this module returns an LLMResponse regardless of provider.
# The caller never needs to know which SDK produced the response.
#
#   text        — model's text reply (None if it called a tool instead)
#   tool_calls  — list of {"name": str, "arguments": dict} if tools requested
#   raw         — original provider response object (escape hatch)
# =============================================================================

class LLMResponse:
    def __init__(
        self,
        text:       str | None       = None,
        tool_calls: list[dict]       = None,
        raw:        Any              = None,
    ):
        self.text       = text
        self.tool_calls = tool_calls or []
        self.raw        = raw

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_text(self) -> bool:
        return bool(self.text and self.text.strip())

    def __repr__(self) -> str:
        if self.has_tool_calls:
            names = [tc.get("name", "?") for tc in self.tool_calls]
            return f"LLMResponse(tool_calls={names})"
        preview = (self.text or "")[:60].replace("\n", " ")
        return f"LLMResponse(text={preview!r})"


# =============================================================================
# Provider Initialisation
# =============================================================================

def _get_google_client():
    """Returns an initialised google.genai Client."""
    if not GOOGLE_AVAILABLE:
        raise ImportError(
            "google-genai not installed. Run: pip install google-genai"
        )
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")
    return google_genai.Client(api_key=GOOGLE_API_KEY)


def _get_anthropic_client():
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic not installed. Run: pip install anthropic")
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _get_openai_client():
    if not OPENAI_AVAILABLE:
        raise ImportError("openai not installed. Run: pip install openai")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


# =============================================================================
# Google Format Converters
#
# The new google.genai SDK uses a different structure than the old one:
#   - Client is instantiated with api_key, not configured globally
#   - System instruction goes into GenerateContentConfig, not the model init
#   - Messages use Content(role, parts=[Part(text=...)]) objects
#   - Tool schemas use FunctionDeclaration inside Tool inside a config
#   - "assistant" role is "model" in Google's convention
# =============================================================================

def _messages_to_google(
    messages: list[dict],
) -> tuple[str | None, list]:
    """
    Splits our standard message list into:
      - system_instruction (str | None) — extracted from role="system" message
      - contents (list) — remaining messages as google_types.Content objects

    Google's API takes the system instruction separately from the conversation
    history, so we need to pull it out before building the contents list.
    """
    system_instruction = None
    contents           = []

    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "") or ""

        if role == "system":
            system_instruction = content
            continue

        # Google uses "model" for assistant turns
        google_role = "model" if role == "assistant" else "user"

        if isinstance(content, str) and content.strip():
            contents.append(
                google_types.Content(
                    role=google_role,
                    parts=[google_types.Part(text=content)],
                )
            )
        elif role == "tool":
            # Tool results get wrapped as user-side context
            contents.append(
                google_types.Content(
                    role="user",
                    parts=[google_types.Part(text=f"[Tool result]: {content}")],
                )
            )

    return system_instruction, contents


def _tools_to_google(tools: list[dict]) -> list:
    """
    Converts our standard tool schema list into a list containing one
    google_types.Tool object with all FunctionDeclarations inside it.

    Our standard format:
        [{"type": "function", "function": {"name": ..., "parameters": ...}}]

    Google's format:
        [Tool(function_declarations=[FunctionDeclaration(name=..., parameters=Schema(...))])]
    """
    if not tools:
        return []

    declarations = []
    for tool in tools:
        fn     = tool.get("function", tool)
        name   = fn.get("name", "")
        desc   = fn.get("description", "")
        params = fn.get("parameters", {"type": "object", "properties": {}})

        # Build property schemas
        properties = {}
        for prop_name, prop_info in params.get("properties", {}).items():
            type_str = prop_info.get("type", "string")

            # Handle array type with items
            if type_str == "array":
                items_schema = google_types.Schema(
                    type=_str_to_google_type("string")
                )
                if isinstance(prop_info.get("items"), dict):
                    items_schema = google_types.Schema(
                        type=_str_to_google_type(
                            prop_info["items"].get("type", "string")
                        )
                    )
                properties[prop_name] = google_types.Schema(
                    type=google_types.Type.ARRAY,
                    description=prop_info.get("description", ""),
                    items=items_schema,
                )
            else:
                schema_kwargs = dict(
                    type=_str_to_google_type(type_str),
                    description=prop_info.get("description", ""),
                )
                # Handle enum values if present
                if "enum" in prop_info:
                    schema_kwargs["enum"] = prop_info["enum"]

                properties[prop_name] = google_types.Schema(**schema_kwargs)

        param_schema = google_types.Schema(
            type=google_types.Type.OBJECT,
            properties=properties,
            required=params.get("required", []),
        )

        declarations.append(
            google_types.FunctionDeclaration(
                name=name,
                description=desc,
                parameters=param_schema,
            )
        )

    return [google_types.Tool(function_declarations=declarations)]


def _str_to_google_type(type_str: str):
    """Maps JSON schema type strings to google_types.Type enum values."""
    mapping = {
        "string":  google_types.Type.STRING,
        "integer": google_types.Type.INTEGER,
        "number":  google_types.Type.NUMBER,
        "boolean": google_types.Type.BOOLEAN,
        "array":   google_types.Type.ARRAY,
        "object":  google_types.Type.OBJECT,
    }
    return mapping.get(type_str, google_types.Type.STRING)


def _parse_google_response(response) -> LLMResponse:
    """
    Converts a google.genai GenerateContentResponse into our LLMResponse.

    Checks for function calls first (tool use), then falls back to text.
    Handles multi-part responses by concatenating all text parts.
    """
    try:
        candidate = response.candidates[0]
        parts      = candidate.content.parts

        # Check for function call in any part
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                # fc.args is a MapComposite — convert to plain dict
                arguments = dict(fc.args) if fc.args else {}
                return LLMResponse(
                    tool_calls=[{"name": fc.name, "arguments": arguments}],
                    raw=response,
                )

        # No function call — concatenate all text parts
        text = "".join(
            part.text for part in parts
            if hasattr(part, "text") and part.text
        )
        return LLMResponse(text=text.strip(), raw=response)

    except (IndexError, AttributeError) as e:
        # Malformed response — return empty text rather than crashing
        return LLMResponse(text="", raw=response)


# =============================================================================
# Public Interface — The Three Core Functions
# =============================================================================

def chat(
    messages:   list[dict],
    model:      str  = None,
    provider:   str  = None,
    max_tokens: int  = 1024,
) -> LLMResponse:
    """
    Send a conversation and get a text response.

    Args:
        messages:   List of {"role": "system"|"user"|"assistant", "content": str}
        model:      Model identifier. Defaults to ORCHESTRATOR_MODEL.
        provider:   "google", "anthropic", "openai", or "ollama".
                    Defaults to ACTIVE_PROVIDER from config.
        max_tokens: Maximum tokens in the response.

    Returns:
        LLMResponse with .text populated.
    """
    _provider = provider or ACTIVE_PROVIDER
    _model    = model    or ORCHESTRATOR_MODEL

    if _provider == "google":
        return _google_chat(messages, _model, max_tokens)
    elif _provider == "anthropic":
        return _anthropic_chat(messages, _model, max_tokens)
    elif _provider == "openai":
        return _openai_chat(messages, _model, max_tokens)
    elif _provider == "ollama":
        return _ollama_chat(messages, _model)
    else:
        raise ValueError(
            f"Unknown provider {_provider!r}. "
            f"Choose from: google, anthropic, openai, ollama"
        )


def tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str  = None,
    provider:   str  = None,
    max_tokens: int  = 1024,
) -> LLMResponse:
    """
    Send a conversation with tool schemas and get back a tool decision or text.

    Returns LLMResponse with either .tool_calls or .text populated.
    Check .has_tool_calls to decide which to use.
    """
    _provider = provider or ACTIVE_PROVIDER
    _model    = model    or ORCHESTRATOR_MODEL

    if _provider == "google":
        return _google_tool_call(messages, tools, _model, max_tokens)
    elif _provider == "anthropic":
        return _anthropic_tool_call(messages, tools, _model, max_tokens)
    elif _provider == "openai":
        return _openai_tool_call(messages, tools, _model, max_tokens)
    elif _provider == "ollama":
        return _ollama_tool_call(messages, tools, _model)
    else:
        raise ValueError(f"Unknown provider {_provider!r}")


def structured(
    prompt:      str,
    schema_hint: str  = "",
    model:       str  = None,
    provider:    str  = None,
    max_tokens:  int  = 1024,
) -> dict:
    """
    Send a prompt and get back a guaranteed parsed Python dict.

    Appends JSON-only instructions to the prompt, then attempts to parse
    the response through three increasingly lenient strategies before giving up.

    Returns:
        Parsed dict.

    Raises:
        ValueError if the response cannot be parsed as JSON after all attempts.
    """
    _provider = provider or ACTIVE_PROVIDER
    _model    = model    or ORCHESTRATOR_MODEL

    full_prompt = prompt
    if schema_hint:
        full_prompt += f"\n\nRespond ONLY with valid JSON matching this structure:\n{schema_hint}"
    else:
        full_prompt += "\n\nRespond ONLY with valid JSON. No explanation, no markdown."

    messages = [{"role": "user", "content": full_prompt}]

    if _provider == "google":
        response = _google_chat(messages, _model, max_tokens, json_mode=True)
    elif _provider == "anthropic":
        response = _anthropic_chat(messages, _model, max_tokens)
    elif _provider == "openai":
        response = _openai_chat(messages, _model, max_tokens, json_mode=True)
    elif _provider == "ollama":
        response = _ollama_chat(messages, _model, json_mode=True)
    else:
        raise ValueError(f"Unknown provider {_provider!r}")

    return _parse_json_response(response.text or "")


# =============================================================================
# Agent Convenience Wrappers
# =============================================================================

def agent_chat(messages: list[dict], max_tokens: int = 1024) -> LLMResponse:
    """Always uses the local Ollama AGENT_MODEL. Use inside specialist agents."""
    return chat(messages, model=AGENT_MODEL, provider="ollama", max_tokens=max_tokens)


def agent_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    max_tokens: int = 1024,
) -> LLMResponse:
    """Always uses the local Ollama AGENT_MODEL. Use inside specialist agents."""
    return tool_call(
        messages, tools,
        model=AGENT_MODEL,
        provider="ollama",
        max_tokens=max_tokens,
    )


# =============================================================================
# Google Gemini Adapters (google.genai — new SDK)
# =============================================================================

def _google_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
    json_mode:  bool = False,
) -> LLMResponse:
    """
    Chat call using the new google.genai SDK.

    Key differences from old SDK:
    - Client instantiated with api_key, not global configure()
    - System instruction goes into GenerateContentConfig
    - Contents are typed Content objects, not plain dicts
    - Single client.models.generate_content() call instead of
      model.generate_content() or chat_session.send_message()
    """
    client                    = _get_google_client()
    system_instruction, contents = _messages_to_google(messages)

    config_kwargs = {"max_output_tokens": max_tokens}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    config   = google_types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return _parse_google_response(response)


def _google_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    """
    Tool call using the new google.genai SDK.

    Tools are passed inside GenerateContentConfig rather than as a
    separate constructor argument (as in the old SDK).
    AUTO mode lets the model decide when to call tools vs respond in text.
    """
    client                       = _get_google_client()
    system_instruction, contents = _messages_to_google(messages)
    google_tools                 = _tools_to_google(tools)

    config_kwargs = {
        "max_output_tokens": max_tokens,
        "tools":             google_tools,
        "tool_config":       google_types.ToolConfig(
            function_calling_config=google_types.FunctionCallingConfig(
                mode="AUTO"
            )
        ),
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    config   = google_types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    return _parse_google_response(response)


# =============================================================================
# Anthropic Claude Adapters
# =============================================================================

def _anthropic_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    client   = _get_anthropic_client()
    system   = ""
    filtered = []

    for msg in messages:
        if msg["role"] == "system":
            system = msg.get("content", "")
        else:
            filtered.append({"role": msg["role"], "content": msg.get("content", "")})

    kwargs = dict(model=model, max_tokens=max_tokens, messages=filtered)
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text     = response.content[0].text if response.content else ""
    return LLMResponse(text=text, raw=response)


def _anthropic_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    client   = _get_anthropic_client()
    system   = ""
    filtered = []

    for msg in messages:
        if msg["role"] == "system":
            system = msg.get("content", "")
        else:
            filtered.append({"role": msg["role"], "content": msg.get("content", "")})

    # Anthropic's tool schema format differs slightly from the OpenAI standard
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name":         fn["name"],
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=filtered,
        tools=anthropic_tools,
    )
    if system:
        kwargs["system"] = system

    response   = client.messages.create(**kwargs)
    tool_calls = []
    text       = ""

    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({"name": block.name, "arguments": block.input})
        elif block.type == "text":
            text += block.text

    return LLMResponse(text=text.strip() or None, tool_calls=tool_calls, raw=response)


# =============================================================================
# OpenAI Adapters
# =============================================================================

def _openai_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
    json_mode:  bool = False,
) -> LLMResponse:
    client = _get_openai_client()
    kwargs = dict(model=model, max_tokens=max_tokens, messages=messages)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    text     = response.choices[0].message.content or ""
    return LLMResponse(text=text, raw=response)


def _openai_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    client     = _get_openai_client()
    response   = client.chat.completions.create(
        model=model, max_tokens=max_tokens, messages=messages, tools=tools
    )
    msg        = response.choices[0].message
    tool_calls = []

    if msg.tool_calls:
        for tc in msg.tool_calls:
            tool_calls.append({
                "name":      tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })

    return LLMResponse(
        text=msg.content or None,
        tool_calls=tool_calls,
        raw=response,
    )


# =============================================================================
# Ollama (Local) Adapters
# =============================================================================

def _ollama_chat(
    messages:  list[dict],
    model:     str,
    json_mode: bool = False,
) -> LLMResponse:
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama not installed. Run: pip install ollama")

    kwargs = dict(model=model, messages=messages, stream=False)
    if json_mode:
        kwargs["format"] = "json"

    response = ollama.chat(**kwargs)
    text     = response["message"]["content"]
    return LLMResponse(text=text, raw=response)


def _ollama_tool_call(
    messages: list[dict],
    tools:    list[dict],
    model:    str,
) -> LLMResponse:
    if not OLLAMA_AVAILABLE:
        raise ImportError("ollama not installed. Run: pip install ollama")

    response  = ollama.chat(model=model, messages=messages, tools=tools, stream=False)
    message   = response.get("message", {})
    raw_calls = message.get("tool_calls", [])

    tool_calls = [
        {
            "name":      tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        }
        for tc in raw_calls
    ]

    return LLMResponse(
        text=message.get("content") or None,
        tool_calls=tool_calls,
        raw=response,
    )


# =============================================================================
# JSON Parsing Helper
# =============================================================================

def _parse_json_response(raw: str) -> dict:
    """
    Parses a JSON string from an LLM response through three attempts:
      1. Direct json.loads()
      2. Strip markdown code fences then parse
      3. Regex extract first {...} block then parse

    IMPORTANT: always returns a dict. If parsing succeeds but produces a
    non-dict (e.g. the model returned the string "direct" as valid JSON),
    that attempt is skipped and the next is tried. This prevents callers
    from getting a string or list back when they expect a dict.

    Raises ValueError with a clear message if all three fail.
    """
    if not raw.strip():
        raise ValueError("LLM returned an empty response when JSON was expected.")

    def _load_as_dict(text: str):
        """Parse text as JSON and return only if the result is a dict."""
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        # Valid JSON but not a dict — reject and try next strategy
        return None

    # Attempt 1: direct parse
    try:
        result = _load_as_dict(raw)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        result = _load_as_dict(cleaned)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract the first {...} block with regex
    # Note: we only look for objects ({}), not arrays ([]),
    # since callers always expect a dict
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        # Try a greedy match for nested objects
        match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            result = _load_as_dict(match.group(0))
            if result is not None:
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse LLM response as a JSON object after 3 attempts.\n"
        f"First 300 chars: {raw[:300]}"
    )