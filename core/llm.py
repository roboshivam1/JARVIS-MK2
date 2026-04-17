# =============================================================================
# core/llm.py — The LLM Abstraction Layer
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Every LLM provider (Google, Anthropic, OpenAI) has a completely different
# Python SDK with different method names, message formats, tool schemas,
# and response structures. Without this file, every agent and the orchestrator
# would contain provider-specific code scattered everywhere.
#
# This file implements the PROVIDER PATTERN — a single consistent interface
# that the rest of JARVIS talks to. Swap the provider here, nothing else
# changes. The orchestrator, planner, and every agent call the same three
# functions regardless of which company's model is running underneath.
#
# THE THREE CALL SHAPES:
# Every LLM interaction in JARVIS is one of:
#   1. chat()       — conversation history in, text response out
#   2. tool_call()  — conversation + tool schemas in, tool decisions out
#   3. structured() — prompt in, guaranteed parsed JSON out
#
# CURRENT STATE:
#   Provider: Google Gemini (works with UPI, no international card needed)
#   Orchestrator model: gemini-2.0-flash
#   Agent model: llama3.1:8b (local Ollama — unchanged)
#
# SWITCHING TO ANTHROPIC LATER:
#   1. pip install anthropic
#   2. Add ANTHROPIC_API_KEY to .env
#   3. In config.py: set ACTIVE_PROVIDER = "anthropic"
#   4. That's it. Zero other changes needed.
#
# SWITCHING TO OPENAI:
#   Same process — set ACTIVE_PROVIDER = "openai"
#
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Any

# ── Google Gemini ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
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
# LLMResponse — The Unified Response Object
# =============================================================================
#
# Every call to this module returns an LLMResponse regardless of provider.
# This means the orchestrator never needs to know whether the response came
# from Google or Anthropic — it always accesses the same fields.
#
# Fields:
#   text        — the model's text reply (None if it chose to call tools instead)
#   tool_calls  — list of tool call dicts if model requested tools (else empty list)
#   raw         — the original provider response object, in case you need it
#
# A response will have EITHER text OR tool_calls populated, rarely both.
# This mirrors how LLMs actually work — they either reply in text or request
# a tool call. They don't usually do both in one response.
# =============================================================================

class LLMResponse:
    def __init__(
        self,
        text: str | None           = None,
        tool_calls: list[dict]     = None,
        raw: Any                   = None,
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
# Message Format
# =============================================================================
#
# Internally we use a simple, provider-agnostic message format:
#   {"role": "user" | "assistant" | "tool", "content": "..."}
#
# Each provider adapter converts this to their own format before calling
# the API, and converts the response back to LLMResponse.
#
# This dict format is close to the OpenAI/Anthropic standard, which makes
# the adapters simple. Google is the odd one out — it uses a different
# structure — so the Google adapter does the most conversion work.
# =============================================================================


# =============================================================================
# Provider Initialisation
# =============================================================================

def _init_google():
    if not GOOGLE_AVAILABLE:
        raise ImportError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")
    genai.configure(api_key=GOOGLE_API_KEY)


def _init_anthropic():
    if not ANTHROPIC_AVAILABLE:
        raise ImportError(
            "anthropic not installed. Run: pip install anthropic"
        )
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _init_openai():
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "openai not installed. Run: pip install openai"
        )
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")
    return openai.OpenAI(api_key=OPENAI_API_KEY)


# =============================================================================
# Tool Schema Conversion
# =============================================================================
#
# WHY THIS IS NEEDED:
# Different providers require tool/function schemas in different formats.
# Our internal format (matching Anthropic/OpenAI) looks like:
#   {
#     "type": "function",
#     "function": {
#       "name": "search_web",
#       "description": "...",
#       "parameters": { "type": "object", "properties": {...} }
#     }
#   }
#
# Google Gemini expects a completely different structure using their own
# FunctionDeclaration objects. The converters below handle this translation
# so the agents can define tools once in the standard format and use them
# with any provider.
# =============================================================================

def _tools_to_google(tools: list[dict]) -> list:
    """
    Converts our standard tool schema list into Google's FunctionDeclaration
    format. Returns a list of genai.Tool objects.
    """
    if not tools:
        return []

    declarations = []
    for tool in tools:
        fn     = tool.get("function", tool)  # handle both wrapped and unwrapped
        name   = fn["name"]
        desc   = fn.get("description", "")
        params = fn.get("parameters", {"type": "object", "properties": {}})

        declarations.append(
            genai.protos.FunctionDeclaration(
                name=name,
                description=desc,
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        k: genai.protos.Schema(
                            type=_python_type_to_google(v.get("type", "string")),
                            description=v.get("description", ""),
                        )
                        for k, v in params.get("properties", {}).items()
                    },
                    required=params.get("required", []),
                ),
            )
        )

    return [genai.Tool(function_declarations=declarations)]


def _python_type_to_google(type_str: str):
    """Maps JSON schema type strings to Google's Type enum."""
    mapping = {
        "string":  genai.protos.Type.STRING,
        "integer": genai.protos.Type.INTEGER,
        "number":  genai.protos.Type.NUMBER,
        "boolean": genai.protos.Type.BOOLEAN,
        "array":   genai.protos.Type.ARRAY,
        "object":  genai.protos.Type.OBJECT,
    }
    return mapping.get(type_str, genai.protos.Type.STRING)


def _messages_to_google(messages: list[dict]) -> tuple[str | None, list]:
    """
    Converts our standard message list into Google's format.

    Google separates the system prompt from the conversation history.
    It also uses "model" instead of "assistant" for the role name, and
    uses a Content/Part structure instead of plain dicts.

    Returns: (system_instruction_str, google_history_list)
    """
    system_instruction = None
    history            = []

    for msg in messages:
        role    = msg["role"]
        content = msg.get("content", "")

        # Google takes system prompt separately, not as a message
        if role == "system":
            system_instruction = content
            continue

        # Google uses "model" not "assistant"
        google_role = "model" if role == "assistant" else "user"

        # Convert to Google's Content structure
        if isinstance(content, str):
            history.append({
                "role":  google_role,
                "parts": [{"text": content}],
            })
        # Tool result messages need special handling
        elif role == "tool":
            history.append({
                "role":  "user",
                "parts": [{"text": f"[Tool result]: {content}"}],
            })

    return system_instruction, history


def _parse_google_response(response) -> LLMResponse:
    """
    Parses a Google GenerateContentResponse into our unified LLMResponse.
    Handles both text responses and function call responses.
    """
    candidate = response.candidates[0]
    part       = candidate.content.parts[0]

    # Check if this is a function call response
    if hasattr(part, "function_call") and part.function_call.name:
        fc = part.function_call
        return LLMResponse(
            tool_calls=[{
                "name":      fc.name,
                "arguments": dict(fc.args),
            }],
            raw=response,
        )

    # Otherwise it's a text response
    text = ""
    for p in candidate.content.parts:
        if hasattr(p, "text"):
            text += p.text

    return LLMResponse(text=text.strip(), raw=response)


# =============================================================================
# Core Public Functions
# =============================================================================
#
# These three functions are the ENTIRE public interface of this module.
# Everything else in JARVIS imports from here and calls only these.
#
# Usage:
#   from core.llm import chat, tool_call, structured
# =============================================================================


def chat(
    messages:   list[dict],
    model:      str  = None,
    provider:   str  = None,
    max_tokens: int  = 1024,
) -> LLMResponse:
    """
    Send a conversation and get a text response.
    Use this for: orchestrator responses, agent summaries, TTS text generation.

    Args:
        messages:   List of {"role": ..., "content": ...} dicts.
                    Include {"role": "system", "content": ...} as the first
                    message for the system prompt.
        model:      Override the default model. If None, uses ORCHESTRATOR_MODEL
                    from config.
        provider:   Override the active provider. If None, uses ACTIVE_PROVIDER.
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
        raise ValueError(f"Unknown provider: {_provider!r}. Choose from: google, anthropic, openai, ollama")


def tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str  = None,
    provider:   str  = None,
    max_tokens: int  = 1024,
) -> LLMResponse:
    """
    Send a conversation with available tools and get back either a tool
    call decision OR a text response (if the model decides no tool is needed).

    Use this for: agent execution loops, orchestrator delegation calls.

    Args:
        messages:   Conversation history including system prompt.
        tools:      List of tool schemas in standard format.
        model:      Override model. Defaults to ORCHESTRATOR_MODEL.
        provider:   Override provider. Defaults to ACTIVE_PROVIDER.
        max_tokens: Max response tokens.

    Returns:
        LLMResponse with either .tool_calls or .text populated.
        Check .has_tool_calls to decide which path to take.
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
        raise ValueError(f"Unknown provider: {_provider!r}")


def structured(
    prompt:     str,
    schema_hint: str = "",
    model:      str  = None,
    provider:   str  = None,
    max_tokens: int  = 1024,
) -> dict:
    """
    Send a prompt and get back a guaranteed parsed JSON dict.
    Use this for: Planner (task graph), memory consolidation extraction.

    Unlike chat() and tool_call() which return LLMResponse objects,
    this returns a plain Python dict because the caller always needs
    structured data, not a response object.

    Args:
        prompt:      The full prompt including any examples or schema instructions.
        schema_hint: Optional description of the expected JSON structure,
                     appended to the prompt to help the model.
        model:       Override model.
        provider:    Override provider.
        max_tokens:  Max response tokens.

    Returns:
        Parsed dict. Raises ValueError if response cannot be parsed as JSON
        after cleanup attempts.
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
        raise ValueError(f"Unknown provider: {_provider!r}")

    return _parse_json_response(response.text or "")


# =============================================================================
# Agent-Specific Convenience Wrapper
# =============================================================================

def agent_chat(messages: list[dict], max_tokens: int = 1024) -> LLMResponse:
    """
    Convenience wrapper that always uses the AGENT_MODEL and ollama provider.
    Use this inside specialist agents (web_agent, memory_agent, etc.) for
    their internal reasoning — keeps them local and free.
    """
    return chat(messages, model=AGENT_MODEL, provider="ollama", max_tokens=max_tokens)


def agent_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    max_tokens: int = 1024,
) -> LLMResponse:
    """
    Convenience wrapper for agent tool calls — always uses local Ollama.
    """
    return tool_call(messages, tools, model=AGENT_MODEL, provider="ollama", max_tokens=max_tokens)


# =============================================================================
# Google Gemini Adapters
# =============================================================================

def _google_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
    json_mode:  bool = False,
) -> LLMResponse:
    _init_google()
    system_instruction, history = _messages_to_google(messages)

    generation_config = genai.GenerationConfig(
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    # Start a chat session with history (all but the last message)
    # then send the last message as the current turn
    if len(history) > 1:
        chat_session = client.start_chat(history=history[:-1])
        last_message = history[-1]["parts"][0]["text"]
        response     = chat_session.send_message(last_message)
    else:
        last_message = history[0]["parts"][0]["text"] if history else ""
        response     = client.generate_content(last_message)

    return _parse_google_response(response)


def _google_tool_call(
    messages:   list[dict],
    tools:      list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    _init_google()
    system_instruction, history = _messages_to_google(messages)
    google_tools = _tools_to_google(tools)

    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        tools=google_tools,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
    )

    if len(history) > 1:
        chat_session = client.start_chat(history=history[:-1])
        last_message = history[-1]["parts"][0]["text"]
        response     = chat_session.send_message(last_message)
    else:
        last_message = history[0]["parts"][0]["text"] if history else ""
        response     = client.generate_content(last_message)

    return _parse_google_response(response)


# =============================================================================
# Anthropic Claude Adapters
# =============================================================================

def _anthropic_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
) -> LLMResponse:
    client = _init_anthropic()

    # Anthropic takes system prompt separately
    system = ""
    filtered = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            filtered.append({"role": msg["role"], "content": msg["content"]})

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
    client = _init_anthropic()

    system   = ""
    filtered = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            filtered.append({"role": msg["role"], "content": msg["content"]})

    # Anthropic tool schema is slightly different — extract the inner "function" dict
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name":         fn["name"],
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })

    kwargs = dict(model=model, max_tokens=max_tokens, messages=filtered, tools=anthropic_tools)
    if system:
        kwargs["system"] = system

    response    = client.messages.create(**kwargs)
    tool_calls  = []
    text        = ""

    for block in response.content:
        if block.type == "tool_use":
            tool_calls.append({"name": block.name, "arguments": block.input})
        elif block.type == "text":
            text += block.text

    return LLMResponse(
        text=text.strip() or None,
        tool_calls=tool_calls,
        raw=response,
    )


# =============================================================================
# OpenAI Adapters
# =============================================================================

def _openai_chat(
    messages:   list[dict],
    model:      str,
    max_tokens: int,
    json_mode:  bool = False,
) -> LLMResponse:
    client = _init_openai()

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
    client = _init_openai()

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
    messages:   list[dict],
    model:      str,
    json_mode:  bool = False,
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

    response   = ollama.chat(model=model, messages=messages, tools=tools, stream=False)
    message    = response.get("message", {})
    raw_calls  = message.get("tool_calls", [])
    tool_calls = []

    for tc in raw_calls:
        tool_calls.append({
            "name":      tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        })

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
    Robustly parses a JSON string from an LLM response.

    LLMs sometimes wrap JSON in markdown code fences like:
      ```json
      {"key": "value"}
      ```
    This function handles that, plus trailing text after the JSON object,
    plus minor formatting issues.

    Raises ValueError with a clear message if parsing genuinely fails.
    """
    if not raw.strip():
        raise ValueError("LLM returned an empty response when JSON was expected.")

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract first JSON object/array using regex
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse LLM response as JSON after 3 attempts.\n"
        f"Raw response (first 300 chars): {raw[:300]}"
    )