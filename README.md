# JARVIS MK2 — Multi-Agent Autonomous AI Assistant

> *"Keep your friends rich, and your enemies rich, and wait to find out which is which..."*
>
> *MK1 was a chatbot that could use tools. MK2 is an autonomous system that happens to have a voice interface.*

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [System Diagram](#system-diagram)
  - [The Three Operating Modes](#the-three-operating-modes)
  - [Component Breakdown](#component-breakdown)
- [Agents](#agents)
- [Memory System](#memory-system)
- [LLM Layer](#llm-layer)
- [Voice I/O Pipeline](#voice-io-pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running JARVIS](#running-jarvis)
- [Project Structure](#project-structure)
- [Extending JARVIS](#extending-jarvis)

---

## Overview

JARVIS MK2 is a **fully autonomous, voice-driven, multi-agent, multi-modal AI assistant** for macOS. It is not a chatbot with plugins — it is an orchestrated system where a cloud LLM acts as the strategic brain, and a fleet of local specialist agents each own a domain: web research, long-term memory, system control, music playback, and deep analysis (soon browser handling and extended. vision control too!).

**What makes it different from a simple tool-calling assistant:**

- **True task planning.** Complex requests are decomposed into a dependency graph of tasks — not a flat list of tool calls. Results flow between tasks as context.
- **Autonomous re-planning.** A Critic evaluates whether the goal was achieved after execution. If not, the Planner generates a revised plan for the remaining work.
- **Persistent memory.** Long-term memory survives across sessions. At the end of each session, an LLM extracts meaningful user facts from the conversation transcript and stores them in a vault. At next boot, JARVIS already knows who you are.
- **Pipelined voice I/O.** TTS runs two concurrent threads — synthesis and playback — so JARVIS starts speaking the first sentence while still synthesising the rest. Latency is perceptually near-zero.
- **Provider-agnostic LLM layer.** One config value switches the entire system between Google Gemini, Anthropic Claude, OpenAI, and local Ollama. No other code changes required.

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VOICE INTERFACE                             │
│   STT (Groq Whisper)         ←→         TTS (Kokoro ONNX)          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR  (JARVIS)                           │
│                                                                     │
│   classify(input) → direct | delegate | plan                        │
│                                                                     │
│   direct   ──→  chat() with conversation history                    │
│   delegate ──→  Dispatcher (single task, no planning)               │
│   plan     ──→  Planner → Dispatcher → Critic → (replan loop)       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴─────────────┐
              │                          │
              ▼                          ▼
┌─────────────────────┐    ┌────────────────────────────┐
│       PLANNER       │    │        DISPATCHER           │
│                     │    │                             │
│  goal → TaskPlan    │    │  TaskPlan → route → execute │
│  (LLM structured    │    │  context injection between  │
│   JSON output)      │    │  dependent tasks            │
└─────────────────────┘    └──────────┬─────────────────┘
                                      │
         ┌────────────┬───────────────┼──────────────┬─────────────┐
         ▼            ▼               ▼              ▼             ▼
   ┌──────────┐ ┌──────────┐  ┌───────────┐  ┌──────────┐ ┌───────────┐
   │  HERMES  │ │MNEMOSYNE │  │HEPHAESTUS │  │  APOLLO  │ │  ATHENA   │
   │web_agent │ │mem_agent │  │sys_agent  │  │mus_agent │ │res_agent  │
   └──────────┘ └──────────┘  └───────────┘  └──────────┘ └───────────┘
         │            │               │              │             │
         ▼            ▼               ▼              ▼             ▼
   DuckDuckGo    JSON Vault      osascript      Apple Music    Multi-source
   + requests    (fuzzy search)  + subprocess   iTunes API     synthesis
```

### The Three Operating Modes

The Orchestrator classifies every input into one of three modes before deciding what to do:

| Mode | When Used | What Happens |
|------|-----------|--------------|
| **direct** | Greetings, general knowledge, conversation | JARVIS answers from its own knowledge using conversation history. No agents involved. |
| **delegate** | Single-domain requests: *"play some jazz"*, *"what time is it"*, *"remember that I prefer Python"* | Skips the Planner. A single Task is routed directly to the correct agent via the Dispatcher. Faster than full planning. |
| **plan** | Multi-step goals: *"research FastAPI vs Django and save a summary to memory"* | Full Planner → Dispatcher → Critic → (optional replan) cycle. Task graph with dependency injection. |

Classification uses a plain `chat()` call with `max_tokens=20` and scans the response text for the keyword — no JSON parsing, no structured output, no failure modes from malformed responses. A keyword fallback fires if the LLM call itself fails.

### Component Breakdown

#### Orchestrator (`core/orchestrator.py`)
The top-level coordinator. Holds JARVIS's persona, conversation memory (`ShortTermMemory`), and routes between the three operating modes. The `_synthesise_response()` method converts raw agent results into natural spoken JARVIS responses after execution. The `_critique()` method evaluates whether the goal was achieved post-execution.

#### Planner (`core/planner.py`)
Decomposes a goal into a `TaskPlan` using a structured LLM call. Produces a JSON task graph where each task has an `id`, `description`, `agent_hint`, and `depends_on` list. The Planner knows nothing about execution — it only produces the graph. A `replan()` method generates a revised plan for remaining work after partial failure, told what was already completed so it doesn't duplicate effort.

#### Dispatcher (`core/dispatcher.py`)
Executes the `TaskPlan` by routing each `Task` to the correct agent. Three-tier agent resolution: (1) use `assigned_agent` if already set and valid, (2) keyword match against `AGENT_REGISTRY`'s `best_for` lists, (3) LLM-assisted routing for ambiguous tasks, (4) fallback to `web_agent`. The **scratchpad** (`dict[task_id → result]`) carries output between tasks — dependent tasks receive earlier results as injected context.

#### Task / TaskPlan (`core/task.py`)
The atomic data structures. `Task` carries `id`, `description`, `assigned_agent`, `status` (PENDING → RUNNING → DONE/FAILED/SKIPPED), `result`, `error`, `context`, and `depends_on`. `TaskPlan` is a container with goal metadata and helper methods (`success_rate()`, `completed_tasks()`, `failed_tasks()`). The `TaskStatus` enum prevents silent string typo bugs.

---

## Agents

Each agent inherits from `BaseAgent` and must implement three abstract methods:

```python
class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str: ...   # Agent's persona and rules
    def get_tools(self) -> list[dict]: ...    # JSON schema tool definitions
    def get_tool_map(self) -> dict: ...       # name → callable mapping
```

`BaseAgent.run()` handles the full inner tool loop automatically — calling the LLM, executing tool calls, looping until a text response arrives, enforcing `MAX_AGENT_ITERATIONS`, and returning a standardised result dict. All agents return the same shape: `{"status", "summary", "result", "agent"}`.

| Agent | Alias | Domain | Key Tools |
|-------|-------|--------|-----------|
| `web_agent` | HERMES | Internet | `search_web` (DuckDuckGo), `fetch_page` (BeautifulSoup), `search_and_summarise` |
| `memory_agent` | MNEMOSYNE | Long-term memory | `search_memory`, `store_memory`, `forget_memory`, `memory_stats` |
| `system_agent` | HEPHAESTUS | macOS control | `open_application`, `set_volume`, `get_battery_status`, `take_screenshot`, `analyze_screen`, `run_shortcut` |
| `music_agent` | APOLLO | Apple Music | `play_local_track`, `play_playlist` (fuzzy match), `play_global_search` (iTunes API), `playback_control`, `get_now_playing` |
| `research_agent` | ATHENA | Deep research | `search_multiple` (parallel queries), `fetch_page`, `synthesise` (LLM-in-tool) |

### Notable Design Choices

**`research_agent` — `synthesise()` as a tool:**
The research agent has a tool that itself calls the LLM. After gathering raw findings via `search_multiple`, it calls `synthesise(findings, question)` which runs a focused summarisation prompt on all gathered content. This produces consistently better structured output than asking the agent model to both research and synthesise in its final response — separation of concerns applied to prompting.

**`music_agent` — local-first fallback:**
The system prompt instructs the agent to try `play_local_track` first and automatically fall back to `play_global_search` if the local library doesn't have it. This multi-step fallback logic is exactly why agentic loops exist — a single tool call can't express conditional branching.

**`system_agent` — two control mechanisms:**
- `subprocess` + shell commands for OS-level actions (`open -a AppName`, `pmset -g batt`)
- `osascript` (AppleScript) for app-specific control (`set volume output volume 50`, music commands)

---

## Memory System

JARVIS has three memory layers:

### Short-Term Memory (`memory/short_term.py`)
A sliding window of active conversation history. LLMs are stateless — every API call receives the full history. `ShortTermMemory` manages this window with:

- **Compression, not truncation.** When the turn limit is exceeded, the oldest half of history is summarised into a compact paragraph by the LLM. This summary is injected as established context on every subsequent API call. Facts mentioned 20 minutes ago are preserved, just compressed.
- **Tool message segregation.** Tool call and result messages are tracked separately via `_tool_indices`. They're included in LLM context (so the model knows what tools ran) but excluded from the transcript saved for long-term memory extraction.

### Long-Term Memory (`memory/long_term.py`)
A flat-list JSON vault at `memory/jarvis_memory.json`. Each entry carries `fact`, `category`, `added`, `source`, `importance` (0.0–1.0), and `access_count`.

**Search scoring** combines three weighted components:
1. Word match score (primary signal)
2. Importance bonus (high-importance facts surface over trivial ones)
3. Recency bonus (last 30 days get a small boost)

**Deduplication** uses `difflib.SequenceMatcher`. Facts with ≥75% similarity are rejected as duplicates without storage. **Atomic writes** use write-to-temp-then-`os.replace()` — the file is never in a partial state if Python crashes mid-write.

**Thread safety:** all write operations acquire `self._lock` (a `threading.Lock`). Safe to write from both the main thread and the background consolidation thread simultaneously.

### Memory Consolidation (`memory/consolidator.py`)
Runs at the **start of the next boot** in a background daemon thread — JARVIS greets you immediately while consolidation happens silently.

Process:
1. Read the pending transcript from `memory/pending_transcript.txt`
2. Split into chunks of `CONSOLIDATION_CHUNK_SIZE` lines (default: 30)
3. For each chunk, call a local Ollama LLM with an extraction prompt
4. Store extracted facts in the vault (deduplication prevents duplicates)
5. Delete the transcript only if **all** chunks succeeded — partial failures preserve the transcript for retry on the next boot

Consolidation always uses local Ollama (`llama3.1:8b`) regardless of the cloud provider setting — it's a batch background job, not latency-sensitive, and should not consume cloud API quota.

---

## LLM Layer

`core/llm.py` provides a fully provider-agnostic interface. Change one value in `config.py` to switch providers:

```python
ACTIVE_PROVIDER = "google"   # ← change this
```

**Public interface (three functions):**

```python
chat(messages, model, provider, max_tokens) → LLMResponse
tool_call(messages, tools, model, provider, max_tokens) → LLMResponse
structured(prompt, schema_hint, model, provider, max_tokens) → dict
```

`structured()` guarantees a parsed Python `dict` through three escalating parse strategies: direct `json.loads()`, strip markdown fences then parse, regex-extract first `{...}` block then parse.

**Convenience wrappers for agents (always use local Ollama):**
```python
agent_chat(messages)
agent_tool_call(messages, tools)
```

**`LLMResponse` — unified return object:**
```python
response.text           # str | None — model's text reply
response.tool_calls     # list[{"name": str, "arguments": dict}]
response.has_tool_calls # bool
response.has_text       # bool
response.raw            # original provider response (escape hatch)
```

| Provider | Orchestrator Model | Notes |
|----------|--------------------|-------|
| Google | `gemini-2.5-flash` | Current default. UPI-compatible, no international card needed. |
| Anthropic | `claude-sonnet-4-6` | Uncomment in `config.py` when API key available. |
| OpenAI | `gpt-4o` | Alternative cloud option. |
| Ollama | `llama3.1:8b` | All specialist agents. Always local. |

---

## Voice I/O Pipeline

### Speech-to-Text (`in_out/stt.py`)
**Push-to-talk** via Right Shift key. Hold → speak → release → transcription returned.

Uses **Groq's Whisper API** (`whisper-large-v3-turbo`). Groq runs Whisper on custom LPUs — a 5-second clip typically transcribes in under 300ms. The free tier is generous for personal use. The API is format-compatible with OpenAI's Whisper endpoint.

Minimum recording duration: `0.5s` — accidental key taps are silently discarded without an API call.

### Text-to-Speech (`in_out/tts.py`)
**Kokoro ONNX** (local, high-quality neural TTS). macOS `say` command as zero-dependency fallback.

The pipelined architecture eliminates the inter-sentence gap that makes most TTS feel robotic:

```
Sequential (old — has gaps):
  synthesise(S1) → play(S1) → [GAP] → synthesise(S2) → play(S2)

Pipelined (current — no gaps):
  [Synthesis thread]   synthesise(S1) → synthesise(S2) → synthesise(S3)
  [Playback thread]                  play(S1) ──────→ play(S2) ──────→ play(S3)
```

While sentence N plays, sentence N+1 is already synthesised and waiting in the audio queue. The gap becomes zero.

`wait_until_done()` uses the **sentinel pattern**: a `_Sentinel` object flows through both queues; when the playback thread sees it, it sets a `threading.Event` that unblocks the caller.

### Think-and-Speak threading (`main.py`)
Each conversation turn runs two concurrent threads:

- **Think thread:** calls `orchestrator.process()`, splits response into sentences, puts them in a `queue.Queue`
- **Main thread:** reads sentences from the queue as they arrive, feeds each to `mouth.speak()`

JARVIS starts speaking the first sentence while still processing the rest. Perceived latency is the time to produce the first sentence, not the full response.

---

## Installation

### Prerequisites

- macOS (system agent and music agent use macOS-specific APIs)
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- An active internet connection for cloud LLM and Groq STT

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/jarvis-mk2.git
cd jarvis-mk2

# 2. Create and activate a virtual environment
python3 -m venv jarvis_env
source jarvis_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull local Ollama models
ollama pull llama3.1:8b          # Agent model (required)
ollama pull llava                # Vision model (optional — for analyze_screen)

# 5. Download Kokoro TTS model files
# Place kokoro-v1.0.onnx and voices-v1.0.bin in the project root
# Download from: https://github.com/thewh1teagle/kokoro-onnx/releases

# 6. Create your .env file
cp .env.example .env             # or create manually
```

### `.env` file

```env
GOOGLE_API_KEY=AIza...           # Required (current provider)
GROQ_API_KEY=gsk_...             # Required (Whisper STT)
ANTHROPIC_API_KEY=sk-ant-...     # Optional — for future switch
OPENAI_API_KEY=sk-...            # Optional — for future switch
```

Get API keys:
- **Google Gemini (free tier):** https://aistudio.google.com/app/apikey
- **Groq (free tier):** https://console.groq.com

---

## Configuration

All tuneable values live in `config.py`. Key settings:

```python
# Switch the entire system to a different cloud LLM provider
ACTIVE_PROVIDER = "google"          # "google" | "anthropic" | "openai" | "ollama"

# Models
ORCHESTRATOR_MODEL = "gemini-2.5-flash"   # Cloud model for planning, orchestration
AGENT_MODEL        = "llama3.1:8b"        # Local model for all specialist agents
CONSOLIDATION_MODEL = "llama3.1:8b"       # Local model for memory consolidation
VISION_MODEL       = "llava"              # Local vision model for screen analysis

# Voice
STT_MODEL  = "whisper-large-v3-turbo"
TTS_VOICE  = "am_michael"                 # Kokoro voice ID

# Agent execution
MAX_AGENT_ITERATIONS = 5    # Max tool-call loops per agent task
MAX_REPLAN_ATTEMPTS  = 3    # Max times the Critic can trigger a replan

# Memory
MAX_CONVERSATION_TURNS   = 10   # Short-term memory window before compression
CONSOLIDATION_CHUNK_SIZE = 30   # Transcript lines per consolidation chunk
```

---

## Running JARVIS

```bash
# Activate the virtual environment first
source jarvis_env/bin/activate

# Start JARVIS
python main.py
```

**Boot sequence:**
1. Memory vault loads (or creates fresh if first run)
2. Background memory consolidation starts (processes last session's transcript)
3. All five specialist agents initialise
4. Orchestrator boots with JARVIS persona
5. Speech I/O initialises
6. JARVIS greets you and enters the voice loop

**Push-to-talk:** Hold **Right Shift** → speak → release.

**Shutdown phrases:** *"sleep JARVIS"*, *"shut down"*, *"goodbye JARVIS"*, *"power down"*, *"go to sleep"*

**Example voice commands:**

| Request | Mode | Agent(s) |
|---------|------|----------|
| *"How are you doing today?"* | direct | — |
| *"What time is it?"* | delegate | system_agent |
| *"Play Bohemian Rhapsody"* | delegate | music_agent |
| *"Search the web for the latest Python release"* | delegate | web_agent |
| *"Remember that I prefer tabs over spaces"* | delegate | memory_agent |
| *"Research FastAPI vs Django and save your findings to memory"* | plan | research_agent → memory_agent |
| *"Find the current Bitcoin price and open a calculator"* | plan | web_agent → system_agent |

---

## Project Structure

```
jarvis-mk2/
│
├── main.py                    # Entry point — boot sequence and voice loop
├── config.py                  # All configuration (models, paths, limits, registry)
├── requirements.txt
│
├── core/
│   ├── llm.py                 # Provider-agnostic LLM abstraction layer
│   ├── orchestrator.py        # JARVIS — classify, route, synthesise response
│   ├── planner.py             # Goal → TaskPlan (LLM structured output)
│   ├── dispatcher.py          # TaskPlan → route → execute → scratchpad
│   └── task.py                # Task and TaskPlan dataclasses, TaskStatus enum
│
├── agents/
│   ├── base_agent.py          # Abstract base — tool loop, result format
│   ├── web_agent.py           # HERMES — DuckDuckGo + BeautifulSoup
│   ├── memory_agent.py        # MNEMOSYNE — long-term vault read/write
│   ├── system_agent.py        # HEPHAESTUS — macOS subprocess + osascript
│   ├── music_agent.py         # APOLLO — Apple Music + iTunes Search API
│   └── research_agent.py      # ATHENA — multi-source research + synthesis
│
├── memory/
│   ├── long_term.py           # MemoryVault — persistent JSON store
│   ├── short_term.py          # ShortTermMemory — sliding window + compression
│   ├── consolidator.py        # End-of-session LLM fact extraction
│   └── jarvis_memory.json     # The vault (auto-created on first run)
│
├── in_out/
│   ├── stt.py                 # SpeechToText — Groq Whisper push-to-talk
│   └── tts.py                 # TextToSpeech — Kokoro ONNX + afplay pipeline
│
└── tools/                     # Reserved for future shared tool modules
    ├── web_tools.py
    ├── memory_tools.py
    ├── system_tools.py
    └── vision_tools.py
```

---

## Extending JARVIS

### Adding a New Agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent`
2. Implement `get_system_prompt()`, `get_tools()`, `get_tool_map()`
3. Register in `config.py`:

```python
AGENT_REGISTRY = {
    ...
    "my_agent": {
        "description": "Does X, Y, Z.",
        "best_for": ["keyword1", "keyword2"],
    }
}
```

4. Instantiate and add to the `agents` dict in `main.py`:

```python
from agents.my_agent import MyAgent

agents = {
    ...
    "my_agent": MyAgent(),
}
```

5. Optionally add a Greek alias in `AGENT_ALIASES` in `config.py`.

The `BaseAgent.run()` loop, tool execution, error handling, result formatting, and iteration limits are all inherited automatically.

### Switching LLM Providers

Change one line in `config.py`:

```python
ACTIVE_PROVIDER = "anthropic"       # was "google"
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
```

Add the corresponding API key to `.env`. Nothing else changes.

---

## Acknowledgements

- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx) — local neural TTS
- [Groq](https://groq.com) — ultra-fast Whisper transcription
- [Ollama](https://ollama.ai) — local LLM inference
- [Google Gemini](https://aistudio.google.com) — orchestrator LLM (current default)
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — no-key web search