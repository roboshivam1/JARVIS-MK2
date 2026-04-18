# =============================================================================
# config.py — The Single Source of Truth for JARVIS MK2
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# PROVIDER SELECTION
#
# This is the ONE value you change to switch the entire system between
# LLM providers. Everything else in the codebase imports from core/llm.py
# which reads this value and routes accordingly.
#
# Options:  "google"     ← current (works with UPI, no intl card needed)
#           "anthropic"  ← switch to this when you get international access
#           "openai"     ← alternative to anthropic
#           "ollama"     ← fully local (for testing without any API)
# -----------------------------------------------------------------------------

ACTIVE_PROVIDER = "google"


# -----------------------------------------------------------------------------
# API KEYS — read from .env, never hardcoded
#
# Your .env file should look like:
#   GOOGLE_API_KEY=AIza...
#   GROQ_API_KEY=gsk_...
#   ANTHROPIC_API_KEY=sk-ant-...   ← add this later
#   OPENAI_API_KEY=sk-...          ← add this later
#
# Keys that aren't set yet are just None — no error at load time.
# The LLM adapter raises a clear error only when you actually try to use
# a provider whose key is missing.
# -----------------------------------------------------------------------------

GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")   # Not needed yet
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")       # Not needed yet

# Validate only the keys that are actually needed right now
if ACTIVE_PROVIDER == "google" and not GOOGLE_API_KEY:
    raise EnvironmentError(
        "[Config] GOOGLE_API_KEY not found. Add it to your .env file.\n"
        "Get one free at: https://aistudio.google.com/app/apikey"
    )

if ACTIVE_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    raise EnvironmentError("[Config] ANTHROPIC_API_KEY not found in .env")

if ACTIVE_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise EnvironmentError("[Config] OPENAI_API_KEY not found in .env")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "[Config] GROQ_API_KEY not found. This is needed for speech-to-text.\n"
        "Get one free at: https://console.groq.com"
    )


# -----------------------------------------------------------------------------
# MODEL NAMES
#
# These are the actual model identifier strings passed to each provider's API.
# When you switch ACTIVE_PROVIDER, also update ORCHESTRATOR_MODEL to the
# equivalent model on the new provider.
#
# Current:  Google Gemini
# Future:   Uncomment the Anthropic line and comment the Google one
# -----------------------------------------------------------------------------

# ── Current (Google) ──────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL = "gemini-2.5-flash"

# ── Future (Anthropic) — swap these when you have the API key ─────────────────
# ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

# ── Future (OpenAI) ───────────────────────────────────────────────────────────
# ORCHESTRATOR_MODEL = "gpt-4o"

# ── Local Ollama — agents always run locally regardless of cloud provider ──────
AGENT_MODEL         = "llama3.1:8b"
CONSOLIDATION_MODEL = "llama3.1:8b"
VISION_MODEL        = "llava"


# -----------------------------------------------------------------------------
# SPEECH I/O
# -----------------------------------------------------------------------------

STT_MODEL       = "whisper-large-v3-turbo"
STT_SAMPLE_RATE = 16000
TTS_MODEL       = "kokoro"
TTS_VOICE       = "am_michael"
TTS_FALLBACK    = "say"             # macOS built-in, zero-dependency fallback


# -----------------------------------------------------------------------------
# MEMORY
# -----------------------------------------------------------------------------

LONG_TERM_MEMORY_FILE   = "memory/jarvis_memory.json"
PENDING_TRANSCRIPT_FILE = "memory/pending_transcript.txt"
FULL_HISTORY_LOG_FILE   = "logs/full_history.txt"
MAX_CONVERSATION_TURNS  = 10
CONSOLIDATION_CHUNK_SIZE = 30


# -----------------------------------------------------------------------------
# AGENT SYSTEM
# -----------------------------------------------------------------------------

MAX_AGENT_ITERATIONS = 5
MAX_REPLAN_ATTEMPTS  = 3

# Greek name → internal agent name
# Dispatcher uses this to resolve aliases so JARVIS understands both
# "ask web_agent to..." and "ask HERMES to..."
AGENT_ALIASES = {
    "hermes":     "web_agent",
    "mnemosyne":  "memory_agent",
    "hephaestus": "system_agent",
    "apollo":     "music_agent",
    "athena":     "research_agent",
}

AGENT_REGISTRY = {
    "web_agent": {
        "description": "Searches the internet, scrapes web pages, retrieves live online information.",
        "best_for":    ["web search", "current news", "live data", "reading a webpage"],
    },
    "memory_agent": {
        "description": "Reads and writes to JARVIS's long-term memory vault.",
        "best_for":    ["remembering facts", "recalling past conversations", "storing preferences"],
    },
    "system_agent": {
        "description": "Controls macOS — opens apps, adjusts volume, takes screenshots.",
        "best_for":    ["opening applications", "system commands", "screenshots"],
    },
    "music_agent": {
        "description": "Controls music playback — plays songs, playlists, artists via Apple Music.",
        "best_for":    ["playing music", "pause", "skip", "search songs", "playlists"],
    },
    "research_agent": {
        "description": "Performs deep multi-step research across multiple sources and synthesises findings.",
        "best_for":    ["research", "comparisons", "detailed analysis", "summarising topics"],
    },
}


# -----------------------------------------------------------------------------
# DIRECTORY SETUP
# -----------------------------------------------------------------------------

REQUIRED_DIRECTORIES = ["memory", "logs"]

def ensure_directories():
    """Creates required directories if they don't exist. Call once at boot."""
    for directory in REQUIRED_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)