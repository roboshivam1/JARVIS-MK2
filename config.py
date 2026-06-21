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

ACTIVE_PROVIDER = "anthropic"


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
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

# ── Future (Anthropic) — swap these when you have the API key ─────────────────
# ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

# ── Future (OpenAI) ───────────────────────────────────────────────────────────
# ORCHESTRATOR_MODEL = "gpt-4o"

# ── Local Ollama — agents always run locally regardless of cloud provider ──────
AGENT_MODEL         = "llama3.1:8b"
CONSOLIDATION_MODEL = "llama3.1:8b"
VISION_MODEL        = "llava"

# ── CALLIOPE (scribe_agent) uses cloud model for writing quality ──────────────
# Writing quality is the entire value of the scribe — local models produce
# flat, generic prose. Claude Sonnet produces documents worth reading.
# Falls back to Gemini if Anthropic is not yet configured.
CALLIOPE_MODEL    = "claude-sonnet-4-6"
CALLIOPE_PROVIDER = "anthropic"

# ── DAEDALUS (coding_agent) uses cloud model — debugging is multi-step ────────
# causal reasoning where the gap between local and frontier models is largest.
# A misdiagnosed bug produces confidently WORSE code, not just a weaker answer —
# reliability matters more here than almost anywhere else in the system.
DAEDALUS_MODEL    = "claude-sonnet-4-6"
DAEDALUS_PROVIDER = "anthropic"

# Coding tasks legitimately need more rounds than other agents — write, run,
# read error, fix, run again is a normal cycle. 20 gives enough room for
# real debugging without letting a stuck loop run forever.
DAEDALUS_MAX_ITERATIONS = 20

# Per-command/script execution timeout (seconds). Prevents a hung or
# infinite-looping script from blocking the agent indefinitely.
DAEDALUS_EXEC_TIMEOUT = 30


# -----------------------------------------------------------------------------
# SPEECH I/O
# -----------------------------------------------------------------------------

STT_MODEL       = "whisper-large-v3-turbo"
STT_SAMPLE_RATE = 16000
TTS_MODEL       = "kokoro"
TTS_VOICE       = "af_sarah"
TTS_FALLBACK    = "say"             # macOS built-in, zero-dependency fallback


# -----------------------------------------------------------------------------
# MEMORY
# -----------------------------------------------------------------------------

LONG_TERM_MEMORY_FILE   = "memory/jarvis_memory.json"
PENDING_TRANSCRIPT_FILE = "memory/pending_transcript.txt"
FULL_HISTORY_LOG_FILE   = "logs/full_history.txt"
MAX_CONVERSATION_TURNS  = 10

# How many conversation turns between working memory refreshes.
# Working memory (vault.get_core_profile()) is injected directly into
# JARVIS's own system prompt at boot. This interval controls how often
# it's rebuilt mid-session so newly stored facts become part of JARVIS's
# own context without requiring a restart. Refresh is cheap (no LLM call,
# just a vault read) so a moderate interval like this has negligible cost.
WORKING_MEMORY_REFRESH_INTERVAL = 6
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
    "proteus":    "browser_agent",
    "calliope":   "scribe_agent",
    "daedalus":   "coding_agent",
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
    "browser_agent": {
        "description": (
            "Controls a real browser to complete tasks on any website — accepts connections, "
            "fills forms, downloads files, logs into accounts, navigates portals and web apps. "
            "Use for anything requiring actual browser interaction that cannot be done via API."
        ),
        "best_for": [
            "browser agent", "proteus",
            "accept", "connection requests", "linkedin",
            "portal", "college portal", "university portal",
            "google classroom", "classroom",
            "download from", "upload to",
            "fill form", "fill out", "submit form",
            "log into", "sign into", "login to",
            "booking", "checkout",
            "web app", "web application",
        ],
    },
    "scribe_agent": {
        "description": (
            "Writes, saves, and manages markdown documents. Creates project briefs, "
            "captures brainstorm sessions, writes notes and decision logs. "
            "Always documents from conversation context — reads what was discussed and "
            "structures it into a well-written markdown file."
        ),
        "best_for": [
            # Write triggers
            "write up", "write down", "document", "save this",
            "capture", "note this", "make a note",
            "create a brief", "project brief", "write a plan",
            "calliope", "scribe",
            # Document types
            "notes", "brief", "plan", "summary", "decision log",
            "brainstorm", "write these thoughts",
            # Append triggers
            "add to", "append to", "update the document",
        ],
    },
    "coding_agent": {
        "description": (
            "Writes, runs, and debugs code in a sandboxed environment. Iterates "
            "until the code actually works — writes, executes, reads errors, "
            "fixes, and re-runs. Use for scripts, automation, debugging existing "
            "code, or any task requiring real code execution."
        ),
        "best_for": [
            "write a script", "write code", "write a function",
            "debug", "fix this code", "fix the bug",
            "run this", "execute", "test this code",
            "daedalus", "coding agent",
            "automation script", "python script",
            "build a tool", "write a program",
        ],
    },
}


# -----------------------------------------------------------------------------
# DIRECTORY SETUP
# -----------------------------------------------------------------------------

# Workspace — where CALLIOPE (scribe_agent) writes all documents
WORKSPACE_DIR = "workspace"

# Sandbox — where DAEDALUS (coding_agent) reads, writes, and executes code.
# Kept entirely separate from WORKSPACE_DIR — code execution and document
# writing have different risk profiles and shouldn't share a directory.
SANDBOX_DIR = "sandbox"

REQUIRED_DIRECTORIES = [
    "memory",
    "logs",
    "workspace",
    "workspace/projects",
    "workspace/decisions",
    "workspace/brainstorms",
    "workspace/notes",
    "workspace/research",
    "sandbox",
]

def ensure_directories():
    """Creates required directories if they don't exist. Call once at boot."""
    for directory in REQUIRED_DIRECTORIES:
        os.makedirs(directory, exist_ok=True)