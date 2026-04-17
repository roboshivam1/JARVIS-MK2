# =============================================================================
# memory/consolidator.py — End-of-Session Memory Consolidation
# =============================================================================
#
# WHAT THIS DOES:
# At the end of each session, JARVIS has had a conversation containing
# potentially useful long-term facts — your preferences, your projects,
# things you mentioned about yourself. The consolidator reads the session
# transcript, extracts those facts using an LLM, and stores them in the
# MemoryVault so JARVIS remembers them in future sessions.
#
# WHEN IT RUNS:
# At the START of the next boot, in a background thread. This means:
#   - JARVIS boots immediately and greets you
#   - Consolidation happens silently in parallel
#   - You never wait for it
#
# 
# =============================================================================

import os
import json
import threading

from config import (
    PENDING_TRANSCRIPT_FILE,
    CONSOLIDATION_CHUNK_SIZE,
    AGENT_MODEL,
)
from memory.long_term import MemoryVault


# =============================================================================
# Extraction Prompt
#
# This prompt is the most critical part of the consolidator. The quality
# of everything that gets permanently stored in the vault depends on how
# well this prompt instructs the model.
#
# Key design choices:
# - Explicit DO NOT EXTRACT list prevents the model from trying to extract
#   facts from tool results and raw JSON that ends up in the transcript
# - Importance scoring enables the vault's search ranking to work well
# - Category inference keeps the vault organised without hardcoded categories
# - "Return empty list if no facts" prevents the model from hallucinating
#   facts when a chunk has no meaningful content
# =============================================================================

EXTRACTION_PROMPT = """\
You are a precise fact extractor. Read the conversation transcript below and \
extract permanent, meaningful facts about the USER ONLY.

EXTRACT facts about:
- Personal information (name, location, occupation, relationships)
- Preferences and habits (likes, dislikes, routines)
- Active projects and goals
- Technical context (tools, languages, systems they work with)
- Important decisions or plans they mentioned

DO NOT EXTRACT:
- Anything JARVIS said (only facts the user stated or implied)
- Temporary information ("I'm busy today", "just woke up")
- Questions the user asked
- Content from tool results, system messages, or JSON data
- Generic observations ("user asked a question")

For each fact assign:
- category: short snake_case label (e.g. "preferences", "projects", "personal", "technical")
- importance: 0.1 (trivial) to 1.0 (critical, must always remember)

If there are NO extractable facts in this chunk, return {"facts": []}.

Respond ONLY with valid JSON. No explanation, no markdown.

Required format:
{
  "facts": [
    {
      "fact": "The user is building a multi-agent AI system called JARVIS.",
      "category": "projects",
      "importance": 0.9
    }
  ]
}

Transcript:
"""


# =============================================================================
# Core Consolidation Logic
# =============================================================================

def consolidate_session(vault: MemoryVault) -> None:
    """
    Reads the pending transcript, processes it in chunks, extracts facts,
    and stores them in the vault.

    Safe to call from a background thread. MemoryVault has its own internal
    lock so concurrent writes from this thread and the main thread are safe.

    WHY CHUNKS?
    - Local models have limited context windows (8K tokens for llama3.1:8b)
    - A full session transcript can be thousands of lines
    - Smaller chunks produce higher-quality extraction than one huge prompt
    - If one chunk fails, others still succeed

    Args:
        vault: The shared MemoryVault instance. Passed in so all writes
               go to the same in-memory object — no file re-read needed.
    """
    if not os.path.exists(PENDING_TRANSCRIPT_FILE):
        return  # Nothing to consolidate — first run or already processed

    with open(PENDING_TRANSCRIPT_FILE, "r") as f:
        lines = f.readlines()

    # Filter out blank lines before chunking
    lines = [l for l in lines if l.strip()]

    if not lines:
        _safe_delete_transcript()
        return

    chunks = [
        "".join(lines[i: i + CONSOLIDATION_CHUNK_SIZE])
        for i in range(0, len(lines), CONSOLIDATION_CHUNK_SIZE)
    ]

    print(f"[consolidation] Processing {len(chunks)} chunk(s) from last session...")

    total_stored  = 0
    failed_chunks = 0

    for i, chunk in enumerate(chunks):
        print(f"[consolidation] Chunk {i + 1}/{len(chunks)}...")

        try:
            # Import here to avoid potential circular import issues at module load.
            # consolidator → llm → config works fine, but keeping imports lazy
            # in utility modules is a good habit.
            from core.llm import structured

            # We use provider="ollama" and the agent model explicitly here.
            # Consolidation is a batch background job — we don't want it
            # competing for the cloud provider quota or adding API cost.
            # Local Ollama is fast enough and free for this purpose.
            data = structured(
                prompt=EXTRACTION_PROMPT + chunk,
                provider="ollama",
                model=AGENT_MODEL,
                max_tokens=1024,
            )

            facts = data.get("facts", [])

            for item in facts:
                fact       = str(item.get("fact", "")).strip()
                category   = str(item.get("category", "general"))
                importance = float(item.get("importance", 0.5))

                if not fact:
                    continue

                result = vault.add(
                    fact=fact,
                    category=category,
                    source="consolidation",
                    importance=importance,
                )
                # Only print non-duplicate results to keep output clean
                if "Already stored" not in result:
                    print(f"  {result}")
                    total_stored += 1

        except ValueError as e:
            # structured() raises ValueError when JSON parsing fails completely
            print(f"[consolidation] Chunk {i + 1} — JSON parse failed: {e}")
            failed_chunks += 1

        except Exception as e:
            print(f"[consolidation] Chunk {i + 1} — unexpected error: {e}")
            failed_chunks += 1

    # Only delete the transcript if ALL chunks succeeded.
    #
    # WHY: If some chunks failed, those turns haven't been processed yet.
    # Keeping the transcript means the next boot gets another attempt.
    # The vault's deduplication ensures already-extracted facts are skipped
    # on the retry, so there's no risk of duplication.
    if failed_chunks == 0:
        _safe_delete_transcript()
        print(
            f"[consolidation] Done. {total_stored} new facts stored. "
            f"Transcript cleared."
        )
    else:
        print(
            f"[consolidation] Partial. {total_stored} facts stored. "
            f"{failed_chunks} chunk(s) failed — transcript preserved for retry."
        )


def _safe_delete_transcript() -> None:
    """Deletes the pending transcript file, silently ignoring if already gone."""
    try:
        if os.path.exists(PENDING_TRANSCRIPT_FILE):
            os.remove(PENDING_TRANSCRIPT_FILE)
    except OSError as e:
        print(f"[consolidation] Could not delete transcript: {e}")


# =============================================================================
# Background Thread Launcher
# =============================================================================

def consolidate_in_background(vault: MemoryVault) -> threading.Thread:
    """
    Launches consolidation in a background daemon thread so it doesn't
    block JARVIS's boot sequence.

    WHY A DAEMON THREAD:
    A daemon thread is automatically killed when the main program exits.
    This means if you quit JARVIS while consolidation is still running,
    the thread dies cleanly without keeping the process alive.

    The trade-off: if Python exits mid-chunk, that chunk's work is lost.
    But the transcript file is still intact, so the next boot retries it.
    In practice, consolidation finishes in a few seconds — the window
    where this matters is tiny.

    Returns the thread object so callers can optionally join() it in tests.
    """
    thread = threading.Thread(
        target=consolidate_session,
        args=(vault,),
        daemon=True,
        name="MemoryConsolidator",
    )
    thread.start()
    print("[consolidation] Running in background...")
    return thread


# =============================================================================
# Transcript Writing
# =============================================================================

def append_to_transcript(lines: list[str]) -> None:
    """
    Appends conversational lines to the pending transcript file.

    Called by the Orchestrator at the end of each conversation turn
    using the clean lines from ShortTermMemory.get_transcript_lines().

    WHY APPEND LINE-BY-LINE instead of writing everything at the end:
    If JARVIS crashes mid-session, an append-based transcript contains
    everything up to the crash. Writing at session end would lose the
    entire session's content. Appending is crash-safe.

    Args:
        lines: Output of ShortTermMemory.get_transcript_lines() —
               clean "ROLE: content" strings, no tool messages.
    """
    if not lines:
        return

    try:
        with open(PENDING_TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except OSError as e:
        print(f"[consolidation] Failed to write transcript line: {e}")