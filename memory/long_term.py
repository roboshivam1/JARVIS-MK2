# =============================================================================
# memory/long_term.py — Persistent Long-Term Memory Vault
# =============================================================================
#
# WHAT THIS IS:
# A persistent JSON store that survives across sessions. While short-term
# memory resets every time JARVIS restarts, this vault remembers facts about
# the user indefinitely — preferences, projects, personal details, anything
# worth keeping.
#
# This is managed by the memory_agent at runtime. The vault itself is just
# a data store — it has no LLM logic of its own. All intelligence (deciding
# what to store, what to search for) lives in the memory_agent.
#
# DATA STRUCTURE:
# The vault is a flat JSON file structured as:
# {
#   "metadata": {
#     "created": "2025-04-15T10:00:00",
#     "last_updated": "2025-04-15T10:30:00",
#     "total_facts": 42
#   },
#   "memories": [
#     {
#       "fact":         "The user prefers dark mode in all applications.",
#       "category":     "preferences",
#       "added":        "2025-04-15T10:00:00",
#       "source":       "conversation",
#       "importance":   0.7,
#       "access_count": 3
#     },
#     ...
#   ]
# }
#
# WHY A FLAT LIST INSTEAD OF DICT-OF-CATEGORIES (as in MK1)?
# MK1 stored facts as {"preferences": ["fact1"], "user_profile": ["fact2"]}.
# This looks organised but:
#   - You still iterate everything to search
#   - Adding new categories requires restructuring
#   - Sorting by importance across categories means flattening first anyway
# A flat list with a "category" field on each entry does everything the dict
# did, with simpler code and better flexibility.
# =============================================================================

from __future__ import annotations

import json
import os
import difflib
import threading
from datetime import datetime
from typing import Optional

from config import LONG_TERM_MEMORY_FILE


# =============================================================================
# Memory Entry Helpers
# =============================================================================

def _make_entry(
    fact:       str,
    category:   str,
    source:     str   = "conversation",
    importance: float = 0.5,
) -> dict:
    """
    Creates a new memory entry dict with the current timestamp.

    Args:
        fact:       The fact to store. Should be a complete, standalone sentence.
        category:   Snake_case label. e.g. "preferences", "projects", "personal"
        source:     Where this came from. "conversation", "consolidation", "manual"
        importance: 0.0 (trivial) to 1.0 (always remember this).
                    Affects search ranking and system prompt injection.
    """
    return {
        "fact":         fact.strip(),
        "category":     category.lower().strip().replace(" ", "_"),
        "added":        datetime.now().isoformat(),
        "source":       source,
        "importance":   max(0.0, min(1.0, float(importance))),
        "access_count": 0,
    }


def _similarity(a: str, b: str) -> float:
    """
    Returns a 0.0-1.0 similarity score between two strings using
    difflib's SequenceMatcher — no external dependencies required.

    Used for deduplication: two facts above DUPLICATE_THRESHOLD are
    treated as the same fact and the second one is not stored.

    Examples:
      "User likes coffee"  vs "User likes coffee"     → 1.00 (identical)
      "User enjoys coffee" vs "User likes coffee"     → ~0.73 (near-duplicate)
      "User likes coffee"  vs "User owns a MacBook"   → ~0.24 (different)
    """
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Facts with similarity above this are considered duplicates
DUPLICATE_THRESHOLD = 0.75


# =============================================================================
# MemoryVault
# =============================================================================

class MemoryVault:
    """
    Persistent long-term memory storage for JARVIS.

    Create one instance at boot and share it across the application.
    Multiple instances pointing at the same file are safe (file lock
    prevents corruption) but wasteful.

    Thread-safe: all write operations acquire self._lock before modifying
    the in-memory vault or writing to disk. Concurrent reads are safe
    without the lock since Python dict reads are atomic.
    """

    def __init__(self, filepath: str = LONG_TERM_MEMORY_FILE):
        self.filepath = filepath
        self._lock    = threading.Lock()
        self._vault   = self._load()

    # -------------------------------------------------------------------------
    # Load / Save
    # -------------------------------------------------------------------------

    def _load(self) -> dict:
        """
        Loads the vault from disk.
        Creates a fresh default vault if the file doesn't exist or is corrupted.
        Never raises — always returns a valid dict.
        """
        if not os.path.exists(self.filepath):
            print(f"[memory] No vault at {self.filepath}. Creating fresh vault.")
            return self._create_fresh()

        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            total = len(data.get("memories", []))
            print(f"[memory] Vault loaded. {total} memories found.")
            return data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[memory] Vault corrupted ({e}). Rebuilding.")
            # Back up the corrupted file — don't silently destroy data
            if os.path.exists(self.filepath):
                backup = self.filepath + ".corrupted"
                os.rename(self.filepath, backup)
                print(f"[memory] Corrupted file backed up to: {backup}")
            return self._create_fresh()

    def _create_fresh(self) -> dict:
        """
        Returns the default vault structure with two seed memories.
        The seeds give JARVIS minimal baseline context out of the box.
        """
        fresh = {
            "metadata": {
                "created":      datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_facts":  2,
            },
            "memories": [
                _make_entry(
                    fact="The user is building JARVIS — a multi-agent AI assistant system.",
                    category="user_projects",
                    importance=0.9,
                ),
                _make_entry(
                    fact=(
                        "JARVIS uses specialist agents for different tasks: web_agent for "
                        "internet search, memory_agent for recall, system_agent for macOS "
                        "control, music_agent for playback, research_agent for deep research."
                    ),
                    category="system_context",
                    importance=0.8,
                ),
            ]
        }
        self._save(fresh)
        return fresh

    def _save(self, data: Optional[dict] = None) -> None:
        """
        Atomically saves the vault to disk.

        Uses write-to-temp-then-rename pattern. This means if Python crashes
        mid-write, the old file is still intact. The rename is atomic on
        POSIX systems (macOS included) — the file either fully exists or
        doesn't, never in a partial state.
        """
        if data is None:
            data = self._vault

        data["metadata"]["last_updated"] = datetime.now().isoformat()
        data["metadata"]["total_facts"]  = len(data["memories"])

        tmp = self.filepath + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.filepath)  # Atomic on macOS/Linux
        except Exception as e:
            print(f"[memory] Save failed: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def add(
        self,
        fact:       str,
        category:   str   = "general",
        source:     str   = "conversation",
        importance: float = 0.5,
    ) -> str:
        """
        Stores a new fact in the vault.

        Runs a deduplication check first — if a similar fact already exists
        (similarity >= DUPLICATE_THRESHOLD), the new fact is not stored and
        a message indicating the existing entry is returned instead.

        Returns a status string. This is what the memory_agent returns to
        JARVIS after a store operation — JARVIS can relay this to the user
        if relevant ("I already had that noted, sir.").

        Thread-safe.
        """
        fact = fact.strip()
        if not fact:
            return "[memory] Nothing to store — empty fact provided."

        with self._lock:
            for entry in self._vault["memories"]:
                sim = _similarity(fact, entry["fact"])
                if sim >= DUPLICATE_THRESHOLD:
                    return (
                        f"[memory] Already stored (similarity {sim:.0%}): "
                        f"'{entry['fact']}'"
                    )

            new_entry = _make_entry(fact, category, source, importance)
            self._vault["memories"].append(new_entry)
            self._save()

            return f"[memory] Stored in '{new_entry['category']}': '{fact}'"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Searches the vault and returns the most relevant memory entries.

        SCORING ALGORITHM:
        Each memory is scored by three components:

        1. Word match score (0.0–1.0)
           The query is split into words. Each word is checked against the
           fact text. Score = matched_words / total_query_words.
           "coffee morning" against "user drinks espresso each morning" would
           score 0.5 (1 of 2 words matched: "morning").

        2. Importance bonus (0.0–0.3)
           Scales with the fact's stored importance value. A fact with
           importance=1.0 gets +0.3 added to its score. This ensures that
           highly important facts surface over trivial ones when scores
           are otherwise close.

        3. Recency bonus (0.0–0.1)
           Facts added in the last 30 days get a small boost. This helps
           recent information surface over stale information when both are
           equally relevant.

        Returns entries sorted by final score, highest first.
        Returns empty list if no facts match any query word.
        """
        if not query.strip():
            return []

        query_words = set(query.lower().split())
        scored      = []

        for entry in self._vault["memories"]:
            fact_lower   = entry["fact"].lower()
            word_matches = sum(1 for w in query_words if w in fact_lower)
            if word_matches == 0:
                continue

            word_score        = word_matches / len(query_words)
            importance_bonus  = entry.get("importance", 0.5) * 0.3

            try:
                added     = datetime.fromisoformat(entry["added"])
                days_old  = (datetime.now() - added).days
                recency   = max(0.0, (30 - days_old) / 30) * 0.1
            except (ValueError, KeyError):
                recency   = 0.0

            scored.append((word_score + importance_bonus + recency, entry))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:top_k]]

        # Increment access_count for returned facts
        with self._lock:
            for entry in results:
                entry["access_count"] = entry.get("access_count", 0) + 1
            self._save()

        return results

    def search_formatted(self, query: str, top_k: int = 5) -> str:
        """
        Same as search() but returns a formatted string for the LLM to read.
        Used by memory_agent when it needs to pass search results back to JARVIS.

        Format example:
          Found 2 memories for 'coffee':
            [preferences] User prefers espresso in the morning. (added 3 days ago)
            [health] User is cutting back on caffeine. (added 1 day ago)
        """
        results = self.search(query, top_k)

        if not results:
            return f"[memory] No memories found matching '{query}'."

        lines = [f"Found {len(results)} memories for '{query}':"]
        for entry in results:
            try:
                added    = datetime.fromisoformat(entry["added"])
                days_old = (datetime.now() - added).days
                if days_old == 0:
                    age = "today"
                elif days_old == 1:
                    age = "yesterday"
                else:
                    age = f"{days_old} days ago"
            except (ValueError, KeyError):
                age = "unknown date"

            lines.append(
                f"  [{entry['category']}] {entry['fact']} (added {age})"
            )

        return "\n".join(lines)

    def get_core_profile(self) -> str:
        """
        Returns the most important user facts for injection into JARVIS's
        system prompt at boot.

        Pulls the top 10 highest-importance facts from "user_profile" and
        "preferences" categories. This gives JARVIS baseline knowledge about
        the user without overloading the system prompt with everything in
        the vault.

        Called once at startup by the Orchestrator when building JARVIS's
        initial system prompt.
        """
        relevant = [
            e for e in self._vault["memories"]
            if e.get("category") in ("user_profile", "preferences")
        ]
        relevant.sort(key=lambda e: e.get("importance", 0.5), reverse=True)
        top = relevant[:10]

        if not top:
            return ""

        lines = ["[Established facts about the user:]"]
        for entry in top:
            lines.append(f"  • {entry['fact']}")

        return "\n".join(lines)

    def remove(self, fact: str) -> str:
        """
        Removes a fact by fuzzy match. Used when the user says
        "forget that I mentioned X."
        """
        with self._lock:
            before = len(self._vault["memories"])
            self._vault["memories"] = [
                e for e in self._vault["memories"]
                if _similarity(fact, e["fact"]) < DUPLICATE_THRESHOLD
            ]
            removed = before - len(self._vault["memories"])
            if removed:
                self._save()
                return f"[memory] Removed {removed} matching entr{'y' if removed == 1 else 'ies'}."
            return f"[memory] No memory found matching '{fact}'."

    def get_all_categories(self) -> list[str]:
        """Returns a sorted list of all category names currently in the vault."""
        return sorted(set(e["category"] for e in self._vault["memories"]))

    def stats(self) -> str:
        """Returns a human-readable summary of vault contents."""
        total      = len(self._vault["memories"])
        categories = self.get_all_categories()
        breakdown  = {
            cat: sum(1 for e in self._vault["memories"] if e["category"] == cat)
            for cat in categories
        }
        lines = [f"[memory] {total} memories across {len(categories)} categories:"]
        for cat, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MemoryVault("
            f"path={self.filepath!r}, "
            f"memories={len(self._vault['memories'])}, "
            f"categories={len(self.get_all_categories())}"
            f")"
        )