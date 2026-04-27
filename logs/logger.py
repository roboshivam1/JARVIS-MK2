# =============================================================================
# logs/logger.py — JARVIS MK2 Session Logger
# =============================================================================
#
# WHAT THIS DOES:
# Writes a permanent, human-readable record of every JARVIS interaction to
# disk. Two formats run in parallel:
#
#   1. CONVERSATION LOG (logs/conversation.log)
#      Plain text, easy to read, one entry per turn.
#      Looks like a transcript — what you said, what JARVIS said,
#      which agents were involved, how long it took.
#
#   2. STRUCTURED JSON LOG (logs/sessions.jsonl)
#      One JSON object per line (JSONL format). Machine-readable.
#      Useful for later analysis — "how often does JARVIS use each agent?",
#      "which queries take the longest?", "what errors occurred this week?"
#      Each record contains the full turn data.
#
# WHY JSONL NOT JSON?
# A single JSON file gets corrupt if JARVIS crashes mid-write (the file
# is left without a closing bracket). JSONL (one JSON object per line)
# means each line is independent — a crash during write only loses the
# current line, never the previous history. Also trivially appendable.
#
# THREAD SAFETY:
# JARVIS's think thread and main thread both touch the orchestrator.
# The logger uses a threading.Lock so concurrent writes don't interleave
# and corrupt the log files.
#
# USAGE:
#   from logs.logger import JarvisLogger
#   logger = JarvisLogger()                    # call once at boot
#   logger.log_turn(user_input, response, ...) # call after each turn
#   logger.log_error(error_msg)                # call on exceptions
# =============================================================================

from __future__ import annotations

import os
import json
import threading
from datetime import datetime
from pathlib import Path

from config import FULL_HISTORY_LOG_FILE


# =============================================================================
# Log File Paths
# =============================================================================

LOGS_DIR             = Path("logs")
CONVERSATION_LOG     = LOGS_DIR / "conversation.log"
STRUCTURED_LOG       = LOGS_DIR / "sessions.jsonl"
ERROR_LOG            = LOGS_DIR / "errors.log"


# =============================================================================
# JarvisLogger
# =============================================================================

class JarvisLogger:
    """
    Persistent session logger for JARVIS MK2.

    Creates the logs directory and files on first use.
    Thread-safe for concurrent access from think thread and main thread.

    Usage:
        logger = JarvisLogger()

        # After each conversation turn:
        logger.log_turn(
            user_input   = "Play Bohemian Rhapsody",
            response     = "Playing Bohemian Rhapsody by Queen.",
            classification = "delegate",
            agents_used  = ["music_agent"],
            duration_ms  = 1240,
        )

        # On errors:
        logger.log_error("Classification failed: ...", context="classify")
    """

    def __init__(self):
        # Create logs directory if it doesn't exist
        LOGS_DIR.mkdir(exist_ok=True)

        # Threading lock — prevents interleaved writes from concurrent threads
        self._lock = threading.Lock()

        # Write a session start marker so log files show clear session boundaries
        self._write_session_header()

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def log_turn(
        self,
        user_input:      str,
        response:        str,
        classification:  str            = "unknown",
        agents_used:     list[str]      = None,
        duration_ms:     int            = 0,
        plan_steps:      int            = 0,
        extra:           dict           = None,
    ) -> None:
        """
        Logs a complete conversation turn to both log files.

        Args:
            user_input:     What the user said.
            response:       What JARVIS responded.
            classification: "direct", "delegate", or "plan".
            agents_used:    List of agent names that were invoked.
            duration_ms:    Total processing time in milliseconds.
            plan_steps:     Number of tasks in the plan (0 for direct/delegate).
            extra:          Any additional metadata to include in the JSON log.
        """
        now       = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        agents    = agents_used or []

        with self._lock:
            self._write_conversation_entry(
                timestamp, user_input, response, classification,
                agents, duration_ms
            )
            self._write_json_entry(
                timestamp=now.isoformat(),
                user_input=user_input,
                response=response,
                classification=classification,
                agents_used=agents,
                duration_ms=duration_ms,
                plan_steps=plan_steps,
                extra=extra or {},
            )

    def log_error(self, message: str, context: str = "") -> None:
        """
        Logs an error to the error log file and the conversation log.

        Args:
            message: The error message or exception string.
            context: Where the error occurred (e.g. "classify", "synthesis").
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line      = f"[{timestamp}] ERROR"
        if context:
            line += f" ({context})"
        line += f": {message}\n"

        with self._lock:
            try:
                with open(ERROR_LOG, "a", encoding="utf-8") as f:
                    f.write(line)
                # Also note in conversation log so errors appear in context
                with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
                    f.write(line)
            except OSError as e:
                print(f"[logger] Failed to write error log: {e}")

    def log_status(self, message: str) -> None:
        """
        Logs a status/debug message to the conversation log.
        Used for notable events like agent delegations, replanning etc.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            try:
                with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
                    f.write(f"  [{timestamp}] {message}\n")
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # Private Write Methods
    # -------------------------------------------------------------------------

    def _write_session_header(self) -> None:
        """
        Writes a visible session boundary to both log files.
        Makes it easy to find the start of each JARVIS session when reading logs.
        """
        now    = datetime.now()
        border = "=" * 70
        header = (
            f"\n{border}\n"
            f"  JARVIS MK2 SESSION — {now.strftime('%A, %B %d %Y at %I:%M %p')}\n"
            f"{border}\n\n"
        )

        with self._lock:
            try:
                with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
                    f.write(header)
                # JSON log gets a lightweight session marker
                with open(STRUCTURED_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type":      "session_start",
                        "timestamp": now.isoformat(),
                    }) + "\n")
            except OSError as e:
                print(f"[logger] Could not write session header: {e}")

    def _write_conversation_entry(
        self,
        timestamp:      str,
        user_input:     str,
        response:       str,
        classification: str,
        agents_used:    list[str],
        duration_ms:    int,
    ) -> None:
        """
        Writes a human-readable turn entry to conversation.log.

        Format:
        ──────────────────────────────────────────
        [14:32:07] YOU → delegate (music_agent) [1240ms]
        Can you play Bohemian Rhapsody?

        JARVIS:
        Playing Bohemian Rhapsody by Queen, via APOLLO.
        ──────────────────────────────────────────
        """
        # Build the header line — shows classification and agents at a glance
        agent_str = ", ".join(agents_used) if agents_used else "none"
        mode_str  = classification
        if agents_used:
            mode_str += f" → {agent_str}"

        duration_str = f" [{duration_ms}ms]" if duration_ms else ""

        divider = "─" * 60
        entry   = (
            f"{divider}\n"
            f"[{timestamp}] YOU ({mode_str}){duration_str}\n"
            f"{user_input}\n\n"
            f"JARVIS:\n"
            f"{response}\n\n"
        )

        try:
            with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
                f.write(entry)
        except OSError as e:
            print(f"[logger] Failed to write conversation log: {e}")

    def _write_json_entry(self, **kwargs) -> None:
        """
        Writes a single JSON object as one line to sessions.jsonl.
        Adds a "type": "turn" field so session_start markers are distinguishable.
        """
        record = {"type": "turn", **kwargs}
        try:
            with open(STRUCTURED_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[logger] Failed to write JSON log: {e}")

    # -------------------------------------------------------------------------
    # Convenience: Read Recent Log
    # -------------------------------------------------------------------------

    def get_recent_turns(self, n: int = 10) -> list[dict]:
        """
        Returns the last n turn records from the JSON log.
        Useful for JARVIS to reference what happened recently.
        """
        if not STRUCTURED_LOG.exists():
            return []

        try:
            lines = STRUCTURED_LOG.read_text(encoding="utf-8").strip().splitlines()
            turns = []
            for line in reversed(lines):
                try:
                    record = json.loads(line)
                    if record.get("type") == "turn":
                        turns.append(record)
                        if len(turns) >= n:
                            break
                except json.JSONDecodeError:
                    continue
            return list(reversed(turns))
        except OSError:
            return []