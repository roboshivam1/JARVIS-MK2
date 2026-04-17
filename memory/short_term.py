# =============================================================================
# memory/short_term.py — Short-Term Conversational Memory
# =============================================================================
#
# WHAT THIS IS:
# Manages the sliding window of active conversation history that gets sent
# to the LLM on every call. LLMs are stateless — they remember nothing
# between API calls. The only reason a conversation feels continuous is
# because we re-send the full history on every call. This class manages
# that history.
#
# =============================================================================

from __future__ import annotations

from typing import Optional

from config import MAX_CONVERSATION_TURNS


# -----------------------------------------------------------------------------
# Token Estimation
#
# We estimate token count rather than counting exactly. Exact counting
# requires a tokeniser library (adds a dependency) and is slower. The
# rough rule of 1 token per 4 characters is accurate enough to prevent
# context window overflows without adding complexity.
#
# The practical limit we enforce is well below any provider's hard limit,
# so estimation errors of ±20% don't matter.
# -----------------------------------------------------------------------------

CHARS_PER_TOKEN    = 4
MAX_HISTORY_TOKENS = 8000  # Soft limit — triggers compression above this


def _estimate_tokens(text: str) -> int:
    """Returns a rough token count: 1 token per 4 characters."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _message_tokens(msg: dict) -> int:
    """Estimates the token cost of a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        return sum(_estimate_tokens(str(block)) for block in content)
    return 1


# =============================================================================
# ShortTermMemory
# =============================================================================

class ShortTermMemory:
    """
    Manages the active message history for a single conversation session.

    Used by:
    - The Orchestrator for JARVIS's own conversation with the user
    - Each specialist agent for its isolated task execution context

    WHY EACH AGENT GETS ITS OWN INSTANCE:
    Agents run isolated tasks. If a web agent's tool results (potentially
    thousands of tokens of scraped content) mixed into JARVIS's conversation
    history, it would bloat the context window and confuse JARVIS's responses.
    Each component creates its own ShortTermMemory so contexts stay clean.

    MESSAGE FORMAT:
    We use the provider-agnostic format that llm.py understands:
        {"role": "user" | "assistant" | "system" | "tool", "content": "..."}

    The system prompt is stored separately and never mixed into self.history.
    This prevents it from being accidentally trimmed or summarised away.
    """

    def __init__(
        self,
        system_prompt: str,
        max_turns:     int  = MAX_CONVERSATION_TURNS,
        label:         str  = "memory",
    ):
        """
        Args:
            system_prompt: The persona/instructions for this context.
                           Never trimmed — always present in every API call.
            max_turns:     Number of full exchanges (user + assistant pairs)
                           to keep in active memory before compressing older ones.
            label:         Human-readable identifier for debug output.
                           Use something like "orchestrator", "web_agent", etc.
        """
        self.label      = label
        self.max_turns  = max_turns
        self._system    = system_prompt
        self.history: list[dict] = []

        # Set of message indices (into self.history) that are tool-related.
        # We track these separately so we can exclude them from the transcript
        # that gets saved to disk for end-of-session consolidation.
        # Tool messages contain raw API data and JSON blobs — not useful for
        # long-term memory extraction and actively harmful to the LLM doing
        # the extraction.
        self._tool_indices: set[int] = set()

        # A compressed summary of turns that were pushed out of the active
        # window. If present, injected at the start of history as context.
        self._summary: Optional[str] = None

        # Running count of completed turns (user + assistant pairs)
        self._turn_count: int = 0

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def add(self, role: str, content: str | list, is_tool: bool = False) -> None:
        """
        Appends a message to the conversation history.

        Args:
            role:    "user", "assistant", or "tool"
            content: The message text, or a list of content blocks
            is_tool: Set True for tool call and tool result messages.
                     These are included in the LLM context (so the model
                     knows what tools were called) but excluded from the
                     transcript saved for long-term memory extraction.
        """
        index = len(self.history)
        self.history.append({"role": role, "content": content})

        if is_tool:
            self._tool_indices.add(index)

        # A turn completes when the assistant produces a non-tool response
        if role == "assistant" and not is_tool:
            self._turn_count += 1
            if self._turn_count > self.max_turns:
                self._compress_old_turns()

    def get_messages(self) -> list[dict]:
        """
        Returns the full message list ready to pass to llm.chat() or
        llm.tool_call() as the `messages` parameter.

        The system prompt is included as the first message. If a summary
        of older turns exists, it's injected right after the system prompt
        as a synthetic exchange so the LLM treats it as established context.

        All messages are clean dicts with only "role" and "content" fields —
        no internal metadata that would confuse the provider APIs.
        """
        messages = [{"role": "system", "content": self._system}]

        # Inject compressed history summary if one exists
        if self._summary:
            messages.append({
                "role":    "user",
                "content": "[Earlier context — treat as established background]"
            })
            messages.append({
                "role":    "assistant",
                "content": self._summary
            })

        messages.extend(self.history)
        return messages

    def get_transcript_lines(self) -> list[str]:
        """
        Returns only the human-readable conversational lines for the
        end-of-session transcript.

        Excludes tool messages (raw API data, JSON blobs) because:
        1. They're not meaningful facts about the user
        2. They confuse the LLM doing memory extraction
        3. They inflate the transcript size unnecessarily

        The returned lines are what get appended to the pending transcript
        file and later processed by the consolidator.
        """
        lines = []
        for i, msg in enumerate(self.history):
            if i in self._tool_indices:
                continue
            role    = msg["role"].upper()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                lines.append(f"{role}: {content.strip()}")
        return lines

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Replaces the system prompt.

        Called at boot after long-term memory loads the user's profile —
        JARVIS's system prompt gets updated to include those established facts
        so he has context from previous sessions from the very first message.
        """
        self._system = new_prompt

    def get_system_prompt(self) -> str:
        return self._system

    def clear(self) -> None:
        """Resets the memory. Called at clean session end after transcript is saved."""
        self.history       = []
        self._tool_indices = set()
        self._summary      = None
        self._turn_count   = 0

    # -------------------------------------------------------------------------
    # Compression — Summarise Old Turns Instead of Deleting Them
    #
    # WHY SUMMARISE INSTEAD OF TRUNCATE?
    # MK1 used brutal truncation:
    #   self.history = [self.history[0]] + self.history[-16:]
    # This silently deleted facts the user had mentioned 10 minutes ago.
    # JARVIS would forget things mid-session with no indication it had done so.
    #
    # Compression calls the LLM to summarise the oldest half of history into
    # a paragraph, stores that as self._summary, and keeps only the recent half
    # as active history. get_messages() then prepends the summary so the LLM
    # always has access to earlier context, just in compressed form.
    #
    # USING llm.chat() INSTEAD OF A DIRECT API CALL:
    # The previous version imported anthropic directly here. This version
    # calls llm.chat() which means compression works with whatever provider
    # is currently active — Google, Anthropic, or any future provider.
    # -------------------------------------------------------------------------

    def _compress_old_turns(self) -> None:
        """
        Summarises the oldest half of history and stores it as self._summary.
        Keeps only the recent half as active history.
        Falls back to truncation if the LLM call fails.
        """
        if len(self.history) < 4:
            return

        # Import here to avoid circular imports at module load time
        # (llm.py imports from config.py, memory files import from config.py —
        # importing llm at the top of this file would create a potential cycle)
        from core.llm import chat

        midpoint     = len(self.history) // 2
        old_messages = self.history[:midpoint]
        new_messages = self.history[midpoint:]

        # Build plain text of old messages for summarisation
        # We skip tool messages — they're not meaningful for a summary
        old_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for i, m in enumerate(old_messages)
            if i not in self._tool_indices
            and isinstance(m.get("content"), str)
        ])

        if not old_text.strip():
            # Nothing worth summarising — just truncate
            self.history = new_messages
            return

        print(f"[{self.label}] Compressing {midpoint} messages into summary...")

        try:
            summary_messages = [
                {
                    "role":    "system",
                    "content": (
                        "You compress conversation history into a brief factual summary. "
                        "Preserve: key facts the user mentioned, decisions made, tasks completed, "
                        "important context. Discard: pleasantries, filler, repetition. "
                        "Write in third person. Be concise — 3 to 6 sentences maximum."
                    )
                },
                {
                    "role":    "user",
                    "content": f"Summarise this conversation history:\n\n{old_text}"
                }
            ]

            response    = chat(messages=summary_messages, max_tokens=400)
            new_summary = (response.text or "").strip()

            if new_summary:
                # Append to existing summary rather than replacing —
                # there may have been a previous compression round already
                if self._summary:
                    self._summary = f"{self._summary}\n\nLater: {new_summary}"
                else:
                    self._summary = new_summary

                # Recalculate tool indices for the new shorter history
                # Old indices no longer valid after slicing
                old_tool_count  = sum(1 for i in self._tool_indices if i < midpoint)
                self._tool_indices = {
                    i - midpoint
                    for i in self._tool_indices
                    if i >= midpoint
                }

                self.history     = new_messages
                self._turn_count = sum(
                    1 for i, m in enumerate(new_messages)
                    if m["role"] == "assistant"
                    and i not in self._tool_indices
                )

                print(f"[{self.label}] Compression done. Active: {len(self.history)} messages.")
            else:
                raise ValueError("Compression returned empty summary")

        except Exception as e:
            # Graceful fallback — truncation is worse than compression
            # but better than crashing
            print(f"[{self.label}] Compression failed ({e}). Falling back to truncation.")
            keep             = self.max_turns * 2
            self.history     = self.history[-keep:]
            self._turn_count = self.max_turns
            # Rebuild tool indices for the truncated list
            self._tool_indices = {
                i for i in self._tool_indices
                if i >= len(self.history) - keep
            }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def token_estimate(self) -> int:
        """Rough estimate of total tokens currently in active history."""
        return sum(_message_tokens(m) for m in self.history)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"label={self.label!r}, "
            f"turns={self._turn_count}, "
            f"messages={len(self.history)}, "
            f"~{self.token_estimate()} tokens, "
            f"summarised={self._summary is not None}"
            f")"
        )