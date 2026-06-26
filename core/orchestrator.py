# =============================================================================
# core/orchestrator.py — The Orchestrator (JARVIS)
# =============================================================================
#
# THREE OPERATING MODES:
#   direct   — answers from own knowledge, no agents needed
#   delegate — single agent, single task
#   plan     — multi-step, multiple agents, full workflow
#
# CLASSIFICATION APPROACH:
# Uses plain chat() with max_tokens=20 and scans the response text for the
# keyword. No JSON involved — eliminates all JSON parsing failure modes.
# "How are you?" → model replies "direct" → we find "direct" in text → done.
# Even if the model replies "I think this is direct" we still find the word.
# Keyword fallback runs if the LLM call fails entirely.
# =============================================================================

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from core.task import TaskPlan, TaskStatus
from core.planner import Planner
from core.dispatcher import Dispatcher
from memory.short_term import ShortTermMemory
from memory.long_term import MemoryVault
from memory.consolidator import append_to_transcript
from logs.logger import JarvisLogger
from config import (
    AGENT_REGISTRY,
    AGENT_ALIASES,
    MAX_REPLAN_ATTEMPTS,
)


# =============================================================================
# JARVIS System Prompt
# =============================================================================

JARVIS_SYSTEM_PROMPT = (
    "You are JARVIS — a sophisticated, intelligent AI assistant with dry wit, "
    "quiet confidence, and genuine capability. You are not a chatbot. "
    "You are an autonomous system.\n\n"
    "You have a team of specialist agents:\n"
    "  - HERMES    (web_agent):       Internet research and live web content\n"
    "  - MNEMOSYNE (memory_agent):    Long-term memory — deep archive search and storing facts\n"
    "  - HEPHAESTUS (system_agent):   macOS control — apps, volume, screenshots\n"
    "  - APOLLO    (music_agent):     Music playback and Apple Music control\n"
    "  - ATHENA    (research_agent):  Deep multi-source research and synthesis\n"
    "  - PROTEUS   (browser_agent):   Controls any website — downloads, forms, logged-in accounts\n"
    "  - CALLIOPE  (scribe_agent):    Writes and saves markdown documents from conversation\n"
    "  - DAEDALUS  (coding_agent):    Writes, runs, and debugs code in a sandbox\n\n"
    "Refer to agents by their Greek names in speech. Speak as JARVIS in first person.\n\n"
    "MEMORY — IMPORTANT:\n"
    "The 'Working memory' section below (if present) contains facts you already know "
    "about the user. When asked something covered there, answer directly from it — "
    "do NOT delegate to MNEMOSYNE for facts you already have in working memory. "
    "Only delegate to MNEMOSYNE for deep, specific recall not covered in working memory, "
    "or to store a brand new fact.\n\n"
    "STYLE:\n"
    "- Sharp, concise, direct. No filler. No hollow openers like 'Certainly!'.\n"
    "- Dry wit when appropriate — never forced.\n"
    "- Lead with the answer. Stop when you have said what needs saying.\n"
)


# =============================================================================
# Keyword Fallback Classifier
#
# Runs when the LLM classification call fails entirely.
# Priority: plan > delegate > direct
# =============================================================================

_DELEGATE_KEYWORDS = [
    "search", "look up", "find", "latest", "current", "news",
    "google", "browse", "web", "internet", "online",
    "remember", "recall", "forget", "store", "save that",
    "what did i", "do you know my", "what's my",
    "open ", "launch", "battery", "volume", "screenshot",
    "what time", "what's the time",
    "play ", "pause", "skip", "next track", "previous track",
    "music", "song", "playlist",
    "research ", "analyse ", "analyze ",
]

_PLAN_KEYWORDS = [
    "and then", "and save", "and store", "and remember",
    "compare", "find and", "search and", "research and",
    "look up and",
]


def _keyword_classify(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in _PLAN_KEYWORDS):
        return "plan"
    if any(kw in lower for kw in _DELEGATE_KEYWORDS):
        return "delegate"
    return "direct"


# =============================================================================
# Remember-Shortcut Pattern
#
# WHY THIS EXISTS:
# "Remember that X" is one of the most common, most unambiguous things
# said to JARVIS — there is exactly one correct interpretation and exactly
# one agent involved. Routing it through full classification (an LLM call)
# and then dispatcher agent-resolution (more routing logic, possibly
# another LLM call) is pure overhead for a phrase pattern this clear-cut.
#
# This regex matches optional leading "hey"/"jarvis"/"please" politeness
# words, then "remember" + optional "that", capturing everything after as
# the fact to store. If it matches, process() bypasses _classify() and
# the dispatcher entirely and stores the fact directly via the shared
# MemoryVault instance — see _try_remember_shortcut below.
#
# Deliberately conservative: only matches when "remember" is clearly the
# primary verb near the start of the utterance. Something like "do you
# remember what we discussed" should NOT match this (that's a recall
# question, not a store request) — the pattern requires "remember" to be
# followed directly by the fact, not preceded by "do you".
# =============================================================================

_REMEMBER_PATTERN = re.compile(
    r"^(?:hey\s+)?(?:jarvis\s*,?\s+)?(?:please\s+)?remember\s+(?:that\s+)?(.+)$",
    re.IGNORECASE,
)


# =============================================================================
# Orchestrator
# =============================================================================

class Orchestrator:
    """
    JARVIS — the top-level coordinator that talks to the user and manages agents.

    Usage:
        orchestrator = Orchestrator(agents, vault)
        response     = orchestrator.process("How are you?")
    """

    def __init__(
        self,
        agents:    dict,
        vault:     MemoryVault,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            agents:    Dict of agent_name → agent instance.
            vault:     Shared MemoryVault instance.
            on_status: Optional callback for progress narration.
                       Called with a short status string before and during
                       agent execution. In main.py this is wired to
                       mouth.speak() so JARVIS narrates what he is doing.
                       Signature: on_status(message: str) -> None
        """
        self.agents     = agents
        self.vault      = vault
        self.planner    = Planner()
        self.dispatcher = Dispatcher(agents)
        self.on_status  = on_status   # Progress narration callback
        self.logger     = JarvisLogger()

        # Working memory injected directly into JARVIS's OWN system prompt.
        # This is the fix for memory recall latency — a rich set of facts
        # lives in context with zero retrieval step, so most "what do you
        # know about X" queries get answered in _handle_direct() without
        # ever delegating to memory_agent. See long_term.get_core_profile().
        core_profile = vault.get_core_profile()
        self._base_system_prompt = JARVIS_SYSTEM_PROMPT  # without memory — kept separate so refresh can rebuild cleanly
        system = JARVIS_SYSTEM_PROMPT
        if core_profile:
            system += "\n\n" + core_profile

        self.memory = ShortTermMemory(
            system_prompt=system,
            label="orchestrator",
        )

        # Tracks turns since working memory was last refreshed.
        # New facts stored mid-session (via memory_agent.store_memory or
        # background consolidation) wouldn't otherwise reach JARVIS's own
        # context until a restart. Periodic refresh closes that gap.
        self._turns_since_memory_refresh = 0

    # -------------------------------------------------------------------------
    # Working Memory Refresh
    # -------------------------------------------------------------------------

    def _maybe_refresh_working_memory(self) -> None:
        """
        Periodically rebuilds JARVIS's system prompt with fresh working
        memory from the vault.

        WHY THIS IS NEEDED:
        Working memory is loaded once at __init__ time. If memory_agent
        stores a new fact mid-session (you say "remember that I prefer X"),
        that fact lives in the vault immediately — but JARVIS's own system
        prompt was already built and won't reflect it until this refresh runs.

        Runs every WORKING_MEMORY_REFRESH_INTERVAL turns. Cheap — it's just
        a vault read (no LLM call) and a string rebuild, not worth doing
        every single turn but worth doing periodically during long sessions.
        """
        from config import WORKING_MEMORY_REFRESH_INTERVAL

        self._turns_since_memory_refresh += 1
        if self._turns_since_memory_refresh < WORKING_MEMORY_REFRESH_INTERVAL:
            return

        self._turns_since_memory_refresh = 0
        core_profile = self.vault.refresh_working_memory()
        system = self._base_system_prompt
        if core_profile:
            system += "\n\n" + core_profile

        self.memory.update_system_prompt(system)
        print("[orchestrator] Working memory refreshed.")

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def _narrate(self, message: str) -> None:
        """
        Sends a status message to the on_status callback (mouth.speak).
        Silent no-op if no callback was provided.
        Only called for non-direct queries — direct responses need no narration.
        """
        if self.on_status:
            try:
                self.on_status(message)
            except Exception as e:
                print(f"[orchestrator] Narration callback error: {e}")

    def process(self, user_input: str) -> str:
        """
        Takes user input, classifies it, routes to the right handler,
        returns a spoken response string.

        Also handles:
        - Timing: records how long the full turn takes
        - Logging: writes every turn to conversation.log and sessions.jsonl
        - Narration: calls on_status before agent tasks so JARVIS speaks
                     a brief status line rather than going silent
        """
        print(f"\n[orchestrator] Input: {user_input}")
        start_ms = int(time.time() * 1000)

        # Refresh working memory periodically so facts stored mid-session
        # (via memory_agent or background consolidation) reach JARVIS's
        # own context without requiring a restart
        self._maybe_refresh_working_memory()

        # Fast path: explicit "remember that X" requests skip classification
        # and dispatcher routing entirely — see _try_remember_shortcut.
        shortcut_response = self._try_remember_shortcut(user_input)
        if shortcut_response is not None:
            classification = "delegate"  # for logging — this IS a memory_agent action, just fast-pathed
            response       = shortcut_response
        else:
            classification = self._classify(user_input)
            print(f"[orchestrator] Classification: {classification}")

            if classification == "direct":
                response = self._handle_direct(user_input)
            elif classification == "delegate":
                response = self._handle_delegate(user_input)
            else:
                response = self._handle_plan(user_input)

        duration_ms = int(time.time() * 1000) - start_ms

        append_to_transcript(self.memory.get_transcript_lines())

        # Log the completed turn
        agents_used = getattr(self, "_last_agents_used", [])
        plan_steps  = getattr(self, "_last_plan_steps", 0)
        self.logger.log_turn(
            user_input=user_input,
            response=response,
            classification=classification,
            agents_used=agents_used,
            duration_ms=duration_ms,
            plan_steps=plan_steps,
        )
        # Reset per-turn tracking
        self._last_agents_used = []
        self._last_plan_steps  = 0

        print(f"[orchestrator] Response ({duration_ms}ms): {response[:100]}{'...' if len(response) > 100 else ''}")
        return response

    # -------------------------------------------------------------------------
    # Classification — plain text, no JSON
    # -------------------------------------------------------------------------

    def _classify(self, user_input: str) -> str:
        """
        Classifies input as 'direct', 'delegate', or 'plan'.

        Uses a plain chat() call with max_tokens=20.
        We then scan the entire response text for the three valid words.
        This means the model can reply "direct", "I'd say direct", or
        "The classification is: direct" and all three work correctly.

        No JSON. No structured(). No parsing errors.

        Falls back to keyword matching if the LLM call fails entirely
        or returns a response containing none of the expected words.
        """
        from core.llm import chat

        prompt = (
            "Classify this user message into one category.\n\n"
            "direct   = answer from knowledge, no tools needed\n"
            "           (greetings, questions you know, explanations, math)\n\n"
            "delegate = one specific agent action needed\n"
            "           (web search, play music, open app, set volume,\n"
            "            remember something, check battery, recall a fact)\n\n"
            "plan     = multiple steps or agents needed\n"
            "           (research AND save, compare AND recommend,\n"
            "            find THEN do something else)\n\n"
            "User message: " + user_input + "\n\n"
            "Reply with a single word only: direct, delegate, or plan"
        )

        try:
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )

            text = (response.text or "").strip().lower()

            # Check for each word in the response.
            # We check "plan" first because "plan" does not appear inside
            # "delegate" or "direct", but want to be careful about ordering.
            for word in ("plan", "delegate", "direct"):
                if word in text:
                    return word

            # Model returned something but none of the three words appear in it
            print(f"[orchestrator] Unclear classification ({text!r}). Keyword fallback.")

        except Exception as e:
            print(f"[orchestrator] Classification error ({e!r}). Keyword fallback.")

        return _keyword_classify(user_input)

    # -------------------------------------------------------------------------
    # Mode 1: Direct
    # -------------------------------------------------------------------------

    def _try_remember_shortcut(self, user_input: str) -> Optional[str]:
        """
        Fast path for explicit "remember that X" requests. Bypasses
        _classify() and the dispatcher's agent-resolution entirely — we
        already know definitively this is a memory store request, so
        going through an LLM classification call and then agent routing
        is pure overhead for a phrase pattern this unambiguous.

        Calls self.vault.add() DIRECTLY rather than going through
        memory_agent's own tool-call loop. This is an intentional
        architectural shortcut: memory_agent's loop exists so an LLM can
        decide HOW to store something (what category, what importance) —
        but for "remember that X" there's nothing to decide. self.vault is
        the exact same MemoryVault instance memory_agent was constructed
        with in main.py, so this is equivalent to what memory_agent would
        ultimately do, just without the unnecessary LLM round-trip.

        Returns the spoken response string if user_input matched the
        remember-shortcut pattern, or None if it didn't (caller should
        fall through to normal classification).
        """
        match = _REMEMBER_PATTERN.match(user_input.strip())
        if not match:
            return None

        fact = match.group(1).strip().rstrip(".")
        if not fact:
            return None

        print(f"[orchestrator] Remember-shortcut matched: '{fact}'")

        self.memory.add("user", user_input)

        # importance=0.5 matches memory_agent's own store_memory tool
        # default — keeps facts stored via either path scored consistently
        result = self.vault.add(
            fact=fact,
            category="general",
            source="conversation",
            importance=0.5,
        )

        if result.startswith("[memory] Stored"):
            response = f"Got it — I will remember that {fact}."
        elif result.startswith("[memory] Already stored"):
            response = "I already had that noted, sir."
        else:
            response = "I tried to remember that but ran into an issue."

        self.memory.add("assistant", response)
        self._last_agents_used = ["memory_agent"]
        return response

    def _handle_direct(self, user_input: str) -> str:
        """Answers from JARVIS's own knowledge using conversation history."""
        from core.llm import chat

        self.memory.add("user", user_input)

        try:
            response = chat(messages=self.memory.get_messages(), max_tokens=512)
            text     = (response.text or "I am not sure how to respond to that.").strip()
        except Exception as e:
            print(f"[orchestrator] Direct response error: {e}")
            text = "I encountered an issue. Could you rephrase that, sir?"

        self.memory.add("assistant", text)
        return text

    # -------------------------------------------------------------------------
    # Mode 2: Delegate
    # -------------------------------------------------------------------------

    def _handle_delegate(self, user_input: str) -> str:
        """Routes a single-agent task directly without full planning."""
        from core.task import Task, TaskPlan

        task = Task(id=1, description=user_input, depends_on=[])
        plan = TaskPlan(goal=user_input, tasks=[task])

        # Narrate before delegating so JARVIS doesn't go silent
        self.dispatcher._detect_explicit_agent(task)
        agent_name   = self.dispatcher._resolve_agent(task)
        display_name = self._agent_display_name(agent_name)
        self._narrate(f"Let me have {display_name} handle that.")

        # ── Scribe context injection ───────────────────────────────────────────
        # CALLIOPE needs the conversation history to know what to document.
        # No other agent needs this — only the scribe synthesises from chat.
        # We inject the last 60 transcript lines as context so CALLIOPE has
        # the full recent conversation available when writing.
        if agent_name == "scribe_agent":
            transcript_lines = self.memory.get_transcript_lines()
            if transcript_lines:
                # Take last 60 lines (~30 exchanges) — enough context without
                # overwhelming the model with irrelevant earlier conversation
                recent  = transcript_lines[-60:]
                history = "\n".join(recent)
                task.context = (
                    "CONVERSATION HISTORY (most recent exchanges):\n"
                    + history
                    + "\n\nUse this conversation as the source material for the document."
                )

        completed_plan = self.dispatcher.execute(plan)
        result_task    = completed_plan.tasks[0]

        # Track which agents were used for logging
        self._last_agents_used = [result_task.assigned_agent] if result_task.assigned_agent else []

        if result_task.is_done:
            response = self._synthesise_response(
                goal=user_input,
                plan=completed_plan,
                mode="delegate",
            )
        else:
            response = (
                "I was not able to complete that. "
                + (result_task.error or "The agent encountered an issue.")
            )

        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # -------------------------------------------------------------------------
    # Mode 3: Full Plan
    # -------------------------------------------------------------------------

    def _handle_plan(self, user_input: str) -> str:
        """Full Plan → Execute → Critique → (Replan) → Respond workflow."""
        self._narrate("Let me think through this and put together a plan.")

        # Recent conversation history, used by the planner to resolve
        # ambiguous references like "that" or "this project" in the goal.
        # Without this, the planner has no way to know what the user is
        # referring to and may invent unnecessary tasks (e.g. asking
        # memory_agent to "retrieve" something that's actually just a
        # reference to what was discussed a moment earlier in this session).
        recent_lines  = self.memory.get_transcript_lines()
        conv_context  = "\n".join(recent_lines[-40:]) if recent_lines else ""

        plan         = self.planner.plan(user_input, conversation_context=conv_context)
        replan_count = 0

        # Narrate the plan overview if it has multiple steps
        if len(plan.tasks) > 1:
            step_count = len(plan.tasks)
            self._narrate(f"I have broken this into {step_count} steps. Working through them now.")

        while True:
            # Narrate each task before it runs
            for task in plan.tasks:
                if task.is_pending:
                    agent_name   = task.assigned_agent or "the right agent"
                    display_name = self._agent_display_name(agent_name)
                    self._narrate(f"{display_name} is working on step {task.id}.")

            plan     = self.dispatcher.execute(plan)
            critique = self._critique(user_input, plan)
            print(f"[orchestrator] Critique: achieved={critique['goal_achieved']}")

            if critique["goal_achieved"]:
                break

            if not critique["needs_replan"] or replan_count >= MAX_REPLAN_ATTEMPTS:
                print(f"[orchestrator] Stopping after {replan_count} replan(s).")
                break

            replan_count += 1
            print(f"[orchestrator] Replanning ({replan_count}/{MAX_REPLAN_ATTEMPTS})...")
            self._narrate("Some steps didn't go as planned. Let me adjust my approach.")

            new_plan          = self.planner.replan(
                original_goal=user_input,
                completed_tasks=plan.completed_tasks(),
                failed_tasks=plan.failed_tasks(),
                failure_reason=critique.get("failure_reason", "Unknown"),
                conversation_context=conv_context,
            )
            plan.tasks        = plan.completed_tasks() + new_plan.tasks
            plan.replan_count = replan_count

        # Track agents used and plan size for logging
        self._last_agents_used = list(set(
            t.assigned_agent for t in plan.tasks if t.assigned_agent
        ))
        self._last_plan_steps = len(plan.tasks)

        response = self._synthesise_response(goal=user_input, plan=plan, mode="plan")
        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # -------------------------------------------------------------------------
    # Critic
    # -------------------------------------------------------------------------

    def _critique(self, goal: str, plan: TaskPlan) -> dict:
        """
        Evaluates whether the plan achieved the goal.
        Uses plain chat() — no JSON — same reasoning as _classify.
        Falls back to a success-rate heuristic if parsing fails.
        """
        from core.llm import chat

        task_summary = "\n".join([
            "  Task " + str(t.id) + " [" + t.status.value + "]: " + t.description
            + "\n    Result: " + str(t.result or t.error or "")[:200]
            for t in plan.tasks
        ])

        prompt = (
            "Did this task plan fully achieve the goal? "
            "Answer with ONLY 'yes' or 'no'.\n\n"
            "Goal: " + goal + "\n\n"
            "Task results:\n" + task_summary
        )

        try:
            response = chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            text     = (response.text or "").strip().lower()
            achieved = "yes" in text

            return {
                "goal_achieved":  achieved,
                "needs_replan":   not achieved and plan.replan_count < MAX_REPLAN_ATTEMPTS,
                "failure_reason": "" if achieved else "Critic determined goal not fully achieved.",
            }

        except Exception as e:
            print(f"[orchestrator] Critique error ({e}). Using heuristic.")
            success_rate = plan.success_rate()
            return {
                "goal_achieved":  success_rate > 0.5,
                "needs_replan":   success_rate <= 0.5 and plan.replan_count < MAX_REPLAN_ATTEMPTS,
                "failure_reason": f"Heuristic: {success_rate:.0%} tasks succeeded.",
            }

    # -------------------------------------------------------------------------
    # Response Synthesis
    # -------------------------------------------------------------------------

    def _synthesise_response(self, goal: str, plan: TaskPlan, mode: str) -> str:
        """Converts task results into a natural spoken JARVIS response."""
        from core.llm import chat

        if mode == "delegate" and plan.tasks:
            task    = plan.tasks[0]
            result  = str(task.result or task.error or "No result.")
            context = "Task: " + task.description + "\nResult: " + result[:1500]
        else:
            parts = []
            for t in plan.tasks:
                if t.is_done and t.result:
                    parts.append(
                        "[" + (t.assigned_agent or "agent") + "] "
                        + t.description + ":\n" + str(t.result)[:600]
                    )
                elif t.is_failed:
                    parts.append(
                        "[" + (t.assigned_agent or "agent") + "] "
                        + t.description + ": FAILED — " + (t.error or "unknown error")
                    )
            context = "\n\n".join(parts) if parts else "No tasks completed."

        messages = [
            {
                "role":    "system",
                "content": (
                    JARVIS_SYSTEM_PROMPT
                    + "\nSynthesise the agent results into a spoken response. "
                    "Lead with the answer. Be concise. Speak as JARVIS."
                ),
            },
            {
                "role":    "user",
                "content": (
                    "Request: " + goal
                    + "\n\nAgent results:\n" + context
                    + "\n\nRespond as JARVIS."
                ),
            },
        ]

        try:
            response = chat(messages=messages, max_tokens=400)
            return (response.text or "Task complete but no summary available.").strip()
        except Exception as e:
            print(f"[orchestrator] Synthesis error: {e}")
            for task in plan.tasks:
                if task.is_done and task.result:
                    return str(task.result)[:500]
            return "The task ran but I could not produce a clean response."

    # -------------------------------------------------------------------------
    # Agent Display Name
    # -------------------------------------------------------------------------

    def _agent_display_name(self, agent_name: str) -> str:
        """
        Converts a functional agent name to the Greek display name
        that JARVIS uses in speech.

        web_agent → HERMES, memory_agent → MNEMOSYNE, etc.
        Falls back to a cleaned version of the functional name if not found.
        """
        # Build reverse map: functional_name → Greek name
        reverse = {v: k.upper() for k, v in AGENT_ALIASES.items()}
        if agent_name in reverse:
            return reverse[agent_name]

        # Fallback: clean up the functional name for speech
        # "browser_agent" → "the browser agent"
        cleaned = agent_name.replace("_agent", "").replace("_", " ")
        return f"the {cleaned} agent"

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def get_conversation_history(self) -> list[dict]:
        return self.memory.get_messages()

    def reset_conversation(self) -> None:
        self.memory.clear()
        self._last_agents_used = []
        self._last_plan_steps  = 0