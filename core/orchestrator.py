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

from core.task import TaskPlan, TaskStatus
from core.planner import Planner
from core.dispatcher import Dispatcher
from memory.short_term import ShortTermMemory
from memory.long_term import MemoryVault
from memory.consolidator import append_to_transcript
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
    "  - MNEMOSYNE (memory_agent):    Long-term memory — stores and recalls facts\n"
    "  - HEPHAESTUS (system_agent):   macOS control — apps, volume, screenshots\n"
    "  - APOLLO    (music_agent):     Music playback and Apple Music control\n"
    "  - ATHENA    (research_agent):  Deep multi-source research and synthesis\n\n"
    "Refer to agents by their Greek names in speech. Speak as JARVIS in first person.\n\n"
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
# Orchestrator
# =============================================================================

class Orchestrator:
    """
    JARVIS — the top-level coordinator that talks to the user and manages agents.

    Usage:
        orchestrator = Orchestrator(agents, vault)
        response     = orchestrator.process("How are you?")
    """

    def __init__(self, agents: dict, vault: MemoryVault):
        self.agents     = agents
        self.vault      = vault
        self.planner    = Planner()
        self.dispatcher = Dispatcher(agents)

        core_profile = vault.get_core_profile()
        system       = JARVIS_SYSTEM_PROMPT
        if core_profile:
            system += "\n\n" + core_profile

        self.memory = ShortTermMemory(
            system_prompt=system,
            label="orchestrator",
        )

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        """
        Takes user input, classifies it, routes to the right handler,
        returns a spoken response string.
        """
        print(f"\n[orchestrator] Input: {user_input}")

        classification = self._classify(user_input)
        print(f"[orchestrator] Classification: {classification}")

        if classification == "direct":
            response = self._handle_direct(user_input)
        elif classification == "delegate":
            response = self._handle_delegate(user_input)
        else:
            response = self._handle_plan(user_input)

        append_to_transcript(self.memory.get_transcript_lines())
        print(f"[orchestrator] Response: {response[:100]}{'...' if len(response) > 100 else ''}")
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

        task           = Task(id=1, description=user_input, depends_on=[])
        plan           = TaskPlan(goal=user_input, tasks=[task])
        completed_plan = self.dispatcher.execute(plan)
        result_task    = completed_plan.tasks[0]

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
        plan         = self.planner.plan(user_input)
        replan_count = 0

        while True:
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

            new_plan          = self.planner.replan(
                original_goal=user_input,
                completed_tasks=plan.completed_tasks(),
                failed_tasks=plan.failed_tasks(),
                failure_reason=critique.get("failure_reason", "Unknown"),
            )
            plan.tasks        = plan.completed_tasks() + new_plan.tasks
            plan.replan_count = replan_count

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
    # Session Management
    # -------------------------------------------------------------------------

    def get_conversation_history(self) -> list[dict]:
        return self.memory.get_messages()

    def reset_conversation(self) -> None:
        self.memory.clear()