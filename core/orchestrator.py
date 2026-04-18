# =============================================================================
# core/orchestrator.py — The Orchestrator (JARVIS)
# =============================================================================
#
# WHAT THIS IS:
# The Orchestrator IS JARVIS. Every other component — agents, planner,
# dispatcher, memory — is infrastructure. This is the entity that talks
# to the user, decides what to do, coordinates everything, and responds.
#
# THE ORCHESTRATOR'S THREE MODES:
#
# 1. DIRECT RESPONSE (no agents needed)
#    For simple conversational turns: greetings, quick factual questions,
#    clarifications, things JARVIS can answer from his own knowledge.
#    Just calls the LLM and returns the text. Fast, no overhead.
#    Examples: "How are you?", "What's 15% of 240?", "Explain recursion."
#
# 2. SINGLE-AGENT DELEGATION (one agent, one task)
#    For tasks that clearly belong to one agent domain.
#    JARVIS delegates directly without a full planning cycle.
#    Examples: "Play something relaxing", "What's my battery?",
#              "Remember that I prefer tabs over spaces."
#    Skips the Planner — goes straight to routing via the Dispatcher.
#
# 3. FULL AGENTIC WORKFLOW (planner → dispatcher → critic → respond)
#    For complex multi-step goals requiring multiple agents or sequential
#    information gathering.
#    Examples: "Research Rust vs Go and tell me which suits my project",
#              "Find today's AI news and save the highlights to memory"
#
# HOW JARVIS DECIDES WHICH MODE:
# The classifier — a lightweight LLM call — categorises every incoming
# message as "direct", "delegate", or "plan". The classifier uses a
# tight prompt focused purely on classification, not execution.
#
# THE CRITIC:
# After a full workflow runs, the Critic evaluates whether the goal was
# actually achieved. If not, it triggers replanning (up to MAX_REPLAN_ATTEMPTS).
# This is what gives JARVIS self-correction capability.
#
# SHORT-TERM MEMORY:
# JARVIS maintains a rolling conversation history (ShortTermMemory).
# This is used for all direct responses so JARVIS has conversational
# continuity — he knows what was said earlier in the session.
# Agents do NOT share this memory — they have their own isolated contexts.
#
# PERSONALITY:
# JARVIS's system prompt is where his personality lives. The Greek agent
# names appear here so JARVIS knows to refer to his agents by those names
# in speech, even though the code uses functional names internally.
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
#
# This is JARVIS's personality, knowledge of his own capabilities,
# and the rules that govern his behaviour.
#
# Note: Greek agent names appear HERE (in JARVIS's spoken persona) but
# nowhere else in the codebase. This is the only place lore lives in code.
# =============================================================================

JARVIS_SYSTEM_PROMPT = """You are JARVIS — a sophisticated, intelligent AI assistant with a dry wit, \
quiet confidence, and genuine capability. You are not a chatbot. You are an autonomous system.

You have a team of specialist agents who handle specific domains on your behalf:
  - HERMES   (web_agent):      Internet research and web content retrieval
  - MNEMOSYNE (memory_agent):  Long-term memory — stores and recalls facts about the user
  - HEPHAESTUS (system_agent): macOS system control — apps, volume, screenshots
  - APOLLO   (music_agent):    Music playback and Apple Music control
  - ATHENA   (research_agent): Deep multi-source research and synthesis

When you delegate to an agent, you may refer to them by their Greek names in speech.
When reporting results, speak in first person as JARVIS — not "the web agent found..."
but "I found..." or "HERMES retrieved..."

COMMUNICATION STYLE:
- Sharp, concise, and direct. No filler. No unnecessary padding.
- Dry wit when appropriate — never forced, never at the user's expense.
- Speak like a capable entity that gets things done, not like a helpful chatbot.
- For task completion: confirm what was done, report the key finding, stop.
- For conversation: engage genuinely, be brief.

WHAT YOU NEVER DO:
- Announce that you are "calling an agent" or "processing a request"
- Use hollow phrases like "Certainly!", "Of course!", "Great question!"
- Ramble. If you've said what needs saying, stop.
- Pretend you cannot do something you can do via your agents.
"""


# =============================================================================
# Classification Prompt
# =============================================================================

CLASSIFICATION_PROMPT = """Classify this user message into exactly one category.

Categories:
  direct   — Can be answered directly from knowledge. No tools, no agents needed.
              Examples: greetings, opinions, explanations, math, coding help,
              general knowledge, "how are you", "what is X"

  delegate — Clearly belongs to one specific agent domain. Single action needed.
              Examples: "play X", "open Y", "remember Z", "what's my battery",
              "search for X", "what's the weather", "set volume to 50"

  plan     — Multi-step goal requiring multiple agents or sequential information
              gathering. Involves research, comparison, storing AND retrieving,
              or performing multiple distinct actions.
              Examples: "research X and save highlights", "find X then do Y",
              "compare A and B and recommend one"

User message: "{message}"

Respond with ONLY a JSON object:
{{ "classification": "direct" | "delegate" | "plan", "reason": "one sentence" }}
"""


# =============================================================================
# Orchestrator
# =============================================================================

class Orchestrator:
    """
    JARVIS — the top-level coordinator that talks to the user and manages agents.

    Usage:
        orchestrator = Orchestrator(agents, vault)
        response     = orchestrator.process("Research Rust vs Go")
        # response is a string JARVIS can speak aloud
    """

    def __init__(self, agents: dict, vault: MemoryVault):
        """
        Args:
            agents: Dict of agent_name → agent instance for all loaded agents.
            vault:  The shared MemoryVault instance for long-term memory.
        """
        self.agents     = agents
        self.vault      = vault
        self.planner    = Planner()
        self.dispatcher = Dispatcher(agents)

        # Build JARVIS's short-term memory with his persona + user profile
        core_profile = vault.get_core_profile()
        system       = JARVIS_SYSTEM_PROMPT
        if core_profile:
            system += f"\n\n{core_profile}"

        self.memory = ShortTermMemory(
            system_prompt=system,
            label="orchestrator",
        )

    def process(self, user_input: str) -> str:
        """
        Main entry point. Takes user input, routes it, executes, returns response.

        This is the function called by main.py on every voice input.

        Args:
            user_input: The transcribed text of what the user said.

        Returns:
            A string response for JARVIS to speak aloud.
        """
        print(f"\n[orchestrator] Input: {user_input}")

        # ── Classify ──────────────────────────────────────────────────────────
        classification = self._classify(user_input)
        print(f"[orchestrator] Classification: {classification}")

        # ── Route to appropriate handler ──────────────────────────────────────
        if classification == "direct":
            response = self._handle_direct(user_input)

        elif classification == "delegate":
            response = self._handle_delegate(user_input)

        else:  # "plan" — full agentic workflow
            response = self._handle_plan(user_input)

        # ── Save conversation turn to transcript for consolidation ─────────────
        append_to_transcript(self.memory.get_transcript_lines())

        print(f"[orchestrator] Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        return response

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def _classify(self, user_input: str) -> str:
        """
        Classifies the user input as 'direct', 'delegate', or 'plan'.

        Uses a lightweight structured LLM call with a tight classification
        prompt. Falls back to 'direct' on failure — always safe.
        """
        from core.llm import structured

        prompt = CLASSIFICATION_PROMPT.format(message=user_input)

        try:
            data   = structured(prompt=prompt, max_tokens=100)
            result = data.get("classification", "direct").strip().lower()

            if result in ("direct", "delegate", "plan"):
                return result

            # LLM returned something unexpected — default to direct
            print(f"[orchestrator] Unexpected classification: {result!r}. Defaulting to direct.")
            return "direct"

        except Exception as e:
            print(f"[orchestrator] Classification failed ({e}). Defaulting to direct.")
            return "direct"

    # -------------------------------------------------------------------------
    # Mode 1: Direct Response
    # -------------------------------------------------------------------------

    def _handle_direct(self, user_input: str) -> str:
        """
        Answers directly using JARVIS's own knowledge.
        Uses short-term memory so JARVIS has conversational context.
        """
        from core.llm import chat

        self.memory.add("user", user_input)

        try:
            response = chat(messages=self.memory.get_messages(), max_tokens=512)
            text     = (response.text or "I'm not sure how to respond to that.").strip()
        except Exception as e:
            print(f"[orchestrator] Direct response failed: {e}")
            text = "I encountered an issue processing that. Could you rephrase?"

        self.memory.add("assistant", text)
        return text

    # -------------------------------------------------------------------------
    # Mode 2: Single-Agent Delegation
    # -------------------------------------------------------------------------

    def _handle_delegate(self, user_input: str) -> str:
        """
        Routes a single-agent task directly to the best agent.
        Skips the Planner — creates a single Task and dispatches it.

        Faster than full planning for clear single-domain requests.
        """
        from core.task import Task, TaskPlan

        # Create a minimal single-task plan
        task = Task(id=1, description=user_input, depends_on=[])
        plan = TaskPlan(goal=user_input, tasks=[task])

        # Dispatcher handles agent resolution and execution
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
                f"I wasn't able to complete that. "
                f"{result_task.error or 'The agent encountered an issue.'}"
            )

        # Add to conversation memory so JARVIS can reference this later
        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # -------------------------------------------------------------------------
    # Mode 3: Full Agentic Workflow (Plan → Execute → Critique → Respond)
    # -------------------------------------------------------------------------

    def _handle_plan(self, user_input: str) -> str:
        """
        Full multi-step agentic workflow with replanning on failure.

        Loop:
        1. Planner creates task graph
        2. Dispatcher executes all tasks
        3. Critic evaluates outcome
        4. If goal not achieved and retries remain → replan and repeat
        5. Synthesise final spoken response from all results
        """
        plan         = self.planner.plan(user_input)
        replan_count = 0

        while True:
            # Execute the current plan
            plan = self.dispatcher.execute(plan)

            # Evaluate outcome
            critique = self._critique(user_input, plan)
            print(f"[orchestrator] Critique: achieved={critique['goal_achieved']}")

            if critique["goal_achieved"]:
                break

            # Goal not achieved — should we replan?
            if not critique["needs_replan"] or replan_count >= MAX_REPLAN_ATTEMPTS:
                print(
                    f"[orchestrator] Stopping after {replan_count} replan(s). "
                    f"Proceeding with partial results."
                )
                break

            # Replan with context from what was already completed
            replan_count += 1
            print(f"[orchestrator] Replanning (attempt {replan_count}/{MAX_REPLAN_ATTEMPTS})...")

            new_tasks = self.planner.replan(
                original_goal=user_input,
                completed_tasks=plan.completed_tasks(),
                failed_tasks=plan.failed_tasks(),
                failure_reason=critique.get("failure_reason", "Unknown failure"),
            )

            # Merge: keep completed tasks, replace rest with new plan
            plan.tasks = plan.completed_tasks() + new_tasks.tasks
            plan.replan_count = replan_count

        # Synthesise and return the final spoken response
        response = self._synthesise_response(
            goal=user_input,
            plan=plan,
            mode="plan",
        )

        self.memory.add("user", user_input)
        self.memory.add("assistant", response)
        return response

    # -------------------------------------------------------------------------
    # Critic
    # -------------------------------------------------------------------------

    def _critique(self, goal: str, plan: TaskPlan) -> dict:
        """
        Evaluates whether the executed plan achieved the user's goal.

        Returns a dict with:
          goal_achieved:  bool
          needs_replan:   bool
          failure_reason: str (if not achieved)

        Falls back to a simple heuristic if the LLM call fails:
        if >50% of tasks succeeded, consider the goal achieved.
        """
        from core.llm import structured

        task_summary = "\n".join([
            f"  Task {t.id} [{t.status.value}]: {t.description}\n"
            f"    Result: {str(t.result or t.error or '')[:200]}"
            for t in plan.tasks
        ])

        prompt = (
            f"You are evaluating whether a task plan achieved its goal.\n\n"
            f"Goal: {goal}\n\n"
            f"Task results:\n{task_summary}\n\n"
            f"Did the plan achieve the goal? Consider:\n"
            f"- Were the key tasks completed successfully?\n"
            f"- Does the output actually answer/address the goal?\n"
            f"- Would the user be satisfied with this outcome?\n\n"
            f"Respond ONLY with JSON:\n"
            f'{{"goal_achieved": true/false, "needs_replan": true/false, '
            f'"failure_reason": "explanation if not achieved"}}'
        )

        try:
            data = structured(prompt=prompt, max_tokens=200)
            return {
                "goal_achieved":  bool(data.get("goal_achieved", False)),
                "needs_replan":   bool(data.get("needs_replan", False)),
                "failure_reason": str(data.get("failure_reason", "")),
            }
        except Exception as e:
            print(f"[orchestrator] Critique failed ({e}). Using heuristic.")
            # Heuristic fallback: success if majority of tasks succeeded
            success_rate = plan.success_rate()
            return {
                "goal_achieved":  success_rate > 0.5,
                "needs_replan":   success_rate <= 0.5 and plan.replan_count < MAX_REPLAN_ATTEMPTS,
                "failure_reason": f"Heuristic: {success_rate:.0%} task success rate.",
            }

    # -------------------------------------------------------------------------
    # Response Synthesis
    # -------------------------------------------------------------------------

    def _synthesise_response(
        self,
        goal:  str,
        plan:  TaskPlan,
        mode:  str,
    ) -> str:
        """
        Converts task results into a natural spoken response.

        This is JARVIS's voice — the final LLM call that takes all the
        raw data gathered by agents and produces what gets spoken aloud.

        For delegation (single task): brief confirmation + key finding.
        For full plans: summary of what was done + synthesised answer.
        """
        from core.llm import chat

        # Build a compact summary of results to give the LLM
        if mode == "delegate" and plan.tasks:
            task   = plan.tasks[0]
            result = str(task.result or task.error or "No result.")
            context = f"Task: {task.description}\nResult: {result[:1000]}"
        else:
            parts = []
            for t in plan.tasks:
                if t.is_done and t.result:
                    parts.append(f"[{t.assigned_agent}] {t.description}:\n{str(t.result)[:500]}")
                elif t.is_failed:
                    parts.append(f"[{t.assigned_agent}] {t.description}: FAILED — {t.error}")
            context = "\n\n".join(parts) if parts else "No tasks were completed."

        synthesis_messages = [
            {
                "role":    "system",
                "content": (
                    JARVIS_SYSTEM_PROMPT +
                    "\n\nYou are synthesising the results of completed agent work into "
                    "a spoken response. Be concise. Lead with the answer, not the process. "
                    "Speak as JARVIS, not as a report generator."
                ),
            },
            {
                "role":    "user",
                "content": (
                    f"Original request: {goal}\n\n"
                    f"Agent results:\n{context}\n\n"
                    f"Now give the user your response as JARVIS."
                ),
            },
        ]

        try:
            response = chat(messages=synthesis_messages, max_tokens=400)
            return (response.text or "The task completed but I have no summary to offer.").strip()
        except Exception as e:
            print(f"[orchestrator] Synthesis failed: {e}")
            # Fallback: extract the most useful result directly
            for task in plan.tasks:
                if task.is_done and task.result:
                    return str(task.result)[:500]
            return "The task ran but I couldn't synthesise a clean response."

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def get_conversation_history(self) -> list[dict]:
        """Returns the current conversation history for logging or debugging."""
        return self.memory.get_messages()

    def reset_conversation(self) -> None:
        """Clears short-term memory. Called between sessions if needed."""
        self.memory.clear()