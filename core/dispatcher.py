# =============================================================================
# core/dispatcher.py — The Dispatcher
# =============================================================================
#
# WHAT THIS DOES:
# Takes a TaskPlan from the Planner and executes it — routing each Task
# to the right specialist agent, injecting context from completed tasks
# into dependent ones, and collecting all results.
#
# The Dispatcher answers: "Who does what, and in what order?"
#
# THREE RESPONSIBILITIES:
# 1. ROUTING — decides which agent gets each task (using agent_hint from
#    the Planner, falling back to LLM-assisted routing for unassigned tasks)
# 2. SEQUENCING — respects depends_on by checking all dependencies are
#    done before starting a task
# 3. CONTEXT INJECTION — passes the results of completed dependency tasks
#    as context into dependent tasks, so each agent has what it needs
#
# THE SCRATCHPAD:
# A dict mapping task_id → result string. As tasks complete, their results
# are written here. When a task with dependencies starts, the Dispatcher
# reads those dependency results from the scratchpad and injects them as
# context. This is how information flows between tasks.
#
# ROUTING LOGIC:
# Priority order for deciding which agent gets a task:
#   1. Task already has assigned_agent set (from Planner's agent_hint,
#      if it was a valid agent name)
#   2. Keyword matching against AGENT_REGISTRY's best_for lists — fast,
#      no LLM call needed
#   3. LLM-assisted routing — ask the model to pick the right agent
#      (only for ambiguous tasks where keyword matching fails)
#   4. Fallback to web_agent — something is always better than nothing
# =============================================================================

from __future__ import annotations

from core.task import Task, TaskPlan, TaskStatus
from config import AGENT_REGISTRY, AGENT_ALIASES


# =============================================================================
# Dispatcher
# =============================================================================

class Dispatcher:
    """
    Routes and executes a TaskPlan by sending each Task to the right agent.

    Usage:
        dispatcher = Dispatcher(agents)
        plan       = dispatcher.execute(plan)
        # plan.tasks now have status, result fields populated
    """

    def __init__(self, agents: dict):
        """
        Args:
            agents: Dict mapping agent name → agent instance.
                    e.g. {"web_agent": WebAgent(), "memory_agent": MemoryAgent()}
                    Must include all agents referenced in AGENT_REGISTRY.
        """
        self.agents = agents

    def execute(self, plan: TaskPlan) -> TaskPlan:
        """
        Executes all tasks in the plan in dependency order.

        For each task:
        1. Checks all dependencies are DONE
        2. Resolves which agent should handle it
        3. Builds context from dependency results
        4. Calls agent.run(task, context)
        5. Updates task status and result

        Tasks whose dependencies failed are marked SKIPPED rather than
        attempted — no point running "summarise the search results" if
        the search itself failed.

        Args:
            plan: A TaskPlan from the Planner. Tasks have status=PENDING.

        Returns:
            The same TaskPlan with all tasks' status and result fields updated.
        """
        scratchpad: dict[int, str] = {}

        print(f"\n[dispatcher] Executing plan: {len(plan.tasks)} tasks")

        for task in plan.tasks:
            # ── Check dependencies ────────────────────────────────────────────
            if task.depends_on:
                dep_statuses = {
                    dep_id: self._get_task_status(dep_id, plan)
                    for dep_id in task.depends_on
                }
                failed_deps = [
                    dep_id for dep_id, status in dep_statuses.items()
                    if status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
                ]

                if failed_deps:
                    print(
                        f"[dispatcher] Skipping task {task.id} — "
                        f"dependencies {failed_deps} failed."
                    )
                    task.status = TaskStatus.SKIPPED
                    task.error  = f"Dependencies {failed_deps} did not complete."
                    continue

                # Check if dependencies are even done yet
                # (in a sequential plan they always are, but good to verify)
                pending_deps = [
                    dep_id for dep_id, status in dep_statuses.items()
                    if status == TaskStatus.PENDING
                ]
                if pending_deps:
                    print(
                        f"[dispatcher] Warning: task {task.id} has pending "
                        f"dependencies {pending_deps} — skipping for now."
                    )
                    task.status = TaskStatus.SKIPPED
                    task.error  = f"Dependencies {pending_deps} not yet complete."
                    continue

            # ── Resolve agent ─────────────────────────────────────────────────
            agent_name = self._resolve_agent(task)
            task.assigned_agent = agent_name

            agent = self.agents.get(agent_name)
            if not agent:
                print(f"[dispatcher] Agent '{agent_name}' not found. Skipping task {task.id}.")
                task.status = TaskStatus.FAILED
                task.error  = f"Agent '{agent_name}' is not registered."
                continue

            # ── Build context from dependencies ───────────────────────────────
            context = self._build_context(task, scratchpad)

            # ── Execute ───────────────────────────────────────────────────────
            task.status = TaskStatus.RUNNING
            print(f"\n[dispatcher] → [{agent_name}] Task {task.id}: {task.description[:60]}")

            try:
                result_dict = agent.run(task=task.description, context=context)

                if result_dict.get("status") == "done":
                    task.status = TaskStatus.DONE
                    task.result = result_dict.get("result", "")
                    scratchpad[task.id] = str(task.result)
                    print(f"[dispatcher] Task {task.id} ✓ — {result_dict.get('summary', '')[:80]}")
                else:
                    task.status = TaskStatus.FAILED
                    task.error  = result_dict.get("summary", "Agent returned failure status.")
                    print(f"[dispatcher] Task {task.id} ✗ — {task.error[:80]}")

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error  = str(e)
                print(f"[dispatcher] Task {task.id} ✗ — unexpected error: {e}")

        return plan

    # -------------------------------------------------------------------------
    # Agent Resolution
    #
    # Three-tier lookup so we never silently drop a task.
    # -------------------------------------------------------------------------

    def _resolve_agent(self, task: Task) -> str:
        """
        Determines which agent should handle this task.

        Priority:
        0. Explicit agent naming in the task description
           ("tell PROTEUS to...", "ask the browser agent to...",
            "have HERMES search for...", "tell web_agent to...")
           This is the most common case when the user explicitly addresses
           an agent by name or Greek name.
        1. task.assigned_agent (set by Planner — trusted if valid)
        2. Keyword matching against AGENT_REGISTRY best_for lists
        3. LLM-assisted routing (for genuinely ambiguous tasks)
        4. Fallback to "web_agent"
        """
        # ── Tier 0: Explicit agent name in task description ───────────────────
        # When the user says "tell Proteus to X" or "ask the browser agent to X"
        # we detect that and route directly — no scoring needed.
        # Also strips the routing instruction from the description so the
        # receiving agent only sees the actual task (not "tell X agent to do Y").
        explicit = self._detect_explicit_agent(task)
        if explicit:
            return explicit

        # ── Tier 1: Already assigned and valid ────────────────────────────────
        if task.assigned_agent:
            resolved = AGENT_ALIASES.get(
                task.assigned_agent.lower(), task.assigned_agent
            )
            if resolved in self.agents:
                return resolved

        # ── Tier 2: Keyword matching ───────────────────────────────────────────
        matched = self._keyword_match(task.description)
        if matched:
            return matched

        # ── Tier 3: LLM routing ────────────────────────────────────────────────
        llm_choice = self._llm_route(task.description)
        if llm_choice and llm_choice in self.agents:
            return llm_choice

        # ── Tier 4: Fallback ──────────────────────────────────────────────────
        print(f"[dispatcher] Could not confidently route task {task.id}. Defaulting to web_agent.")
        return "web_agent"

    def _detect_explicit_agent(self, task: Task) -> str | None:
        """
        Tier 0 routing: detects when the user explicitly names an agent in
        the task description and routes directly to that agent.

        Handles all of these patterns:
          "tell Proteus to go to LinkedIn..."
          "ask the browser agent to download..."
          "have HERMES search for..."
          "tell web_agent to look up..."
          "ask APOLLO to play something"
          "get MNEMOSYNE to remember this"

        When an explicit agent name is found:
        1. Routes to that agent
        2. Strips the delegation preamble ("tell X to", "ask X to", etc.)
           from task.description so the agent receives a clean task,
           not "tell Proteus to go to LinkedIn" but "go to LinkedIn"

        Returns the resolved agent name, or None if no explicit name found.
        """
        import re

        desc       = task.description
        desc_lower = desc.lower()

        # Build a lookup of all recognisable agent names:
        # functional names ("browser_agent", "web agent") and
        # Greek aliases ("proteus", "hermes") and their display variants
        name_to_agent = {}

        # Functional names — both underscore and space versions
        for agent_name in self.agents:
            name_to_agent[agent_name.lower()]              = agent_name  # "browser_agent"
            name_to_agent[agent_name.lower().replace("_", " ")] = agent_name  # "browser agent"

        # Greek aliases from config
        for greek, functional in AGENT_ALIASES.items():
            if functional in self.agents:
                name_to_agent[greek.lower()] = functional  # "proteus" → "browser_agent"

        # Delegation trigger phrases — patterns like "tell X to", "ask X to"
        # We look for these followed by any known agent name
        DELEGATION_PATTERNS = [
            r"tell\s+(?:the\s+)?({name})\s+(?:agent\s+)?(?:\w+\s+)?to\s+",
            r"ask\s+(?:the\s+)?({name})\s+(?:agent\s+)?(?:\w+\s+)?to\s+",
            r"have\s+(?:the\s+)?({name})\s+(?:agent\s+)?(?:\w+\s+)?(?:to\s+)?",
            r"get\s+(?:the\s+)?({name})\s+(?:agent\s+)?(?:to\s+)?",
            r"use\s+(?:the\s+)?({name})\s+(?:agent\s+)?(?:to\s+)?",
            r"(?:the\s+)?({name})\s+(?:agent\s+)?(?:should|needs\s+to|can you)\s+",
        ]

        for name, agent_name in name_to_agent.items():
            # Escape the name for use in regex (handles underscores, spaces)
            escaped = re.escape(name)

            for pattern_template in DELEGATION_PATTERNS:
                pattern = pattern_template.format(name=escaped)
                m = re.search(pattern, desc_lower)
                if m:
                    # Found a match — extract the actual task after the preamble
                    end_pos  = m.end()
                    raw_task = desc[end_pos:].strip()

                    if raw_task:
                        # Update the task description to be just the actual task
                        task.description = raw_task[0].upper() + raw_task[1:]
                        print(
                            f"[dispatcher] Explicit agent '{name}' detected → {agent_name}. "
                            f"Task rewritten to: {task.description[:60]}"
                        )
                    else:
                        print(f"[dispatcher] Explicit agent '{name}' detected → {agent_name}.")

                    return agent_name

        return None

    def _keyword_match(self, description: str) -> str | None:
        """
        Matches the task description against AGENT_REGISTRY's best_for lists.

        Scores each agent by how many of its best_for keywords appear in
        the task description. Returns the highest-scoring agent name,
        or None if no agent scores above 0.

        MUSIC PRIORITY RULE:
        If the description contains a music action word ("play", "pause",
        "skip") alongside any other word like "search", "globally", "find",
        music_agent always wins. The user saying "search globally and play X"
        is a music request, not a web search request.
        """
        desc_lower = description.lower()

        # Hard priority: if any music action word is present, route to music_agent
        # before doing general scoring. "play", "pause", "skip", "next track"
        # are unambiguous music intents regardless of other words in the sentence.
        MUSIC_ACTION_WORDS = ["play ", "pause the", "skip track", "next track",
                              "previous track", "stop the music", "resume music"]
        if any(w in desc_lower for w in MUSIC_ACTION_WORDS):
            if "music_agent" in self.agents:
                print(f"[dispatcher] Music action detected — routing to music_agent.")
                return "music_agent"

        scores = {}
        for agent_name, info in AGENT_REGISTRY.items():
            if agent_name not in self.agents:
                continue

            score = sum(
                1 for keyword in info.get("best_for", [])
                if keyword.lower() in desc_lower
            )
            if score > 0:
                scores[agent_name] = score

        if not scores:
            return None

        best = max(scores, key=lambda k: scores[k])
        print(f"[dispatcher] Keyword match: '{best}' (score {scores[best]}) for task.")
        return best

    def _llm_route(self, task_description: str) -> str | None:
        """
        Asks the LLM to pick the best agent for a task.
        Only called when keyword matching produces no result.

        Returns the agent name string, or None if routing fails.
        """
        from core.llm import structured

        agent_options = "\n".join([
            f"  - {name}: {info['description']}"
            for name, info in AGENT_REGISTRY.items()
            if name in self.agents
        ])

        prompt = (
            f"You are routing a task to the best specialist agent.\n\n"
            f"Available agents:\n{agent_options}\n\n"
            f"Task: {task_description}\n\n"
            f"Which agent should handle this task? "
            f"Respond with ONLY a JSON object:\n"
            f'{{ "agent": "agent_name" }}'
        )

        try:
            data   = structured(prompt=prompt, max_tokens=50)
            choice = data.get("agent", "").strip().lower()

            # Resolve alias in case LLM used a Greek name
            choice = AGENT_ALIASES.get(choice, choice)

            if choice in self.agents:
                print(f"[dispatcher] LLM routing: '{choice}' for task.")
                return choice

        except Exception as e:
            print(f"[dispatcher] LLM routing failed: {e}")

        return None

    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------

    def _build_context(self, task: Task, scratchpad: dict[int, str]) -> str:
        """
        Assembles context from the results of this task's dependencies.

        If task 3 depends on tasks 1 and 2, this function returns a
        formatted string containing tasks 1 and 2's results. That string
        is passed to the agent as the `context` argument so it knows what
        earlier steps found.

        This is the information pipeline between tasks.
        """
        if not task.depends_on:
            return ""

        context_parts = []
        for dep_id in task.depends_on:
            if dep_id in scratchpad:
                context_parts.append(
                    f"Result from step {dep_id}:\n{scratchpad[dep_id]}"
                )

        return "\n\n".join(context_parts) if context_parts else ""

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_task_status(self, task_id: int, plan: TaskPlan) -> TaskStatus:
        """Returns the status of a task by ID, or PENDING if not found."""
        for task in plan.tasks:
            if task.id == task_id:
                return task.status
        return TaskStatus.PENDING