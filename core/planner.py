# =============================================================================
# core/planner.py — The Planner
# =============================================================================
#
# WHAT THIS DOES:
# Takes a user's goal (a plain English string) and breaks it into an ordered
# list of Tasks that can be assigned to specialist agents and executed.
#
# The Planner answers the question: "What needs to happen, in what order,
# to achieve this goal?"
#
# It does NOT decide which agent does what — that's the Dispatcher's job.
# It does NOT execute anything — that's the agents' job.
# It only produces the task graph.
#
# SINGLE RESPONSIBILITY:
# This separation matters. A Planner that also assigns agents and executes
# tasks becomes tangled — changes to agent capabilities leak into planning
# logic. By keeping planning pure (just task decomposition), you can:
# - Improve planning quality without touching execution
# - Change which agents exist without rewriting planning prompts
# - Test planning independently of execution
#
# HOW IT WORKS:
# 1. Builds a prompt describing the goal + available agents
# 2. Calls llm.structured() to get a guaranteed JSON task graph
# 3. Parses the JSON into a TaskPlan of Task objects
# 4. Validates the plan (checks dependency IDs are valid, etc.)
#
# WHAT THE LLM PRODUCES:
# {
#   "tasks": [
#     {
#       "id": 1,
#       "description": "Search the web for Python 3.13 release notes",
#       "agent_hint": "web_agent",
#       "depends_on": []
#     },
#     {
#       "id": 2,
#       "description": "Summarise the key changes from the search results",
#       "agent_hint": "research_agent",
#       "depends_on": [1]
#     }
#   ]
# }
#
# agent_hint is a suggestion, not a command. The Dispatcher makes the
# final decision on which agent gets each task.
# =============================================================================

from __future__ import annotations

from core.task import Task, TaskPlan
from config import AGENT_REGISTRY, MAX_REPLAN_ATTEMPTS


# =============================================================================
# Planning Prompt
#
# This prompt is the heart of the planner. It teaches the LLM what a good
# task graph looks like for JARVIS's specific agent ecosystem.
#
# Key design decisions:
# - We tell the model exactly which agents exist and what they do
#   so agent_hints are grounded in reality, not hallucinated
# - We emphasise atomicity (one tool call per task) to prevent
#   tasks that are actually multiple tasks merged together
# - We require depends_on to be explicit so the Dispatcher can
#   sequence tasks correctly and inject context between them
# - We specify "if goal is simple, return a single task" to prevent
#   over-engineering simple requests into multi-step plans
# =============================================================================

def _build_planning_prompt(goal: str, agent_descriptions: str) -> str:
    return f"""
You are a task planning AI for JARVIS, a multi-agent assistant system.
Break the following goal into an ordered list of atomic tasks.

AVAILABLE AGENTS:
{agent_descriptions}

PLANNING RULES:
1. Each task must be atomic — completable in a single agent session with
   one or a few tool calls. If a task requires multiple distinct actions,
   split it into separate tasks.
2. Tasks must be concrete and specific. Not "research Python" but
   "Search the web for Python 3.13 release notes and return key changes."
3. Order tasks so earlier results feed into later tasks naturally.
4. Use depends_on to express dependencies — task 3 depending on task 1
   means task 3 will receive task 1's result as context.
5. agent_hint should be the most appropriate agent name from the list above.
   It's a suggestion — the system may override it.
6. If the goal is simple (one agent, one action), return a single task.
   Do not over-engineer simple requests.
7. Maximum 6 tasks per plan. If a goal seems to need more, find a way
   to consolidate.

GOAL: {goal}

Respond ONLY with valid JSON in exactly this format:
{{
  "tasks": [
    {{
      "id": 1,
      "description": "Specific, concrete description of what to do",
      "agent_hint": "agent_name",
      "depends_on": []
    }},
    {{
      "id": 2,
      "description": "Another specific task",
      "agent_hint": "agent_name",
      "depends_on": [1]
    }}
  ]
}}
"""


def _build_replan_prompt(
    original_goal:  str,
    agent_descriptions: str,
    completed_summary:  str,
    failure_reason:     str,
) -> str:
    """
    Builds a prompt for replanning after partial failure.
    Tells the model what was already accomplished so it doesn't redo it.
    """
    return f"""
You are a task planning AI for JARVIS. A previous plan partially failed.
Create a NEW plan covering only what still needs to be done.

AVAILABLE AGENTS:
{agent_descriptions}

ORIGINAL GOAL: {original_goal}

ALREADY COMPLETED:
{completed_summary}

REASON PREVIOUS PLAN FAILED:
{failure_reason}

Create a focused plan for the REMAINING work only.
Do not re-do what was already completed successfully.

{_build_planning_prompt.__doc__}

Respond ONLY with valid JSON in the same format as before.
"""


# =============================================================================
# Planner
# =============================================================================

class Planner:
    """
    Breaks a user goal into an ordered TaskPlan using the LLM.

    Usage:
        planner = Planner()
        plan    = planner.plan("Research FastAPI vs Django and recommend one")
        # plan is a TaskPlan with a list of Task objects
    """

    def __init__(self):
        # Build the agent description string once at construction time.
        # This gets injected into every planning prompt so the LLM knows
        # what agents are available.
        self._agent_descriptions = self._build_agent_descriptions()

    def _build_agent_descriptions(self) -> str:
        """
        Formats AGENT_REGISTRY into a readable string for the planning prompt.

        Output example:
          - web_agent: Searches the internet and retrieves web content.
          - memory_agent: Reads and writes to long-term memory.
          ...
        """
        lines = []
        for name, info in AGENT_REGISTRY.items():
            lines.append(f"  - {name}: {info['description']}")
        return "\n".join(lines)

    def plan(self, goal: str) -> TaskPlan:
        """
        Creates a TaskPlan from a goal string.

        Args:
            goal: Plain English description of what to accomplish.

        Returns:
            TaskPlan containing ordered Task objects.
            If planning fails, returns a single-task fallback plan.
        """
        from core.llm import structured

        print(f"\n[planner] Planning goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")

        prompt = _build_planning_prompt(goal, self._agent_descriptions)

        try:
            data  = structured(prompt=prompt, max_tokens=1024)
            tasks = self._parse_tasks(data, goal)
            plan  = TaskPlan(goal=goal, tasks=tasks)

            print(f"[planner] Generated {len(tasks)}-step plan:")
            for task in tasks:
                deps = f" (after {task.depends_on})" if task.depends_on else ""
                print(f"  Step {task.id}: [{task.assigned_agent or 'unassigned'}] "
                      f"{task.description[:60]}{'...' if len(task.description) > 60 else ''}{deps}")

            return plan

        except Exception as e:
            print(f"[planner] Planning failed ({e}). Using fallback single-task plan.")
            return self._fallback_plan(goal)

    def replan(
        self,
        original_goal:     str,
        completed_tasks:   list[Task],
        failed_tasks:      list[Task],
        failure_reason:    str,
    ) -> TaskPlan:
        """
        Creates a new TaskPlan for the remaining work after partial failure.

        Called by the Orchestrator when the Critic decides the goal hasn't
        been achieved and the plan needs revision.

        Args:
            original_goal:   The user's original goal string.
            completed_tasks: Tasks that finished successfully.
            failed_tasks:    Tasks that failed (tells the LLM what went wrong).
            failure_reason:  The Critic's explanation of why things failed.

        Returns:
            A new TaskPlan for the remaining work only.
        """
        from core.llm import structured

        print(f"[planner] Replanning after failure: {failure_reason[:80]}")

        # Summarise what was already accomplished
        if completed_tasks:
            completed_summary = "\n".join([
                f"  - {t.description}: {str(t.result)[:100]}"
                for t in completed_tasks
            ])
        else:
            completed_summary = "  Nothing was completed successfully."

        prompt = _build_replan_prompt(
            original_goal=original_goal,
            agent_descriptions=self._agent_descriptions,
            completed_summary=completed_summary,
            failure_reason=failure_reason,
        )

        try:
            data  = structured(prompt=prompt, max_tokens=1024)
            tasks = self._parse_tasks(data, original_goal)
            plan  = TaskPlan(goal=original_goal, tasks=tasks)

            print(f"[planner] Replan: {len(tasks)} new steps.")
            return plan

        except Exception as e:
            print(f"[planner] Replan failed ({e}). Returning empty plan.")
            return TaskPlan(goal=original_goal, tasks=[])

    # -------------------------------------------------------------------------
    # Parsing and Validation
    # -------------------------------------------------------------------------

    def _parse_tasks(self, data: dict, goal: str) -> list[Task]:
        """
        Converts the LLM's JSON response into a list of Task objects.
        Validates structure, fills defaults, checks dependency references.

        Raises ValueError if the data structure is unusable.
        """
        raw_tasks = data.get("tasks", [])

        if not raw_tasks:
            raise ValueError("LLM returned empty task list.")

        if not isinstance(raw_tasks, list):
            raise ValueError(f"Expected 'tasks' to be a list, got {type(raw_tasks)}")

        valid_ids = {t.get("id") for t in raw_tasks if isinstance(t, dict)}
        tasks     = []

        for raw in raw_tasks:
            if not isinstance(raw, dict):
                continue

            task_id     = int(raw.get("id", len(tasks) + 1))
            description = str(raw.get("description", "")).strip()
            agent_hint  = str(raw.get("agent_hint", "")).strip() or None
            depends_on  = raw.get("depends_on", [])

            if not description:
                print(f"[planner] Warning: skipping task {task_id} with empty description.")
                continue

            # Validate dependency IDs — skip invalid ones rather than crashing
            if isinstance(depends_on, list):
                valid_deps = [d for d in depends_on if d in valid_ids and d != task_id]
            else:
                valid_deps = []

            # agent_hint goes into assigned_agent — Dispatcher may change it
            task = Task(
                id=task_id,
                description=description,
                assigned_agent=agent_hint if agent_hint in AGENT_REGISTRY else None,
                depends_on=valid_deps,
            )
            tasks.append(task)

        if not tasks:
            raise ValueError("No valid tasks could be parsed from LLM response.")

        # Sort by ID to ensure correct execution order
        tasks.sort(key=lambda t: t.id)
        return tasks

    def _fallback_plan(self, goal: str) -> TaskPlan:
        """
        Returns a single-task plan as a last resort when planning fails.
        The task is unassigned — the Dispatcher will figure out the agent.
        """
        fallback_task = Task(
            id=1,
            description=goal,
            assigned_agent=None,
            depends_on=[],
        )
        return TaskPlan(goal=goal, tasks=[fallback_task])