# =============================================================================
# core/task.py — The Task: The Atom of All Work in JARVIS MK2
# =============================================================================
#
# WHY THIS FILE EXISTS:
# Everything that JARVIS does - whether it takes 1 step or 5 - is broken down into
# Tasks. A task is the smallest unit of work that can be assigned to an agent. 
#
# It has:
#   - What needs to be done (description)
#   - Who should do it (assigned_agent)
#   - What context they need (context)
#   - Whether its done yet (status)
#   - What they found out (result)
#   - What other tasks it depends upon (depends_on)
#
# WHY USE A DATACLASS?
# A dataclass is just a clean way to define a data structure in Python.
# Instead of using a plain dict like {"id": 1, "description": "..."} —
# which gives you no autocomplete, no type checking, and easy typos —
# a dataclass gives you a proper object with named fields, defaults, and
# a nice __repr__ for printing. It's a dict that grew up.
#
# WHY USE AN ENUM FOR STATUS?
# Instead of using raw strings like "done", "failed", "pending" — which means
# a typo like "Done" or "DONE" silently breaks your logic — an Enum gives you
# a fixed set of valid values. TaskStatus.DONE is always exactly that value.
# Python will throw an error if you try to use a status that doesn't exist.
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# =============================================================================
# TaskStatus - The Lifecycle of a Task...
#
# Every task moves through these states in order:
#
#   PENDING → RUNNING → DONE
#                    ↘ FAILED
#
# PENDING:  Task has been created by the Planner but not started yet.
#           This is the default state. Tasks sit here until the Dispatcher
#           picks them up and sends them to an agent.
#
# RUNNING:  An agent has received the task and is currently working on it.
#           If JARVIS crashes while a task is RUNNING, we know exactly
#           which task was in-progress.
#
# DONE:     The agent completed the task successfully.
#           task.result will contain the output.
#
# FAILED:   The agent tried and could not complete the task.
#           task.error will contain the reason.
#           The Critic will see this and decide whether to replan.
#
# SKIPPED:  The task was never attempted, usually because a task it
#           depended on (depends_on) failed first. No point running
#           "summarise search results" if the search itself failed.
# -----------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"

# -----------------------------------------------------------------------------
# Task — The Core Data Structure
#
# Every field explained:
#
# id: int
#   A unique number for this task within a plan. Used by depends_on to
#   reference other tasks. The Planner assigns these sequentially: 1, 2, 3...
#
# description: str
#   A plain English description of what needs to be done. This is what gets
#   sent to the agent. It should be specific enough that an agent can execute
#   it without needing to ask clarifying questions.
#   Good:  "Search the web for Python 3.13 release notes and return a summary"
#   Bad:   "Find Python stuff"
#
# assigned_agent: Optional[str]
#   The name of the agent responsible for this task ("web_agent", "memory_agent").
#   Set by the Dispatcher after the Planner creates the task.
#   None means unassigned — the Dispatcher hasn't processed it yet.
#
# status: TaskStatus
#   Current lifecycle state. Defaults to PENDING.
#
# result: Any
#   Whatever the agent returned when it completed the task.
#   Usually a string (a summary, a piece of data), but could be anything.
#   None if the task hasn't been completed yet.
#
# error: Optional[str]
#   If status is FAILED, this explains why.
#   None if the task succeeded or hasn't run yet.
#
# context: str
#   Additional background information the agent might need.
#   The Dispatcher populates this with results from earlier tasks that
#   this task depends on — so agents don't have to re-do earlier work.
#   Example: task 2 depends on task 1, so task 2's context contains
#            task 1's result before it's sent to the agent.
#
# depends_on: list[int]
#   Task IDs that must be DONE before this task can start.
#   This lets the Planner express dependencies:
#   "Step 3 can't run until Steps 1 and 2 are done."
#   An empty list means no dependencies — the task can start immediately.
# -----------------------------------------------------------------------------

@dataclass
class Task:
    id:             int
    description:    str
    assigned_agent: Optional[str]       = None
    status:         TaskStatus          = TaskStatus.PENDING
    result:         Any                 = None
    error:          Optional[str]       = None
    context:        str                 = ""
    depends_on:     list[int]           = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Helper properties — convenient ways to check state without importing
    # TaskStatus everywhere.
    #
    # Usage:
    #   if task.is_done: ...
    #   if task.is_failed: ...
    #
    # This is cleaner than: if task.status == TaskStatus.DONE: ...
    # -------------------------------------------------------------------------
 
    @property
    def is_done(self) -> bool:
        return self.status == TaskStatus.DONE
 
    @property
    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED
 
    @property
    def is_pending(self) -> bool:
        return self.status == TaskStatus.PENDING
 
    @property
    def is_runnable(self) -> bool:
        """
        A task is runnable if it's PENDING and not waiting on unfinished
        dependencies. The Dispatcher calls this to decide if a task is
        ready to be sent to an agent.
        """
        return self.status == TaskStatus.PENDING
 
    # -------------------------------------------------------------------------
    # __repr__ — How a Task prints itself for debugging.
    #
    # Without this, print(task) gives you something unreadable like:
    #   Task(id=1, description='...', assigned_agent=None, status=<TaskStatus.PENDING>...)
    #
    # With this, you get a clean one-liner:
    #   [Task 1 | PENDING | web_agent] Search the web for Python 3.13 release notes
    # -------------------------------------------------------------------------
 
    def __repr__(self) -> str:
        agent_str = self.assigned_agent or "unassigned"
        return (
            f"[Task {self.id} | {self.status.value.upper()} | {agent_str}] "
            f"{self.description}"
        )

# -----------------------------------------------------------------------------
# TaskPlan — A Container for a Full Plan
#
# When the Planner creates a plan, it returns a TaskPlan — an ordered list
# of Tasks with some metadata about the original goal.
#
# WHY NOT JUST USE A LIST?
# A plain list[Task] works but loses the original goal string — you'd have to
# pass it separately everywhere. Wrapping it in a TaskPlan keeps goal + tasks
# together as a unit, which makes the Dispatcher and Critic code cleaner.
#
# goal: str
#   The original user request that generated this plan.
#   Used by the Critic to evaluate whether the outcome actually satisfied
#   what the user asked for.
#
# tasks: list[Task]
#   The ordered list of tasks. Execute in order, respecting depends_on.
#
# replan_count: int
#   How many times this plan has been revised by the Critic.
#   Compared against MAX_REPLAN_ATTEMPTS in config.py to prevent
#   infinite replanning loops.
# -----------------------------------------------------------------------------

@dataclass
class TaskPlan:
    goal:          str
    tasks:         list[Task]          = field(default_factory=list)
    replan_count:  int                 = 0
 
    def pending_tasks(self) -> list[Task]:
        """Returns only the tasks that haven't been started yet."""
        return [t for t in self.tasks if t.is_pending]
 
    def completed_tasks(self) -> list[Task]:
        """Returns only the tasks that finished successfully."""
        return [t for t in self.tasks if t.is_done]
 
    def failed_tasks(self) -> list[Task]:
        """Returns only the tasks that failed."""
        return [t for t in self.tasks if t.is_failed]
 
    def is_complete(self) -> bool:
        """
        Returns True if every task is in a terminal state — meaning
        there's nothing left to execute (though some may have failed).
        Terminal states are: DONE, FAILED, SKIPPED.
        """
        terminal = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(t.status in terminal for t in self.tasks)
 
    def success_rate(self) -> float:
        """
        Returns the fraction of tasks that succeeded (0.0 to 1.0).
        Used by the Critic to judge overall plan quality.
        A plan where 4/5 tasks succeeded is much better than 1/5.
        """
        if not self.tasks:
            return 0.0
        return len(self.completed_tasks()) / len(self.tasks)

    def __repr__(self) -> str:
        return (
            f"TaskPlan(goal='{self.goal[:50]}...', "
            f"tasks={len(self.tasks)}, "
            f"done={len(self.completed_tasks())}, "
            f"failed={len(self.failed_tasks())})"
        )
