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
        return TaskStatus.PENDING# =============================================================================
# agents/base_agent.py — Abstract Base Agent
# =============================================================================
#
# WHAT THIS IS:
# The template that every specialist agent inherits from. It defines the
# standard interface and implements the shared execution logic so each
# specialist only needs to declare what makes it unique.
#
# WHAT IS AN ABSTRACT BASE CLASS?
# An abstract base class (ABC) is a class that cannot be instantiated
# directly — you can't do `agent = BaseAgent()`. It exists only to be
# inherited from. It defines a contract: any class that inherits from it
# MUST implement certain methods, or Python will raise a TypeError the
# moment you try to create an instance.
#
# This is enforced by the @abstractmethod decorator. If WebAgent forgets
# to implement get_system_prompt(), Python tells you immediately when the
# app boots — not silently at runtime when the method gets called.
#
# WHY THIS PATTERN?
# Without a base class, each agent would implement its own loop from scratch.
# That means 5 copies of the same tool execution logic — if you find a bug
# or want to add logging, you fix it in 5 places. With a base class, you
# fix it once and all agents inherit the improvement.
#
# WHAT EACH AGENT MUST PROVIDE (abstract methods):
#   get_system_prompt() → str
#     The persona and instructions for this specific agent.
#     Tells the model what kind of specialist it is and what it should focus on.
#
#   get_tools() → list[dict]
#     The JSON schema list of tools this agent can use.
#     Agents only get the tools relevant to their domain —
#     web_agent gets web tools, system_agent gets OS tools, etc.
#
#   get_tool_map() → dict[str, callable]
#     Maps tool names (strings) to actual Python functions.
#     When the model says "call search_web", this dict lets the executor
#     find and call the right function.
#
# WHAT THE BASE CLASS PROVIDES (implemented methods):
#   run(task, context) → dict
#     The full execution loop. Calls the model, handles tool calls,
#     loops until the model produces text, returns a standard result dict.
#
#   _execute_tool(name, arguments) → str
#     Looks up a tool by name in get_tool_map() and calls it safely.
#     Catches and formats exceptions so a tool failure doesn't crash the loop.
#
#   _build_result(status, summary, result) → dict
#     Builds the standardised return dict that JARVIS reads.
#
# THE STANDARD RESULT FORMAT:
# Every agent.run() returns a dict with exactly these keys:
# {
#     "status":  "done" | "failed",
#     "summary": "One sentence JARVIS can read aloud or relay.",
#     "result":  <the actual data — string, list, dict, etc.>
#     "agent":   "web_agent"   ← which agent produced this
# }
# JARVIS reads "summary" when reporting to the user.
# JARVIS passes "result" as context to subsequent tasks that depend on this one.
# =============================================================================


from abc import ABC, abstractmethod
from typing import Any

from config import MAX_AGENT_ITERATIONS


# =============================================================================
# BaseAgent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all JARVIS specialist agents.

    Subclass this and implement the three abstract methods.
    Call agent.run(task, context) to execute a task.

    Example subclass:
        class WebAgent(BaseAgent):
            def get_system_prompt(self): return "You are a web search specialist..."
            def get_tools(self): return WEB_TOOLS_SCHEMA
            def get_tool_map(self): return WEB_TOOLS_MAP
    """

    def __init__(self, name: str):
        """
        Args:
            name: The functional name of this agent, matching the key in
                  AGENT_REGISTRY in config.py. e.g. "web_agent", "music_agent".
                  Used in log output and result dicts.
        """
        self.name = name

    # -------------------------------------------------------------------------
    # Abstract Methods — Every Subclass MUST Implement These
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for this agent's LLM calls.

        This defines the agent's persona, its domain of expertise,
        and any rules specific to its job. Keep it focused — an agent
        that thinks it can do everything does nothing well.

        Example (for web_agent):
            "You are a web research specialist. Your only job is to find
             information on the internet and return accurate summaries.
             Always cite your sources. Never make up information."
        """
        ...

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """
        Returns the list of tool schemas this agent can use.

        These are the JSON schema dicts in the standard format:
            [{"type": "function", "function": {"name": ..., "parameters": ...}}]

        Agents only receive tools relevant to their domain.
        web_agent gets web search tools. system_agent gets OS tools.
        An agent cannot call tools outside this list.
        """
        ...

    @abstractmethod
    def get_tool_map(self) -> dict[str, callable]:
        """
        Returns a dict mapping tool name strings to callable Python functions.

        When the LLM says "call search_web with query='...'", the executor
        looks up "search_web" in this dict and calls the function.

            {
                "search_web":  search_web,    ← the actual function
                "open_website": open_website,
            }

        Every name in get_tools() must have a corresponding entry here.
        If a name is in get_tools() but not here, the agent will log an
        error when the model tries to call it.
        """
        ...

    # -------------------------------------------------------------------------
    # run() — The Main Entry Point
    #
    # This is what the Orchestrator calls. It runs the full inner agent loop:
    #
    # 1. Build an initial message list with the system prompt and task
    # 2. Call the LLM with available tools
    # 3. If the LLM returned tool calls → execute them, append results, loop
    # 4. If the LLM returned text → that's the answer, package and return it
    # 5. If we hit MAX_AGENT_ITERATIONS without an answer → return failure
    #
    # The context parameter carries results from earlier tasks in the plan.
    # For example, if task 2 depends on task 1's result, the Dispatcher
    # passes task 1's result as context when calling agent.run() for task 2.
    # -------------------------------------------------------------------------

    def run(self, task: str, context: str = "") -> dict:
        """
        Executes a task using the inner tool loop.

        Args:
            task:    Plain English description of what to do.
                     e.g. "Search the web for the latest Python release notes."
            context: Results from previous tasks this one depends on.
                     Empty string if no dependencies.

        Returns:
            A standardised result dict:
            {
                "status":  "done" | "failed",
                "summary": "One-sentence description of what happened.",
                "result":  <the actual output data>,
                "agent":   self.name
            }
        """
        print(f"\n[{self.name}] Starting task: {task[:80]}{'...' if len(task) > 80 else ''}")

        # ── Build the initial message list ────────────────────────────────────
        # We include the task and any context from prior tasks.
        # The system prompt is included as the first message.
        user_content = f"Task: {task}"
        if context.strip():
            user_content += f"\n\nContext from previous steps:\n{context}"

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user",   "content": user_content},
        ]

        tools    = self.get_tools()
        tool_map = self.get_tool_map()

        # ── Inner Tool Loop ───────────────────────────────────────────────────
        # We loop up to MAX_AGENT_ITERATIONS times.
        # Each iteration: call the LLM → handle tool calls or text response.
        # The loop exits when the LLM produces a text response (the answer)
        # or when we've exhausted the iteration limit.
        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            print(f"  [{self.name}] Iteration {iteration}/{MAX_AGENT_ITERATIONS}")

            # Import inside the method to avoid circular imports.
            # base_agent.py → llm.py → config.py is fine, but keeping
            # heavy imports lazy in abstract modules is good practice.
            from core.llm import agent_tool_call

            try:
                response = agent_tool_call(messages=messages, tools=tools)
            except Exception as e:
                # LLM call itself failed (network error, bad response, etc.)
                error_msg = f"LLM call failed on iteration {iteration}: {e}"
                print(f"  [{self.name}] Error: {error_msg}")
                return self._build_result(
                    status="failed",
                    summary=f"{self.name} encountered an error: {e}",
                    result=error_msg,
                )

            # ── Case A: Model wants to call tools ─────────────────────────────
            if response.has_tool_calls:
                # Append the assistant's tool-call decision to message history
                # so the model remembers what it already decided to do
                messages.append({
                    "role":    "assistant",
                    "content": response.text or "",
                })

                # Execute each requested tool and append the results
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    arguments = tc.get("arguments", {})

                    result = self._execute_tool(tool_name, arguments)
                    print(f"  [{self.name}] Tool: {tool_name}({arguments}) → {str(result)[:100]}")

                    messages.append({
                        "role":    "tool",
                        "content": str(result),
                    })

                # Loop back — model will now read the tool results and either
                # call more tools or produce its final answer
                continue

            # ── Case B: Model produced a text response — task complete ────────
            answer = (response.text or "").strip()

            if not answer:
                print(f"  [{self.name}] Warning: empty response on iteration {iteration}")
                continue

            # ── Llama tool-call leak detection ────────────────────────────────────────────
            # llama3.1:8b sometimes outputs its tool-call intent as plain text
            # instead of structured tool_calls, using its special token syntax:
            #   <|python_tag|>play_global_search(query="Thunderstruck")
            # When we detect this, we parse and execute the tool ourselves
            # rather than treating it as the final answer.
            leaked = self._extract_leaked_tool_call(answer, tool_map)
            if leaked:
                tool_name, arguments = leaked
                print(f"  [{self.name}] Leaked tool call detected: {tool_name}({arguments})")
                result = self._execute_tool(tool_name, arguments)
                print(f"  [{self.name}] Tool: {tool_name}({arguments}) → {str(result)[:100]}")
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "tool",      "content": str(result)})
                continue  # Loop back so model can produce a clean text summary

            print(f"  [{self.name}] Done.")
            return self._build_result(
                status="done",
                summary=self._make_summary(answer),
                result=answer,
            )

        # ── Iteration limit reached without a text response ───────────────────
        # The model got stuck in a tool-calling loop without converging.
        # This can happen with small models on complex tasks.
        print(f"  [{self.name}] Failed: exhausted {MAX_AGENT_ITERATIONS} iterations.")
        return self._build_result(
            status="failed",
            summary=(
                f"{self.name} could not complete the task within "
                f"{MAX_AGENT_ITERATIONS} steps."
            ),
            result="Max iterations reached without producing a response.",
        )

    # -------------------------------------------------------------------------
    # _execute_tool() — Safe Tool Execution
    #
    # Looks up a tool by name in the tool map and calls it with the
    # provided arguments. Catches all exceptions and returns them as
    # formatted strings rather than crashing the loop.
    #
    # WHY RETURN ERRORS AS STRINGS?
    # The model needs to see what went wrong so it can decide what to do
    # next — retry with different arguments, try a different tool, or give
    # up and explain the failure. If we raised an exception here, the loop
    # would crash and the model would never get that context.
    # -------------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Executes a tool by name and returns the result as a string.
        Never raises — errors are returned as descriptive strings.

        Args:
            tool_name:  The function name the model requested.
            arguments:  Dict of arguments to pass to the function.

        Returns:
            String result of the tool call, or an error description.
        """
        tool_map = self.get_tool_map()

        if tool_name not in tool_map:
            return (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tool_map.keys())}"
            )

        try:
            result = tool_map[tool_name](**arguments)
            # Ensure result is a string — some tools return dicts or lists
            if isinstance(result, (dict, list)):
                import json
                return json.dumps(result, ensure_ascii=False)
            return str(result)

        except TypeError as e:
            # Argument mismatch — model passed wrong args to the function
            return (
                f"Error calling '{tool_name}': incorrect arguments. "
                f"Detail: {e}"
            )

        except Exception as e:
            return f"Error executing '{tool_name}': {e}"

    # -------------------------------------------------------------------------
    # _make_summary() — Extract a Short Summary From the Full Answer
    #
    # The full answer from an agent might be several paragraphs.
    # The "summary" field in the result dict is what JARVIS reads when
    # reporting back — it should be one sentence, not the full answer.
    #
    # We use the first sentence of the response as a heuristic.
    # Subclasses can override this for more sophisticated summarisation.
    # -------------------------------------------------------------------------

    def _extract_leaked_tool_call(self, text: str, tool_map: dict):
        # Detects Llama tool-call syntax leaking into text output.
        # llama3.1:8b sometimes emits: <|python_tag|>play_global_search(query="X")
        # Returns (tool_name, arguments_dict) or None.
        import re
        import json

        if "<|python_tag|>" not in text:
            return None

        after_tag = text.split("<|python_tag|>", 1)[1].strip()

        for tool_name in tool_map:
            if tool_name not in after_tag:
                continue

            escaped = re.escape(tool_name)

            # Pattern A: tool_name(key="value") or tool_name(key='value')
            m = re.search(escaped + r"\(([^)]*)\)", after_tag)
            if m:
                args_str = m.group(1).strip()
                args = {}
                for kv in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                    args[kv.group(1)] = kv.group(2)
                if not args:
                    for kv in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
                        args[kv.group(1)] = kv.group(2)
                if args:
                    return tool_name, args
                if args_str:
                    first_param = self._get_first_param(tool_name)
                    if first_param:
                        val = args_str.strip('"').strip("'").strip()
                        return tool_name, {first_param: val}

            # Pattern B: "tool_name", {...}  (tool.call style)
            json_pat = '"' + re.escape(tool_name) + r'",\s*(\{[^}]*\})'
            m = re.search(json_pat, after_tag)
            if m:
                try:
                    args = json.loads(m.group(1))
                    if isinstance(args, dict):
                        return tool_name, args
                except (json.JSONDecodeError, ValueError):
                    pass

        return None

    def _get_first_param(self, tool_name: str):
        # Returns the first parameter name of a tool schema, or None.
        for t in self.get_tools():
            fn = t.get("function", {})
            if fn.get("name") == tool_name:
                props = fn.get("parameters", {}).get("properties", {})
                if props:
                    return list(props.keys())[0]
        return None


    def _make_summary(self, full_answer: str) -> str:
        """
        Extracts a one-sentence summary from the full answer.
        Used to populate the "summary" field of the result dict.

        Subclasses can override for custom summarisation logic.
        """
        # Take the first sentence (split on . ? or !)
        for terminator in [". ", "? ", "! ", ".\n"]:
            if terminator in full_answer:
                return full_answer.split(terminator)[0].strip() + terminator.strip()

        # No sentence terminator found — truncate at 120 chars
        if len(full_answer) > 120:
            return full_answer[:117].strip() + "..."

        return full_answer.strip()

    # -------------------------------------------------------------------------
    # _build_result() — Standardised Return Dict Builder
    # -------------------------------------------------------------------------

    def _build_result(
        self,
        status:  str,
        summary: str,
        result:  Any,
    ) -> dict:
        """
        Builds the standardised result dict returned by every agent.

        Args:
            status:  "done" if the task succeeded, "failed" if not.
            summary: One sentence JARVIS can read when reporting status.
            result:  The actual output data — string, list, dict, etc.

        Returns:
            {
                "status":  "done" | "failed",
                "summary": "...",
                "result":  <data>,
                "agent":   self.name
            }
        """
        return {
            "status":  status,
            "summary": summary,
            "result":  result,
            "agent":   self.name,
        }

    # -------------------------------------------------------------------------
    # __repr__
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"