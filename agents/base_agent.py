# =============================================================================
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

from __future__ import annotations

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
                # Model returned neither tools nor text — unusual but handle it
                print(f"  [{self.name}] Warning: empty response on iteration {iteration}")
                continue

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