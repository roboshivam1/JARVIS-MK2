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

    def __init__(
        self,
        name:           str,
        model:          str = None,
        provider:       str = None,
        max_iterations: int = None,
        max_tokens:     int = None,
    ):
        """
        Args:
            name:           Functional agent name matching AGENT_REGISTRY key.
                            e.g. "web_agent", "scribe_agent"
            model:          LLM model override. Defaults to AGENT_MODEL (local Ollama).
                            Pass ORCHESTRATOR_MODEL to use the cloud model for this agent.
            provider:       LLM provider override. Defaults to "ollama".
                            Pass "anthropic" or "google" for cloud model agents.
            max_iterations: Per-agent override for how many tool-call rounds
                            this agent gets before giving up. Defaults to the
                            global MAX_AGENT_ITERATIONS from config. Coding
                            tasks legitimately need more rounds than a single
                            web search or system command — write, run, read
                            error, fix, run again is a normal 6+ step cycle —
                            so coding_agent passes a higher value here.
            max_tokens:     Per-agent override for the LLM response token budget
                            PER TOOL CALL. Defaults to 1024 (core/llm.py's
                            default), which is fine for agents whose tool
                            arguments are short (a search query, a file path).
                            It is NOT enough for an agent whose tool arguments
                            include entire file contents — write_file's
                            "content" value competes with everything else in
                            the same 1024-token budget. If a tool call runs out
                            of tokens mid-generation, the incomplete final
                            argument is silently dropped from the parsed JSON
                            rather than raising a clear error — which looks
                            exactly like the model "forgetting" a required
                            argument, repeating identically every iteration
                            since the same prompt hits the same ceiling every
                            time. coding_agent passes a much higher value here
                            for exactly this reason.

        Most agents use the defaults (local Ollama, fast and free).
        Agents that require higher writing or reasoning quality — like
        scribe_agent — can opt into the cloud model by passing model and provider.
        """
        from config import AGENT_MODEL, MAX_AGENT_ITERATIONS
        self.name           = name
        self.model          = model          or AGENT_MODEL
        self.provider       = provider       or "ollama"
        self.max_iterations = max_iterations or MAX_AGENT_ITERATIONS
        self.max_tokens     = max_tokens     or 1024

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
        # Accumulates every tool result so task.result contains actual
        # data, not just the model's confirmation summary.
        tool_results: list[str] = []

        # or when we've exhausted the iteration limit.
        for iteration in range(1, self.max_iterations + 1):
            print(f"  [{self.name}] Iteration {iteration}/{self.max_iterations}")

            # Import inside the method to avoid circular imports.
            # base_agent.py → llm.py → config.py is fine, but keeping
            # heavy imports lazy in abstract modules is good practice.
            from core.llm import tool_call, agent_tool_call

            try:
                # Use cloud model if this agent was constructed with one,
                # otherwise use the default local Ollama agent model.
                # max_tokens is always passed explicitly here — see the
                # __init__ docstring for why the 1024-token default silently
                # truncates tool arguments that contain file content.
                if self.provider != "ollama":
                    response = tool_call(
                        messages=messages,
                        tools=tools,
                        model=self.model,
                        provider=self.provider,
                        max_tokens=self.max_tokens,
                    )
                else:
                    response = agent_tool_call(
                        messages=messages,
                        tools=tools,
                        max_tokens=self.max_tokens,
                    )
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
                # so the model remembers what it already decided to do.
                #
                # We preserve the full tool_calls list (including each call's
                # "id") on this message. Providers that require strict id
                # matching between a tool_use block and its tool_result —
                # Anthropic and OpenAI — read this back out in their message
                # conversion (see core/llm.py _messages_to_anthropic /
                # _messages_to_openai). Providers that don't need ids
                # (Ollama, Google) simply ignore this extra field.
                messages.append({
                    "role":       "assistant",
                    "content":    response.text or "",
                    "tool_calls": response.tool_calls,
                })

                # Execute each requested tool and append the results
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    arguments = tc.get("arguments", {})

                    result = self._execute_tool(tool_name, arguments)
                    print(f"  [{self.name}] Tool: {tool_name}({arguments}) → {str(result)[:100]}")

                    # Capture tool result for inclusion in task.result
                    tool_results.append(f"[{tool_name}]:\n{str(result)}")

                    # tool_call_id carries the matching id forward so
                    # Anthropic/OpenAI can pair this result with the
                    # tool_use block above when the next request is built.
                    messages.append({
                        "role":         "tool",
                        "content":      str(result),
                        "tool_call_id": tc.get("id"),
                        "name":         tool_name,
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
                tool_results.append(f"[{tool_name}]:\n{str(result)}")
                # No real id exists here (the model never made a structured
                # tool call) — tool_call_id is left absent. This path is only
                # ever hit with Ollama-based agents (Llama leaking text), which
                # don't require id matching, so this is safe.
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "tool",      "content": str(result)})
                continue  # Loop back so model can produce a clean text summary

            print(f"  [{self.name}] Done.")

            # Build the full result from tool outputs + model's summary.
            # tool_results contains the raw data (screen analysis, web content,
            # etc). answer is the model's natural language summary of it.
            # We combine both so the orchestrator's _synthesise_response()
            # gets the actual data, not just a one-line confirmation.
            if tool_results:
                full_result = "\n\n".join(tool_results)
                if answer and answer.lower() not in ("confirmed.", "done.", "ok."):
                    full_result += "\n\nAgent summary: " + answer
            else:
                full_result = answer

            return self._build_result(
                status="done",
                summary=self._make_summary(answer),
                result=full_result,
            )

        # ── Iteration limit reached without a text response ───────────────────
        # The model got stuck in a tool-calling loop without converging.
        # This can happen with small models on complex tasks.
        print(f"  [{self.name}] Failed: exhausted {self.max_iterations} iterations.")
        return self._build_result(
            status="failed",
            summary=(
                f"{self.name} could not complete the task within "
                f"{self.max_iterations} steps."
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