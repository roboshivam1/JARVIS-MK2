# =============================================================================
# agents/memory_agent.py — Memory Specialist Agent
# =============================================================================
#
# MM    MM NN   NN EEEEEEE MM    MM  OOOOO   SSSSS  YY   YY NN   NN EEEEEEE
# MMM  MMM NNN  NN EE      MMM  MMM OO   OO SS      YY   YY NNN  NN EE     
# MM MM MM NN N NN EEEEE   MM MM MM OO   OO  SSSSS   YYYYY  NN N NN EEEEE  
# MM    MM NN  NNN EE      MM    MM OO   OO      SS   YYY   NN  NNN EE     
# MM    MM NN   NN EEEEEEE MM    MM  OOOO0   SSSSS    YYY   NN   NN EEEEEEE
#
# WHAT THIS AGENT DOES:
# Handles all read and write operations on JARVIS's long-term memory vault.
# When JARVIS needs to remember something or recall something from the past,
# he delegates that task here.
#
# TOOLS THIS AGENT HAS:
#   search_memory(query)           — fuzzy search the vault
#   store_memory(fact, category,   — add a new fact to the vault
#                importance)
#   forget_memory(fact)            — remove a fact by fuzzy match
#   memory_stats()                 — return vault stats (count, categories)
#
# DESIGN NOTE — WHY A SHARED VAULT INSTANCE?
# The MemoryVault object is created once in main.py and passed into this agent.
# We do NOT create a new MemoryVault() inside the agent itself, because:
#   1. Each MemoryVault() call loads the JSON file from disk
#   2. Multiple instances of MemoryVault pointing at the same file would have
#      out-of-sync in-memory state between writes
#   3. The consolidator also writes to the vault — they need to share the same
#      in-memory object to stay consistent
#
# The vault is injected at construction time (dependency injection pattern).
# =============================================================================

from agents.base_agent import BaseAgent
from memory.long_term import MemoryVault


# =============================================================================
# Tool Functions
#
# These are the actual Python functions the agent can call.
# They're defined as closures below in _make_tool_functions() so they
# can close over the vault instance. This is cleaner than making them
# module-level functions that need a global vault reference.
# =============================================================================

def _make_tool_functions(vault: MemoryVault) -> dict:
    """
    Creates the tool function map closed over the provided vault instance.
    Returns a dict mapping tool name → callable.
    """

    def search_memory(query: str) -> str:
        """Searches long-term memory for facts matching the query."""
        return vault.search_formatted(query, top_k=5)

    def store_memory(
        fact:       str,
        category:   str   = "general",
        importance: float = 0.5,
    ) -> str:
        """Stores a new fact in long-term memory."""
        return vault.add(
            fact=fact,
            category=category,
            source="agent_store",
            importance=importance,
        )

    def forget_memory(fact: str) -> str:
        """Removes a fact from long-term memory by fuzzy match."""
        return vault.remove(fact)

    def memory_stats() -> str:
        """Returns a summary of vault contents — total count and categories."""
        return vault.stats()

    return {
        "search_memory": search_memory,
        "store_memory":  store_memory,
        "forget_memory": forget_memory,
        "memory_stats":  memory_stats,
    }


# =============================================================================
# Tool Schemas
#
# JSON schema definitions for the tools above.
# These are what get passed to the LLM so it knows what tools are available
# and what arguments each one expects.
# =============================================================================

MEMORY_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "search_memory",
            "description": (
                "Searches JARVIS's long-term memory vault for facts matching "
                "the query. Use this to recall past conversations, user "
                "preferences, project details, or anything previously stored."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Keywords to search for. e.g. 'favorite music', 'current project', 'home location'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "store_memory",
            "description": (
                "Stores a new permanent fact in long-term memory. "
                "Use this when the user shares something worth remembering. "
                "Each fact should be a complete, standalone sentence."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "fact": {
                        "type":        "string",
                        "description": "The fact to store as a complete sentence. e.g. 'The user prefers Python over JavaScript.'",
                    },
                    "category": {
                        "type":        "string",
                        "description": "Snake_case category label. e.g. 'preferences', 'projects', 'personal', 'technical'",
                    },
                    "importance": {
                        "type":        "number",
                        "description": "Importance from 0.1 (trivial) to 1.0 (critical). Default 0.5.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "forget_memory",
            "description": (
                "Removes a fact from long-term memory. "
                "Use when the user explicitly asks JARVIS to forget something."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "fact": {
                        "type":        "string",
                        "description": "The fact to remove. Fuzzy matched — exact wording not required.",
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "memory_stats",
            "description": "Returns a summary of vault contents — total memories and category breakdown.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
]


# =============================================================================
# MemoryAgent
# =============================================================================

class MemoryAgent(BaseAgent):
    """
    Specialist agent for reading and writing long-term memory.

    Receives a shared MemoryVault instance at construction so all agents
    and the consolidator work with the same in-memory state.
    """

    def __init__(self, vault: MemoryVault):
        super().__init__(name="memory_agent")
        self._vault    = vault
        self._tool_map = _make_tool_functions(vault)

    def get_system_prompt(self) -> str:
        return (
            "You are a memory specialist. Your only job is to read from and "
            "write to JARVIS's long-term memory vault accurately.\n\n"
            "Rules:\n"
            "1. For recall tasks: search memory with specific, targeted keywords. "
            "If the first search returns nothing useful, try different keywords "
            "before concluding the memory doesn't exist.\n"
            "2. For store tasks: write facts as complete, standalone sentences. "
            "Include enough context that the fact is meaningful on its own — "
            "not just 'Python' but 'The user prefers Python for backend work.'\n"
            "3. For forget tasks: confirm what you removed.\n"
            "4. Never invent or assume facts. Only report what the vault actually contains.\n"
            "5. Be concise in your final response — one clear paragraph maximum."
        )

    def get_tools(self) -> list[dict]:
        return MEMORY_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return self._tool_map