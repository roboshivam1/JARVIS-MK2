# =============================================================================
# agents/research_agent.py — Deep Research Specialist Agent
# =============================================================================
#
#        d8888 88888888888 888    888 8888888888 888b    888        d8888
#       d88888     888     888    888 888        8888b   888       d88888
#      d88P888     888     888    888 888        88888b  888      d88P888
#     d88P 888     888     8888888888 8888888    888Y88b 888     d88P 888
#    d88P  888     888     888    888 888        888 Y88b888    d88P  888
#   d88P   888     888     888    888 888        888  Y88888   d88P   888
#  d8888888888     888     888    888 888        888   Y8888  d8888888888
# d88P     888     888     888    888 8888888888 888    Y888 d88P     888
# 
# WHAT THIS AGENT DOES:
# Performs multi-source, multi-step research on a topic and synthesises
# the findings into a structured report. This is the "think hard" agent —
# for when a single web search isn't enough and you need JARVIS to actually
# investigate something thoroughly.
#
# HOW IT DIFFERS FROM web_agent:
# web_agent is a tactical tool — it fetches one thing and returns it.
# research_agent is a strategic tool — it plans a research approach,
# gathers from multiple angles, cross-references, and synthesises.
#
# Example distinction:
#   "What is the latest Python version?"   → web_agent (one search)
#   "Compare FastAPI vs Django for my use  → research_agent (multiple
#    case and recommend one"                  searches, structured analysis)
#
# TOOLS THIS AGENT HAS:
#   web_search(query)               — DuckDuckGo search (same as web_agent)
#   fetch_page(url)                 — fetch and extract page content
#   search_multiple(queries)        — runs several searches in one call
#   synthesise(findings, question)  — uses LLM to synthesise gathered info
#
# THE KEY TOOL: search_multiple()
# Research requires gathering from multiple angles. Without search_multiple,
# the agent would need one tool call per search query, each using a full
# iteration. search_multiple runs several queries and returns all results
# in one iteration, making research significantly faster and more thorough.
#
# THE synthesise() TOOL:
# This is unusual — it's a tool that calls the LLM. Agents can call the LLM
# from inside their tool execution. The research agent gathers raw findings
# then calls synthesise() to get a clean, structured response. This produces
# better output than asking the agent model to both research and synthesise
# in its final response — separation of concerns applies to prompting too.
# =============================================================================

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent


# =============================================================================
# Reuse web_agent's fetch tools rather than duplicating them
# =============================================================================
from agents.web_agent import search_web, fetch_page


# =============================================================================
# Research-Specific Tool Functions
# =============================================================================

def search_multiple(queries: list[str]) -> str:
    """
    Runs multiple web searches and returns all results concatenated.

    This is the core efficiency tool for research — instead of one search
    per iteration, gather information from multiple angles in a single step.

    Example use: researching a framework comparison by searching for:
      - "FastAPI performance benchmarks 2024"
      - "Django REST framework pros cons"
      - "FastAPI vs Django real world comparison"
    ...all in one tool call.
    """
    if not queries:
        return "No queries provided."

    if isinstance(queries, str):
        # Handle case where model passes a string instead of a list
        queries = [queries]

    all_results = []
    for i, query in enumerate(queries[:5], 1):  # Cap at 5 to avoid overload
        result = search_web(query, max_results=3)
        all_results.append(f"=== Search {i}: '{query}' ===\n{result}")

    return "\n\n".join(all_results)


def synthesise(findings: str, question: str) -> str:
    """
    Uses the LLM to synthesise raw research findings into a structured answer.

    This tool calls the LLM directly — it's a "meta-tool" that uses AI
    to process the information gathered by other tools.

    WHY CALL THE LLM FROM INSIDE A TOOL?
    The research agent model (local llama3.1:8b) is good at following
    instructions and deciding what to search, but producing a highly
    structured analytical synthesis is harder for a small model in
    a single response. By explicitly calling synthesise(), we:
    1. Give the synthesiser a clean, focused prompt with all findings
    2. Use the full context window just for synthesis, not cluttered
       with tool call history
    3. Get consistently better structured output

    Args:
        findings: All the raw text gathered by search and fetch tools.
        question: The original research question to answer.
    """
    from core.llm import chat

    # Truncate findings if they're too long for the model's context
    max_findings_chars = 6000
    if len(findings) > max_findings_chars:
        findings = findings[:max_findings_chars] + "\n[...truncated for length]"

    messages = [
        {
            "role":    "system",
            "content": (
                "You are a research synthesiser. You receive raw research findings "
                "and a question, then produce a clear, well-structured answer.\n\n"
                "Structure your response with:\n"
                "1. A direct answer to the question (2-3 sentences)\n"
                "2. Key findings that support the answer (bullet points)\n"
                "3. Any important caveats or limitations\n\n"
                "Be factual. Only use information from the provided findings. "
                "If the findings don't fully answer the question, say so."
            ),
        },
        {
            "role":    "user",
            "content": (
                f"Question: {question}\n\n"
                f"Research findings:\n{findings}"
            ),
        },
    ]

    try:
        response = chat(messages=messages, max_tokens=800)
        return response.text or "Synthesis produced no output."
    except Exception as e:
        return f"Synthesis failed: {e}. Raw findings:\n{findings[:1000]}"


# =============================================================================
# Tool Map and Schema
# =============================================================================

RESEARCH_TOOLS_MAP = {
    "web_search":       search_web,
    "fetch_page":       fetch_page,
    "search_multiple":  search_multiple,
    "synthesise":       synthesise,
}

RESEARCH_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "web_search",
            "description": (
                "Search the web for a single query. Returns titles, URLs and snippets. "
                "Use for targeted lookups. For broad research, prefer search_multiple."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type":        "integer",
                        "description": "Number of results, default 4.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "fetch_page",
            "description": (
                "Fetches and extracts the text content of a specific URL. "
                "Use after web_search to get full content of a promising result."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "url": {
                        "type":        "string",
                        "description": "Full URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "search_multiple",
            "description": (
                "Runs multiple web searches at once and returns all results. "
                "Use this for research tasks requiring multiple angles — "
                "it saves iterations by gathering from several queries in one step. "
                "Ideal for comparisons, pros/cons, and multi-faceted topics."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "queries": {
                        "type":        "array",
                        "items":       {"type": "string"},
                        "description": "List of search queries to run. Maximum 5. Each should target a different angle of the research topic.",
                    }
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "synthesise",
            "description": (
                "Uses the AI to synthesise raw research findings into a structured answer. "
                "Call this as your FINAL step after gathering all needed information. "
                "Pass all the findings you collected and the original question."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "findings": {
                        "type":        "string",
                        "description": "All raw research content gathered — search results, page content, etc. Concatenated together.",
                    },
                    "question": {
                        "type":        "string",
                        "description": "The original research question to answer.",
                    },
                },
                "required": ["findings", "question"],
            },
        },
    },
]


# =============================================================================
# ResearchAgent
# =============================================================================

class ResearchAgent(BaseAgent):
    """
    Specialist agent for deep multi-source research and synthesis.
    Uses search_multiple for efficiency and synthesise for structured output.
    """

    def __init__(self):
        super().__init__(name="research_agent")

    def get_system_prompt(self) -> str:
        return (
            "You are a deep research specialist. Your job is to thoroughly "
            "investigate topics and produce well-structured, accurate answers.\n\n"
            "Research process — follow this approach:\n"
            "1. PLAN: Identify 2-4 search angles that would give a complete "
            "   picture of the topic.\n"
            "2. GATHER: Use search_multiple to collect information from all "
            "   angles at once. Fetch full pages for the most relevant results.\n"
            "3. DEEPEN: If a critical angle has insufficient information, "
            "   do a targeted web_search for that specific gap.\n"
            "4. SYNTHESISE: Once you have enough information, call synthesise() "
            "   with ALL your gathered findings and the original question. "
            "   This is always your final tool call.\n\n"
            "Rules:\n"
            "- Always call synthesise() as your last step — never just return "
            "  raw search results as your answer.\n"
            "- Gather from at least 2 different sources before synthesising.\n"
            "- If information is conflicting across sources, note the conflict "
            "  in your synthesis rather than picking one side silently.\n"
            "- Do not invent information not found in your research."
        )

    def get_tools(self) -> list[dict]:
        return RESEARCH_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return RESEARCH_TOOLS_MAP

    def _make_summary(self, full_answer: str) -> str:
        """
        Override the base summary extractor.
        Research answers are structured with sections — extract the
        first paragraph (the direct answer) rather than just the first sentence.
        """
        lines = [l.strip() for l in full_answer.strip().split("\n") if l.strip()]

        # Skip any header lines (lines starting with # or numbers)
        for line in lines:
            if not line.startswith("#") and not line[0].isdigit():
                # Return up to 150 chars of the first content line
                if len(line) > 150:
                    return line[:147] + "..."
                return line

        return super()._make_summary(full_answer)