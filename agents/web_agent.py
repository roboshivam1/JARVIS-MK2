# =============================================================================
# agents/web_agent.py — Web Research Specialist Agent
# =============================================================================
#
# 888    888 8888888888 8888888b.  888b     d888 8888888888 .d8888b. 
# 888    888 888        888   Y88b 8888b   d8888 888       d88P  Y88b
# 888    888 888        888    888 88888b.d88888 888       Y88b.     
# 8888888888 8888888    888   d88P 888Y88888P888 8888888    "Y888b.  
# 888    888 888        8888888P"  888 Y888P 888 888           "Y88b.
# 888    888 888        888 T88b   888  Y8P  888 888             "888
# 888    888 888        888  T88b  888   "   888 888       Y88b  d88P
# 888    888 8888888888 888   T88b 888       888 8888888888 "Y8888P" 
#
# WHAT THIS AGENT DOES:
# Handles all internet-facing tasks — searching the web, fetching page
# content, and returning clean summaries. When JARVIS needs live information
# (current news, real-time data, a specific webpage), he delegates here.
#
# TOOLS THIS AGENT HAS:
#   search_web(query)          — DuckDuckGo search, returns top results
#   fetch_page(url)            — fetches and extracts text from a URL
#   search_and_summarise(query) — combined search + fetch top result
#
# WHY DUCKDUCKGO INSTEAD OF GOOGLE?
# DuckDuckGo's search API (via the `duckduckgo-search` / `ddgs` package)
# requires no API key and has no rate limit for personal use. Google Custom
# Search requires an API key and has a 100 query/day free limit. For a
# personal assistant, DuckDuckGo is the pragmatic choice.
#
# WHY NOT SELENIUM HERE?
# Selenium is the right tool for browser automation — clicking buttons,
# filling forms, JavaScript-heavy pages. For simple text extraction from
# static or lightly dynamic pages, the `requests` + `BeautifulSoup` combo
# is faster, lighter, and doesn't require Chrome to be installed.
# Selenium belongs in a separate browser_agent when we build that later.
#
# CONTENT TRUNCATION:
# Web pages can contain tens of thousands of tokens. Local models have
# small context windows (~8K tokens for llama3.1:8b). We truncate fetched
# content aggressively to prevent context overflow. The agent's job is to
# extract and summarise, not to pass entire pages to the model.
# =============================================================================

from __future__ import annotations
import re
import json

from agents.base_agent import BaseAgent


# =============================================================================
# Constants
# =============================================================================

# Maximum characters of page content to pass to the model
# ~3000 chars ≈ ~750 tokens — leaves plenty of room for the model's response
MAX_PAGE_CHARS = 3000

# Maximum number of search results to return
MAX_SEARCH_RESULTS = 4


# =============================================================================
# Tool Functions
# =============================================================================

def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Searches the web using DuckDuckGo and returns formatted results.

    Returns a formatted string with titles, URLs, and snippets so the
    model can decide which result is most relevant for a follow-up fetch.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return (
            "Error: duckduckgo-search not installed. "
            "Run: pip install duckduckgo-search"
        )

    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No results found for '{query}'."

        lines = [f"Search results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"Result {i}:\n"
                f"  Title:   {r.get('title', 'No title')}\n"
                f"  URL:     {r.get('href', 'No URL')}\n"
                f"  Snippet: {r.get('body', 'No snippet')[:200]}\n"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Search failed: {e}"


def fetch_page(url: str) -> str:
    """
    Fetches a URL and returns the main text content.

    Strips HTML tags, scripts, styles, and navigation boilerplate.
    Truncates to MAX_PAGE_CHARS to stay within model context limits.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return (
            "Error: requests or beautifulsoup4 not installed. "
            "Run: pip install requests beautifulsoup4"
        )

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove elements that don't contain meaningful text content
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text).strip()

        if not text:
            return f"Could not extract text from {url}."

        # Truncate to stay within model context limits
        if len(text) > MAX_PAGE_CHARS:
            text = text[:MAX_PAGE_CHARS] + f"\n\n[Content truncated at {MAX_PAGE_CHARS} chars]"

        return f"Content from {url}:\n\n{text}"

    except requests.Timeout:
        return f"Timeout fetching {url} — page took too long to respond."
    except requests.HTTPError as e:
        return f"HTTP error fetching {url}: {e}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def search_and_summarise(query: str) -> str:
    """
    Convenience tool: searches, fetches the top result, returns combined output.

    Useful when the model wants information but doesn't know the exact URL.
    Combines search_web + fetch_page in one tool call, saving an iteration.
    """
    search_result = search_web(query, max_results=1)

    # Extract the URL from the search result using a simple parse
    url_match = re.search(r"URL:\s+(https?://\S+)", search_result)
    if not url_match:
        return search_result  # Return just the search results if no URL found

    url         = url_match.group(1)
    page_content = fetch_page(url)

    return f"{search_result}\n\n---\n\n{page_content}"


# =============================================================================
# Tool Map and Schema
# =============================================================================

WEB_TOOLS_MAP = {
    "search_web":          search_web,
    "fetch_page":          fetch_page,
    "search_and_summarise": search_and_summarise,
}

WEB_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "search_web",
            "description": (
                "Searches the internet using DuckDuckGo and returns titles, "
                "URLs, and snippets for the top results. Use this first to "
                "find relevant URLs, then use fetch_page for full content."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "The search query. Be specific — 'Python 3.13 release notes' not just 'Python'.",
                    },
                    "max_results": {
                        "type":        "integer",
                        "description": f"Number of results to return. Default {MAX_SEARCH_RESULTS}, max 10.",
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
                "Fetches a specific URL and returns the main text content. "
                "Use after search_web to get the full content of a promising result. "
                "Not suitable for JavaScript-heavy pages or pages requiring login."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "url": {
                        "type":        "string",
                        "description": "The full URL to fetch. Must start with http:// or https://",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "search_and_summarise",
            "description": (
                "Combines a web search with fetching the top result in one step. "
                "Use this when you want content from the web but don't have a "
                "specific URL. More efficient than calling search_web then fetch_page separately."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "What to search for and summarise.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# =============================================================================
# WebAgent
# =============================================================================

class WebAgent(BaseAgent):
    """
    Specialist agent for internet research and web content retrieval.
    """

    def __init__(self):
        super().__init__(name="web_agent")

    def get_system_prompt(self) -> str:
        return (
            "You are a web research specialist. Your job is to find accurate, "
            "up-to-date information from the internet and return clear summaries.\n\n"
            "Rules:\n"
            "1. Always search before concluding information isn't available.\n"
            "2. If a search result is insufficient, fetch the full page content.\n"
            "3. Prefer authoritative sources — official docs, reputable news, "
            "   official announcements — over forums and opinion pieces.\n"
            "4. Summarise what you find clearly and factually. Do not invent "
            "   information that wasn't in the search results or page content.\n"
            "5. If you cannot find reliable information, say so clearly rather "
            "   than speculating.\n"
            "6. Keep your final response focused and concise — the task description "
            "   tells you exactly what's needed. Don't pad with extra content."
        )

    def get_tools(self) -> list[dict]:
        return WEB_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return WEB_TOOLS_MAP