# =============================================================================
# agents/scribe_agent.py — CALLIOPE, The Scribe Agent
# =============================================================================
#
#  dP""b8    db    88     88     88  dP"Yb  88""Yb 888888
# dP   `"   dPYb   88     88     88 dP   Yb 88__dP 88__  
# Yb       dP__Yb  88  .o 88  .o 88 Yb   dP 88"""  88""  
#  YboodP dP""""Yb 88ood8 88ood8 88  YbodP  88     888888
#
# GREEK LORE:
# Calliope is the Muse of eloquence and epic poetry — chief among the nine
# Muses in Greek mythology. Her name means "beautiful voice." She presided
# over all writing and knowledge preservation, and was the muse who inspired
# Homer to capture the greatest stories of the ancient world in written form.
# An agent whose job is to listen to verbal thinking and translate it into
# well-structured written documents is a natural fit for her name.
#
# WHAT THIS AGENT DOES:
# Writes, reads, and updates markdown documents in JARVIS's workspace.
# Captures brainstorms, writes project briefs, maintains notes and decision
# logs — always working from conversation context rather than producing
# generic content from nothing.
#
# V1 SCOPE — DELIBERATELY SIMPLE:
# Four tools only: write_document, read_document, append_to_document,
# list_documents. No versioning, no cross-referencing, no templates yet.
# The value of v1 is clean, high-quality writing from conversation context —
# everything else (versioning, search, ambient capture) is a v2+ addition
# once this foundation is proven useful in daily use.
#
# WHY THIS OVERRIDES BaseAgent's STANDARD TOOL LOOP:
# It doesn't, actually — unlike browser_agent, CALLIOPE uses the normal
# BaseAgent.run() tool loop. The only thing unusual about this agent is
# WHICH MODEL it runs on (see below) and HOW IT RECEIVES CONTEXT (the
# orchestrator injects conversation history into task.context before
# calling run() — see orchestrator._handle_delegate()).
#
# WHY CLAUDE SONNET, NOT LOCAL OLLAMA:
# CALLIOPE's entire value proposition is writing quality. A document that
# reads like generic AI output is worse than no document at all — you'd
# rather write it yourself. Local llama3.1:8b produces competent but flat
# prose. Claude Sonnet genuinely understands document structure, preserves
# nuance from conversation, and writes the way a thoughtful person would.
# This is one of the few agents where the cost of a cloud model call is
# clearly justified — CALLIOPE is invoked occasionally (not constantly like
# a monitoring loop), and the quality bar is load-bearing: a document
# written today may be read and built upon six months from now.
# =============================================================================

from __future__ import annotations

import os
import re
import json
from datetime import datetime
from pathlib import Path

from agents.base_agent import BaseAgent
from config import WORKSPACE_DIR, CALLIOPE_MODEL, CALLIOPE_PROVIDER


# =============================================================================
# Document Registry
#
# A simple JSON index mapping document names to their paths and metadata.
# Lets CALLIOPE (and you) find a document by name without scanning the
# filesystem, and gives a queryable index of everything that's been written.
# =============================================================================

REGISTRY_PATH = os.path.join(WORKSPACE_DIR, "registry.json")


def _load_registry() -> dict:
    """Loads the document registry, creating an empty one if missing."""
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_registry(registry: dict) -> None:
    """Saves the document registry atomically."""
    tmp_path = REGISTRY_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
        os.replace(tmp_path, REGISTRY_PATH)
    except OSError as e:
        print(f"[scribe_agent] Failed to save registry: {e}")


def _register_document(name: str, path: str, doc_type: str, description: str) -> None:
    """Adds or updates an entry in the document registry."""
    registry = _load_registry()
    now = datetime.now().isoformat()

    if name in registry:
        registry[name]["modified"] = now
    else:
        registry[name] = {
            "path":        path,
            "type":        doc_type,
            "description": description,
            "created":     now,
            "modified":    now,
        }
    _save_registry(registry)


def _slugify(title: str) -> str:
    """
    Converts a document title into a safe filename.
    "School Content Engine Brief" -> "school_content_engine_brief"
    """
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug[:80] or "untitled"


# =============================================================================
# Tool Functions
#
# All paths are resolved relative to WORKSPACE_DIR and validated to prevent
# writing outside the workspace — a basic but important safety boundary
# since this agent has filesystem write access.
# =============================================================================

def _resolve_safe_path(relative_path: str) -> str:
    """
    Resolves a path relative to WORKSPACE_DIR and ensures it doesn't escape
    the workspace directory (no '../' tricks, no absolute path overrides).

    Raises ValueError if the resolved path would land outside WORKSPACE_DIR.
    """
    workspace_abs = os.path.abspath(WORKSPACE_DIR)
    target_abs    = os.path.abspath(os.path.join(workspace_abs, relative_path))

    if not target_abs.startswith(workspace_abs):
        raise ValueError(
            f"Path '{relative_path}' resolves outside the workspace directory. "
            f"Documents must stay within {WORKSPACE_DIR}/."
        )
    return target_abs


def write_document(
    title:       str,
    content:     str,
    category:    str = "notes",
) -> str:
    """
    Creates a new markdown document in the workspace.

    Args:
        title:    Human-readable document title. Used to generate the filename
                  and stored in the registry. e.g. "School Content Engine Brief"
        content:  Full markdown content to write. Should be complete,
                  well-structured markdown — headers, bullet points, etc.
        category: Subdirectory within workspace/. One of:
                  "projects", "decisions", "brainstorms", "notes", "research".
                  Defaults to "notes" if uncertain.

    Returns a status string confirming the save location.
    """
    category = category.strip().lower()
    valid_categories = {"projects", "decisions", "brainstorms", "notes", "research"}
    if category not in valid_categories:
        category = "notes"

    slug     = _slugify(title)
    filename = f"{slug}.md"
    rel_path = os.path.join(category, filename)

    try:
        abs_path = _resolve_safe_path(rel_path)
    except ValueError as e:
        return str(e)

    # Don't silently overwrite — if it exists, tell the caller to use append/edit
    if os.path.exists(abs_path):
        return (
            f"A document named '{title}' already exists at {rel_path}. "
            f"Use append_to_document to add to it, or choose a different title."
        )

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        _register_document(
            name=slug,
            path=rel_path,
            doc_type=category,
            description=title,
        )

        return f"Document saved: {rel_path}"

    except OSError as e:
        return f"Failed to write document: {e}"


def read_document(name_or_path: str) -> str:
    """
    Reads an existing document by name (looked up in the registry) or by
    direct relative path within the workspace.

    Args:
        name_or_path: Either a document name as it appears in the registry
                      (e.g. "school_content_engine_brief") or a relative path
                      (e.g. "projects/school_content_engine_brief.md").

    Returns the document content, or an error message if not found.
    """
    registry = _load_registry()

    # Try registry lookup first
    slug = _slugify(name_or_path) if not name_or_path.endswith(".md") else None
    rel_path = None

    if name_or_path in registry:
        rel_path = registry[name_or_path]["path"]
    elif slug and slug in registry:
        rel_path = registry[slug]["path"]
    else:
        # Treat as a direct path
        rel_path = name_or_path

    try:
        abs_path = _resolve_safe_path(rel_path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        # Offer a helpful list of what does exist
        available = ", ".join(registry.keys()) if registry else "none yet"
        return (
            f"No document found matching '{name_or_path}'. "
            f"Available documents: {available}"
        )

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Failed to read document: {e}"


def append_to_document(name_or_path: str, content: str, section_title: str = "") -> str:
    """
    Appends new content to an existing document.

    Args:
        name_or_path:  Document name (registry lookup) or relative path.
        content:       Markdown content to append.
        section_title: Optional header to prepend before the new content,
                       e.g. "Pricing" produces a "## Pricing" header.
                       Leave empty to append without a new section header.

    Returns a status string. Fails clearly if the document doesn't exist —
    use write_document to create new documents, append is only for existing ones.
    """
    registry = _load_registry()
    slug = _slugify(name_or_path) if not name_or_path.endswith(".md") else None

    if name_or_path in registry:
        rel_path = registry[name_or_path]["path"]
    elif slug and slug in registry:
        rel_path = registry[slug]["path"]
    else:
        rel_path = name_or_path

    try:
        abs_path = _resolve_safe_path(rel_path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return (
            f"No document found matching '{name_or_path}'. "
            f"Use write_document to create it first."
        )

    addition = ""
    if section_title.strip():
        addition += f"\n\n## {section_title.strip()}\n\n"
    else:
        addition += "\n\n"
    addition += content

    try:
        with open(abs_path, "a", encoding="utf-8") as f:
            f.write(addition)

        # Update registry's modified timestamp
        key = slug if slug in registry else name_or_path
        if key in registry:
            registry[key]["modified"] = datetime.now().isoformat()
            _save_registry(registry)

        return f"Appended to {rel_path}."

    except OSError as e:
        return f"Failed to append to document: {e}"


def list_documents(category: str = "") -> str:
    """
    Lists all documents in the workspace, optionally filtered by category.

    Args:
        category: Optional filter — "projects", "decisions", "brainstorms",
                  "notes", or "research". Empty string lists everything.

    Returns a formatted string of document names, types, and descriptions.
    """
    registry = _load_registry()

    if not registry:
        return "No documents in the workspace yet."

    category = category.strip().lower()
    items = [
        (name, info) for name, info in registry.items()
        if not category or info.get("type") == category
    ]

    if not items:
        return f"No documents found in category '{category}'."

    lines = [f"Documents{f' in {category}' if category else ''}:"]
    for name, info in sorted(items, key=lambda x: x[1].get("modified", ""), reverse=True):
        modified = info.get("modified", "")[:10]  # just the date part
        lines.append(
            f"  • {info.get('description', name)} "
            f"[{info.get('type', 'notes')}] — {info.get('path')} (modified {modified})"
        )

    return "\n".join(lines)


# =============================================================================
# Tool Map and Schema
# =============================================================================

SCRIBE_TOOLS_MAP = {
    "write_document":      write_document,
    "read_document":       read_document,
    "append_to_document":  append_to_document,
    "list_documents":      list_documents,
}

SCRIBE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "write_document",
            "description": (
                "Creates a new markdown document in the workspace. Use this to "
                "capture project briefs, brainstorm sessions, decision logs, or "
                "notes. Fails if a document with this title already exists — "
                "use append_to_document instead for existing documents."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "title": {
                        "type":        "string",
                        "description": "Human-readable document title, e.g. 'School Content Engine Brief'.",
                    },
                    "content": {
                        "type":        "string",
                        "description": (
                            "Full markdown content. Write well-structured markdown with "
                            "headers (##), bullet points, and clear sections. This should "
                            "read like a document a thoughtful person wrote, not a transcript."
                        ),
                    },
                    "category": {
                        "type":        "string",
                        "enum":        ["projects", "decisions", "brainstorms", "notes", "research"],
                        "description": "Which workspace subdirectory this belongs in. Default 'notes' if unsure.",
                    },
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "read_document",
            "description": (
                "Reads an existing document by name or path. Use this before "
                "appending to or referencing an existing document, to see its "
                "current content first."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "name_or_path": {
                        "type":        "string",
                        "description": "Document name (as registered) or relative workspace path.",
                    }
                },
                "required": ["name_or_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "append_to_document",
            "description": (
                "Adds new content to an existing document without rewriting "
                "the rest of it. Use when the user wants to add a section or "
                "update an existing document rather than create a new one."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "name_or_path": {
                        "type":        "string",
                        "description": "Document name or relative path to append to.",
                    },
                    "content": {
                        "type":        "string",
                        "description": "Markdown content to add.",
                    },
                    "section_title": {
                        "type":        "string",
                        "description": "Optional heading for the new section, e.g. 'Pricing'. Leave empty for no new header.",
                    },
                },
                "required": ["name_or_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "list_documents",
            "description": "Lists documents in the workspace, optionally filtered by category. Use to find what already exists.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "category": {
                        "type":        "string",
                        "enum":        ["projects", "decisions", "brainstorms", "notes", "research"],
                        "description": "Optional category filter. Leave empty to list everything.",
                    }
                },
                "required": [],
            },
        },
    },
]


# =============================================================================
# ScribeAgent
# =============================================================================

class ScribeAgent(BaseAgent):
    """
    CALLIOPE — Writes, reads, and updates markdown documents in the workspace.

    Unlike most agents, this one runs on Claude Sonnet rather than local
    Ollama — see module docstring for why writing quality justifies the
    cloud model cost here.

    Receives conversation history as task.context, injected by the
    Orchestrator before delegation (see orchestrator._handle_delegate()).
    Without that context, CALLIOPE has nothing to write from — she does not
    independently reach into JARVIS's memory or conversation history.
    """

    def __init__(self):
        super().__init__(
            name="scribe_agent",
            model=CALLIOPE_MODEL,
            provider=CALLIOPE_PROVIDER,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are a writing specialist. Your job is to read conversation "
            "context and produce clear, well-structured markdown documents — "
            "the kind a thoughtful person would write, not an AI transcript dump.\n\n"
            "Process:\n"
            "1. Read the conversation context provided. Identify what was actually "
            "   decided, the key ideas, and the structure that makes sense — not "
            "   a chronological retelling of how the conversation unfolded.\n"
            "2. Check list_documents or read_document if the task suggests this "
            "   might be updating something that already exists.\n"
            "3. Write using write_document for new documents, or append_to_document "
            "   to add to existing ones.\n\n"
            "Writing standards:\n"
            "- Use clear markdown structure: headers (##), bullet points, short "
            "  paragraphs. Make the document scannable.\n"
            "- Capture the FINAL STATE of thinking, not the journey — if the "
            "  conversation backtracked or changed direction, write up where "
            "  it ended up, not every turn along the way.\n"
            "- Be concrete. Prefer specific details over vague summaries.\n"
            "- Choose a sensible document type: a project brief has Overview/"
            "  Approach/Architecture/Open Questions sections; a decision log has "
            "  Decision/Context/Reasoning; a brainstorm capture has Core Ideas/"
            "  Decisions Made/Open Questions. Pick what fits the content.\n"
            "- Never write generic filler. Every sentence should carry real content.\n\n"
            "After saving, briefly confirm what you captured and where — one or "
            "two sentences, not a recap of the whole document."
        )

    def get_tools(self) -> list[dict]:
        return SCRIBE_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return SCRIBE_TOOLS_MAP