# =============================================================================
# agents/coding_agent.py — DAEDALUS, The Coding Agent
# =============================================================================
#
# 8888b.     db    888888 8888b.     db    88     88   88 .dP"Y8
#  8I  Yb   dPYb   88__    8I  Yb   dPYb   88     88   88 `Ybo."
#  8I  dY  dP__Yb  88""    8I  dY  dP__Yb  88  .o Y8   8P o.`Y8b
# 8888Y"  dP""""Yb 888888 8888Y"  dP""""Yb 88ood8 `YbodP' 8bodP'
#
# GREEK LORE:
# Daedalus was the master craftsman of Greek mythology — he built the
# Labyrinth, designed the wings that let Icarus fly, and was renowned as
# the greatest inventor and engineer of his age. He solved problems by
# building his way out of them. No name fits an agent that writes, runs,
# and debugs code until it works better than the mythological engineer.
#
# WHAT MAKES THIS AGENT DIFFERENT FROM EVERY OTHER AGENT:
# Every other agent in this system is "done" when the model produces a
# sensible-looking answer. DAEDALUS is done when the code actually RUNS —
# an objective, externally verifiable fact, not something the model just
# asserts. This changes the shape of the work: instead of one or two tool
# calls, a task typically runs write → execute → read error → fix →
# execute again, several times over. This is why DAEDALUS gets a much
# higher max_iterations than other agents (see config.DAEDALUS_MAX_ITERATIONS).
#
# Notably, this agent does NOT override BaseAgent.run() the way browser_agent
# does. The standard tool-call loop already does exactly what's needed here —
# call tools, read results, loop until text. What changes is which tools are
# available and how many iterations are allowed, both handled through the
# normal __init__ parameters (model, provider, max_iterations) added to
# BaseAgent for exactly this purpose.
#
# THE SANDBOX:
# A dedicated directory (config.SANDBOX_DIR, default "sandbox/") completely
# separate from CALLIOPE's workspace/. Code execution and document writing
# have different risk profiles and should never share a folder. All file
# tools resolve paths against this directory and reject anything that would
# escape it — the exact same _resolve_safe_path pattern scribe_agent.py uses
# for workspace/, applied here to sandbox/.
#
# SAFETY MODEL — v1 IS A HARD BLOCK, NOT A CONFIRMATION PROMPT:
# Genuinely interactive "are you sure?" confirmation would require pausing
# the agent loop mid-execution and waiting for a spoken response — that
# needs interrupt/mid-task pause infrastructure this system doesn't have yet
# (see the barge-in support discussed but not yet built). Until that exists,
# dangerous patterns are hard-blocked: the tool refuses to run and returns a
# clear explanation, which DAEDALUS reports back to you in his final answer.
# You can then explicitly ask again with more context if it was a false
# positive, but nothing destructive runs without you reading about it first.
#
# WHY CLAUDE SONNET:
# Debugging is multi-step causal reasoning — forming a hypothesis about WHY
# something failed, not just noticing THAT it failed. This is where local
# models degrade fastest, and the failure mode is worse than a weak answer:
# a misdiagnosed bug produces confidently WRONG code. See config.py for the
# full reasoning — same pattern as CALLIOPE_MODEL, applied here even more
# decisively.
# =============================================================================

from __future__ import annotations

import os
import re
import sys
import subprocess
import tempfile
import uuid

from agents.base_agent import BaseAgent
from config import (
    SANDBOX_DIR,
    DAEDALUS_MODEL,
    DAEDALUS_PROVIDER,
    DAEDALUS_MAX_ITERATIONS,
    DAEDALUS_EXEC_TIMEOUT,
)


# =============================================================================
# Dangerous Pattern Detection
#
# Scoped to genuinely destructive or irreversible operations — filesystem
# deletion outside the sandbox, privilege escalation, fork bombs, piping
# remote scripts straight into a shell. Deliberately NOT banning broad
# things like "subprocess" or "os.system" outright, since legitimate
# scripts use these constructively — the patterns below target the
# specific destructive USE of such calls, not the mere presence of the API.
# =============================================================================

DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",            # rm -rf / or rm -rf /something
    r"rm\s+-rf\s+~",            # rm -rf ~
    r"rm\s+-rf\s+\*",           # rm -rf * (wipes current dir wholesale)
    r"shutil\.rmtree\(\s*['\"]?/",   # rmtree targeting root-ish paths
    r"chmod\s+-R\s+777",
    r"chmod\s+777\s+/",
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;:",  # the classic bash fork bomb
    r"curl[^|]*\|\s*(ba)?sh",         # curl ... | bash / sh
    r"wget[^|]*\|\s*(ba)?sh",
    r"sudo\s+",
    r">\s*/dev/sd[a-z]",         # writing directly to a disk device
    r"mkfs\.",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM\s+\w+\s*;?\s*$",  # unconditional DELETE FROM with no WHERE
]

_DANGEROUS_RE = re.compile("|".join(DANGEROUS_PATTERNS), re.IGNORECASE)


def _check_dangerous(text: str) -> str | None:
    """
    Scans code or command text for genuinely destructive patterns.
    Returns a human-readable reason string if something dangerous is found,
    or None if the text looks safe to execute.
    """
    match = _DANGEROUS_RE.search(text)
    if match:
        return f"Blocked — matched dangerous pattern: '{match.group(0).strip()}'"
    return None


# =============================================================================
# Path Safety — identical pattern to scribe_agent.py, scoped to SANDBOX_DIR
# =============================================================================

def _resolve_safe_path(relative_path: str) -> str:
    """
    Resolves a path relative to SANDBOX_DIR and ensures it doesn't escape
    the sandbox directory. Raises ValueError if it would.
    """
    sandbox_abs = os.path.abspath(SANDBOX_DIR)
    target_abs  = os.path.abspath(os.path.join(sandbox_abs, relative_path))

    if not target_abs.startswith(sandbox_abs):
        raise ValueError(
            f"Path '{relative_path}' resolves outside the sandbox directory. "
            f"Files must stay within {SANDBOX_DIR}/."
        )
    return target_abs


# =============================================================================
# File Tools
# =============================================================================

def read_file(path: str) -> str:
    """Reads a file from the sandbox and returns its content."""
    try:
        abs_path = _resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"File not found: {path}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate very long files so they don't blow the model's context —
        # a model that needs to see more can call read_file with a note to
        # itself, but most debugging needs don't require the whole file.
        if len(content) > 8000:
            return content[:8000] + f"\n\n[... truncated, file is {len(content)} chars total]"
        return content
    except OSError as e:
        return f"Failed to read {path}: {e}"
    except UnicodeDecodeError:
        return f"Could not decode {path} as text — it may be a binary file."


def write_file(path: str, content: str) -> str:
    """
    Creates or overwrites a file in the sandbox.
    Unlike scribe_agent's write_document, this DOES allow overwriting —
    code files get rewritten constantly during iterative debugging, and
    requiring a separate "edit" call for every change would make the
    debug loop needlessly slow. Use edit_file for surgical single-change
    edits when the file is large; write_file for new files or full rewrites.
    """
    danger = _check_dangerous(content)
    if danger:
        return danger

    try:
        abs_path = _resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"
    except OSError as e:
        return f"Failed to write {path}: {e}"


def edit_file(path: str, old_str: str, new_str: str) -> str:
    """
    Surgically replaces an exact string match within an existing file.
    Much more token-efficient than rewriting an entire file to fix one bug.
    Fails clearly if old_str isn't found, or if it matches more than once
    (ambiguous — caller should provide more surrounding context to make
    the match unique).
    """
    try:
        abs_path = _resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"File not found: {path}. Use write_file to create it first."

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"Failed to read {path}: {e}"

    count = content.count(old_str)
    if count == 0:
        return (
            f"Could not find the given text in {path}. "
            f"Check it matches exactly, including whitespace and indentation."
        )
    if count > 1:
        return (
            f"The given text appears {count} times in {path} — ambiguous edit. "
            f"Include more surrounding context so the match is unique."
        )

    new_content = content.replace(old_str, new_str)

    danger = _check_dangerous(new_content)
    if danger:
        return danger

    try:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Edited {path} — replaced 1 occurrence."
    except OSError as e:
        return f"Failed to write {path}: {e}"


def list_files(directory: str = "") -> str:
    """Lists files and folders within the sandbox (or a subdirectory of it)."""
    try:
        abs_path = _resolve_safe_path(directory)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"Directory not found: {directory or '(sandbox root)'}"

    try:
        entries = sorted(os.listdir(abs_path))
        if not entries:
            return f"{directory or '(sandbox root)'} is empty."

        lines = []
        for entry in entries:
            full = os.path.join(abs_path, entry)
            marker = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{marker}")
        return f"Contents of {directory or '(sandbox root)'}:\n" + "\n".join(lines)
    except OSError as e:
        return f"Failed to list {directory}: {e}"


def delete_file(path: str) -> str:
    """Deletes a single file within the sandbox. Cannot delete directories."""
    try:
        abs_path = _resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"File not found: {path}"
    if os.path.isdir(abs_path):
        return f"{path} is a directory, not a file — delete_file only removes single files."

    try:
        os.remove(abs_path)
        return f"Deleted {path}"
    except OSError as e:
        return f"Failed to delete {path}: {e}"


def get_file_tree() -> str:
    """
    Returns the full sandbox directory structure as a string.
    Called by DAEDALUS at the start of a task to understand what already
    exists before deciding whether to create new files or work with
    existing ones — the project-awareness step.
    """
    sandbox_abs = os.path.abspath(SANDBOX_DIR)
    if not os.path.exists(sandbox_abs) or not os.listdir(sandbox_abs):
        return "Sandbox is empty."

    lines = []
    for root, dirs, files in os.walk(sandbox_abs):
        rel_root = os.path.relpath(root, sandbox_abs)
        depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
        indent = "  " * depth
        if rel_root != ".":
            lines.append(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            lines.append(f"{indent}  {f}")

    return "Sandbox structure:\n" + "\n".join(lines)


# =============================================================================
# Execution Tools
# =============================================================================

def run_python(code: str) -> str:
    """
    Executes a Python code snippet in a subprocess and returns stdout/stderr.

    Writes the code to a temporary file inside the sandbox, runs it with
    the same Python interpreter JARVIS is running under, captures output,
    and cleans up. Subprocess isolation means a crashing or hanging script
    can't take down the agent process itself — the timeout below ensures
    it can't hang JARVIS indefinitely either.
    """
    danger = _check_dangerous(code)
    if danger:
        return danger

    os.makedirs(SANDBOX_DIR, exist_ok=True)
    tmp_name = f"_run_{uuid.uuid4().hex[:8]}.py"
    tmp_path = os.path.join(SANDBOX_DIR, tmp_name)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=DAEDALUS_EXEC_TIMEOUT,
            cwd=os.path.abspath(SANDBOX_DIR),
        )

        output = result.stdout
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- exited with code {result.returncode} ---"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {DAEDALUS_EXEC_TIMEOUT}s — check for an infinite loop or blocking call."
    except OSError as e:
        return f"Failed to execute code: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_file(path: str) -> str:
    """
    Executes an existing Python file within the sandbox by path.
    Use this after write_file has already saved the script, rather than
    pasting the whole file content again through run_python.
    """
    try:
        abs_path = _resolve_safe_path(path)
    except ValueError as e:
        return str(e)

    if not os.path.exists(abs_path):
        return f"File not found: {path}"

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"Failed to read {path}: {e}"

    danger = _check_dangerous(content)
    if danger:
        return danger

    try:
        result = subprocess.run(
            [sys.executable, abs_path],
            capture_output=True,
            text=True,
            timeout=DAEDALUS_EXEC_TIMEOUT,
            cwd=os.path.abspath(SANDBOX_DIR),
        )

        output = result.stdout
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- exited with code {result.returncode} ---"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {DAEDALUS_EXEC_TIMEOUT}s — check for an infinite loop or blocking call."
    except OSError as e:
        return f"Failed to execute {path}: {e}"


def run_command(command: str) -> str:
    """
    Executes a shell command within the sandbox directory — for things
    like 'pip install requests', 'git init', or other tooling that isn't
    directly running a Python file.

    Subject to the same dangerous-pattern check as code execution. Runs
    with cwd set to the sandbox so any relative-path side effects (files
    created by the command) land inside the sandbox, not the project root.
    """
    danger = _check_dangerous(command)
    if danger:
        return danger

    os.makedirs(SANDBOX_DIR, exist_ok=True)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=DAEDALUS_EXEC_TIMEOUT,
            cwd=os.path.abspath(SANDBOX_DIR),
        )

        output = result.stdout
        if result.stderr:
            output += f"\n--- stderr ---\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n--- exited with code {result.returncode} ---"

        return output.strip() or "(no output)"

    except subprocess.TimeoutExpired:
        return f"Command timed out after {DAEDALUS_EXEC_TIMEOUT}s."
    except OSError as e:
        return f"Failed to run command: {e}"


# =============================================================================
# Tool Map and Schema
# =============================================================================

CODING_TOOLS_MAP = {
    "read_file":    read_file,
    "write_file":   write_file,
    "edit_file":    edit_file,
    "list_files":   list_files,
    "delete_file":  delete_file,
    "get_file_tree": get_file_tree,
    "run_python":   run_python,
    "run_file":     run_file,
    "run_command":  run_command,
}

CODING_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "read_file",
            "description": "Reads a file from the sandbox. Use before editing to see current content.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path within the sandbox, e.g. 'scraper.py'."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "write_file",
            "description": (
                "Creates a new file or completely overwrites an existing one. "
                "Use for new files, or full rewrites. For small fixes to an "
                "existing file, prefer edit_file — it's more efficient and safer."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative path within the sandbox, e.g. 'scraper.py'."},
                    "content": {"type": "string", "description": "Full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "edit_file",
            "description": (
                "Replaces an exact string match within an existing file — a surgical "
                "fix rather than a full rewrite. old_str must match exactly once in "
                "the file (including whitespace). If it's not unique, include more "
                "surrounding context to disambiguate."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "path":    {"type": "string", "description": "Relative path within the sandbox."},
                    "old_str": {"type": "string", "description": "Exact existing text to find and replace."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "list_files",
            "description": "Lists files and folders in the sandbox or a subdirectory.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "directory": {"type": "string", "description": "Relative subdirectory path, or empty for sandbox root."}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "delete_file",
            "description": "Deletes a single file within the sandbox.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of the file to delete."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_file_tree",
            "description": "Returns the full sandbox directory structure. Call this first when starting a task to see what already exists.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "run_python",
            "description": (
                "Executes a Python code snippet in a subprocess and returns its "
                "output (stdout, stderr, exit code). Use for quick checks or when "
                "no file has been saved yet. For testing a file that's already "
                "been written, use run_file instead."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "run_file",
            "description": "Executes an existing Python file in the sandbox by path and returns its output.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path of the Python file to run."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "run_command",
            "description": (
                "Executes a shell command within the sandbox directory — for "
                "package installation (pip install X), git commands, or other "
                "tooling. Not for running Python files directly; use run_file for that."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run, e.g. 'pip install requests'."}
                },
                "required": ["command"],
            },
        },
    },
]


# =============================================================================
# CodingAgent
# =============================================================================

class CodingAgent(BaseAgent):
    """
    DAEDALUS — Writes, runs, and debugs code in a sandboxed environment.

    Runs on Claude Sonnet (see config.DAEDALUS_MODEL) with a higher iteration
    budget than other agents (config.DAEDALUS_MAX_ITERATIONS) — debugging
    cycles legitimately need more rounds than a single search or system call.

    Uses the standard BaseAgent.run() tool loop unmodified — no override
    needed, unlike browser_agent. The only customisation is the tool set,
    model, and iteration budget, all handled through __init__.
    """

    def __init__(self):
        super().__init__(
            name="coding_agent",
            model=DAEDALUS_MODEL,
            provider=DAEDALUS_PROVIDER,
            max_iterations=DAEDALUS_MAX_ITERATIONS,
        )

    def get_system_prompt(self) -> str:
        return (
            "You are a coding specialist. You write, run, and debug code until it "
            "actually works — not until it looks plausible.\n\n"
            "Process:\n"
            "1. If the task involves an existing project, call get_file_tree or "
            "   list_files first to see what's already there before writing anything.\n"
            "2. Write code with write_file (new files / full rewrites) or edit_file "
            "   (surgical fixes to existing files).\n"
            "3. Run it with run_file (for saved files) or run_python (for quick "
            "   snippets) and READ THE OUTPUT carefully.\n"
            "4. If it fails: diagnose the actual cause from the error message before "
            "   changing anything. Don't guess-and-check randomly — form a specific "
            "   hypothesis about why it failed, then make a targeted fix.\n"
            "5. Re-run after every fix. Do not declare success without seeing the "
            "   code actually execute correctly.\n"
            "6. Use run_command only for tooling (pip install, git) — never to "
            "   re-implement what run_file/run_python already do.\n\n"
            "Rules:\n"
            "- Never claim a task is complete without having actually run the code "
            "  and observed correct output in this session.\n"
            "- If a tool returns a blocked/dangerous-pattern message, do not attempt "
            "  to route around it — explain the concern in your final answer instead.\n"
            "- Prefer edit_file over write_file once a file already has working code "
            "  in it — full rewrites risk losing unrelated working logic.\n"
            "- Be concise in your final summary: what you built, confirmation it runs, "
            "  one sentence on how to use it. Not a full code walkthrough."
        )

    def get_tools(self) -> list[dict]:
        return CODING_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return CODING_TOOLS_MAP