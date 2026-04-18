# =============================================================================
# agents/system_agent.py — macOS System Control Specialist Agent
# =============================================================================
#
# 8   8 8"""" 8""""8 8   8 8""""8 8"""" 8""""8 ""8"" 8   8 8""""8
# 8   8 8     8    8 8   8 8    8 8     8        8   8   8 8     
# 8eee8 8eeee 8eeee8 8eee8 8eeee8 8eeee 8eeeee   8e  8e  8 8eeeee
# 88  8 88    88     88  8 88   8 88        88   88  88  8     88
# 88  8 88    88     88  8 88   8 88    e   88   88  88  8 e   88
# 88  8 88eee 88     88  8 88   8 88eee 8eee88   88  88ee8 8eee88
#
# WHAT THIS AGENT DOES:
# Controls the macOS operating system — opens applications, adjusts volume,
# takes screenshots, checks battery, manages files. Anything that interacts
# with the machine itself rather than the internet or memory.
#
# TOOLS THIS AGENT HAS:
#   get_current_time()              — returns current date and time
#   get_battery_status()            — checks battery percentage and charge state
#   open_application(app_name)      — launches a macOS application
#   set_volume(level)               — sets system output volume 0-100
#   get_volume()                    — returns current volume level
#   take_screenshot()               — captures screen, returns file path
#   analyze_screen(query)           — screenshot + vision model analysis
#   run_shortcut(shortcut_name)     — runs a macOS Shortcut by name
#
# HOW macOS CONTROL WORKS:
# Two main mechanisms:
#
# 1. subprocess + shell commands
#    For things the OS exposes via the command line:
#      subprocess.run(["open", "-a", "Safari"])
#      subprocess.run(["pmset", "-g", "batt"])
#
# 2. osascript (AppleScript via command line)
#    For things that need to talk to specific apps or system settings:
#      subprocess.run(["osascript", "-e", "set volume output volume 50"])
#    AppleScript is Apple's automation language, built into every Mac.
#    osascript lets us run AppleScript from Python via the terminal.
#
# WHY NOT PyAutoGUI HERE?
# PyAutoGUI simulates mouse and keyboard input at the pixel level.
# It's the right tool when you have no other way to control something.
# For macOS system actions, native APIs (subprocess, osascript) are more
# reliable — they don't break when window positions change or resolutions differ.
# PyAutoGUI belongs in a future browser_agent for web automation.
# =============================================================================

from __future__ import annotations

import subprocess
import os
from datetime import datetime

from agents.base_agent import BaseAgent


# =============================================================================
# Tool Functions
# =============================================================================

def get_current_time() -> str:
    """Returns the current local date and time as a readable string."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def get_battery_status() -> str:
    """
    Returns battery percentage and charging state.
    Uses pmset — macOS's power management command line tool.
    """
    try:
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if not output:
            return "Could not retrieve battery information."
        return output
    except Exception as e:
        return f"Error checking battery: {e}"


def open_application(app_name: str) -> str:
    """
    Opens a macOS application by name.
    Uses `open -a` which searches /Applications and known app locations.
    """
    try:
        subprocess.run(["open", "-a", app_name], check=True, timeout=10)
        return f"Opened {app_name} successfully."
    except subprocess.CalledProcessError:
        return (
            f"Could not open '{app_name}'. "
            f"Make sure the app is installed and the name is correct "
            f"(e.g. 'Visual Studio Code', not 'VSCode')."
        )
    except Exception as e:
        return f"Error opening {app_name}: {e}"


def set_volume(level: int) -> str:
    """
    Sets system output volume to a value between 0 and 100.
    Uses osascript (AppleScript) to control system audio settings.
    """
    try:
        level = max(0, min(100, int(level)))
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {level}"],
            check=True, timeout=5
        )
        return f"Volume set to {level}%."
    except Exception as e:
        return f"Error setting volume: {e}"


def get_volume() -> str:
    """Returns the current system output volume level."""
    try:
        result = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True, timeout=5
        )
        level = result.stdout.strip()
        return f"Current volume is {level}%."
    except Exception as e:
        return f"Error getting volume: {e}"


def take_screenshot() -> str:
    """
    Captures the current screen to a timestamped file in /tmp.
    Returns the file path so the result can be used downstream
    (e.g. passed to analyze_screen).
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = f"/tmp/jarvis_screenshot_{timestamp}.jpg"
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", path],
            check=True, timeout=10
        )
        if os.path.exists(path):
            return f"Screenshot saved to {path}"
        return "Screenshot capture failed — file not created."
    except Exception as e:
        return f"Error taking screenshot: {e}"


def analyze_screen(query: str) -> str:
    """
    Takes a screenshot and analyses it using a local vision model (LLaVA).
    Use when the user asks about what's currently on screen.

    Requires `ollama pull llava` to be run once before first use.
    """
    try:
        import ollama
        from config import VISION_MODEL

        # Capture screen
        path = "/tmp/jarvis_screen_analysis.jpg"
        subprocess.run(
            ["screencapture", "-x", "-t", "jpg", path],
            check=True, timeout=10
        )

        if not os.path.exists(path):
            return "Failed to capture screen for analysis."

        # Analyse with vision model
        response = ollama.generate(
            model=VISION_MODEL,
            prompt=query,
            images=[path],
        )

        # Clean up temp file
        os.remove(path)

        return f"Screen analysis: {response['response']}"

    except ImportError:
        return "Error: ollama not installed. Run: pip install ollama"
    except Exception as e:
        if os.path.exists("/tmp/jarvis_screen_analysis.jpg"):
            os.remove("/tmp/jarvis_screen_analysis.jpg")
        return f"Error analysing screen: {e}"


def run_shortcut(shortcut_name: str) -> str:
    """
    Runs a macOS Shortcut by name.
    The Shortcuts app must have a shortcut with exactly this name.
    Great for triggering complex automations JARVIS can't do directly.
    """
    try:
        subprocess.run(
            ["shortcuts", "run", shortcut_name],
            check=True, timeout=30
        )
        return f"Shortcut '{shortcut_name}' ran successfully."
    except subprocess.CalledProcessError:
        return (
            f"Could not run shortcut '{shortcut_name}'. "
            f"Make sure a shortcut with that exact name exists in the Shortcuts app."
        )
    except FileNotFoundError:
        return "The 'shortcuts' command is not available on this system."
    except Exception as e:
        return f"Error running shortcut '{shortcut_name}': {e}"


# =============================================================================
# Tool Map and Schema
# =============================================================================

SYSTEM_TOOLS_MAP = {
    "get_current_time":  get_current_time,
    "get_battery_status": get_battery_status,
    "open_application":  open_application,
    "set_volume":        set_volume,
    "get_volume":        get_volume,
    "take_screenshot":   take_screenshot,
    "analyze_screen":    analyze_screen,
    "run_shortcut":      run_shortcut,
}

SYSTEM_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "get_current_time",
            "description": "Returns the current local date and time. Use when the user asks what time or date it is.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_battery_status",
            "description": "Returns the current battery percentage and whether the Mac is charging. Use when the user asks about battery or power.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "open_application",
            "description": "Opens a macOS application by its name. Use when the user asks to open, launch, or start an app.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "app_name": {
                        "type":        "string",
                        "description": "Exact application name as it appears in /Applications. e.g. 'Safari', 'Visual Studio Code', 'Spotify'.",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "set_volume",
            "description": "Sets the system output volume. Use when the user asks to change, increase, decrease, or mute the volume.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "level": {
                        "type":        "integer",
                        "description": "Volume level from 0 (mute) to 100 (maximum).",
                    }
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_volume",
            "description": "Returns the current system output volume level.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "take_screenshot",
            "description": "Captures the current screen and saves it to a file. Returns the file path. Use before analyze_screen if you need both.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "analyze_screen",
            "description": "Takes a screenshot and analyses it using a vision model. Use when the user asks what is on screen, to read something visible, or to describe the current display.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "What to look for or answer about the screen. e.g. 'What error message is shown?' or 'Describe what is on screen.'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "run_shortcut",
            "description": "Runs a saved macOS Shortcut by name. Use for complex automations set up in the Shortcuts app.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "shortcut_name": {
                        "type":        "string",
                        "description": "Exact name of the Shortcut as it appears in the Shortcuts app.",
                    }
                },
                "required": ["shortcut_name"],
            },
        },
    },
]


# =============================================================================
# SystemAgent
# =============================================================================

class SystemAgent(BaseAgent):
    """
    Specialist agent for macOS system control.
    Handles time, battery, app launching, volume, screenshots, and shortcuts.
    """

    def __init__(self):
        super().__init__(name="system_agent")

    def get_system_prompt(self) -> str:
        return (
            "You are a macOS system control specialist. Your job is to interact "
            "with the operating system to carry out the user's commands.\n\n"
            "Rules:\n"
            "1. Execute commands directly — don't ask for confirmation unless "
            "   the action is irreversible (e.g. deleting files).\n"
            "2. If an action fails, report the exact error and suggest a fix "
            "   (e.g. 'App not found — check the name in /Applications').\n"
            "3. For volume changes, the user may say 'louder', 'quieter', "
            "   'half volume' etc. — use get_volume first to know the current "
            "   level, then calculate the appropriate target.\n"
            "4. For screen analysis, always use analyze_screen rather than "
            "   take_screenshot alone — analyze_screen does both in one call.\n"
            "5. Be concise — confirm what you did in one sentence."
        )

    def get_tools(self) -> list[dict]:
        return SYSTEM_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return SYSTEM_TOOLS_MAP