# =============================================================================
# agents/music_agent.py — Music Playback Specialist Agent
# =============================================================================
#
#        d8888 8888888b.   .d88888b.  888      888      .d88888b. 
#       d88888 888   Y88b d88P" "Y88b 888      888     d88P" "Y88b
#      d88P888 888    888 888     888 888      888     888     888
#     d88P 888 888   d88P 888     888 888      888     888     888
#    d88P  888 8888888P"  888     888 888      888     888     888
#   d88P   888 888        888     888 888      888     888     888
#  d8888888888 888        Y88b. .d88P 888      888     Y88b. .d88P
# d88P     888 888         "Y88888P"  88888888 88888888 "Y88888P" 
#
# WHAT THIS AGENT DOES:
# Controls music playback entirely. Searches the iTunes/Apple Music catalog,
# plays local library tracks and playlists, handles playback control (pause,
# skip, previous), and queries what's currently playing.
#
# TOOLS THIS AGENT HAS:
#   play_local_track(track_name)        — plays a track from local library
#   play_playlist(playlist_name)        — plays a local playlist (fuzzy match)
#   play_global_search(query)           — searches iTunes catalog and plays result
#   playback_control(command)           — play / pause / next / previous
#   get_now_playing()                   — returns current track info
#   get_playlists()                     — lists all local playlists
#
# WHY SEPARATE FROM system_agent?
# Music control is a distinct domain with its own set of tools, failure modes,
# and decision logic. A dedicated agent means:
# - The system prompt can be finely tuned for music-specific reasoning
#   (e.g. "try local library first, fall back to global search")
# - Music tool schemas don't crowd the system_agent's tool list
# - You can upgrade music logic independently (add Spotify later)
#
# TWO PLAYBACK STRATEGIES:
# 1. Local library (fast, no internet) — via osascript talking to Music.app
# 2. Global iTunes catalog (requires internet) — via iTunes Search API + open URL
#
# The agent's system prompt instructs it to try local first, then fall back
# to global. This is the multi-step fallback logic that requires the agentic
# loop — a single tool call can't express "try this, and if it fails, try that".
#
# FUZZY PLAYLIST MATCHING:
# When the user says "play my chill playlist", they probably don't remember
# the exact name. difflib.get_close_matches() finds the closest playlist name
# so "chill" matches "Chill Evening Vibes" without needing an exact match.
# =============================================================================

from __future__ import annotations

import subprocess
import difflib
import urllib.parse
import urllib.request
import json
import ssl

from agents.base_agent import BaseAgent


# =============================================================================
# Tool Functions
# =============================================================================

def play_local_track(track_name: str) -> str:
    """
    Plays a specific track from the user's local Apple Music library.
    Uses AppleScript to tell Music.app to play by track name.
    Returns success or failure — caller should fall back to global search on failure.
    """
    try:
        script = f'tell application "Music" to play track "{track_name}"'
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"Playing '{track_name}' from local library."
        else:
            # AppleScript error usually means track not found in library
            return f"Track '{track_name}' not found in local library. Try global search."
    except Exception as e:
        return f"Error playing local track: {e}"


def play_playlist(playlist_name: str) -> str:
    """
    Plays a playlist from the local library using fuzzy name matching.

    First fetches all playlist names, then uses difflib to find the
    closest match to what the user said. This handles "play my workout
    stuff" matching "Workout Mix 2024".
    """
    try:
        # Fetch all playlist names from Music.app
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "Music" to get name of user playlists'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0 or not result.stdout.strip():
            return "Could not retrieve playlists from Music app."

        all_playlists = [p.strip() for p in result.stdout.split(",") if p.strip()]

        if not all_playlists:
            return "No playlists found in Music library."

        # Find closest matching playlist name
        matches = difflib.get_close_matches(
            playlist_name, all_playlists, n=1, cutoff=0.35
        )

        if not matches:
            playlist_list = ", ".join(all_playlists[:10])
            return (
                f"No playlist found matching '{playlist_name}'. "
                f"Available playlists include: {playlist_list}"
            )

        best_match = matches[0]
        script     = f'tell application "Music" to play playlist "{best_match}"'
        subprocess.run(["osascript", "-e", script], check=True, timeout=10)
        return f"Playing playlist '{best_match}'."

    except Exception as e:
        return f"Error playing playlist: {e}"


def play_global_search(query: str) -> str:
    """
    Searches the iTunes global catalog and plays the top result.

    Uses the free iTunes Search API (no key required) to find a track,
    then opens the Apple Music URL which macOS routes to the Music app.
    This works for any song in the Apple Music catalog, not just the
    user's local library.
    """
    try:
        safe_query = urllib.parse.quote(query.replace(" ", "+"))
        url        = f"https://itunes.apple.com/search?term={safe_query}&limit=1&entity=song"

        req     = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        context = ssl._create_unverified_context()

        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("resultCount", 0) == 0:
            return f"No results found in iTunes catalog for '{query}'."

        track      = data["results"][0]
        track_name = track["trackName"]
        artist     = track["artistName"]
        track_url  = track["trackViewUrl"]

        # Open the Apple Music URL — macOS routes this to the Music app
        subprocess.run(["open", track_url], check=True, timeout=10)
        # Trigger play in case Music.app opens but doesn't auto-play
        subprocess.run(
            ["osascript", "-e", 'tell application "Music" to play'],
            check=False, timeout=5
        )

        return f"Playing '{track_name}' by {artist} from Apple Music catalog."

    except urllib.error.URLError:
        return "Could not reach iTunes catalog — check internet connection."
    except Exception as e:
        return f"Error searching global catalog: {e}"


def playback_control(command: str) -> str:
    """
    Controls Music.app playback.
    command must be one of: 'play', 'pause', 'next', 'previous', 'stop'
    """
    command = command.lower().strip()

    script_map = {
        "play":     "tell application \"Music\" to play",
        "pause":    "tell application \"Music\" to pause",
        "stop":     "tell application \"Music\" to stop",
        "next":     "tell application \"Music\" to next track",
        "previous": "tell application \"Music\" to previous track",
        "back":     "tell application \"Music\" to previous track",
        "skip":     "tell application \"Music\" to next track",
    }

    script = script_map.get(command)
    if not script:
        return (
            f"Unknown command '{command}'. "
            f"Valid commands: {', '.join(script_map.keys())}"
        )

    try:
        subprocess.run(["osascript", "-e", script], check=True, timeout=5)
        return f"Music: {command}."
    except Exception as e:
        return f"Error executing '{command}': {e}"


def get_now_playing() -> str:
    """Returns the name and artist of the currently playing track."""
    try:
        script = (
            'tell application "Music"\n'
            '  if player state is playing then\n'
            '    set t to name of current track\n'
            '    set a to artist of current track\n'
            '    return t & " by " & a\n'
            '  else\n'
            '    return "Nothing is currently playing."\n'
            '  end if\n'
            'end tell'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output if output else "Nothing is currently playing."
    except Exception as e:
        return f"Error getting now playing: {e}"


def get_playlists() -> str:
    """Returns all playlist names from the local Music library."""
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "Music" to get name of user playlists'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "No playlists found or Music app is not open."

        playlists = [p.strip() for p in result.stdout.split(",") if p.strip()]
        return "Local playlists:\n" + "\n".join(f"  • {p}" for p in playlists)
    except Exception as e:
        return f"Error retrieving playlists: {e}"


# =============================================================================
# Tool Map and Schema
# =============================================================================

MUSIC_TOOLS_MAP = {
    "play_local_track":   play_local_track,
    "play_playlist":      play_playlist,
    "play_global_search": play_global_search,
    "playback_control":   playback_control,
    "get_now_playing":    get_now_playing,
    "get_playlists":      get_playlists,
}

MUSIC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name":        "play_local_track",
            "description": (
                "Plays a specific track from the user's local Apple Music library. "
                "Try this FIRST before global search. If it fails, use play_global_search."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "track_name": {
                        "type":        "string",
                        "description": "The track name to search for in the local library.",
                    }
                },
                "required": ["track_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "play_playlist",
            "description": (
                "Plays a playlist from the local library. Uses fuzzy matching "
                "so the exact name is not required. Use get_playlists first "
                "if you're unsure what playlists exist."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "playlist_name": {
                        "type":        "string",
                        "description": "The playlist name or partial name to search for.",
                    }
                },
                "required": ["playlist_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "play_global_search",
            "description": (
                "Searches the full Apple Music catalog and plays the top result. "
                "Use when the track is not in the local library or when the user "
                "asks for a specific song by a specific artist."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "Search query — song name and/or artist. e.g. 'Bohemian Rhapsody Queen'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "playback_control",
            "description": "Controls Music.app playback. Use for pause, resume, skip, previous, stop.",
            "parameters": {
                "type":       "object",
                "properties": {
                    "command": {
                        "type":        "string",
                        "enum":        ["play", "pause", "stop", "next", "previous", "skip", "back"],
                        "description": "The playback command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_now_playing",
            "description": "Returns the name and artist of the currently playing track.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_playlists",
            "description": "Lists all playlists in the local Music library. Use to find exact playlist names.",
            "parameters":  {"type": "object", "properties": {}, "required": []},
        },
    },
]


# =============================================================================
# MusicAgent
# =============================================================================

class MusicAgent(BaseAgent):
    """
    Specialist agent for Apple Music control and playback management.
    """

    def __init__(self):
        super().__init__(name="music_agent")

    def get_system_prompt(self) -> str:
        return (
            "You are a music playback specialist controlling Apple Music on macOS.\n\n"
            "Playback strategy — follow this order:\n"
            "1. If the user wants a specific song: try play_local_track first.\n"
            "   If it fails, immediately fall back to play_global_search.\n"
            "2. If the user wants a playlist: use play_playlist with the name they gave.\n"
            "   If no match, call get_playlists and suggest the closest one.\n"
            "3. For playback control (pause, skip, etc.): use playback_control directly.\n"
            "4. For 'what's playing': use get_now_playing.\n\n"
            "Rules:\n"
            "- Never ask the user if they want local or global — try local first, "
            "  fall back to global automatically.\n"
            "- When reporting what you're playing, include track name and artist.\n"
            "- Be concise — one sentence confirmation is enough."
        )

    def get_tools(self) -> list[dict]:
        return MUSIC_TOOLS_SCHEMA

    def get_tool_map(self) -> dict:
        return MUSIC_TOOLS_MAP