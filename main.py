# =============================================================================
# main.py — JARVIS MK2 Entry Point
# =============================================================================
#
# WHAT THIS DOES:
# Boots every subsystem in the correct order, then runs the main voice loop.
# This file is intentionally thin — its only job is wiring components together
# and keeping the loop running. All intelligence lives in the components.
#
# BOOT SEQUENCE:
# 1. Ensure required directories exist (memory/, logs/)
# 2. Load long-term memory vault
# 3. Start memory consolidation in background (last session's transcript)
# 4. Initialise all specialist agents
# 5. Boot the Orchestrator (JARVIS) with agents and vault
# 6. Initialise speech I/O (STT and TTS)
# 7. Greet the user and enter the voice loop
#
# THREADING MODEL:
# Each conversation turn runs two threads concurrently:
#
#   Think thread: orchestrator.process() → splits response → puts in queue
#   Main thread:  reads from queue → mouth.speak() each sentence
#
# This means JARVIS starts speaking the first sentence while still generating
# the rest — perceived latency is much lower than actual processing time.
#
# SHUTDOWN:
# Say any shutdown phrase (see SHUTDOWN_PHRASES) to exit cleanly.
# =============================================================================

import threading
import queue
import sys
import re

from config import ensure_directories
from memory.long_term import MemoryVault
from memory.consolidator import consolidate_in_background
from agents.web_agent import WebAgent
from agents.memory_agent import MemoryAgent
from agents.system_agent import SystemAgent
from agents.music_agent import MusicAgent
from agents.research_agent import ResearchAgent
from core.orchestrator import Orchestrator
from in_out.stt import SpeechToText
from in_out.tts import TextToSpeech


# =============================================================================
# Shutdown Triggers
# =============================================================================

SHUTDOWN_PHRASES = [
    "sleep jarvis",
    "shut down",
    "shutdown",
    "goodbye jarvis",
    "power down",
    "go to sleep",
]


# =============================================================================
# Boot
# =============================================================================

def boot() -> tuple:
    """
    Initialises all subsystems in dependency order.

    Returns:
        (orchestrator, ears, mouth)

    Any critical failure raises immediately with a clear error — better to
    crash at boot with a meaningful message than fail silently mid-session.
    """
    print("[boot] JARVIS MK2 initialising...")

    # ── Directories ───────────────────────────────────────────────────────────
    ensure_directories()

    # ── Long-term memory ──────────────────────────────────────────────────────
    print("[boot] Loading memory vault...")
    vault = MemoryVault()

    # ── Memory consolidation (runs in background, doesn't block boot) ─────────
    consolidate_in_background(vault)

    # ── Specialist agents ─────────────────────────────────────────────────────
    print("[boot] Loading agents...")
    agents = {
        "web_agent":      WebAgent(),
        "memory_agent":   MemoryAgent(vault=vault),
        "system_agent":   SystemAgent(),
        "music_agent":    MusicAgent(),
        "research_agent": ResearchAgent(),
    }
    print(f"[boot] Agents loaded: {', '.join(agents.keys())}")

    # ── Orchestrator ──────────────────────────────────────────────────────────
    print("[boot] Starting orchestrator...")
    # on_status is wired after TTS boots — see below
    orchestrator = Orchestrator(agents=agents, vault=vault)

    # ── Speech I/O ────────────────────────────────────────────────────────────
    print("[boot] Initialising speech systems...")
    ears  = SpeechToText()
    mouth = TextToSpeech()

    # Wire up progress narration now that mouth exists
    # on_status calls mouth.speak() — JARVIS narrates what he is doing
    # while agents are working rather than going silent
    orchestrator.on_status = lambda msg: mouth.speak(msg)

    print("[boot] All systems online.")
    return orchestrator, ears, mouth


# =============================================================================
# Banner
# =============================================================================

def print_banner() -> None:
    banner = r"""
    /$$$$$  /$$$$$$  /$$$$$$$  /$$    /$$ /$$$$$$  /$$$$$$ 
   |__  $$ /$$__  $$| $$__  $$| $$   | $$|_  $$_/ /$$__  $$
      | $$| $$  \ $$| $$  \ $$| $$   | $$  | $$  | $$  \__/
      | $$| $$$$$$$$| $$$$$$$/|  $$ / $$/  | $$  |  $$$$$$ 
 /$$  | $$| $$__  $$| $$__  $$ \  $$ $$/   | $$   \____  $$
| $$  | $$| $$  | $$| $$  \ $$  \  $$$/    | $$   /$$  \ $$
|  $$$$$$/| $$  | $$| $$  | $$   \  $/    /$$$$$$|  $$$$$$/
 \______/ |__/  |__/|__/  |__/    \_/    |______/ \______/ 

    ============================================
    [ J.A.R.V.I.S. MK2 — MULTI-AGENT ONLINE ]
    ============================================
    """
    print("\033[96m" + banner + "\033[0m")


# =============================================================================
# Sentence Splitting
# =============================================================================

def _split_sentences(text: str) -> list[str]:
    """
    Splits a response string into individual sentences for streaming TTS.

    Splits on sentence-ending punctuation followed by whitespace.
    Each sentence is yielded with its punctuation attached.
    Empty strings are filtered out.
    """
    parts     = re.split(r'(?<=[.?!])\s+', text.strip())
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text.strip()]


# =============================================================================
# Voice Loop
# =============================================================================

def run_voice_loop(
    orchestrator: Orchestrator,
    ears:         SpeechToText,
    mouth:        TextToSpeech,
) -> None:
    """
    Main loop. Runs forever until a shutdown phrase is spoken.

    Each iteration:
    1. Listen (blocks until user speaks and releases key)
    2. Check for shutdown command
    3. Run orchestrator in background thread, stream sentences to TTS
    4. Wait for audio to finish before listening again
    """
    print("\n[JARVIS] Ready. Hold Right Shift to speak.\n")

    while True:
        try:
            # ── Listen ────────────────────────────────────────────────────────
            user_text = ears.listen()

            if not user_text:
                continue

            print(f"\n[You] {user_text}")

            # ── Shutdown check ────────────────────────────────────────────────
            if any(phrase in user_text.lower() for phrase in SHUTDOWN_PHRASES):
                mouth.speak("Powering down. Goodbye, sir.")
                mouth.wait_until_done()
                print("[boot] Shutdown complete.")
                sys.exit(0)

            # ── Think and speak ───────────────────────────────────────────────
            # sentence_queue carries sentences from the think thread to the
            # main thread. None is the sentinel value signalling completion.
            sentence_queue: queue.Queue = queue.Queue()

            def think_and_queue():
                try:
                    response  = orchestrator.process(user_text)
                    sentences = _split_sentences(response)

                    print(f"\n[JARVIS] ", end="", flush=True)
                    for sentence in sentences:
                        print(sentence, end=" ", flush=True)
                        sentence_queue.put(sentence)
                    print()

                except Exception as e:
                    print(f"\n[ERROR] {e}")
                    # Log the error with context
                    orchestrator.logger.log_error(str(e), context="voice_loop")
                    sentence_queue.put(
                        "I ran into an issue there. Could you repeat that, sir?"
                    )
                finally:
                    sentence_queue.put(None)  # Always signal done

            think_thread = threading.Thread(
                target=think_and_queue,
                daemon=True,
                name="ThinkThread",
            )
            think_thread.start()

            # Read sentences and speak them as they arrive
            while True:
                sentence = sentence_queue.get()
                if sentence is None:
                    break
                mouth.speak(sentence)

            # Block until all audio finishes before listening again
            mouth.wait_until_done()

        except KeyboardInterrupt:
            print("\n[JARVIS] Keyboard interrupt. Shutting down.")
            mouth.speak("Shutting down. Goodbye, sir.")
            mouth.wait_until_done()
            sys.exit(0)

        except Exception as e:
            # Keep the loop alive through unexpected errors
            print(f"\n[ERROR] Voice loop error: {e}")
            mouth.speak("Something went wrong on my end. Ready when you are, sir.")
            mouth.wait_until_done()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        orchestrator, ears, mouth = boot()
        print_banner()
        mouth.speak("I'm on. Good to see you, sir.")
        mouth.wait_until_done()
        run_voice_loop(orchestrator, ears, mouth)

    except EnvironmentError as e:
        print(f"\n[FATAL] Configuration error:\n  {e}")
        sys.exit(1)

    except ImportError as e:
        print(f"\n[FATAL] Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        print(f"\n[FATAL] Boot failed: {e}")
        raise