# =============================================================================
# io/stt.py — Speech-to-Text (The Ears)
# =============================================================================
#
# WHAT THIS DOES:
# Captures audio from the microphone and converts it to text using Groq's
# Whisper API. This is JARVIS's hearing — everything the user says passes
# through here before reaching the Orchestrator.
#
# INPUT METHOD: Push-to-Talk
# Hold Right Shift → speak → release Right Shift → transcription returned.
#
# CHANGE FOR BARGE-IN SUPPORT:
# Previously, listen() created a FRESH pynput.keyboard.Listener every call,
# using on_press/on_release callbacks scoped to that single call. This is an
# EDGE-triggered design — it only notices the key going from up to down
# during the lifetime of that specific Listener. This breaks interrupt
# support: if you press Right Shift WHILE JARVIS is speaking (to interrupt
# him), the key is already held down by the time main.py calls listen()
# again — no new "press" edge ever occurs for the new Listener to catch,
# so recording would never start.
#
# The fix: ONE persistent keyboard listener runs for the entire program
# lifetime, maintaining a threading.Event that reflects the CURRENT state
# of the key (held or not) — a LEVEL-based design, not edge-based. listen()
# now works correctly whether the key was just pressed OR was already held
# down before listen() was called (the barge-in case) — both are handled
# by the same self._key_held.wait() call, since it returns immediately if
# the Event is already set.
#
# is_key_held() exposes this state publicly so main.py's voice loop can
# poll it during TTS playback to detect a barge-in attempt.
#
# WHY GROQ FOR TRANSCRIPTION:
# Groq runs Whisper inference on custom silicon (LPUs) that is dramatically
# faster than CPU or even GPU Whisper. A 5-second clip typically transcribes
# in under 300ms. The API is a drop-in replacement for OpenAI's Whisper API.
# =============================================================================

from __future__ import annotations

import os
import wave
import tempfile
import threading

import numpy as np
import sounddevice as sd
from pynput import keyboard
from groq import Groq

from config import GROQ_API_KEY, STT_MODEL, STT_SAMPLE_RATE


# Minimum recording duration to bother transcribing (seconds).
# Shorter recordings are almost certainly accidental key taps.
MIN_RECORDING_SECONDS = 0.5


class SpeechToText:
    """
    Push-to-talk voice input using Groq Whisper, with barge-in support.

    Usage:
        ears = SpeechToText()
        text = ears.listen()        # blocks until user speaks and releases key
        ears.is_key_held()          # non-blocking — True if currently held down
        # text is "" if nothing was captured or transcription failed
    """

    def __init__(self):
        print("[stt] Initialising speech input (Groq Whisper)...")

        if not GROQ_API_KEY:
            raise EnvironmentError(
                "[stt] GROQ_API_KEY not set. Cannot initialise speech input."
            )

        self._client = Groq(api_key=GROQ_API_KEY)

        # Level-based key state — set while Right Shift is physically held
        # down, cleared on release. This is checked by both listen() (to
        # know when to record) and is_key_held() (for barge-in polling).
        self._key_held = threading.Event()

        # ONE persistent listener for the whole program lifetime — this is
        # what makes barge-in work. A fresh Listener per listen() call
        # would only catch a NEW press, missing the case where the key
        # was already held down (e.g. pressed while JARVIS was mid-sentence).
        self._kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._kb_listener.start()

        print("[stt] Ready. Hold Right Shift to speak.")

    # -------------------------------------------------------------------------
    # Persistent Key State Tracking
    # -------------------------------------------------------------------------

    def _on_press(self, key) -> None:
        if key == keyboard.Key.shift_r:
            self._key_held.set()

    def _on_release(self, key) -> None:
        if key == keyboard.Key.shift_r:
            self._key_held.clear()

    def is_key_held(self) -> bool:
        """
        Non-blocking check of whether the push-to-talk key is CURRENTLY
        held down. Used by main.py's voice loop to detect a barge-in
        attempt while JARVIS is speaking — if this returns True during
        TTS playback, the user is trying to interrupt.
        """
        return self._key_held.is_set()

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def listen(self) -> str:
        """
        Waits for the key to be held (returns immediately if it already is —
        the barge-in case), records audio while held, then transcribes once
        released.

        Returns:
            Transcribed text string. Empty string if nothing usable was captured.
        """
        audio_chunks = self._record()

        if not audio_chunks:
            return ""

        total_samples = sum(len(c) for c in audio_chunks)
        duration_secs = total_samples / STT_SAMPLE_RATE
        if duration_secs < MIN_RECORDING_SECONDS:
            return ""

        return self._transcribe(audio_chunks)

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def _record(self) -> list[np.ndarray]:
        """
        Waits for the key to be held, then records until released.

        self._key_held.wait() is LEVEL-based: if the key is already held
        down (barge-in scenario — user pressed it while JARVIS was still
        speaking), this returns immediately rather than waiting for a new
        press edge. If the key isn't held, it blocks until _on_press sets
        the Event from the persistent listener above.
        """
        print("\n[System] Waiting for Right Shift...", end="", flush=True)
        self._key_held.wait()

        print("\r[🎙️  Recording... release Right Shift to stop]", end="", flush=True)

        audio_chunks: list[np.ndarray] = []
        stream = sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        with stream:
            while self._key_held.is_set():
                chunk, _ = stream.read(1024)
                audio_chunks.append(chunk.copy())

        print()  # Move past the recording status line
        return audio_chunks

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

    def _transcribe(self, audio_chunks: list[np.ndarray]) -> str:
        """
        Writes recorded audio to a temp WAV file and sends it to Groq Whisper.
        Cleans up the temp file whether or not transcription succeeds.
        """
        print("[stt] Transcribing...")

        audio_np = np.concatenate(audio_chunks, axis=0)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(STT_SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            with open(tmp_path, "rb") as audio_file:
                transcription = self._client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_path), audio_file.read()),
                    model=STT_MODEL,
                    language="en",
                    response_format="text",
                )

            return transcription.strip() if isinstance(transcription, str) else ""

        except Exception as e:
            print(f"[stt] Transcription failed: {e}")
            return ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)