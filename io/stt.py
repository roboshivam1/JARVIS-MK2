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
# WHY PUSH-TO-TALK INSTEAD OF WAKE WORD?
# Wake word detection (always-on listening) requires a lightweight model
# running continuously — fine on a dedicated Mac Mini, but on a MacBook
# during development it adds constant CPU load. Push-to-talk is:
# - Zero CPU cost when idle
# - Zero false triggers
# - Simpler to implement and debug
# - Easy to replace later (swap this file for a wake-word version)
#
# WHY GROQ FOR TRANSCRIPTION?
# Groq runs Whisper inference on custom silicon (LPUs) that is dramatically
# faster than CPU or even GPU Whisper. A 5-second clip typically transcribes
# in under 300ms. The free tier is generous for personal use.
# The API is a drop-in replacement for OpenAI's Whisper API — same format.
#
# CHANGES FROM MK1:
# - Cleaner separation of recording and transcription into private methods
# - Explicit silence detection: if you tap the key without speaking,
#   it returns "" rather than sending empty audio to the API
# - Minimum audio length check prevents wasted API calls on accidental taps
# - Console output is cleaner — single-line status updates
# =============================================================================

from __future__ import annotations

import os
import wave
import time
import tempfile

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
    Push-to-talk voice input using Groq Whisper.

    Usage:
        ears = SpeechToText()
        text = ears.listen()   # blocks until user speaks and releases key
        # text is "" if nothing was captured or transcription failed
    """

    def __init__(self):
        print("[stt] Initialising speech input (Groq Whisper)...")

        if not GROQ_API_KEY:
            raise EnvironmentError(
                "[stt] GROQ_API_KEY not set. Cannot initialise speech input."
            )

        self._client       = Groq(api_key=GROQ_API_KEY)
        self._is_recording = False
        self._audio_chunks: list[np.ndarray] = []

        print("[stt] Ready. Hold Right Shift to speak.")

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def listen(self) -> str:
        """
        Waits for the user to hold Right Shift, records audio, releases,
        then transcribes via Groq Whisper.

        Returns:
            Transcribed text string. Empty string if nothing usable was captured.

        Blocks until the key is released.
        """
        self._is_recording = False
        self._audio_chunks = []

        self._wait_and_record()

        if not self._audio_chunks:
            return ""

        # Check minimum duration
        total_samples  = sum(len(c) for c in self._audio_chunks)
        duration_secs  = total_samples / STT_SAMPLE_RATE
        if duration_secs < MIN_RECORDING_SECONDS:
            return ""

        return self._transcribe()

    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------

    def _wait_and_record(self) -> None:
        """
        Opens the mic and records while Right Shift is held.
        Blocks until the key is released.
        """

        def on_press(key):
            if key == keyboard.Key.shift_r and not self._is_recording:
                self._is_recording = True
                print("\r[Recording... release Right Shift to stop]", end="", flush=True)

        def on_release(key):
            if key == keyboard.Key.shift_r:
                self._is_recording = False
                return False  # Stops the listener

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            stream = sd.InputStream(
                samplerate=STT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            with stream:
                while listener.running:
                    if self._is_recording:
                        chunk, _ = stream.read(1024)
                        self._audio_chunks.append(chunk.copy())
                    else:
                        time.sleep(0.02)  # Idle — don't peg the CPU

        print()  # Move past the recording status line

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

    def _transcribe(self) -> str:
        """
        Writes recorded audio to a temp WAV file and sends it to Groq Whisper.
        Cleans up the temp file whether or not transcription succeeds.

        Returns:
            Transcribed text, or "" on failure.
        """
        print("[stt] Transcribing...")

        # Flatten chunks into one array
        audio_np = np.concatenate(self._audio_chunks, axis=0)

        # Write to a named temp file — Groq SDK needs a file-like object
        # with a proper filename so it can detect the audio format
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)        # 2 bytes = int16
                wf.setframerate(STT_SAMPLE_RATE)
                wf.writeframes(audio_np.tobytes())

            with open(tmp_path, "rb") as audio_file:
                transcription = self._client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_path), audio_file.read()),
                    model=STT_MODEL,
                    language="en",
                    response_format="text",
                )

            result = transcription.strip() if isinstance(transcription, str) else ""
            return result

        except Exception as e:
            print(f"[stt] Transcription failed: {e}")
            return ""

        finally:
            # Always clean up the temp file
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)