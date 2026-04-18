# =============================================================================
# io/tts.py — Text-to-Speech (The Mouth)
# =============================================================================
#
# WHAT THIS DOES:
# Converts JARVIS's text responses into spoken audio and plays them through
# the system speakers. Uses a producer-consumer queue architecture so the
# Orchestrator can keep generating text while audio is already playing —
# giving the feel of real-time speech with no gaps between sentences.
#
# PRIMARY ENGINE: Kokoro TTS (local, high quality, no API key)
# FALLBACK ENGINE: macOS `say` command (zero dependencies, always works)
#
# WHY KOKORO OVER COQUI (MK1)?
# Coqui TTS was abandoned by its maintainers in January 2024. Kokoro is:
# - Actively maintained
# - Significantly better audio quality
# - Faster on Apple Silicon (M-series chips)
# - Simpler installation (pip install kokoro-onnx)
# - No CUDA/GPU dependencies — runs purely on CPU/ANE
#
# THE QUEUE ARCHITECTURE:
# Two threads run concurrently:
#
#   Main thread (producer):
#     Orchestrator generates response text sentence by sentence.
#     Each sentence → speak() → dropped into playback_queue.
#     Returns immediately — doesn't wait for audio to finish.
#
#   Speaker thread (consumer):
#     Constantly reads from playback_queue.
#     For each item: synthesise audio → play → delete temp file.
#     Blocks only itself, never the main thread.
#
# This means JARVIS starts speaking the first sentence while still
# generating the second — latency feels much lower than it actually is.
#
# wait_until_done() blocks the main thread until the queue drains,
# ensuring JARVIS finishes speaking before listening again.
#
# CHANGES FROM MK1:
# - Kokoro replaces Coqui
# - macOS `say` fallback added for zero-dependency situations
# - Worker thread wraps each item in try/finally so task_done() is
#   ALWAYS called — MK1 had a subtle bug where a synthesis error would
#   leave task_done() uncalled, causing wait_until_done() to hang forever
# - Temp files use tempfile.mktemp() for safer naming
# =============================================================================

from __future__ import annotations

import os
import subprocess
import threading
import queue
import tempfile
import uuid

from config import TTS_MODEL, TTS_VOICE, TTS_FALLBACK


class TextToSpeech:
    """
    Asynchronous text-to-speech with queued audio playback.

    Usage:
        mouth = TextToSpeech()
        mouth.speak("Hello sir.")          # returns immediately
        mouth.speak("How can I help?")     # queued behind first sentence
        mouth.wait_until_done()            # blocks until both are spoken
    """

    def __init__(self):
        print("[tts] Initialising voice output...")

        self._engine        = self._init_engine()
        self._playback_queue = queue.Queue()

        # Start the background speaker thread.
        # daemon=True means the thread dies when the main program exits —
        # we don't need it to finish gracefully on shutdown.
        self._speaker_thread = threading.Thread(
            target=self._speaker_worker,
            daemon=True,
            name="TTSSpeaker",
        )
        self._speaker_thread.start()
        print(f"[tts] Ready. Engine: {self._engine}")

    # -------------------------------------------------------------------------
    # Engine Initialisation
    # -------------------------------------------------------------------------

    def _init_engine(self) -> str:
        """
        Tries to initialise Kokoro TTS. Falls back to macOS `say` if unavailable.
        Returns a string identifying which engine was loaded.
        """
        if TTS_MODEL == "kokoro":
            try:
                from kokoro_onnx import Kokoro
                # Kokoro needs the model files in the working directory.
                # Download with: python -c "from kokoro_onnx import Kokoro; Kokoro.from_pretrained()"
                self._kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
                return "kokoro"
            except Exception as e:
                print(f"[tts] Kokoro unavailable ({e}). Falling back to '{TTS_FALLBACK}'.")

        # macOS `say` is always available as last resort
        self._kokoro = None
        return TTS_FALLBACK or "say"

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Queues a text string for audio synthesis and playback.

        Returns immediately — audio plays asynchronously in the background.
        Call wait_until_done() if you need to block until speech finishes.

        Args:
            text: The text to speak. Empty strings are silently ignored.
        """
        if not text or not text.strip():
            return
        self._playback_queue.put(text.strip())

    def wait_until_done(self) -> None:
        """
        Blocks the calling thread until all queued speech has finished playing.

        Call this after the last speak() in a response before listening again,
        so JARVIS doesn't start listening while still talking.
        """
        self._playback_queue.join()

    def stop(self) -> None:
        """
        Clears the playback queue immediately.
        Use to interrupt JARVIS mid-speech (e.g. user interrupts).
        """
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
                self._playback_queue.task_done()
            except queue.Empty:
                break

    # -------------------------------------------------------------------------
    # Speaker Worker Thread
    # -------------------------------------------------------------------------

    def _speaker_worker(self) -> None:
        """
        Background thread. Continuously reads from the queue,
        synthesises audio, plays it, cleans up.

        The try/finally around each item is critical:
        task_done() MUST be called even if synthesis or playback fails.
        If it isn't, wait_until_done() will block forever.
        """
        while True:
            text = self._playback_queue.get()

            try:
                if self._engine == "kokoro":
                    self._speak_kokoro(text)
                else:
                    self._speak_say(text)
            except Exception as e:
                print(f"[tts] Playback error: {e}")
            finally:
                # Always mark the item as processed, no matter what happened
                self._playback_queue.task_done()

    # -------------------------------------------------------------------------
    # Engine Implementations
    # -------------------------------------------------------------------------

    def _speak_kokoro(self, text: str) -> None:
        """
        Synthesises text using Kokoro ONNX and plays it via afplay.

        Kokoro returns a numpy array of audio samples and a sample rate.
        We write that to a temp WAV file and play it with afplay (macOS).
        The temp file is deleted immediately after playback.
        """
        import soundfile as sf

        tmp_path = os.path.join(tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex}.wav")

        try:
            samples, sample_rate = self._kokoro.create(
                text=text,
                voice=TTS_VOICE,
                speed=1.0,
                lang="en-us",
            )

            sf.write(tmp_path, samples, sample_rate)

            # afplay is macOS's built-in audio player — no dependencies
            subprocess.run(
                ["afplay", tmp_path],
                check=True,
                timeout=60,  # Safety: never hang longer than a minute
            )

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _speak_say(self, text: str) -> None:
        """
        Speaks text using macOS's built-in `say` command.

        No dependencies, no temp files. Quality is decent for system voices.
        Siri voices (if downloaded) are significantly better than the default.
        """
        # Sanitise text: the `say` command can be confused by some characters
        safe_text = text.replace('"', "'").replace("\\", "")

        subprocess.run(
            ["say", "-v", "Samantha", safe_text],
            check=True,
            timeout=60,
        )