# =============================================================================
# io/tts.py — Text-to-Speech (The Mouth)
# =============================================================================
#
# ENGINE: kokoro_onnx (v1.0 model files)
# FALLBACK: macOS `say` command
#
# THE PIPELINE — why gaps are eliminated:
#
# Old approach (sequential, caused gaps):
#   sentence → synthesise (2s) → play (2s) → [GAP] → synthesise next...
#
# New approach (two concurrent threads, no gaps):
#
#   speak(text) puts sentence in text_queue
#
#   [Synthesis Thread]              [Playback Thread]
#   reads text_queue                reads audio_queue
#   calls kokoro.create()           calls afplay
#   puts numpy array in audio_queue
#
# While sentence N is PLAYING in the playback thread,
# the synthesis thread is already synthesising sentence N+1.
# When N finishes, N+1 is already in audio_queue — zero wait.
#
# SENTINEL PATTERN:
# wait_until_done() needs to know when both threads have fully processed
# the current batch. It pushes a special sentinel object into text_queue.
# The sentinel flows through synthesis → audio_queue unchanged.
# When the playback thread sees it, it sets a threading.Event.
# wait_until_done() blocks on that Event — clean synchronisation.
# =============================================================================

from __future__ import annotations

import os
import subprocess
import threading
import queue
import tempfile
import uuid
from typing import Optional

import numpy as np

from config import TTS_VOICE, TTS_FALLBACK


# =============================================================================
# Internal Data Types
# =============================================================================

class _Sentinel:
    """
    Flows through both queues to mark end of a batch.
    When the playback thread sees this, it sets the done_event
    so wait_until_done() can unblock.
    """
    def __init__(self, done_event: threading.Event):
        self.done_event = done_event


class _AudioItem:
    """Carries a synthesised audio payload from synthesis → playback thread."""
    def __init__(self, samples: np.ndarray, sample_rate: int):
        self.samples     = samples
        self.sample_rate = sample_rate


# =============================================================================
# TextToSpeech
# =============================================================================

class TextToSpeech:
    """
    Pipelined text-to-speech using kokoro_onnx.

    speak(text)         — queues sentence, returns immediately
    wait_until_done()   — blocks until all queued audio has played
    stop()              — clears all queues (interrupt mid-speech)
    """

    def __init__(self):
        print("[tts] Initialising voice output...")

        self._engine = self._init_engine()

        # Two queues: text waiting for synthesis, audio waiting for playback
        self._text_queue:  queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        # Start both worker threads as daemons so they die with the main process
        self._synthesis_thread = threading.Thread(
            target=self._synthesis_worker,
            daemon=True,
            name="TTSSynthesis",
        )
        self._playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True,
            name="TTSPlayback",
        )
        self._synthesis_thread.start()
        self._playback_thread.start()

        print(f"[tts] Ready. Engine: {self._engine}")

    # -------------------------------------------------------------------------
    # Engine Init — kokoro_onnx with say fallback
    # -------------------------------------------------------------------------

    def _init_engine(self) -> str:
        """
        Loads the Kokoro ONNX model.
        Falls back to macOS `say` if model files aren't found or import fails.
        """
        try:
            from kokoro_onnx import Kokoro

            # v1.0 model files — must be in the working directory
            # Download from: https://github.com/thewh1teagle/kokoro-onnx/releases
            self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            return "kokoro_onnx"

        except FileNotFoundError:
            print(
                "[tts] Kokoro model files not found (kokoro-v1.0.onnx / voices-v1.0.bin).\n"
                "[tts] Download from: https://github.com/thewh1teagle/kokoro-onnx/releases\n"
                "[tts] Falling back to macOS 'say' command."
            )
            self._kokoro = None
            return TTS_FALLBACK or "say"

        except ImportError:
            print(
                "[tts] kokoro_onnx not installed. Run: pip install kokoro-onnx\n"
                "[tts] Falling back to macOS 'say' command."
            )
            self._kokoro = None
            return TTS_FALLBACK or "say"

        except Exception as e:
            print(f"[tts] Kokoro init failed ({e}). Falling back to 'say'.")
            self._kokoro = None
            return TTS_FALLBACK or "say"

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """
        Queues text for synthesis and playback. Returns immediately.
        The sentence will be spoken as soon as the synthesis thread
        gets to it and the playback thread is free.
        """
        if not text or not text.strip():
            return
        self._text_queue.put(text.strip())

    def wait_until_done(self) -> None:
        """
        Blocks until all queued sentences have been synthesised AND played.

        Pushes a sentinel into text_queue. The sentinel flows through
        synthesis → audio_queue → playback thread, which sets the
        done_event when it sees it. We block on that event.

        If both queues are already empty this returns almost instantly.
        """
        done_event = threading.Event()
        sentinel   = _Sentinel(done_event)
        self._text_queue.put(sentinel)
        done_event.wait()

    def stop(self) -> None:
        """
        Clears all queued text and audio immediately.
        The sentence currently being synthesised or played finishes —
        we can't interrupt mid-synthesis or mid-afplay easily.
        Useful for barge-in / interrupt support.
        """
        _drain_queue(self._text_queue)
        _drain_queue(self._audio_queue)

    # -------------------------------------------------------------------------
    # Synthesis Worker Thread
    # -------------------------------------------------------------------------

    def _synthesis_worker(self) -> None:
        """
        Reads from text_queue, synthesises audio, puts result in audio_queue.

        Sentinels are passed through unchanged so they reach the playback
        thread and trigger the done_event for wait_until_done().
        """
        while True:
            item = self._text_queue.get()

            # Pass sentinels straight through to playback thread
            if isinstance(item, _Sentinel):
                self._audio_queue.put(item)
                continue

            text = item
            try:
                audio = self._synthesise(text)
                if audio is not None:
                    self._audio_queue.put(audio)
            except Exception as e:
                print(f"[tts/synthesis] Error for '{text[:40]}': {e}")
                # Don't let a synthesis error silently drop the sentence —
                # fall back to say for this specific sentence
                self._audio_queue.put(_FallbackItem(text))

    def _synthesise(self, text: str) -> Optional[_AudioItem]:
        """
        Calls kokoro_onnx to produce a numpy audio array.
        Returns None if synthesis produces no output.
        """
        if self._engine == "kokoro_onnx" and self._kokoro is not None:
            samples, sample_rate = self._kokoro.create(
                text=text,
                voice=TTS_VOICE,
                speed=1.0,
            )
            if samples is None or len(samples) == 0:
                return None
            return _AudioItem(samples=samples, sample_rate=sample_rate)

        # Non-kokoro engine — use fallback
        return _FallbackItem(text)

    # -------------------------------------------------------------------------
    # Playback Worker Thread
    # -------------------------------------------------------------------------

    def _playback_worker(self) -> None:
        """
        Reads from audio_queue and plays each item.
        Sentinels trigger the done_event to unblock wait_until_done().
        """
        while True:
            item = self._audio_queue.get()

            if isinstance(item, _Sentinel):
                item.done_event.set()
                continue

            try:
                if isinstance(item, _FallbackItem):
                    _play_say(item.text)
                elif isinstance(item, _AudioItem):
                    _play_numpy(item.samples, item.sample_rate)
            except Exception as e:
                print(f"[tts/playback] Error: {e}")


# =============================================================================
# Fallback Item — carries raw text for `say` playback
# =============================================================================

class _FallbackItem:
    """Used when kokoro is unavailable for a specific sentence."""
    def __init__(self, text: str):
        self.text = text


# =============================================================================
# Playback Helpers
# =============================================================================

def _play_numpy(samples: np.ndarray, sample_rate: int) -> None:
    """
    Writes audio to a temp WAV file and plays it with afplay.

    afplay blocks until playback finishes — exactly what we want
    in the playback thread so items play sequentially without overlap.
    Temp file is deleted immediately after playback.
    """
    import soundfile as sf

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"jarvis_{uuid.uuid4().hex[:8]}.wav",
    )
    try:
        sf.write(tmp_path, samples, sample_rate, subtype="PCM_16")
        subprocess.run(["afplay", tmp_path], check=True, timeout=60)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _play_say(text: str) -> None:
    """macOS built-in TTS. Used as fallback when kokoro is unavailable."""
    safe = text.replace('"', "'").replace("\\", "")
    subprocess.run(["say", "-v", "Samantha", safe], check=True, timeout=60)


def _drain_queue(q: queue.Queue) -> None:
    """Empties a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break