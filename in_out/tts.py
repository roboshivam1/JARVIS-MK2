# =============================================================================
# io/tts.py — Text-to-Speech (The Mouth)
# =============================================================================
#
# ENGINE: kokoro_onnx (v1.0 model files)
# FALLBACK: macOS `say` command
#
# THE PIPELINE — why gaps are eliminated:
#
#   speak(text) puts sentence in text_queue
#
#   [Synthesis Thread]              [Playback Thread]
#   reads text_queue                reads audio_queue
#   calls kokoro.create()           runs afplay/say as a tracked subprocess
#   puts numpy array in audio_queue
#
# While sentence N is PLAYING in the playback thread,
# the synthesis thread is already synthesising sentence N+1.
#
# CHANGES FOR BARGE-IN SUPPORT:
#
# 1. is_speaking() — a non-blocking check of whether JARVIS currently has
#    anything queued OR actively playing. main.py's voice loop polls this
#    (instead of only ever blocking on wait_until_done()) so it can also
#    check for an interrupt key-press in the same loop.
#
#    Implemented via a counter (_pending_count) rather than just checking
#    "are both queues empty?" — that check alone has a race: right after
#    item N finishes playing, item N+1 might not have been PUT into
#    audio_queue yet (synthesis takes a moment), so the queues could look
#    momentarily empty even though more speech is still coming. The
#    counter increments the instant speak() is called and only decrements
#    after an item has ACTUALLY finished playing, so it can never read
#    "done" while there's genuinely more speech in flight.
#
# 2. stop() now does a TRUE interrupt, not just a queue-drain. Previously
#    stop() cleared both queues but the sentence ALREADY mid-playback via
#    afplay/say would keep going until naturally finished — meaning an
#    "interrupt" still wouldn't actually go silent until the current
#    sentence ended. Both playback helpers now run as a tracked
#    subprocess.Popen (instead of subprocess.run) so stop() can terminate
#    whatever is currently playing immediately.
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


class _FallbackItem:
    """Used when kokoro is unavailable for a specific sentence."""
    def __init__(self, text: str):
        self.text = text


# =============================================================================
# TextToSpeech
# =============================================================================

class TextToSpeech:
    """
    Pipelined text-to-speech using kokoro_onnx, with barge-in support.

    speak(text)         — queues sentence, returns immediately
    wait_until_done()   — blocks until all queued audio has played
    is_speaking()       — non-blocking check, True if anything queued/playing
    stop()              — immediately kills active playback and clears queues
    """

    def __init__(self):
        print("[tts] Initialising voice output...")

        self._engine = self._init_engine()

        self._text_queue:  queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        # Pending-item counter for is_speaking() — see module docstring
        # for why this is more correct than checking queue emptiness alone.
        self._pending_count = 0
        self._pending_lock  = threading.Lock()

        # Tracks the currently-running afplay/say subprocess so stop() can
        # kill it immediately rather than waiting for it to finish naturally.
        self._current_process      = None
        self._current_process_lock = threading.Lock()

        self._synthesis_thread = threading.Thread(
            target=self._synthesis_worker, daemon=True, name="TTSSynthesis",
        )
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="TTSPlayback",
        )
        self._synthesis_thread.start()
        self._playback_thread.start()

        print(f"[tts] Ready. Engine: {self._engine}")

    # -------------------------------------------------------------------------
    # Engine Init — kokoro_onnx with say fallback
    # -------------------------------------------------------------------------

    def _init_engine(self) -> str:
        try:
            from kokoro_onnx import Kokoro
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

        Increments _pending_count BEFORE queuing — this is what makes
        is_speaking() correct from the very moment speak() is called,
        not just once synthesis/playback actually starts.
        """
        if not text or not text.strip():
            return
        with self._pending_lock:
            self._pending_count += 1
        self._text_queue.put(text.strip())

    def wait_until_done(self) -> None:
        """
        Blocks until all queued sentences have been synthesised AND played.
        Unchanged from before — still useful for boot/shutdown messages
        where interrupt-checking isn't needed, just a simple full wait.
        """
        done_event = threading.Event()
        sentinel   = _Sentinel(done_event)
        self._text_queue.put(sentinel)
        done_event.wait()

    def is_speaking(self) -> bool:
        """
        Non-blocking check: True if anything is queued for synthesis,
        queued for playback, or actively playing right now.

        main.py's voice loop polls this in a loop alongside checking for
        an interrupt key-press — replacing a plain wait_until_done() call
        with something that can also notice "the user wants to interrupt"
        partway through.
        """
        with self._pending_lock:
            return self._pending_count > 0

    def stop(self) -> None:
        """
        Immediately interrupts speech: kills whatever is actively playing
        right now (not just clearing what's queued next), drains both
        queues, and resets the pending counter to zero.

        This is what makes barge-in feel instant rather than "JARVIS
        finishes this sentence, then stops" — the in-progress afplay/say
        subprocess is terminated directly.
        """
        with self._current_process_lock:
            if self._current_process is not None:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass

        _drain_queue(self._text_queue)
        _drain_queue(self._audio_queue)

        with self._pending_lock:
            self._pending_count = 0

    # -------------------------------------------------------------------------
    # Synthesis Worker Thread
    # -------------------------------------------------------------------------

    def _synthesis_worker(self) -> None:
        while True:
            item = self._text_queue.get()

            if isinstance(item, _Sentinel):
                self._audio_queue.put(item)
                continue

            text = item
            try:
                audio = self._synthesise(text)
                if audio is not None:
                    self._audio_queue.put(audio)
                else:
                    # Synthesis produced nothing — still need to decrement
                    # the pending count since no playback will happen for it
                    self._mark_item_finished()
            except Exception as e:
                print(f"[tts/synthesis] Error for '{text[:40]}': {e}")
                self._audio_queue.put(_FallbackItem(text))

    def _synthesise(self, text: str) -> Optional[_AudioItem]:
        if self._engine == "kokoro_onnx" and self._kokoro is not None:
            samples, sample_rate = self._kokoro.create(
                text=text, voice=TTS_VOICE, speed=1.0,
            )
            if samples is None or len(samples) == 0:
                return None
            return _AudioItem(samples=samples, sample_rate=sample_rate)

        return _FallbackItem(text)

    # -------------------------------------------------------------------------
    # Playback Worker Thread
    # -------------------------------------------------------------------------

    def _playback_worker(self) -> None:
        while True:
            item = self._audio_queue.get()

            if isinstance(item, _Sentinel):
                item.done_event.set()
                continue

            try:
                if isinstance(item, _FallbackItem):
                    self._play_say(item.text)
                elif isinstance(item, _AudioItem):
                    self._play_numpy(item.samples, item.sample_rate)
            except Exception as e:
                print(f"[tts/playback] Error: {e}")
            finally:
                self._mark_item_finished()

    def _mark_item_finished(self) -> None:
        """Decrements the pending counter — called once per speak() item,
        only after it has genuinely finished (played or failed), never
        just because it was dequeued."""
        with self._pending_lock:
            self._pending_count = max(0, self._pending_count - 1)

    # -------------------------------------------------------------------------
    # Playback — tracked subprocesses so stop() can kill them mid-play
    # -------------------------------------------------------------------------

    def _play_numpy(self, samples: np.ndarray, sample_rate: int) -> None:
        """
        Writes audio to a temp WAV file and plays it with afplay.

        Uses subprocess.Popen (tracked in self._current_process) rather
        than subprocess.run — this is what lets stop() terminate playback
        immediately instead of waiting for it to finish naturally.
        """
        import soundfile as sf

        tmp_path = os.path.join(
            tempfile.gettempdir(), f"jarvis_{uuid.uuid4().hex[:8]}.wav",
        )
        try:
            sf.write(tmp_path, samples, sample_rate, subtype="PCM_16")

            with self._current_process_lock:
                self._current_process = subprocess.Popen(["afplay", tmp_path])
            self._current_process.wait(timeout=60)

        finally:
            with self._current_process_lock:
                self._current_process = None
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _play_say(self, text: str) -> None:
        """macOS built-in TTS fallback. Same tracked-Popen pattern as above."""
        safe = text.replace('"', "'").replace("\\", "")
        try:
            with self._current_process_lock:
                self._current_process = subprocess.Popen(["say", "-v", "Samantha", safe])
            self._current_process.wait(timeout=60)
        finally:
            with self._current_process_lock:
                self._current_process = None


# =============================================================================
# Module Helpers
# =============================================================================

def _drain_queue(q: queue.Queue) -> None:
    """Empties a queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break