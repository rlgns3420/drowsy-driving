"""A background worker plays tones without blocking landmark inference."""
import logging
import os
import threading
import time

import numpy as np

from .config import AlarmConfig


class Alarm:
    def __init__(self, config: AlarmConfig):
        self.config = config
        self.active = threading.Event()
        self.stopped = threading.Event()
        self._deadline = 0.0
        self._lock = threading.Lock()
        self.error: str | None = None
        self.thread: threading.Thread | None = None
        if config.enabled:
            self.thread = threading.Thread(target=self._run, name="drowsiness-alarm", daemon=True)
            self.thread.start()

    def update(self, active: bool) -> None:
        with self._lock:
            # A frozen inference loop cannot leave the alarm repeating forever.
            self._deadline = time.monotonic() + max(1.0, self.config.repeat_seconds)
            if active and self.config.enabled:
                self.active.set()
            else:
                self.active.clear()

    def _play(self) -> None:
        c = self.config
        if os.name == "nt":
            import winsound
            winsound.Beep(c.frequency, round(c.tone_seconds * 1000))
        else:
            import sounddevice as sd
            rate = 44100
            t = np.arange(round(rate * c.tone_seconds)) / rate
            tone = (0.3 * np.sin(2 * np.pi * c.frequency * t)).astype(np.float32)
            sd.play(tone, rate, blocking=True)

    def _run(self) -> None:
        while not self.stopped.is_set():
            if not self.active.wait(timeout=0.1):
                continue
            with self._lock:
                fresh = self.active.is_set() and time.monotonic() < self._deadline
                if not fresh:
                    self.active.clear()
            if not fresh:
                continue
            if self.stopped.is_set():
                break
            started = time.monotonic()
            try:
                self._play()
                self.error = None
            except Exception as exc:
                if self.error is None:
                    logging.error("Alarm output failed: %s", exc)
                self.error = "Audio unavailable: check output device"
            self.stopped.wait(max(0.0, self.config.repeat_seconds - (time.monotonic() - started)))

    def close(self) -> None:
        self.active.clear()
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=self.config.tone_seconds + 1.0)
