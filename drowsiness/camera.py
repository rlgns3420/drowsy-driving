import os
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from .config import CameraConfig


class Webcam:
    def __init__(self, config: CameraConfig):
        names = (["dshow", "msmf", "any"] if os.name == "nt" else ["v4l2", "any"])
        if config.backend != "auto":
            names = [config.backend]
        backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                    "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}
        self.capture = None
        self._first_frame = None
        for name in names:
            candidate = cv2.VideoCapture(config.index, backends[name])
            try:
                if candidate.isOpened():
                    candidate.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                    candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                    candidate.set(cv2.CAP_PROP_FPS, config.fps)
                    candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Some drivers open successfully but never return an image.
                    for _ in range(3):
                        ok, frame = candidate.read()
                        if ok and frame is not None and frame.size:
                            self.capture, self.backend = candidate, name
                            self._first_frame = (frame, time.monotonic())
                            return
            except cv2.error:
                pass
            finally:
                if self.capture is not candidate:
                    candidate.release()
        raise RuntimeError(f"Cannot read camera {config.index}. Connect a USB/built-in webcam, "
                           "close other camera apps, and check Windows camera permissions.")

    def read(self):
        if self._first_frame is not None:
            frame, _ = self._first_frame
            self._first_frame = None
            return frame, time.monotonic()
        ok, frame = self.capture.read()
        timestamp = time.monotonic()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Webcam read failed; reconnect the camera and restart")
        return frame, timestamp

    def close(self) -> None:
        self.capture.release()


@dataclass(frozen=True)
class CameraDevice:
    index: int
    backend: str
    width: int
    height: int


def discover_cameras(config: CameraConfig, limit: int = 4) -> list[CameraDevice]:
    devices = []
    for index in range(limit):
        try:
            camera = Webcam(replace(config, index=index))
        except (RuntimeError, cv2.error):
            continue
        try:
            frame, _ = camera.read()
            height, width = frame.shape[:2]
            devices.append(CameraDevice(index, camera.backend, width, height))
        finally:
            camera.close()
    return devices


class VideoFile:
    """Offline timestamps use video time, never the speed of replay."""
    def __init__(self, path: str | Path, fps: float | None = None):
        if fps is not None and (not math.isfinite(fps) or not 0 < fps <= 240):
            raise ValueError("FPS override must be finite and between 0 and 240")
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open video: {Path(path).name}")
        self.fps = fps if fps is not None else self.capture.get(cv2.CAP_PROP_FPS)
        if not 0 < self.fps <= 240:
            self.capture.release()
            raise ValueError("Video must have a valid frame rate (0 < FPS <= 240)")
        self.index = 0

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None or not frame.size:
            raise EOFError("Video playback finished")
        timestamp = self.index / self.fps
        self.index += 1
        return frame, timestamp

    def close(self):
        self.capture.release()
