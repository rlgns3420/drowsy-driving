import os
import time

import cv2

from .config import CameraConfig


class Webcam:
    def __init__(self, config: CameraConfig):
        name = config.backend
        if name == "auto":
            name = "dshow" if os.name == "nt" else "v4l2"
        backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF,
                   "v4l2": cv2.CAP_V4L2}[name]
        self.capture = cv2.VideoCapture(config.index, backend)
        if not self.capture.isOpened() and config.backend == "auto":
            self.capture.release()
            self.capture = cv2.VideoCapture(config.index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open webcam {config.index}; check camera index/permissions")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.capture.set(cv2.CAP_PROP_FPS, config.fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.capture.read()
        timestamp = time.monotonic()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError("Webcam read failed; reconnect the camera and restart")
        return frame, timestamp

    def close(self) -> None:
        self.capture.release()
