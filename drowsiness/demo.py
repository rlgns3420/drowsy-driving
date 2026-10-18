"""Explicitly synthetic input for UI/decision checks, not accuracy evidence."""
import cv2
import numpy as np

from .models import HeadPose, Metrics


def demo_sample(elapsed: float):
    phase = elapsed % 29
    ear, mar, pitch = 0.30, 0.06, 0.0
    label = "NORMAL"
    if 5 <= phase < 9.8:
        ear, label = 0.05, "EYES CLOSED"
    elif 12 <= phase < 14.5:
        mar, label = 0.60, "YAWNING"
    elif 17 <= phase < 23.5:
        pitch, label = 25.0, "HEAD OFF NEUTRAL"
    elif 25 <= phase < 27:
        label = "TRACKING UNAVAILABLE"
    metrics = None if label == "TRACKING UNAVAILABLE" else Metrics(
        ear, ear, mar, HeadPose(pitch, 0, 0))
    frame = np.full((480, 640, 3), (28, 22, 15), dtype=np.uint8)
    if metrics is not None:
        cv2.ellipse(frame, (320, 232), (108, 140), round(pitch), 0, 360, (120, 156, 156), 2)
        for x in (280, 360):
            cv2.ellipse(frame, (x, 210), (18, 2 if ear < 0.1 else 9), 0, 0, 360,
                        (206, 225, 230), 2)
        cv2.ellipse(frame, (320, 290), (27, 25 if mar > 0.35 else 3), 0, 0, 360,
                    (206, 225, 230), 2)
    cv2.putText(frame, "SIMULATED INPUT", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (170, 220, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, label, (30, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (170, 220, 230), 1, cv2.LINE_AA)
    return frame, metrics
