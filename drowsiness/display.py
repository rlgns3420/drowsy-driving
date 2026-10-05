from collections import deque

import cv2
import numpy as np

from .calibration import Calibrator
from .metrics import LEFT_EYE, MOUTH, RIGHT_EYE
from .models import Detection, Metrics, Profile, State


COLORS = {State.NORMAL: (70, 220, 70), State.WARNING: (0, 200, 255),
          State.DANGER: (60, 60, 255)}


class FrameRate:
    """Completed-frame throughput, including capture, processing and display."""
    def __init__(self):
        self.timestamps = deque(maxlen=60)

    def update(self, now: float) -> None:
        self.timestamps.append(now)

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def render(frame, landmarks, metrics: Metrics | None, calibration: Calibrator,
           profile: Profile | None, detection: Detection | None, fps: float,
           inference_ms: float, mirror: bool, draw_landmarks: bool,
           audio_error: str | None, audio_enabled: bool):
    if draw_landmarks and landmarks is not None:
        for index in set(LEFT_EYE + RIGHT_EYE + MOUTH + (1, 152, 61, 291)):
            x, y = map(round, landmarks[index])
            cv2.circle(frame, (x, y), 2, (255, 220, 0), -1)
    if mirror:
        frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    scale = max(0.4, min(0.50, width / 1280))
    spacing = max(17, round(25 * width / 640))
    color = (0, 200, 255) if detection is None else COLORS[detection.state]
    if metrics is None:
        ear_line, mar_line, pose_line = "EAR: --", "MAR: --", "P/Y/R: -- / -- / --"
    else:
        ear_line = f"EAR: {metrics.ear:.3f}  L:{metrics.left_ear:.3f} R:{metrics.right_ear:.3f}"
        mar_line = f"MAR: {metrics.mar:.3f}"
        pose_line = (f"P/Y/R: {metrics.pose.pitch:+.1f} / "
                     f"{metrics.pose.yaw:+.1f} / {metrics.pose.roll:+.1f} deg")
    lines = [ear_line, mar_line, pose_line]
    if profile is None:
        title = f"CALIBRATING {calibration.progress * 100:.0f}%"
        lines += [f"Neutral data: {calibration.valid_seconds:.1f}/{calibration.config.seconds:.0f}s",
                  calibration.message]
    else:
        title = detection.state.value if detection is not None else "READY"
        lines += [f"Thresholds: EAR<{profile.ear_threshold:.3f} MAR>{profile.mar_threshold:.3f}"]
        if detection is not None:
            lines += [f"Eye:{detection.eye_seconds:.1f}s Yawn:{detection.yawn_seconds:.1f}s "
                      f"Head:{detection.head_seconds:.1f}s",
                      ", ".join(detection.reasons) or "Neutral"]
    lines += [f"FPS: {fps:.1f}  Face Mesh: {inference_ms:.1f}ms"]
    if audio_error:
        lines += [audio_error]
    elif not audio_enabled:
        lines += ["Audio disabled"]
    elif detection is not None and detection.alarm:
        lines += ["ALARM ACTIVE"]
    # A separate side panel keeps the driver's eyes and face visible.
    panel_width = max(330, round(width * 0.65))
    panel = np.full((height, panel_width, 3), 22, dtype=np.uint8)
    cv2.putText(panel, title, (10, spacing), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 1.2, color, 2, cv2.LINE_AA)
    wrapped = []
    for line in lines:
        current = ""
        for word in line.split():
            candidate = (current + " " + word).strip()
            text_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            if current and text_width > panel_width - 20:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
    for row, line in enumerate(wrapped, start=2):
        y = row * spacing
        if y < height - 10:
            cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 3)
    cv2.putText(frame, "Q/Esc: quit   C: recalibrate", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((frame, panel), axis=1)
