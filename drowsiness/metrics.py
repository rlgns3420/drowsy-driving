"""EAR/MAR use 2D pixel coordinates, never uncorrected normalized x/y."""
import math

import numpy as np

from .models import HeadPose, Metrics


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
# Inner lip contour: corners, upper pairs, lower pairs.
MOUTH = (78, 81, 13, 311, 308, 402, 14, 178)


def eye_aspect_ratio(points: np.ndarray) -> float:
    """(distance(p2,p6) + distance(p3,p5)) / (2*distance(p1,p4))."""
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError("EAR needs six finite 2D points")
    width = float(np.linalg.norm(points[0] - points[3]))
    if width < 1e-6:
        raise ValueError("Degenerate eye width")
    return float((np.linalg.norm(points[1] - points[5])
                  + np.linalg.norm(points[2] - points[4])) / (2 * width))


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """Mean of three inner-lip vertical distances / inner-mouth width."""
    points = np.asarray(points, dtype=float)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError("MAR needs eight finite 2D points")
    width = float(np.linalg.norm(points[0] - points[4]))
    if width < 1e-6:
        raise ValueError("Degenerate mouth width")
    vertical = sum(np.linalg.norm(points[a] - points[b])
                   for a, b in ((1, 7), (2, 6), (3, 5)))
    return float(vertical / (3 * width))


def measure(landmarks: np.ndarray, pose: HeadPose) -> Metrics:
    return Metrics(
        eye_aspect_ratio(landmarks[list(LEFT_EYE)]),
        eye_aspect_ratio(landmarks[list(RIGHT_EYE)]),
        mouth_aspect_ratio(landmarks[list(MOUTH)]), pose,
    )


def angle_delta(angle: float, baseline: float) -> float:
    return (angle - baseline + 180.0) % 360.0 - 180.0


def is_valid(metrics: Metrics) -> bool:
    values = (metrics.left_ear, metrics.right_ear, metrics.mar,
              metrics.pose.pitch, metrics.pose.yaw, metrics.pose.roll)
    return (all(math.isfinite(v) for v in values)
            and 0 <= metrics.left_ear <= 1 and 0 <= metrics.right_ear <= 1
            and 0 <= metrics.mar <= 2
            and all(abs(v) <= 90 for v in values[3:]))


class MetricSmoother:
    """Time-based EMA so smoothing behaves consistently at 15 or 30 FPS."""

    def __init__(self, seconds: float, max_gap: float):
        self.seconds = seconds
        self.max_gap = max_gap
        self.reset()

    def reset(self) -> None:
        self.previous: Metrics | None = None
        self.timestamp: float | None = None

    def update(self, metrics: Metrics, now: float) -> Metrics:
        if self.timestamp is not None and now <= self.timestamp:
            raise ValueError("Observation times must strictly increase")
        old = self.previous
        if old is None or now - self.timestamp > self.max_gap:
            result = metrics
        else:
            alpha = 1.0 - math.exp(-(now - self.timestamp) / self.seconds)
            blend = lambda a, b: a + alpha * (b - a)
            pose = HeadPose(*(a + alpha * angle_delta(b, a) for a, b in zip(
                (old.pose.pitch, old.pose.yaw, old.pose.roll),
                (metrics.pose.pitch, metrics.pose.yaw, metrics.pose.roll))))
            result = Metrics(blend(old.left_ear, metrics.left_ear),
                             blend(old.right_ear, metrics.right_ear),
                             blend(old.mar, metrics.mar), pose)
        self.previous, self.timestamp = result, now
        return result
