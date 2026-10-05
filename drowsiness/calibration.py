"""Collect approximately 30 seconds of observed neutral data, then MAD filter."""
import numpy as np

from .config import CalibrationConfig
from .metrics import is_valid
from .models import HeadPose, Metrics, Profile


class Calibrator:
    def __init__(self, config: CalibrationConfig, max_gap: float = 0.5):
        self.config = config
        self.max_gap = max_gap
        self.reset()

    def reset(self) -> None:
        self.samples: list[tuple[float, ...]] = []
        self.weights: list[float] = []
        self.valid_seconds = 0.0
        self.last_timestamp: float | None = None
        self.last_eligible = False
        self.profile: Profile | None = None
        self.message = "Look forward, keep eyes open and mouth relaxed"

    @property
    def progress(self) -> float:
        return min(self.valid_seconds / self.config.seconds, 1.0)

    def eligible(self, metrics: Metrics | None) -> bool:
        c = self.config
        return (metrics is not None and is_valid(metrics)
                and c.min_ear <= metrics.left_ear <= c.max_ear
                and c.min_ear <= metrics.right_ear <= c.max_ear
                and metrics.mar <= c.max_mar
                and max(abs(metrics.pose.pitch), abs(metrics.pose.yaw),
                        abs(metrics.pose.roll)) <= c.max_pose_degrees)

    def update(self, metrics: Metrics | None, now: float) -> Profile | None:
        if self.profile is not None:
            return self.profile
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("Observation times must strictly increase")
        eligible = self.eligible(metrics)
        dt = 0.0 if self.last_timestamp is None else now - self.last_timestamp
        weight = dt if eligible and self.last_eligible and dt <= self.max_gap else 0.0
        self.last_timestamp, self.last_eligible = now, eligible
        if eligible:
            self.samples.append((metrics.ear, metrics.mar, metrics.pose.pitch,
                                 metrics.pose.yaw, metrics.pose.roll))
            self.weights.append(weight)
            self.valid_seconds += weight
            self.message = "Collecting neutral samples"
        else:
            self.message = "Paused: face missing, eyes/mouth/pose not neutral"
        if self.valid_seconds >= self.config.seconds and len(self.samples) >= self.config.min_samples:
            try:
                self.profile = self._build_profile()
                self.message = "Calibration complete"
            except ValueError:
                self.reset()
                self.message = "Too many abnormal samples; collecting again"
        return self.profile

    def _build_profile(self) -> Profile:
        c = self.config
        data = np.array(self.samples, dtype=np.float64)
        center = np.median(data, axis=0)
        mad = np.median(np.abs(data - center), axis=0)
        # Non-zero floors are essential when a dimension is nearly constant.
        scale = np.maximum(1.4826 * mad, [0.008, 0.015, 1.5, 1.5, 1.5])
        mask = np.all(np.abs(data - center) <= c.mad_limit * scale, axis=1)
        kept = data[mask]
        fraction = len(kept) / len(data)
        inlier_seconds = float(np.asarray(self.weights)[mask].sum())
        if (len(kept) < c.min_samples or fraction < c.min_inlier_fraction
                or inlier_seconds < c.seconds * c.min_inlier_fraction):
            raise ValueError("Insufficient neutral inlier data")
        baseline = np.median(kept, axis=0)
        sigma = 1.4826 * np.median(np.abs(kept - baseline), axis=0)
        ear_threshold = float(np.clip(baseline[0] * c.ear_ratio, 0.06, 0.35))
        mar_threshold = float(np.clip(
            baseline[1] + max(c.mar_margin, 5.0 * sigma[1]), 0.35, 1.2))
        pose_threshold = np.clip(
            np.maximum([c.pitch_degrees, c.yaw_degrees, c.roll_degrees],
                       c.pose_sigma * sigma[2:]), [5, 5, 5], [35, 45, 35])
        return Profile(float(baseline[0]), float(baseline[1]),
                       HeadPose(*map(float, baseline[2:])), ear_threshold,
                       mar_threshold, HeadPose(*map(float, pose_threshold)),
                       len(kept), len(data) - len(kept))
