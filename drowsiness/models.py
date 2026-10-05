from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DANGER = "DANGER"


@dataclass(frozen=True)
class HeadPose:
    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class Metrics:
    left_ear: float
    right_ear: float
    mar: float
    pose: HeadPose

    @property
    def ear(self) -> float:
        return (self.left_ear + self.right_ear) / 2.0


@dataclass(frozen=True)
class Profile:
    ear_baseline: float
    mar_baseline: float
    pose_baseline: HeadPose
    ear_threshold: float
    mar_threshold: float
    pose_threshold: HeadPose
    samples_kept: int
    samples_rejected: int


@dataclass(frozen=True)
class Detection:
    state: State
    reasons: tuple[str, ...]
    eye_seconds: float = 0.0
    yawn_seconds: float = 0.0
    head_seconds: float = 0.0
    danger_seconds: float = 0.0
    alarm: bool = False
    tracking_valid: bool = True
