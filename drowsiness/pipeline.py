"""The same calibration and decision pipeline for PC, replay, and Pi."""
from dataclasses import dataclass

from .calibration import Calibrator
from .config import Config
from .detector import DrowsinessDetector
from .metrics import MetricSmoother, is_valid
from .models import Detection, Metrics, Profile


@dataclass(frozen=True)
class Observation:
    metrics: Metrics | None
    profile: Profile | None
    detection: Detection | None
    progress: float
    valid_seconds: float
    calibration_message: str


class MonitoringPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.calibrator = Calibrator(config.calibration, config.detection.max_observation_gap)
        self.smoother = MetricSmoother(config.detection.smoothing_seconds,
                                      config.detection.max_observation_gap)
        self.reset()

    def reset(self) -> None:
        self.calibrator.reset()
        self.smoother.reset()
        self.profile = self.detector = None

    def update(self, raw: Metrics | None, now: float) -> Observation:
        if raw is not None and not is_valid(raw):
            raw = None
        metrics = self.smoother.update(raw, now) if raw is not None else None
        if raw is None:
            self.smoother.reset()
        detection = None
        if self.profile is None:
            # Calibration sees raw values, so smoothing cannot hide outliers.
            self.profile = self.calibrator.update(raw, now)
            if self.profile is not None:
                self.detector = DrowsinessDetector(self.profile, self.config.detection)
                self.smoother.reset()
        else:
            detection = self.detector.update(metrics, now)
        return Observation(metrics, self.profile, detection, self.calibrator.progress,
                           self.calibrator.valid_seconds, self.calibrator.message)
