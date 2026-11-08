"""All camera, timing, and threshold choices are configurable for PC/Pi."""
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    inference_width: int = 480
    backend: str = "auto"
    opencv_threads: int = 1
    mirror: bool = True
    draw_landmarks: bool = True


@dataclass
class CalibrationConfig:
    seconds: float = 30.0
    min_samples: int = 100
    min_inlier_fraction: float = 0.6
    mad_limit: float = 3.5
    min_ear: float = 0.12
    max_ear: float = 0.5
    max_mar: float = 0.30
    max_pose_degrees: float = 25.0
    ear_ratio: float = 0.72
    mar_margin: float = 0.25
    pose_sigma: float = 4.0
    pitch_degrees: float = 15.0
    yaw_degrees: float = 20.0
    roll_degrees: float = 18.0


@dataclass
class DetectionConfig:
    eye_warning_seconds: float = 0.8
    eye_danger_seconds: float = 2.0
    yawn_seconds: float = 1.5
    head_seconds: float = 1.5
    nod_danger_seconds: float = 4.0
    fusion_seconds: float = 1.0
    recovery_seconds: float = 1.0
    alarm_delay_seconds: float = 2.0
    max_observation_gap: float = 0.5
    smoothing_seconds: float = 0.1


@dataclass
class AlarmConfig:
    enabled: bool = True
    frequency: int = 1800
    tone_seconds: float = 0.3
    repeat_seconds: float = 1.0


@dataclass
class DisplayConfig:
    simple: bool = False
    calibration_seconds: float = 10.0


@dataclass
class Config:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        config = cls()
        if path is not None:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Configuration must be a JSON object")
            sections = {item.name for item in fields(config)}
            for section, values in payload.items():
                if section not in sections or not isinstance(values, dict):
                    raise ValueError(f"Invalid configuration section: {section}")
                target = getattr(config, section)
                for key, value in values.items():
                    if key not in {item.name for item in fields(target)}:
                        raise ValueError(f"Unknown configuration key: {section}.{key}")
                    default = getattr(target, key)
                    if type(default) is float:
                        valid = type(value) in (int, float) and math.isfinite(value)
                        value = float(value) if valid else value
                    else:
                        valid = type(value) is type(default)
                    if not valid:
                        raise ValueError(f"Invalid value type: {section}.{key}")
                    setattr(target, key, value)
        config.validate()
        return config

    def validate(self) -> None:
        for section_name in ("camera", "calibration", "detection", "alarm", "display"):
            section = getattr(self, section_name)
            for item in fields(section):
                value = getattr(section, item.name)
                if type(value) in (int, float) and item.name != "index":
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"{section_name}.{item.name} must be positive")
        if self.camera.index < 0:
            raise ValueError("Camera index cannot be negative")
        if self.camera.width < 320 or self.camera.height < 240:
            raise ValueError("Camera size must be at least 320x240")
        if self.camera.inference_width < 160:
            raise ValueError("Inference width must be at least 160")
        if self.camera.backend not in ("auto", "dshow", "msmf", "v4l2"):
            raise ValueError("Unsupported camera backend")
        c = self.calibration
        if not 0 < c.min_inlier_fraction <= 1 or not 0 < c.ear_ratio < 1:
            raise ValueError("Invalid calibration fraction or EAR ratio")
        if c.min_ear >= c.max_ear or c.max_ear > 1 or c.max_mar > 1:
            raise ValueError("Invalid calibration EAR/MAR limits")
        if self.detection.eye_danger_seconds < self.detection.eye_warning_seconds:
            raise ValueError("Eye danger time must be >= eye warning time")
        if not 37 <= self.alarm.frequency <= 20000:
            raise ValueError("Alarm frequency must be between 37 and 20000 Hz")
        if self.alarm.repeat_seconds < self.alarm.tone_seconds:
            raise ValueError("Alarm repeat interval must be >= tone duration")
