"""Duration-based evidence fusion. No counters tied to a particular FPS."""
from .config import DetectionConfig
from .metrics import angle_delta, is_valid
from .models import Detection, Metrics, Profile, State


class DrowsinessDetector:
    def __init__(self, profile: Profile, config: DetectionConfig):
        self.profile = profile
        self.config = config
        self.last_timestamp: float | None = None
        self.state = State.NORMAL
        self.since: dict[str, float] = {}
        self.danger_since: float | None = None
        self.recovery_since: float | None = None
        self.last_reasons: tuple[str, ...] = ()

    def _duration(self, name: str, active: bool, now: float) -> float:
        if not active:
            self.since.pop(name, None)
            return 0.0
        self.since.setdefault(name, now)
        return now - self.since[name]

    def _clear_evidence(self) -> None:
        self.since.clear()
        self.danger_since = None
        self.recovery_since = None
        self.last_reasons = ()

    def update(self, metrics: Metrics | None, now: float) -> Detection:
        c, p = self.config, self.profile
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("Observation times must strictly increase")
        gap = self.last_timestamp is not None and now - self.last_timestamp > c.max_observation_gap
        self.last_timestamp = now
        if gap:
            self._clear_evidence()
            self.state = State.NORMAL
        if metrics is None or not is_valid(metrics):
            self._clear_evidence()
            self.state = State.WARNING
            return Detection(State.WARNING, ("Tracking unavailable",), tracking_valid=False)

        # Both eyes must be closed; a one-eye wink is not drowsiness evidence.
        eyes = metrics.left_ear < p.ear_threshold and metrics.right_ear < p.ear_threshold
        yawn = metrics.mar > p.mar_threshold
        dp = abs(angle_delta(metrics.pose.pitch, p.pose_baseline.pitch))
        dy = abs(angle_delta(metrics.pose.yaw, p.pose_baseline.yaw))
        dr = abs(angle_delta(metrics.pose.roll, p.pose_baseline.roll))
        pitch = dp > p.pose_threshold.pitch
        head = pitch or dy > p.pose_threshold.yaw or dr > p.pose_threshold.roll
        eye_time = self._duration("eyes", eyes, now)
        yawn_time = self._duration("yawn", yawn, now)
        head_time = self._duration("head", head, now)
        nod_time = self._duration("nod", pitch, now)
        flags = (eyes and eye_time >= c.eye_warning_seconds,
                 yawn and yawn_time >= c.yawn_seconds,
                 head and head_time >= c.head_seconds)
        reasons = tuple(reason for active, reason in zip(flags,
            ("Eyes closed", "Yawning", "Head off neutral")) if active)
        fusion_time = self._duration("fusion", sum(flags) >= 2, now)
        danger = ((eyes and eye_time >= c.eye_danger_seconds)
                  or (pitch and nod_time >= c.nod_danger_seconds)
                  or (sum(flags) >= 2 and fusion_time >= c.fusion_seconds))
        candidate = State.DANGER if danger else State.WARNING if any(flags) else State.NORMAL
        rank = {State.NORMAL: 0, State.WARNING: 1, State.DANGER: 2}
        if rank[candidate] >= rank[self.state]:
            self.state = candidate
            self.recovery_since = None
        else:
            if self.recovery_since is None:
                self.recovery_since = now
            if now - self.recovery_since >= c.recovery_seconds:
                self.state = candidate
                self.recovery_since = None
        if reasons:
            self.last_reasons = reasons
        elif self.state is State.NORMAL:
            self.last_reasons = ()

        # Recovery hysteresis may keep the red display briefly, but only
        # continuous CURRENT danger evidence is allowed to trigger an alarm.
        if danger:
            if self.danger_since is None:
                self.danger_since = now
        else:
            self.danger_since = None
        danger_time = 0.0 if self.danger_since is None else now - self.danger_since
        alarm = danger and danger_time >= c.alarm_delay_seconds
        return Detection(self.state, reasons or self.last_reasons,
                         eye_time, yawn_time, head_time, danger_time, alarm)
