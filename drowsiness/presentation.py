"""UI wording is derived from decisions, independently of the buzzer."""
from dataclasses import dataclass

from .models import State
from .pipeline import Observation


REASONS = {"Eyes closed": "눈 감김 지속", "Yawning": "하품 / 입 벌림 지속",
           "Head off neutral": "고개 자세 이탈", "Tracking unavailable": "얼굴 추적 불가"}


def exhibit_view(view):
    """Visitor-facing wording, preserving the shared decision keys and colors."""
    title, detail = {
        "NORMAL": ("정상", "안전한 운전 자세를 유지해 주세요."),
        "WARNING": ("주의", "졸음 징후가 보입니다. 주의해 주세요."),
        "DANGER": ("졸음 위험", "안전한 곳에서 휴식을 취하세요."),
        "CALIBRATING": ("준비 중", view.detail),
    }.get(view.key, (view.title, view.detail))
    return StatusView(view.key, title, detail, view.color, view.background)


@dataclass(frozen=True)
class StatusView:
    key: str
    title: str
    detail: str
    color: str
    background: str


def status_view(observation: Observation) -> StatusView:
    detection = observation.detection
    if observation.profile is None:
        detail = ("정면을 보고 눈을 뜬 채 입을 편안하게 닫아주세요."
                  if observation.metrics is not None else "얼굴이 보이도록 카메라와 조명을 확인해주세요.")
        if observation.calibration_message.startswith("Paused") and observation.metrics is not None:
            detail = "수집 일시중지 · 눈을 뜨고 입을 닫은 정면 자세를 유지해주세요."
        return StatusView("CALIBRATING", "개인 기준값 수집 중", detail, "#60a5fa", "#132641")
    if detection is None:
        return StatusView("READY", "감지 준비 완료", "개인 기준값 설정이 완료되었습니다.",
                          "#2dd4bf", "#103330")
    if not detection.tracking_valid:
        return StatusView("TRACKING", "얼굴 추적 불가", "졸음 판정을 할 수 없습니다. 얼굴 위치와 조명을 확인하세요.",
                          "#fbbf24", "#362b17")
    reasons = " · ".join(REASONS.get(reason, reason) for reason in detection.reasons)
    if detection.state is State.DANGER:
        return StatusView("DANGER", "졸음 위험 감지", reasons + "\n안전한 곳에서 휴식을 취하세요.",
                          "#fb7185", "#3b1b29")
    if detection.state is State.WARNING:
        return StatusView("WARNING", "주의 · 졸음 징후", reasons, "#fbbf24", "#362b17")
    return StatusView("NORMAL", "정상 · 감지 중", "지속되는 눈 감김이나 자세 이탈이 없습니다.",
                      "#2dd4bf", "#103330")
