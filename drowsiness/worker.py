"""Capture/inference run off the Tk thread; only the newest frame is displayed."""
from copy import deepcopy
from dataclasses import dataclass
import queue
import threading
import time

import cv2

from .alarm import Alarm
from .camera import VideoFile, Webcam
from .config import Config
from .demo import demo_sample
from .display import FrameRate
from .pipeline import MonitoringPipeline, Observation
from .presentation import status_view
from .processing import FrameProcessor


@dataclass(frozen=True)
class FramePacket:
    frame: object
    landmarks: object
    observation: Observation
    fps: float
    inference_ms: float
    source: str
    audio_enabled: bool
    audio_error: str | None


class MonitorWorker(threading.Thread):
    def __init__(self, config: Config, mode: str = "camera", video=None):
        super().__init__(name="local-monitor", daemon=True)
        self.config = deepcopy(config)
        self.mode, self.video = mode, video
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.commands = queue.Queue()
        self.stopped = threading.Event()
        self.reconnect_requested = threading.Event()

    def stop(self):
        self.stopped.set()
        self.reconnect_requested.set()

    def reconnect(self):
        self.reconnect_requested.set()

    def run(self):
        while True:
            self.reconnect_requested.clear()
            if not self._run_once() or self.stopped.is_set():
                break
            self.reconnect_requested.wait(self.config.display.camera_retry_seconds)

    def _run_once(self):
        camera = processor = alarm = None
        outcome = "감지를 종료했습니다."
        failed = False
        try:
            cv2.setNumThreads(self.config.camera.opencv_threads)
            if self.mode == "demo":
                self.config.calibration.seconds = 2.0
                self.config.calibration.min_samples = 20
                pipeline = MonitoringPipeline(self.config)
                source = "시뮬레이션 · 실제 카메라 검증 아님"
            else:
                camera = VideoFile(self.video) if self.mode == "video" else Webcam(self.config.camera)
                processor = FrameProcessor(self.config)
                pipeline = processor.pipeline
                source = (f"영상 재생 · {self.video.name}" if self.mode == "video" else
                          f"카메라 {self.config.camera.index} · {camera.backend.upper()}")
            # Simulated inputs must never activate physical audio.
            self.config.alarm.enabled = self.config.alarm.enabled and self.mode != "demo"
            alarm = Alarm(self.config.alarm)
            rates, last_status = FrameRate(), None
            started = time.monotonic()
            while not self.stopped.is_set():
                if self.mode == "camera" and self.reconnect_requested.is_set():
                    raise RuntimeError("카메라 수동 재연결 요청")
                while True:
                    try:
                        command, value = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    if command == "reset":
                        pipeline.reset()
                        alarm.update(False)
                        self.events.put(("info", "개인 기준값 수집을 다시 시작했습니다."))
                    elif command == "audio":
                        alarm.close()
                        self.config.alarm.enabled = bool(value) and self.mode != "demo"
                        alarm = Alarm(self.config.alarm)
                if self.mode == "demo":
                    now = time.monotonic()
                    frame, raw = demo_sample(now - started)
                    observation = pipeline.update(raw, now)
                    landmarks, inference_ms = None, 0.0
                else:
                    try:
                        frame, now = camera.read()
                    except EOFError:
                        outcome = "영상 재생이 끝났습니다."
                        break
                    observation, landmarks, inference_ms = processor.process(frame, now)
                alarm.update(observation.detection is not None and observation.detection.alarm)
                current = status_view(observation)
                if current.key != last_status:
                    self.events.put(("state", current.title))
                    last_status = current.key
                rates.update(time.monotonic())
                packet = FramePacket(frame, landmarks, observation, rates.fps,
                                     inference_ms, source, self.config.alarm.enabled, alarm.error)
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(packet)
                if self.mode == "demo":
                    self.stopped.wait(1 / 30)
                elif self.mode == "video":
                    self.stopped.wait(max(0, started + now + 1 / camera.fps - time.monotonic()))
        except Exception as exc:
            failed = True
            outcome = str(exc)
        finally:
            # Keep closing other resources even if one driver fails to close.
            for resource in (alarm, processor, camera):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        self.events.put(("error", f"자원 종료 오류: {exc}"))
            retry = failed and self.mode == "camera" and self.config.display.simple
            if retry:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            self.events.put(("retrying" if retry else "failed" if failed else "finished", outcome))
        return retry
