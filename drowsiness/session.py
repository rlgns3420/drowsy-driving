"""Shared recorded-video inspection and headless Pi monitoring runtime."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import csv
import json
import logging
import math
from pathlib import Path
import time

import cv2

from .alarm import Alarm
from .display import FrameRate, render
from .evaluation import WindowEvaluator
from .processing import FrameProcessor


def observation_state(observation):
    if observation.detection is None:
        return "CALIBRATING"
    if not observation.detection.tracking_valid:
        return "TRACKING_LOST"
    return observation.detection.state.value


class SessionRecorder:
    """Stream frame/event rows to disk, keeping only counters in memory."""
    def __init__(self, output, config, source, fps, labels=None):
        self.output, self.config, self.source, self.fps = output, config, source, fps
        self.stack = ExitStack()
        try:
            frames = self.stack.enter_context((output / "frames.csv").open("w", encoding="utf-8-sig", newline=""))
            events = self.stack.enter_context((output / "events.csv").open("w", encoding="utf-8-sig", newline=""))
        except BaseException:
            self.stack.close()
            raise
        self.frames = csv.writer(frames)
        self.events = csv.writer(events)
        self.frames.writerow(("seconds", "state", "tracking_valid", "ear", "left_ear", "right_ear",
                              "mar", "pitch", "yaw", "roll", "eye_seconds", "yawn_seconds",
                              "head_seconds", "alarm_requested", "reasons", "inference_ms"))
        self.events.writerow(("seconds", "state", "alarm_requested", "reasons"))
        self.evaluator = WindowEvaluator(labels, fps) if labels is not None else None
        self.counts = Counter()
        self.total = self.tracked = 0
        self.last = self.profile = self.calibrated_at = None
        self.first = {}
        self.last_time = 0.0
        self.started = time.perf_counter()

    def add(self, timestamp, observation, inference_ms):
        state = observation_state(observation)
        metrics, detection = observation.metrics, observation.detection
        alarm = bool(detection and detection.alarm)
        reasons = "; ".join(detection.reasons) if detection else observation.calibration_message
        self.total += 1
        self.tracked += metrics is not None
        self.counts[state] += 1
        self.last_time = timestamp
        if observation.profile is not None and self.profile is None:
            self.profile, self.calibrated_at = observation.profile, timestamp
        if state not in self.first:
            self.first[state] = timestamp
        if alarm:
            self.first.setdefault("ALARM", timestamp)
        if (state, alarm) != self.last:
            self.events.writerow((f"{timestamp:.6f}", state, int(alarm), reasons))
            logging.info("%.2fs %s%s: %s", timestamp, state, " ALARM" if alarm else "", reasons)
            self.last = state, alarm
        self.frames.writerow((f"{timestamp:.6f}", state, int(metrics is not None),
                              *([metrics.ear, metrics.left_ear, metrics.right_ear, metrics.mar,
                                 metrics.pose.pitch, metrics.pose.yaw, metrics.pose.roll] if metrics else [""] * 7),
                              *([detection.eye_seconds, detection.yawn_seconds, detection.head_seconds]
                                if detection else [0] * 3), int(alarm), reasons, round(inference_ms, 3)))
        if self.evaluator is not None:
            self.evaluator.add(timestamp, detection)

    def finish(self, status, error, audio_error):
        self.stack.close()
        elapsed = time.perf_counter() - self.started
        report = {"status": status, "error": error, "source": self.source,
                  "source_fps": self.fps, "frames_processed": self.total,
                  "last_timestamp_seconds": self.last_time,
                  "processing_fps": self.total / elapsed if elapsed else 0,
                  "tracking_fraction_all_frames": self.tracked / self.total if self.total else 0,
                  "state_frame_counts": dict(self.counts),
                  "first_occurrence_seconds": self.first,
                  "calibration_finished_at_seconds": self.calibrated_at,
                  "profile": asdict(self.profile) if self.profile else None,
                  "config": asdict(self.config), "audio_error": audio_error,
                  "scoring": self.evaluator.summarize() if self.evaluator is not None else None,
                  "accuracy_validated": False,
                  "limitations": "Unlabelled footage checks behavior only. Labelled scores apply only "
                                 "to that footage. Acted eye closure does not establish real fatigue. "
                                 "processing_fps includes capture/display/pacing when enabled."}
        (self.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


def run_session(config, source_factory, output: Path, *, source_name="video", labels=None,
                preview=False, save_video=False, realtime=False, live=False,
                duration=None, rotation=0, status_seconds=5.0):
    """Source timestamps drive decisions; wall time only drives display pacing.

    A new output directory prevents overwriting a previous inspection or its input.
    Raw images are saved only when save_video is explicitly requested.
    """
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be positive and finite")
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180 or 270")
    if not math.isfinite(status_seconds) or status_seconds <= 0:
        raise ValueError("Status interval must be positive and finite")
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = None
    with ExitStack() as stack:
        cv2.setNumThreads(config.camera.opencv_threads)
        source = source_factory()
        stack.callback(source.close)
        fps = getattr(source, "fps", config.camera.fps)
        recorder = SessionRecorder(output, config, source_name, fps, labels)
        stack.callback(recorder.stack.close)
        status, error, audio_error = "completed", None, None
        try:
            processor = FrameProcessor(config)
            stack.callback(processor.close)
            alarm = Alarm(config.alarm)
            stack.callback(alarm.close)
            if preview:
                cv2.namedWindow("Drowsiness inspection", cv2.WINDOW_NORMAL)
                stack.callback(cv2.destroyAllWindows)
            writer = None
            origin, wall_start = None, time.monotonic()
            rates, next_status = FrameRate(), 0.0
            rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                         270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            while True:
                if live and duration is not None and time.monotonic() - wall_start >= duration:
                    status = "duration_reached"
                    break
                try:
                    frame, timestamp = source.read()
                except EOFError:
                    break
                if origin is None:
                    origin = timestamp
                seconds = timestamp - origin
                if not live and duration is not None and seconds >= duration:
                    status = "duration_reached"
                    break
                if rotation:
                    frame = cv2.rotate(frame, rotations[rotation])
                observation, landmarks, inference_ms = processor.process(frame, timestamp)
                alarm.update(bool(observation.detection and observation.detection.alarm))
                audio_error = alarm.error
                recorder.add(seconds, observation, inference_ms)
                rates.update(time.monotonic())
                if seconds >= next_status:
                    logging.info("%.1fs | %s | calibration %.0f%% | %.1f FPS | output %s",
                                 seconds, observation_state(observation), observation.progress * 100,
                                 rates.fps, output)
                    next_status = seconds + status_seconds
                if preview or save_video:
                    # Bound export size for phone/4K inputs. Inference still uses source geometry.
                    canvas = render(frame.copy(), landmarks, observation.metrics, processor.pipeline.calibrator,
                                    observation.profile, observation.detection, rates.fps, inference_ms,
                                    config.camera.mirror, config.camera.draw_landmarks, audio_error,
                                    config.alarm.enabled)
                    if canvas.shape[1] > 1280:
                        canvas = cv2.resize(canvas, (1280, max(2, round(canvas.shape[0] * 1280 / canvas.shape[1]))))
                    # MJPG requires even dimensions; avoid silently dropping a row.
                    height, width = canvas.shape[:2]
                    canvas = canvas[:height - height % 2, :width - width % 2]
                    cv2.putText(canvas, f"Time: {seconds:.2f}s | {observation_state(observation)}",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                    if save_video:
                        if writer is None:
                            height, width = canvas.shape[:2]
                            writer = cv2.VideoWriter(str(output / "annotated.avi"),
                                                     cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                            stack.callback(writer.release)
                            if not writer.isOpened():
                                raise RuntimeError("Cannot create annotated.avi (MJPG codec)")
                        writer.write(canvas)
                    if preview:
                        cv2.imshow("Drowsiness inspection", canvas)
                        if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                            status = "stopped"
                            break
                        if cv2.getWindowProperty("Drowsiness inspection", cv2.WND_PROP_VISIBLE) < 1:
                            status = "stopped"
                            break
                if realtime and not live:
                    # Decisions have already used video time, even if the PC is too slow.
                    time.sleep(max(0, wall_start + seconds + 1 / fps - time.monotonic()))
            if not recorder.total:
                raise RuntimeError("No video frames were processed")
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status, error = "failed", str(exc)
            raise
        finally:
            report = recorder.finish(status, error, audio_error)
    return report
