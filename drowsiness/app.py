import logging
import time

import cv2

from .alarm import Alarm
from .camera import Webcam
from .config import Config
from .display import FrameRate, render
from .face_mesh import FaceMeshExtractor
from .head_pose import HeadPoseEstimator
from .metrics import is_valid, measure
from .pipeline import MonitoringPipeline


def run(config: Config) -> None:
    cv2.setNumThreads(config.camera.opencv_threads)
    camera = mesh = alarm = None
    window = "Drowsiness Detection"
    try:
        camera = Webcam(config.camera)
        mesh = FaceMeshExtractor(config.camera.inference_width)
        estimator = HeadPoseEstimator()
        pipeline = MonitoringPipeline(config)
        calibrator = pipeline.calibrator
        alarm = Alarm(config.alarm)
        rates = FrameRate()
        profile = None
        logging.info("Calibration: look forward, eyes open, mouth relaxed for about %.0fs",
                     config.calibration.seconds)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            frame, now = camera.read()
            started = time.perf_counter()
            landmarks = mesh.extract(frame)
            inference_ms = (time.perf_counter() - started) * 1000
            raw = metrics = detection = None
            if landmarks is not None:
                height, width = frame.shape[:2]
                pose = estimator.estimate(landmarks, width, height)
                if pose is not None:
                    try:
                        raw = measure(landmarks, pose)
                        if not is_valid(raw):
                            raw = None
                    except ValueError:
                        raw = None
            previous_profile = profile
            observation = pipeline.update(raw, now)
            metrics, profile, detection = (observation.metrics, observation.profile,
                                            observation.detection)
            if previous_profile is None and profile is not None:
                logging.info("Calibration ready: EAR %.3f (<%.3f), MAR %.3f (>%.3f), "
                             "inliers %d, removed %d", profile.ear_baseline,
                             profile.ear_threshold, profile.mar_baseline,
                             profile.mar_threshold, profile.samples_kept,
                             profile.samples_rejected)
            alarm.update(detection is not None and detection.alarm)
            canvas = render(frame, landmarks, metrics, calibrator, profile, detection,
                            rates.fps, inference_ms, config.camera.mirror,
                            config.camera.draw_landmarks, alarm.error, config.alarm.enabled)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            rates.update(time.monotonic())
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("c"), ord("C")):
                alarm.update(False)
                pipeline.reset()
                profile = None
                logging.info("Calibration restarted")
    finally:
        if alarm is not None:
            alarm.close()
        if mesh is not None:
            mesh.close()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
