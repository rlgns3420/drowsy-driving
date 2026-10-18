from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from drowsiness.camera import VideoFile, Webcam, discover_cameras
from drowsiness.config import Config
from drowsiness.models import HeadPose, Metrics, State
from drowsiness.pipeline import MonitoringPipeline
from drowsiness.presentation import status_view
from drowsiness.worker import MonitorWorker


NORMAL = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
CLOSED = Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0))


class LocalTests(unittest.TestCase):
    def calibrated_pipeline(self):
        config = Config()
        config.calibration.seconds = 0.2
        config.calibration.min_samples = 3
        config.alarm.enabled = False
        pipeline = MonitoringPipeline(config)
        for index in range(6):
            pipeline.update(NORMAL, index / 20)
        self.assertIsNotNone(pipeline.profile)
        return pipeline

    def test_local_config_disables_audio_without_changing_detection_thresholds(self):
        root = Path(__file__).resolve().parents[1]
        local = Config.load(root / "test" / "config.local.json")
        self.assertFalse(local.alarm.enabled)
        self.assertEqual(local.detection, Config().detection)
        self.assertEqual(local.calibration.seconds, 30)

    def test_silent_mode_still_displays_drowsiness_danger(self):
        pipeline = self.calibrated_pipeline()
        for index in range(100):
            observation = pipeline.update(CLOSED, 0.3 + index / 20)
        self.assertEqual(observation.detection.state, State.DANGER)
        self.assertTrue(observation.detection.alarm)
        self.assertEqual(status_view(observation).title, "졸음 위험 감지")
        self.assertFalse(pipeline.config.alarm.enabled)

    def test_tracking_loss_has_separate_ui_and_clears_danger_evidence(self):
        pipeline = self.calibrated_pipeline()
        for index in range(100):
            pipeline.update(CLOSED, 0.3 + index / 20)
        observation = pipeline.update(None, 5.3)
        self.assertEqual(status_view(observation).key, "TRACKING")
        self.assertFalse(observation.detection.alarm)
        self.assertEqual(observation.detection.eye_seconds, 0)

    def test_recalibration_clears_profile_and_decision(self):
        pipeline = self.calibrated_pipeline()
        pipeline.reset()
        observation = pipeline.update(NORMAL, 10)
        self.assertIsNone(observation.profile)
        self.assertIsNone(observation.detection)
        self.assertEqual(status_view(observation).key, "CALIBRATING")

    def test_camera_falls_back_when_opened_driver_has_no_frames(self):
        bad, good = Mock(), Mock()
        bad.isOpened.return_value = True
        bad.read.return_value = (False, None)
        good.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        good.read.return_value = (True, frame)
        with patch("drowsiness.camera.os.name", "nt"), patch("drowsiness.camera.cv2.VideoCapture", side_effect=[bad, good]):
            camera = Webcam(Config().camera)
            self.assertEqual(camera.backend, "msmf")
            self.assertIs(camera.read()[0], frame)
            camera.close()
        bad.release.assert_called_once()
        good.release.assert_called_once()

    def test_explicit_backend_is_respected_and_failed_capture_is_released(self):
        camera = Mock()
        camera.isOpened.return_value = False
        config = Config().camera
        config.backend = "msmf"
        with patch("drowsiness.camera.cv2.VideoCapture", return_value=camera) as constructor:
            with self.assertRaisesRegex(RuntimeError, "Cannot read camera"):
                Webcam(config)
        constructor.assert_called_once_with(0, cv2.CAP_MSMF)
        camera.release.assert_called_once()

    def test_discovery_returns_only_devices_with_actual_images(self):
        fake = Mock()
        fake.backend = "dshow"
        fake.read.return_value = (np.zeros((240, 320, 3), dtype=np.uint8), 0)
        with patch("drowsiness.camera.Webcam", side_effect=[RuntimeError("missing"), fake]):
            devices = discover_cameras(Config().camera, limit=2)
        self.assertEqual([(item.index, item.width, item.height) for item in devices], [(1, 320, 240)])
        fake.close.assert_called_once()

    def test_real_video_reader_uses_frame_time_and_reports_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            video = VideoFile(path)
            try:
                times = [video.read()[1] for _ in range(3)]
                self.assertEqual(times, [0, 1 / 15, 2 / 15])
                with self.assertRaises(EOFError):
                    video.read()
            finally:
                video.close()

    def test_worker_failure_reports_error_and_closes_camera(self):
        camera = Mock()
        camera.backend = "dshow"
        with patch("drowsiness.worker.Webcam", return_value=camera), patch(
                "drowsiness.worker.FrameProcessor", side_effect=RuntimeError("model failed")):
            worker = MonitorWorker(Config())
            worker.run()
        camera.close.assert_called_once()
        self.assertEqual(worker.events.get_nowait(), ("failed", "model failed"))

    def test_worker_processes_danger_silently_and_closes_all_resources(self):
        config = Config()
        config.alarm.enabled = False
        config.calibration.seconds, config.calibration.min_samples = 0.1, 2
        pipeline = MonitoringPipeline(config)
        camera, processor, alarm = Mock(), Mock(), Mock()
        camera.backend = "dshow"
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera.read.side_effect = [(frame, index / 20) for index in range(100)] + [EOFError()]
        processor.pipeline = pipeline
        processor.process.side_effect = lambda frame, now: (pipeline.update(NORMAL if now < 0.25 else CLOSED, now), None, 1)
        alarm.error = None
        with patch("drowsiness.worker.Webcam", return_value=camera), patch(
                "drowsiness.worker.FrameProcessor", return_value=processor), patch(
                "drowsiness.worker.Alarm", return_value=alarm):
            worker = MonitorWorker(config)
            worker.run()
        packet = worker.frames.get_nowait()
        self.assertEqual(packet.observation.detection.state, State.DANGER)
        self.assertEqual(status_view(packet.observation).key, "DANGER")
        self.assertFalse(packet.audio_enabled)
        events = []
        while not worker.events.empty():
            events.append(worker.events.get_nowait())
        self.assertIn(("state", "졸음 위험 감지"), events)
        for resource in (camera, processor, alarm):
            resource.close.assert_called_once()

    def test_demo_never_plays_physical_audio(self):
        config = Config()
        worker = MonitorWorker(config, "demo")
        worker.stopped.set()
        with patch("drowsiness.worker.Alarm") as alarm:
            worker.run()
        self.assertFalse(alarm.call_args.args[0].enabled)
        self.assertTrue(config.alarm.enabled)  # demo must not modify real settings


if __name__ == "__main__":
    unittest.main()
