import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import cv2
import numpy as np

from drowsiness.camera import VideoFile
from drowsiness.config import Config
from drowsiness.evaluation import Label
from drowsiness.models import HeadPose, Metrics
from drowsiness.pipeline import MonitoringPipeline
from drowsiness.session import run_session
from tools.build_pi import build

ROOT = Path(__file__).resolve().parents[1]
NORMAL = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
CLOSED = Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0))


def make_video(path, count=80, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Test MJPG encoder unavailable")
    try:
        for _ in range(count):
            writer.write(np.full((240, 320, 3), 100, dtype=np.uint8))
    finally:
        writer.release()


class SessionTests(unittest.TestCase):
    def config(self):
        config = Config()
        config.alarm.enabled = False
        config.camera.mirror = False
        config.calibration.seconds = 0.2
        config.calibration.min_samples = 3
        return config

    def processor(self, config):
        processor = Mock()
        processor.pipeline = MonitoringPipeline(config)
        processor.process.side_effect = lambda frame, timestamp: (
            processor.pipeline.update(NORMAL if timestamp < 1 else CLOSED if timestamp < 7 else None,
                                      timestamp), None, 1.0)
        return processor

    def test_video_time_drives_decisions_report_events_labels_and_decodable_export(self):
        config = self.config()
        processor = self.processor(config)
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            path = temp / "input.avi"
            make_video(path)
            video = VideoFile(path)
            with patch("drowsiness.session.FrameProcessor", return_value=processor), patch(
                    "drowsiness.session.time.sleep") as sleep:
                report = run_session(config, lambda: video, temp / "result", save_video=True,
                                     labels=[Label(0, 1, "exclude"), Label(1, 2, "normal"),
                                             Label(2, 7, "drowsy"), Label(7, 8, "normal")])
            sleep.assert_not_called()
            self.assertFalse(video.capture.isOpened())
            processor.close.assert_called_once()
            self.assertEqual(report["frames_processed"], 80)
            self.assertEqual(report["status"], "completed")
            self.assertIsNotNone(report["profile"])
            self.assertGreaterEqual(report["first_occurrence_seconds"]["WARNING"], 1.8)
            self.assertLess(report["first_occurrence_seconds"]["WARNING"], 2.1)
            self.assertGreaterEqual(report["first_occurrence_seconds"]["DANGER"], 3)
            self.assertGreaterEqual(report["first_occurrence_seconds"]["ALARM"], 5)
            self.assertEqual(report["state_frame_counts"]["TRACKING_LOST"], 10)
            self.assertTrue(report["scoring"]["both_classes_scored"])
            self.assertEqual(report["scoring"]["skipped_windows"]["tracking_unavailable"], 1)
            rows = list(csv.DictReader((temp / "result" / "frames.csv").read_text(encoding="utf-8-sig").splitlines()))
            self.assertEqual(len(rows), 80)
            self.assertEqual(rows[-1]["state"], "TRACKING_LOST")
            self.assertEqual(rows[-1]["alarm_requested"], "0")
            self.assertTrue(any(row["alarm_requested"] == "1" for row in rows))
            events = list(csv.DictReader((temp / "result" / "events.csv").read_text(encoding="utf-8-sig").splitlines()))
            self.assertLess(len(events), 12)
            output = VideoFile(temp / "result" / "annotated.avi")
            try:
                self.assertEqual(output.fps, 10)
                count = 0
                while True:
                    try:
                        frame, _ = output.read()
                        count += 1
                    except EOFError:
                        break
                self.assertEqual(count, 80)
                self.assertGreater(frame.shape[1], 320)
            finally:
                output.close()

    def test_failure_and_interrupt_save_partial_report_and_release_resources(self):
        for failure in (RuntimeError("model failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                config = self.config()
                processor = self.processor(config)
                camera = Mock(fps=10)
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                camera.read.side_effect = [(frame, 10), failure]
                with patch("drowsiness.session.FrameProcessor", return_value=processor):
                    if isinstance(failure, RuntimeError):
                        with self.assertRaisesRegex(RuntimeError, "model failed"):
                            run_session(config, lambda: camera, Path(directory) / "out")
                    else:
                        run_session(config, lambda: camera, Path(directory) / "out")
                report = json.loads((Path(directory) / "out" / "report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["frames_processed"], 1)
                self.assertEqual(report["status"], "failed" if isinstance(failure, RuntimeError) else "interrupted")
                processor.close.assert_called_once()
                camera.close.assert_called_once()

    def test_output_collision_is_rejected_before_camera_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = Mock()
            with self.assertRaises(FileExistsError):
                run_session(self.config(), factory, Path(directory))
            factory.assert_not_called()

    def test_fps_override_changes_media_time_and_invalid_override_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.avi"
            make_video(path, 3)
            video = VideoFile(path, fps=15)
            try:
                self.assertEqual([video.read()[1] for _ in range(3)], [0, 1 / 15, 2 / 15])
            finally:
                video.close()
            for fps in (0, -1, float("nan"), float("inf"), 241):
                with self.assertRaises(ValueError):
                    VideoFile(path, fps=fps)

    def test_duration_rotation_and_no_optional_gui_or_audio(self):
        config = self.config()
        processor = self.processor(config)
        camera = Mock(fps=10)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        camera.read.side_effect = [(frame, 20 + index / 10) for index in range(10)]
        with tempfile.TemporaryDirectory() as directory, patch(
                "drowsiness.session.FrameProcessor", return_value=processor), patch(
                "drowsiness.session.cv2.namedWindow") as window, patch(
                "drowsiness.alarm.Alarm._play") as play:
            report = run_session(config, lambda: camera, Path(directory) / "out",
                                 duration=0.5, rotation=90)
        self.assertEqual(report["frames_processed"], 5)
        self.assertEqual(report["status"], "duration_reached")
        self.assertEqual(processor.process.call_args.args[0].shape, (320, 240, 3))
        window.assert_not_called()
        play.assert_not_called()

    def test_actual_cli_decodes_and_runs_native_model_then_reports_missing_face_honestly(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            video = temp / "blank.avi"
            make_video(video, 15, 15)
            child = subprocess.run([sys.executable, str(ROOT / "test" / "video.py"),
                                    "--video", str(video), "--save-video", "--output", str(temp / "result")],
                                   cwd=temp, capture_output=True, timeout=40)
            self.assertEqual(child.returncode, 2, child.stderr)
            report = json.loads((temp / "result" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frames_processed"], 15)
            self.assertIsNone(report["calibration_finished_at_seconds"])
            self.assertEqual(report["tracking_fraction_all_frames"], 0)
            self.assertIsNone(report["scoring"])
            self.assertFalse(report["accuracy_validated"])
            self.assertTrue((temp / "result" / "annotated.avi").is_file())


class PiDeploymentTests(unittest.TestCase):
    def test_csi_adapter_selects_opencv_bgr_format_and_cleans_up_on_start_failure(self):
        from drowsiness.pi_camera import PiCamera
        instance = Mock()
        instance.capture_array.return_value = np.zeros((240, 320, 3), dtype=np.uint8)
        module = Mock()
        module.Picamera2.return_value = instance
        with patch.dict(sys.modules, {"picamera2": module}):
            camera = PiCamera(Config().camera)
            self.assertEqual(instance.create_video_configuration.call_args.kwargs["main"]["format"], "RGB888")
            self.assertFalse(instance.create_video_configuration.call_args.kwargs["queue"])
            self.assertEqual(camera.read()[0].shape, (240, 320, 3))
            camera.close()
        instance.stop.assert_called_once()
        instance.close.assert_called_once()
        instance.reset_mock()
        instance.start.side_effect = RuntimeError("CSI start failed")
        with patch.dict(sys.modules, {"picamera2": module}), self.assertRaises(RuntimeError):
            PiCamera(Config().camera)
        instance.close.assert_called_once()

    def test_standalone_bundle_hashes_and_actual_offline_runtime_without_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            archive_path = build(temp / "pi.zip")
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                self.assertNotIn("drowsiness-pi/drowsiness/desktop.py", names)
                self.assertNotIn("drowsiness-pi/drowsiness/worker.py", names)
                self.assertFalse(any(".venv" in name or name.endswith(".avi") for name in names))
                archive.extractall(temp)
            bundle = temp / "drowsiness-pi"
            manifest = json.loads((bundle / "MANIFEST.json").read_text())
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), expected)
            self.assertNotIn(b"\r\n", (bundle / "pi" / "setup.sh").read_bytes())
            video = temp / "blank.avi"
            make_video(video, 3)
            child = subprocess.run([sys.executable, str(bundle / "pi" / "run.py"), "--video", str(video),
                                    "--output", str(temp / "result")], cwd=temp, capture_output=True, timeout=40)
            self.assertEqual(child.returncode, 2, child.stderr)
            report = json.loads((temp / "result" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frames_processed"], 3)
            self.assertEqual(report["config"]["camera"]["inference_width"], 320)
            self.assertFalse(report["config"]["alarm"]["enabled"])


if __name__ == "__main__":
    unittest.main()
