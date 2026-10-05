import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from drowsiness.alarm import Alarm
from drowsiness.config import AlarmConfig, Config
from drowsiness.metrics import MetricSmoother
from drowsiness.models import HeadPose, Metrics


class RuntimeTests(unittest.TestCase):
    def test_pc_and_pi_configs_load(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(Config.load(root / "config.json").camera.inference_width, 480)
        self.assertEqual(Config.load(root / "config.pi.json").camera.inference_width, 320)

    def test_unknown_or_invalid_config_rejected(self):
        for payload in ({"camera": {"fps": 0}}, {"detection": {"typo": 2}},
                        {"alarm": {"enabled": "false"}},
                        {"calibration": {"seconds": float("nan")}}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    Config.load(path)

    def test_smoother_gap_does_not_retain_old_closed_eyes(self):
        smoother = MetricSmoother(0.1, 0.5)
        smoother.update(Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0)), 0)
        fresh = Metrics(0.3, 0.3, 0.06, HeadPose(0, 0, 0))
        self.assertEqual(smoother.update(fresh, 10), fresh)

    def test_alarm_worker_plays_and_closes_without_blocking_inference(self):
        import threading
        played = threading.Event()
        with patch.object(Alarm, "_play", side_effect=played.set):
            alarm = Alarm(AlarmConfig(tone_seconds=0.01, repeat_seconds=0.05))
            try:
                alarm.update(False)
                self.assertFalse(played.wait(0.1))
                alarm.update(True)
                self.assertTrue(played.wait(1))
            finally:
                alarm.close()
            self.assertFalse(alarm.thread.is_alive())

    def test_face_mesh_native_graph_processes_blank_image(self):
        # Integration: actually load the bundled models from this workspace,
        # exercising the Windows Unicode workaround when necessary.
        from drowsiness.face_mesh import FaceMeshExtractor
        mesh = FaceMeshExtractor(320)
        temporary = mesh._temporary_models
        try:
            self.assertIsNone(mesh.extract(np.zeros((480, 640, 3), dtype=np.uint8)))
        finally:
            mesh.close()
        if temporary is not None:
            self.assertFalse(Path(temporary.name).exists())

    def test_landmarks_scaled_to_original_pixels(self):
        from drowsiness.face_mesh import FaceMeshExtractor
        from types import SimpleNamespace
        graph = Mock()
        graph.process.return_value = SimpleNamespace(multi_face_landmarks=[
            SimpleNamespace(landmark=[SimpleNamespace(x=0.25, y=0.25)] * 468)])
        with patch("drowsiness.face_mesh.create_face_mesh", return_value=(graph, None)):
            extractor = FaceMeshExtractor(320)
            try:
                points = extractor.extract(np.zeros((480, 640, 3), dtype=np.uint8))
                self.assertEqual(graph.process.call_args.args[0].shape, (240, 320, 3))
                np.testing.assert_allclose(points, np.tile([160, 120], (468, 1)))
            finally:
                extractor.close()

    def test_full_app_wires_calibration_detection_alarm_display_and_cleanup(self):
        from contextlib import ExitStack
        from drowsiness.app import run
        from drowsiness.models import State
        config = Config()
        config.calibration.seconds = 0.15
        config.calibration.min_samples = 3
        config.detection.smoothing_seconds = 0.01
        config.detection.eye_warning_seconds = 0.1
        config.detection.eye_danger_seconds = 0.2
        config.detection.alarm_delay_seconds = 0.1
        config.detection.recovery_seconds = 0.1
        camera, mesh, alarm, estimator = Mock(), Mock(), Mock(), Mock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera.read.side_effect = [(frame.copy(), i / 20) for i in range(35)]
        mesh.extract.return_value = np.ones((468, 2))
        estimator.estimate.return_value = HeadPose(0, 0, 0)
        alarm.error = None
        normal = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
        closed = Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0))
        samples = [normal] * 5 + [closed] * 20 + [normal] * 10
        with ExitStack() as stack:
            for name, instance in (("Webcam", camera), ("FaceMeshExtractor", mesh),
                                   ("Alarm", alarm), ("HeadPoseEstimator", estimator)):
                stack.enter_context(patch(f"drowsiness.app.{name}", return_value=instance))
            stack.enter_context(patch("drowsiness.app.measure", side_effect=samples))
            renderer = stack.enter_context(patch("drowsiness.app.render", return_value=frame))
            for method in ("namedWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch(f"drowsiness.app.cv2.{method}"))
            stack.enter_context(patch("drowsiness.app.cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("drowsiness.app.cv2.waitKey", side_effect=[-1] * 34 + [ord("q")]))
            run(config)
        self.assertEqual(renderer.call_count, 35)
        detections = [call.args[5] for call in renderer.call_args_list if call.args[5] is not None]
        self.assertIn(State.WARNING, [item.state for item in detections])
        self.assertIn(State.DANGER, [item.state for item in detections])
        self.assertTrue(any(item.alarm for item in detections))
        self.assertEqual(detections[-1].state, State.NORMAL)
        self.assertIn(True, [call.args[0] for call in alarm.update.call_args_list])
        for resource in (camera, mesh, alarm):
            resource.close.assert_called_once()

    def test_frame_rate_includes_elapsed_capture_and_display_time(self):
        from drowsiness.display import FrameRate
        meter = FrameRate()
        for index in range(31):
            meter.update(index / 15)
        self.assertAlmostEqual(meter.fps, 15)

    def test_display_preserves_face_area_and_shows_metrics_in_side_panel(self):
        from drowsiness.calibration import Calibrator
        from drowsiness.config import CalibrationConfig
        from drowsiness.display import render
        image = np.full((480, 640, 3), 100, dtype=np.uint8)
        canvas = render(image, None, None, Calibrator(CalibrationConfig()),
                        None, None, 15.0, 20.0, False, False, None, True)
        self.assertGreater(canvas.shape[1], 640)
        np.testing.assert_array_equal(canvas[100:300, 200:400], image[100:300, 200:400])


if __name__ == "__main__":
    unittest.main()
