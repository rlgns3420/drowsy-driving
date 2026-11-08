import unittest
from unittest.mock import Mock, patch

from drowsiness.config import Config
from drowsiness.models import HeadPose, Metrics
from drowsiness.pipeline import MonitoringPipeline
from drowsiness.worker import MonitorWorker


class RecoveryTests(unittest.TestCase):
    def test_camera_open_failure_read_failure_then_recovery(self):
        config = Config()
        config.display.simple = True
        config.display.camera_retry_seconds = 0.001
        worker = MonitorWorker(config)
        broken, recovered = Mock(backend="any"), Mock(backend="any")
        broken.read.side_effect = RuntimeError("unplugged")
        recovered.read.return_value = (object(), 1.0)
        processor = Mock()
        observation = Mock()
        observation.detection = None
        processor.process.return_value = (observation, None, 0)
        processor.process.side_effect = lambda *args: (worker.stop() or observation, None, 0)
        with patch("drowsiness.worker.Webcam", side_effect=[RuntimeError("missing"), broken, recovered]) as webcam, \
                patch("drowsiness.worker.FrameProcessor", return_value=processor), \
                patch("drowsiness.worker.Alarm"), patch("drowsiness.worker.status_view", return_value=Mock(key="READY", title="ready")):
            worker.run()
        self.assertEqual(webcam.call_count, 3)
        broken.close.assert_called_once()
        recovered.close.assert_called_once()
        self.assertEqual(worker.frames.qsize(), 1)
        kinds = [item[0] for item in list(worker.events.queue)]
        self.assertEqual(kinds.count("retrying"), 2)
        self.assertIn("finished", kinds)

    def test_stop_interrupts_retry_wait(self):
        config = Config()
        config.display.simple = True
        worker = MonitorWorker(config)
        with patch("drowsiness.worker.Webcam", side_effect=RuntimeError("missing")):
            worker.start()
            self.assertEqual(worker.events.get(timeout=2)[0], "retrying")
            worker.stop()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())

    def test_manual_reconnect_interrupts_retry_wait(self):
        config = Config()
        config.display.simple = True
        worker = MonitorWorker(config)
        def open_camera(*args):
            if camera.call_count == 2:
                worker.stop()
            raise RuntimeError("missing")
        with patch("drowsiness.worker.Webcam", side_effect=open_camera) as camera:
            worker.start()
            worker.events.get(timeout=2)
            worker.reconnect()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(camera.call_count, 2)

    def test_face_return_resets_only_after_configured_absence_in_exhibit(self):
        normal = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
        for simple, enabled, duration, expected in [(True, True, 5, True), (True, True, 4.9, False),
                (False, True, 6, False), (True, False, 6, False)]:
            with self.subTest(simple=simple, enabled=enabled, duration=duration):
                config = Config()
                config.display.simple, config.display.auto_recalibrate = simple, enabled
                pipeline = MonitoringPipeline(config)
                with patch.object(pipeline, "reset", wraps=pipeline.reset) as reset:
                    pipeline.update(None, 1)
                    pipeline.update(None, 2)
                    pipeline.update(normal, 1 + duration)
                    self.assertEqual(reset.called, expected)
                    pipeline.update(normal, 8)
                    self.assertEqual(reset.call_count, int(expected))

    def test_escape_requires_second_press_within_two_seconds(self):
        from drowsiness.desktop import MonitorApp
        app = Mock(simple=True, escape_at=None)
        with patch("drowsiness.desktop.time.monotonic", side_effect=[1, 4, 5]):
            MonitorApp.escape(app)
            app.close.assert_not_called()
            MonitorApp.escape(app)
            app.close.assert_not_called()
            MonitorApp.escape(app)
            app.close.assert_called_once()

    def test_reconnect_button_signals_existing_camera_worker(self):
        from drowsiness.desktop import MonitorApp
        app = Mock()
        app.worker.mode = "camera"
        MonitorApp.reconnect_camera(app)
        app.worker.reconnect.assert_called_once()
        app.worker = None
        MonitorApp.reconnect_camera(app)
        app.start_camera.assert_called_once()

    def test_calibrated_visitor_profile_is_discarded_on_return(self):
        config = Config()
        config.display.simple = True
        config.calibration.seconds, config.calibration.min_samples = 0.2, 3
        pipeline = MonitoringPipeline(config)
        normal = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
        for index in range(20):
            observation = pipeline.update(normal, index / 20)
        self.assertIsNotNone(observation.profile)
        pipeline.update(None, 1)
        observation = pipeline.update(normal, 6)
        self.assertIsNone(observation.profile)
        self.assertIsNone(observation.detection)
