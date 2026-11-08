"""Exhibition wording uses the real pipeline's state transitions."""
import json
from pathlib import Path
import tempfile
import unittest

from drowsiness.config import Config
from drowsiness.models import HeadPose, Metrics
from drowsiness.pipeline import MonitoringPipeline
from drowsiness.presentation import exhibit_view, status_view


class ExhibitTests(unittest.TestCase):
    def test_decision_transitions_have_short_visitor_titles(self):
        config = Config()
        config.calibration.seconds, config.calibration.min_samples = 0.2, 3
        pipeline = MonitoringPipeline(config)
        normal = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
        closed = Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0))
        seen = {}
        for index in range(120):
            observation = pipeline.update(normal if index < 20 else closed, index / 20)
            detailed = status_view(observation)
            simple = exhibit_view(detailed)
            seen[simple.key] = simple.title
            self.assertEqual(simple.key, detailed.key)
            self.assertEqual(simple.color, detailed.color)
        self.assertEqual({key: seen[key] for key in ("NORMAL", "WARNING", "DANGER")},
                         {"NORMAL": "정상", "WARNING": "주의", "DANGER": "졸음 위험"})
        pipeline.reset()
        self.assertEqual(exhibit_view(status_view(pipeline.update(normal, 10))).title, "준비 중")

    def test_config_preserves_detailed_defaults_and_validates_exhibit_duration(self):
        self.assertFalse(Config().display.simple)
        self.assertEqual(Config().calibration.seconds, 30)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"display": {"simple": True, "calibration_seconds": 8}}))
            config = Config.load(path)
            self.assertTrue(config.display.simple)
            self.assertEqual(config.display.calibration_seconds, 8)
            config.display.calibration_seconds = 0
            with self.assertRaises(ValueError):
                config.validate()
