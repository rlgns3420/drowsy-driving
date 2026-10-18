from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from drowsiness.evaluation import Label, WindowEvaluator, load_labels
from drowsiness.models import Detection, State


class EvaluationTests(unittest.TestCase):
    def test_false_positive_rate_and_recall_include_missed_drowsiness(self):
        evaluator = WindowEvaluator([Label(0, 2, "normal"), Label(2, 4, "drowsy")], 15)
        for second, state in enumerate((State.NORMAL, State.WARNING, State.NORMAL, State.DANGER)):
            for frame in range(15):
                evaluator.add(second + frame / 15, Detection(state, ()))
        report = evaluator.summarize()
        self.assertEqual(report["confusion"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertEqual(report["false_positive_rate"], 0.5)
        self.assertEqual(report["recall"], 0.5)

    def test_tracking_failure_is_excluded_and_reported_not_scored_as_normal(self):
        evaluator = WindowEvaluator([Label(0, 1, "normal")], 15)
        for frame in range(15):
            evaluator.add(frame / 15, Detection(State.WARNING, (), tracking_valid=False))
        report = evaluator.summarize()
        self.assertEqual(report["scored_windows"], 0)
        self.assertEqual(report["skipped_windows"]["tracking_unavailable"], 1)
        self.assertIsNone(report["false_positive_rate"])

    def test_calibration_and_partial_last_second_are_excluded(self):
        evaluator = WindowEvaluator([Label(0, 2, "normal")], 30)
        for frame in range(30):
            evaluator.add(frame / 30, None)
        evaluator.add(1, Detection(State.NORMAL, ()))
        report = evaluator.summarize()
        self.assertEqual(report["skipped_windows"]["uncalibrated"], 1)
        self.assertEqual(report["skipped_windows"]["incomplete_window"], 1)

    def test_overlapping_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            path.write_text("start,end,label\n0,5,normal\n4,8,drowsy\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_labels(path)

    def test_labels_beyond_video_are_reported_as_missing_coverage(self):
        evaluator = WindowEvaluator([Label(0, 3, "normal")], 15)
        for frame in range(15):
            evaluator.add(frame / 15, Detection(State.NORMAL, ()))
        report = evaluator.summarize()
        self.assertEqual(report["skipped_windows"]["no_video_frames"], 2)
        self.assertAlmostEqual(report["scored_label_coverage"], 1 / 3)

    def test_evaluation_cli_processes_real_video_through_native_face_mesh(self):
        # This tests decoding/model wiring and honest incomplete results, not face accuracy.
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            video = temp / "blank.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(15):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            labels = temp / "labels.csv"
            labels.write_text("start,end,label\n0,1,normal\n", encoding="utf-8")
            child = subprocess.run([sys.executable, str(root / "test" / "evaluate.py"),
                                    "--video", str(video), "--labels", str(labels),
                                    "--output", str(temp / "output")], capture_output=True, timeout=30)
            self.assertEqual(child.returncode, 2, child.stderr)
            report = json.loads((temp / "output" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["frames_processed"], 15)
            self.assertIsNone(report["calibration_finished_at_seconds"])
            self.assertIsNone(report["scoring"]["false_positive_rate"])
            self.assertEqual(report["scoring"]["scored_windows"], 0)
            self.assertFalse(report["prevention_effectiveness_validated"])


if __name__ == "__main__":
    unittest.main()
