import unittest

from drowsiness.config import DetectionConfig
from drowsiness.detector import DrowsinessDetector
from drowsiness.models import HeadPose, Metrics, Profile, State


PROFILE = Profile(0.3, 0.06, HeadPose(0, 0, 0), 0.216, 0.35,
                  HeadPose(15, 20, 18), 600, 20)
NORMAL = Metrics(0.30, 0.30, 0.06, HeadPose(0, 0, 0))
CLOSED = Metrics(0.05, 0.05, 0.06, HeadPose(0, 0, 0))
YAWN = Metrics(0.30, 0.30, 0.60, HeadPose(0, 0, 0))
TURN = Metrics(0.30, 0.30, 0.06, HeadPose(0, 30, 0))
NOD = Metrics(0.30, 0.30, 0.06, HeadPose(25, 0, 0))


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = DrowsinessDetector(PROFILE, DetectionConfig())

    def feed(self, sample, duration, start=0, fps=20):
        result = None
        for index in range(round(duration * fps) + 1):
            result = self.detector.update(sample, start + index / fps)
        return result

    def test_brief_blink_remains_normal(self):
        result = self.feed(CLOSED, 0.2)
        self.assertEqual(result.state, State.NORMAL)
        self.assertEqual(self.detector.update(NORMAL, 0.25).eye_seconds, 0)

    def test_unilateral_wink_remains_normal(self):
        wink = Metrics(0.03, 0.30, 0.06, HeadPose(0, 0, 0))
        self.assertEqual(self.feed(wink, 5).state, State.NORMAL)

    def test_eye_warning_then_danger_then_delayed_alarm(self):
        result = self.feed(CLOSED, 0.85)
        self.assertEqual(result.state, State.WARNING)
        result = self.feed(CLOSED, 1.20, start=0.90)
        self.assertEqual(result.state, State.DANGER)
        self.assertFalse(result.alarm)
        result = self.feed(CLOSED, 1.95, start=2.15)
        self.assertTrue(result.alarm)

    def test_yawn_alone_warns_without_danger(self):
        result = self.feed(YAWN, 6)
        self.assertEqual(result.state, State.WARNING)
        self.assertFalse(result.alarm)

    def test_yaw_alone_warns_without_danger(self):
        result = self.feed(TURN, 6)
        self.assertEqual(result.state, State.WARNING)
        self.assertFalse(result.alarm)

    def test_persistent_pitch_is_danger(self):
        result = self.feed(NOD, 6.1)
        self.assertEqual(result.state, State.DANGER)
        self.assertTrue(result.alarm)

    def test_fusion_of_yawn_and_head(self):
        combined = Metrics(0.30, 0.30, 0.60, HeadPose(0, 30, 0))
        result = self.feed(combined, 2.6)
        self.assertEqual(result.state, State.DANGER)
        self.assertFalse(result.alarm)

    def test_missing_face_resets_danger_evidence(self):
        self.assertTrue(self.feed(CLOSED, 4.1).alarm)
        missing = self.detector.update(None, 4.15)
        self.assertEqual(missing.state, State.WARNING)
        self.assertFalse(missing.tracking_valid)
        self.assertFalse(missing.alarm)
        fresh = self.detector.update(CLOSED, 4.20)
        self.assertEqual(fresh.eye_seconds, 0)
        self.assertFalse(fresh.alarm)

    def test_long_processing_gap_resets_timers(self):
        self.feed(CLOSED, 1)
        result = self.detector.update(CLOSED, 10)
        self.assertEqual(result.eye_seconds, 0)
        self.assertEqual(result.state, State.NORMAL)

    def test_recovery_holds_display_but_stops_audio_immediately(self):
        self.feed(CLOSED, 4.1)
        result = self.detector.update(NORMAL, 4.15)
        self.assertEqual(result.state, State.DANGER)
        self.assertFalse(result.alarm)
        result = self.feed(NORMAL, 1.1, start=4.20)
        self.assertEqual(result.state, State.NORMAL)

    def test_same_outcome_at_15_and_30_fps(self):
        for fps in (15, 30):
            self.detector = DrowsinessDetector(PROFILE, DetectionConfig())
            result = self.feed(CLOSED, 4.2, fps=fps)
            self.assertEqual(result.state, State.DANGER)
            self.assertTrue(result.alarm)

    def test_personal_head_offset_is_subtracted(self):
        offset = Profile(0.3, 0.06, HeadPose(15, 20, 10), 0.216, 0.35,
                         HeadPose(15, 20, 18), 600, 0)
        self.detector = DrowsinessDetector(offset, DetectionConfig())
        sample = Metrics(0.30, 0.30, 0.06, offset.pose_baseline)
        self.assertEqual(self.feed(sample, 5).state, State.NORMAL)

    def test_nan_measurement_is_tracking_warning(self):
        sample = Metrics(float("nan"), 0.3, 0.06, HeadPose(0, 0, 0))
        result = self.detector.update(sample, 0)
        self.assertFalse(result.tracking_valid)
        self.assertEqual(result.state, State.WARNING)


if __name__ == "__main__":
    unittest.main()
