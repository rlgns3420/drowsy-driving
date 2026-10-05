import unittest

from drowsiness.calibration import Calibrator
from drowsiness.config import CalibrationConfig
from drowsiness.models import HeadPose, Metrics


def neutral(ear=0.30, mar=0.06, pose=HeadPose(2., -3., 1.)):
    return Metrics(ear, ear, mar, pose)


class CalibrationTests(unittest.TestCase):
    def test_collects_30_seconds_and_removes_in_range_outliers(self):
        calibrator = Calibrator(CalibrationConfig())
        for index in range(601):
            sample = neutral()
            if index % 10 == 5:
                sample = neutral(0.48, 0.29, HeadPose(24., 24., 24.))
            result = calibrator.update(sample, index / 20)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.ear_baseline, 0.3)
        self.assertAlmostEqual(result.ear_threshold, 0.216)
        self.assertGreater(result.samples_rejected, 50)
        self.assertEqual(result.pose_baseline, HeadPose(2., -3., 1.))

    def test_missing_face_and_blinks_pause_collection(self):
        calibrator = Calibrator(CalibrationConfig())
        for index in range(200):
            calibrator.update(None if index % 2 else neutral(ear=0.04), index / 20)
        self.assertEqual(calibrator.valid_seconds, 0)
        self.assertEqual(len(calibrator.samples), 0)
        self.assertIsNone(calibrator.profile)

    def test_unobserved_time_is_not_calibration_time(self):
        calibrator = Calibrator(CalibrationConfig())
        calibrator.update(neutral(), 0)
        calibrator.update(neutral(), 40)
        self.assertEqual(calibrator.valid_seconds, 0)
        self.assertIsNone(calibrator.profile)

    def test_small_eyes_get_personal_threshold(self):
        config = CalibrationConfig(seconds=1, min_samples=10)
        calibrator = Calibrator(config)
        for index in range(21):
            result = calibrator.update(neutral(ear=0.16), index / 20)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.ear_threshold, 0.1152)

    def test_too_many_outliers_restart_collection(self):
        config = CalibrationConfig(seconds=1, min_samples=10, min_inlier_fraction=0.99)
        calibrator = Calibrator(config)
        for index in range(21):
            sample = neutral(0.48, 0.29) if index == 10 else neutral()
            calibrator.update(sample, index / 20)
        self.assertIsNone(calibrator.profile)
        self.assertEqual(calibrator.valid_seconds, 0)
        self.assertIn("collecting again", calibrator.message)

    def test_time_must_increase(self):
        calibrator = Calibrator(CalibrationConfig())
        calibrator.update(neutral(), 1)
        with self.assertRaises(ValueError):
            calibrator.update(neutral(), 1)


if __name__ == "__main__":
    unittest.main()
