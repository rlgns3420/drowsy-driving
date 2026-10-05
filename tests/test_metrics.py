import unittest

import cv2
import numpy as np

from drowsiness.head_pose import HeadPoseEstimator, MODEL_POINTS, POSE_INDICES
from drowsiness.metrics import angle_delta, eye_aspect_ratio, mouth_aspect_ratio


class MetricTests(unittest.TestCase):
    def test_ear_scale_and_rotation_invariant(self):
        points = np.array([(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)])
        self.assertAlmostEqual(eye_aspect_ratio(points), 0.5)
        rotation = np.array([[0, -1], [1, 0]])
        self.assertAlmostEqual(eye_aspect_ratio(points @ rotation * 10 + 50), 0.5)

    def test_mar_uses_three_inner_lip_pairs(self):
        points = np.array([(0, 0), (1, -1), (2, -1), (3, -1),
                           (4, 0), (3, 1), (2, 1), (1, 1)])
        self.assertAlmostEqual(mouth_aspect_ratio(points), 0.5)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            eye_aspect_ratio(np.zeros((6, 2)))
        with self.assertRaises(ValueError):
            mouth_aspect_ratio(np.full((8, 2), np.nan))

    def test_angle_wrap(self):
        self.assertAlmostEqual(angle_delta(-179, 179), 2)


class PoseTests(unittest.TestCase):
    def test_known_3d_pose_recovered_in_degrees(self):
        estimator = HeadPoseEstimator()
        for expected in ((0, 0, 0), (10, 20, -15), (-15, -25, 10)):
            x, y, z = np.radians(expected)
            rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
            ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
            rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
            rvec, _ = cv2.Rodrigues(rz @ ry @ rx)
            projected, _ = cv2.projectPoints(MODEL_POINTS, rvec,
                np.array([0., 0., 600.]), estimator.intrinsics(640, 480), estimator.distortion)
            landmarks = np.zeros((468, 2))
            landmarks[list(POSE_INDICES)] = projected.reshape(-1, 2)
            pose = estimator.estimate(landmarks, 640, 480)
            self.assertIsNotNone(pose)
            np.testing.assert_allclose([pose.pitch, pose.yaw, pose.roll], expected, atol=0.01)

    def test_degenerate_pose_rejected(self):
        self.assertIsNone(HeadPoseEstimator().estimate(np.zeros((468, 2)), 640, 480))


if __name__ == "__main__":
    unittest.main()
