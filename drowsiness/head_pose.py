"""PnP against a fixed generic 3D face; output angles are already degrees."""
import cv2
import numpy as np

from .models import HeadPose


POSE_INDICES = (1, 152, 33, 263, 61, 291)
# Coordinates: x right, y down, z away from camera, arbitrary fixed units.
# Nose, chin, left/right outer eye corners, left/right mouth corners.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, 63.6, 12.5),
    (-43.3, -32.7, 26.0), (43.3, -32.7, 26.0),
    (-28.9, 28.9, 24.1), (28.9, 28.9, 24.1),
], dtype=np.float64)


class HeadPoseEstimator:
    def __init__(self):
        self.size: tuple[int, int] | None = None
        self.camera_matrix: np.ndarray | None = None
        self.distortion = np.zeros((4, 1), dtype=np.float64)

    def intrinsics(self, width: int, height: int) -> np.ndarray:
        if self.size != (width, height):
            # Approximation; personal neutral pose is subtracted by detector.
            focal = float(width)
            self.camera_matrix = np.array([
                [focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1],
            ], dtype=np.float64)
            self.size = width, height
        return self.camera_matrix

    def estimate(self, landmarks: np.ndarray, width: int, height: int) -> HeadPose | None:
        points = np.asarray(landmarks[list(POSE_INDICES)], dtype=np.float64)
        if not np.isfinite(points).all():
            return None
        eye_span = np.linalg.norm(points[2] - points[3])
        if eye_span < 20:
            return None
        camera = self.intrinsics(width, height)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS, points, camera, self.distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                return None
            if tvec[2, 0] <= 0:
                return None
            projected, _ = cv2.projectPoints(
                MODEL_POINTS, rvec, tvec, camera, self.distortion)
            error = np.sqrt(np.mean(np.sum(
                (projected.reshape(-1, 2) - points) ** 2, axis=1)))
            if error / eye_span > 0.15:
                return None
            rotation, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = cv2.RQDecomp3x3(rotation)[0]
            if not np.isfinite([pitch, yaw, roll]).all():
                return None
            return HeadPose(pitch, yaw, roll)
        except cv2.error:
            return None
