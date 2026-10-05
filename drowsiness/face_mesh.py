import cv2
import numpy as np

from .mediapipe_compat import create_face_mesh


class FaceMeshExtractor:
    def __init__(self, inference_width: int = 480):
        self.inference_width = inference_width
        self.mesh, self._temporary_models = create_face_mesh()

    def extract(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        if width > self.inference_width:
            frame = cv2.resize(frame, (self.inference_width,
                round(height * self.inference_width / width)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        # Convert back to ORIGINAL frame pixels, correcting the aspect ratio.
        points = np.array([(p.x * width, p.y * height)
                           for p in result.multi_face_landmarks[0].landmark],
                          dtype=np.float64)
        if not np.isfinite(points).all():
            return None
        return points

    def close(self) -> None:
        try:
            self.mesh.close()
        finally:
            if self._temporary_models is not None:
                self._temporary_models.cleanup()
