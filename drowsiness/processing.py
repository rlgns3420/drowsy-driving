"""Actual video -> landmarks -> metrics -> calibrated decisions."""
import time

from .config import Config
from .face_mesh import FaceMeshExtractor
from .head_pose import HeadPoseEstimator
from .metrics import measure
from .pipeline import MonitoringPipeline


class FrameProcessor:
    def __init__(self, config: Config):
        self.pipeline = MonitoringPipeline(config)
        self.estimator = HeadPoseEstimator()
        self.mesh = FaceMeshExtractor(config.camera.inference_width)

    def process(self, frame, timestamp):
        started = time.perf_counter()
        landmarks = self.mesh.extract(frame)
        raw = None
        if landmarks is not None:
            height, width = frame.shape[:2]
            pose = self.estimator.estimate(landmarks, width, height)
            if pose is not None:
                try:
                    raw = measure(landmarks, pose)
                except ValueError:
                    pass
        result = self.pipeline.update(raw, timestamp)
        return result, landmarks, (time.perf_counter() - started) * 1000

    def close(self):
        self.mesh.close()
