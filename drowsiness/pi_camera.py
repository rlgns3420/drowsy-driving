"""Optional CSI adapter. Import Picamera2 only when a CSI source is selected."""
import time

from .config import CameraConfig


class PiCamera:
    def __init__(self, config: CameraConfig):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("CSI requires apt install python3-picamera2 and a "
                               "Python 3.11 venv with --system-site-packages") from exc
        self.camera = Picamera2(camera_num=config.index)
        self.fps = config.fps
        self.backend = "picamera2"
        try:
            # Picamera2's RGB888 gives B,G,R bytes, as required by OpenCV.
            settings = self.camera.create_video_configuration(
                main={"size": (config.width, config.height), "format": "RGB888"},
                controls={"FrameRate": float(config.fps)}, buffer_count=4,
                queue=False)
            self.camera.configure(settings)
            self.camera.start()
        except BaseException:
            self.camera.close()
            raise

    def read(self):
        # No preview window or display server is needed for capture_array.
        frame = self.camera.capture_array("main")
        now = time.monotonic()
        if frame is None or not frame.size:
            raise RuntimeError("CSI camera returned an empty image")
        return frame, now

    def close(self):
        try:
            self.camera.stop()
        finally:
            self.camera.close()
