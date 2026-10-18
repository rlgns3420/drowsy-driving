"""Check the actual camera, native Face Mesh model and optional audio on Pi."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pi camera/model connection check (not accuracy scoring)")
    parser.add_argument("--source", choices=("usb", "csi"), default="usb")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--audio", action="store_true", help="Play one audible speaker tone")
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    camera = processor = None
    try:
        from drowsiness.alarm import Alarm
        from drowsiness.camera import Webcam
        from drowsiness.config import Config
        from drowsiness.processing import FrameProcessor
        config = Config.load(ROOT / "config.pi.json")
        config.camera.index = args.camera
        config.validate()
        if args.source == "csi":
            from drowsiness.pi_camera import PiCamera
            camera = PiCamera(config.camera)
        else:
            camera = Webcam(config.camera)
        processor = FrameProcessor(config)
        tracked = 0
        for _ in range(args.frames):
            frame, timestamp = camera.read()
            observation, _, _ = processor.process(frame, timestamp)
            tracked += observation.metrics is not None
        if args.audio:
            # A deliberate one-shot device test; do not wait for drowsiness.
            config.alarm.enabled = False
            Alarm(config.alarm)._play()
        print(json.dumps({"camera": args.source, "backend": camera.backend,
                          "frames_received": args.frames, "tracked_frames": tracked,
                          "resolution": list(frame.shape[1::-1]), "audio_tested": args.audio,
                          "accuracy_validated": False}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Pi check failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for resource in (processor, camera):
            if resource is not None:
                resource.close()


if __name__ == "__main__":
    raise SystemExit(main())
