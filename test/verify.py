"""Reproducible software checks and bounded real-camera probes; no GUI."""
import argparse
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def probe(index, backend):
    from drowsiness.camera import Webcam
    from drowsiness.config import Config
    from drowsiness.processing import FrameProcessor
    config = Config.load(ROOT / "test" / "config.local.json")
    config.camera.index, config.camera.backend = index, backend
    camera = processor = None
    try:
        camera = Webcam(config.camera)
        processor = FrameProcessor(config)
        started = time.perf_counter()
        tracked = 0
        for _ in range(45):
            frame, timestamp = camera.read()
            observation, _, _ = processor.process(frame, timestamp)
            tracked += observation.metrics is not None
        height, width = frame.shape[:2]
        return {"index": index, "status": "frames_received", "backend": camera.backend,
                "resolution": [width, height], "frames": 45, "tracked_frames": tracked,
                "processing_fps": round(45 / (time.perf_counter() - started), 2),
                "accuracy_validated": False}
    except Exception as exc:
        return {"index": index, "status": "unavailable", "error": str(exc),
                "accuracy_validated": False}
    finally:
        for resource in (processor, camera):
            if resource is not None:
                resource.close()


def main():
    parser = argparse.ArgumentParser(description="Software verification and real camera connection checks")
    parser.add_argument("--camera", action="store_true", help="Also check cameras 0..3 (no images are saved)")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "v4l2"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "verification.json")
    parser.add_argument("--probe-camera", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe_camera is not None:
        print(json.dumps(probe(args.probe_camera, args.backend), ensure_ascii=True))
        return 0
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    report = {"checked_at_local": datetime.now().isoformat(timespec="seconds"),
              "software": {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
                           "failures": len(result.failures), "errors": len(result.errors),
                           "details": stream.getvalue()},
              "camera_checks": [], "real_drowsiness_accuracy_validated": False,
              "prevention_effectiveness_validated": False}
    if args.camera:
        for index in range(4):
            # Isolate native drivers: a camera stuck in read() cannot hang verification.
            try:
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--probe-camera", str(index), "--backend", args.backend],
                                       capture_output=True, text=True, timeout=25)
                if child.returncode:
                    check = {"index": index, "status": "probe_failed", "error": child.stderr[-1000:]}
                else:
                    check = json.loads(child.stdout.strip().splitlines()[-1])
            except subprocess.TimeoutExpired:
                check = {"index": index, "status": "timeout", "error": "Camera driver exceeded 25s; probe terminated"}
            report["camera_checks"].append(check)
            print(json.dumps(check, ensure_ascii=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verification report: {args.output}")
    if not result.wasSuccessful():
        return 1
    if args.camera and not any(item["status"] == "frames_received" for item in report["camera_checks"]):
        print("Software checks passed; no usable camera. Actual drowsiness accuracy remains unverified.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
