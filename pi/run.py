"""Raspberry Pi entry point: no Tkinter, no display required by default."""
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Raspberry Pi USB/CSI drowsiness monitor")
    parser.add_argument("--config", type=Path, default=ROOT / "config.pi.json")
    parser.add_argument("--source", choices=("usb", "csi"), default="usb")
    parser.add_argument("--camera", type=int, help="USB device or CSI camera index")
    parser.add_argument("--video", type=Path, help="Offline check using the Pi thresholds/resolution")
    parser.add_argument("--preview", action="store_true", help="OpenCV display; requires desktop")
    parser.add_argument("--mute", action="store_true", help="Disable speaker output, retain alarm decisions")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--rotation", choices=(0, 90, 180, 270), type=int, default=0)
    parser.add_argument("--output", type=Path, help="New results directory")
    args = parser.parse_args(argv)
    if args.video and args.source != "usb":
        parser.error("--video cannot be combined with --source csi")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        from drowsiness.camera import VideoFile, Webcam
        from drowsiness.config import Config
        from drowsiness.session import run_session
        config = Config.load(args.config)
        if args.camera is not None:
            config.camera.index = args.camera
        if args.mute or args.video:
            config.alarm.enabled = False
        if args.video:
            if not args.video.is_file():
                raise ValueError(f"Video file does not exist: {args.video}")
            config.camera.mirror = False
            factory = lambda: VideoFile(args.video)
            name = args.video.name
        elif args.source == "csi":
            from drowsiness.pi_camera import PiCamera
            factory = lambda: PiCamera(config.camera)
            name = f"csi:{config.camera.index}"
        else:
            factory = lambda: Webcam(config.camera)
            name = f"usb:{config.camera.index}"
        config.validate()
        output = args.output or ROOT / "results" / datetime.now().strftime("pi-%Y%m%d-%H%M%S-%f")
        report = run_session(config, factory, output, source_name=name, live=not bool(args.video),
                             duration=args.duration, preview=args.preview, rotation=args.rotation)
        print(f"Results: {output.resolve()}")
        if report["calibration_finished_at_seconds"] is None:
            logging.warning("Calibration incomplete. Keep eyes open and face forward for 30 valid seconds.")
            return 2
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop_on_signal)
    raise SystemExit(main())
