"""Run from any directory: python path/to/test/local.py [--demo]."""
import argparse
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description="Local Korean drowsiness monitor")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.local.json"))
    parser.add_argument("--camera", type=int, help="USB/built-in webcam index")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "v4l2"))
    parser.add_argument("--demo", action="store_true", help="Synthetic input; not a camera accuracy test")
    parser.add_argument("--video", type=Path, help="Replay a local recorded video")
    parser.add_argument("--start", action="store_true", help="Start the selected camera immediately")
    parser.add_argument("--ui-smoke", action="store_true", help="Check UI states and close automatically")
    parser.add_argument("--exhibit", action="store_true", help="Fullscreen exhibition dashboard")
    args = parser.parse_args()
    if args.demo and args.video:
        parser.error("--demo and --video cannot be combined")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from drowsiness.config import Config
        from drowsiness.desktop import MonitorApp
        config = Config.load(args.config)
        if args.camera is not None:
            config.camera.index = args.camera
        if args.backend is not None:
            config.camera.backend = args.backend
        if args.exhibit:
            config.display.simple = True
            config.alarm.enabled = True
        config.validate()
        app = MonitorApp(config, demo=args.demo, video=args.video)
        if args.ui_smoke:
            return app.smoke_check(ROOT / ".cache" / ("exhibit-ui" if config.display.simple else "ui"))
        if args.start or args.demo or args.video or config.display.simple:
            app.root.after(150, app.start)
        app.root.mainloop()
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
