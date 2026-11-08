"""Run with Python 3.11: python main.py [--config config.pi.json]."""
import argparse
import logging
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam drowsiness detection using Face Mesh")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.json"),
                        help="JSON settings file (default: config.json next to main.py)")
    parser.add_argument("--ui", choices=("desktop", "opencv"),
                        default="desktop" if os.name == "nt" else "opencv",
                        help="Windows defaults to the Korean dashboard; Pi/Linux use OpenCV")
    parser.add_argument("--simple", "--exhibit", action="store_true", help="Fullscreen exhibition dashboard")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from drowsiness.config import Config
        config = Config.load(args.config)
        if args.simple:
            config.display.simple = True
        if args.ui == "opencv" and config.display.simple:
            logging.warning("--simple/--exhibit requires --ui desktop; exhibition UI is unavailable in OpenCV")
        if args.ui == "desktop":
            from drowsiness.desktop import MonitorApp
            app = MonitorApp(config)
            if config.display.simple:
                app.root.after(150, app.start)
            app.root.mainloop()
        else:
            from drowsiness.app import run
            run(config)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
