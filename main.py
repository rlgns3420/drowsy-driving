"""Run with Python 3.11: python main.py [--config config.pi.json]."""
import argparse
import logging
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam drowsiness detection using Face Mesh")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.json"),
                        help="JSON settings file (default: config.json next to main.py)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from drowsiness.config import Config
        config = Config.load(args.config)
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
