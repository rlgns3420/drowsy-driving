"""Inspect real footage, optionally export an annotated video and score labels."""
import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="실제 촬영 영상 졸음 감지 점검 및 결과 저장")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "test" / "config.local.json")
    parser.add_argument("--labels", type=Path, help="Optional CSV: start,end,label")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--preview", action="store_true", help="Show results during processing; Q/Esc stops")
    parser.add_argument("--save-video", action="store_true", help="Save annotated.avi without audio")
    parser.add_argument("--realtime", action="store_true", help="Pace preview to video time")
    parser.add_argument("--fps", type=float, help="Override incorrect/missing constant FPS metadata")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Additional clockwise rotation after decoder orientation")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from drowsiness.camera import VideoFile
        from drowsiness.config import Config
        from drowsiness.evaluation import load_labels
        from drowsiness.session import run_session
        if not args.video.is_file():
            raise ValueError(f"Video file does not exist: {args.video}")
        config = Config.load(args.config)
        config.alarm.enabled = False
        config.camera.mirror = False
        labels = load_labels(args.labels) if args.labels else None
        output = args.output or ROOT / "results" / datetime.now().strftime("video-%Y%m%d-%H%M%S-%f")
        report = run_session(config, lambda: VideoFile(args.video, fps=args.fps), output,
                             source_name=args.video.name, labels=labels, preview=args.preview,
                             save_video=args.save_video, realtime=args.realtime, rotation=args.rotation)
        print(f"Results: {output.resolve()}")
        if report["status"] != "completed":
            logging.warning("Analysis did not reach EOF: %s", report["status"])
            return 2
        if report["calibration_finished_at_seconds"] is None:
            logging.warning("Calibration incomplete: provide at least 35s of neutral, visible face footage")
            return 2
        if labels is not None and not report["scoring"]["both_classes_scored"]:
            logging.warning("Both normal and drowsy labelled windows are required for scoring")
            return 2
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
