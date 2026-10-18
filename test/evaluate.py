"""Run the real Face Mesh pipeline on a labelled local video; save metrics only."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description="Evaluate labelled real footage before Raspberry Pi migration")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="CSV: start,end,label (normal/drowsy/exclude)")
    parser.add_argument("--config", type=Path, default=ROOT / "test" / "config.local.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "evaluation")
    args = parser.parse_args()
    from drowsiness.camera import VideoFile
    from drowsiness.config import Config
    from drowsiness.evaluation import WindowEvaluator, load_labels
    from drowsiness.processing import FrameProcessor
    video = processor = None
    try:
        labels = load_labels(args.labels)
        config = Config.load(args.config)
        video = VideoFile(args.video)
        processor = FrameProcessor(config)
        evaluator = WindowEvaluator(labels, video.fps)
        args.output.mkdir(parents=True, exist_ok=True)
        frames = tracked = 0
        calibration_finished_at = None
        started = time.perf_counter()
        with (args.output / "frames.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("seconds", "state", "tracking_valid", "ear", "mar", "pitch", "yaw", "roll"))
            while True:
                try:
                    frame, timestamp = video.read()
                except EOFError:
                    break
                observation, _, _ = processor.process(frame, timestamp)
                if observation.profile and calibration_finished_at is None:
                    calibration_finished_at = timestamp
                metrics, detection = observation.metrics, observation.detection
                frames += 1
                tracked += metrics is not None
                evaluator.add(timestamp, detection)
                writer.writerow((round(timestamp, 4), detection.state.value if detection else "CALIBRATING",
                                 bool(detection and detection.tracking_valid),
                                 *([metrics.ear, metrics.mar, metrics.pose.pitch, metrics.pose.yaw,
                                    metrics.pose.roll] if metrics else [""] * 5)))
        summary = evaluator.summarize()
        elapsed = time.perf_counter() - started
        report = {"video": args.video.name, "frames_processed": frames,
                  "tracking_fraction_all_frames": tracked / frames if frames else 0,
                  "calibration_finished_at_seconds": calibration_finished_at,
                  "processing_fps": frames / elapsed if elapsed else 0,
                  "scoring": summary,
                  "prevention_effectiveness_validated": False,
                  "limitations": "Scores apply only to this labelled footage and annotated definitions. "
                                 "Untracked seconds are excluded and must be reviewed separately. "
                                 "Acted eye closure does not establish real fatigue or accident prevention."}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0 if summary["both_classes_scored"] else 2
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for resource in (processor, video):
            if resource is not None:
                resource.close()


if __name__ == "__main__":
    raise SystemExit(main())
