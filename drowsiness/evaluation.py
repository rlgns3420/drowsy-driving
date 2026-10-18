"""Score labelled real video in one-second windows, with explicit coverage."""
from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from .models import State


@dataclass(frozen=True)
class Label:
    start: float
    end: float
    label: str


def load_labels(path: Path) -> list[Label]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["start", "end", "label"]:
            raise ValueError("Label CSV header must be start,end,label")
        labels = []
        for row in reader:
            start, end = float(row["start"]), float(row["end"])
            label = row["label"].strip().lower()
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError("Labels need finite times with 0 <= start < end")
            if label not in ("normal", "drowsy", "exclude"):
                raise ValueError("Labels must be normal, drowsy, or exclude")
            labels.append(Label(start, end, label))
    labels.sort(key=lambda item: item.start)
    if not labels:
        raise ValueError("At least one label is required")
    if any(left.end > right.start for left, right in zip(labels, labels[1:])):
        raise ValueError("Label intervals must not overlap")
    return labels


class WindowEvaluator:
    """A majority of tracked frames decides each fully labelled second."""
    def __init__(self, labels: list[Label], fps: float):
        self.labels, self.fps = labels, fps
        self.windows = defaultdict(list)

    def add(self, timestamp: float, detection):
        self.windows[math.floor(timestamp)].append(detection)

    def summarize(self):
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        skipped = {"unlabelled_or_boundary": 0, "excluded": 0,
                   "incomplete_window": 0, "uncalibrated": 0, "tracking_unavailable": 0,
                   "no_video_frames": 0}
        labelled_windows_present = 0
        for second, observations in sorted(self.windows.items()):
            label = next((item.label for item in self.labels
                          if item.start <= second and second + 1 <= item.end), None)
            if label is None:
                skipped["unlabelled_or_boundary"] += 1
                continue
            if label == "exclude":
                skipped["excluded"] += 1
                continue
            labelled_windows_present += 1
            if len(observations) < math.floor(self.fps):
                skipped["incomplete_window"] += 1
                continue
            calibrated = [item for item in observations if item is not None]
            if len(calibrated) * 2 < len(observations):
                skipped["uncalibrated"] += 1
                continue
            tracked = [item for item in calibrated if item.tracking_valid]
            if len(tracked) * 2 < len(observations):
                skipped["tracking_unavailable"] += 1
                continue
            positive = sum(item.state is not State.NORMAL for item in tracked) * 2 >= len(tracked)
            key = ("tp" if positive else "fn") if label == "drowsy" else ("fp" if positive else "tn")
            counts[key] += 1
        total = sum(counts.values())
        labelled_windows_expected = sum(max(0, math.floor(item.end) - math.ceil(item.start))
                                        for item in self.labels if item.label != "exclude")
        skipped["no_video_frames"] = max(0, labelled_windows_expected - labelled_windows_present)
        negative, positive = counts["fp"] + counts["tn"], counts["tp"] + counts["fn"]
        return {"unit": "one_second_window", "positive_prediction": "WARNING or DANGER",
                "minimum_tracking_fraction": 0.5, "confusion": counts, "scored_windows": total,
                "labelled_windows_expected": labelled_windows_expected,
                "scored_label_coverage": total / labelled_windows_expected if labelled_windows_expected else 0,
                "skipped_windows": skipped,
                "false_positive_rate": counts["fp"] / negative if negative else None,
                "recall": counts["tp"] / positive if positive else None,
                "accuracy": (counts["tp"] + counts["tn"]) / total if total else None,
                "both_classes_scored": bool(negative and positive)}
