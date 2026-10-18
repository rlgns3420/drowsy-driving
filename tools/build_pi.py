"""Produce a standalone Pi ZIP from shared sources, excluding PC UI and footage."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"desktop.py", "worker.py", "demo.py", "presentation.py"}


def build(output):
    output = Path(output)
    files = [ROOT / "requirements.txt", ROOT / "config.pi.json"]
    files += sorted(path for path in (ROOT / "drowsiness").glob("*.py") if path.name not in EXCLUDED)
    files += sorted(path for path in (ROOT / "pi").iterdir() if path.is_file())
    # Versioned source hashes make the deployed copy identifiable without Git.
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr("drowsiness-pi/" + name, data)
        archive.writestr("drowsiness-pi/MANIFEST.json", json.dumps(manifest, indent=2))
        archive.writestr("drowsiness-pi/README.md", (ROOT / "pi" / "README.md").read_text(encoding="utf-8"))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a standalone Raspberry Pi deployment ZIP")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "drowsiness-pi.zip")
    args = parser.parse_args(argv)
    print(build(args.output).resolve())


if __name__ == "__main__":
    main()
