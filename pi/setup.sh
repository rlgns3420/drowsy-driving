#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Requires 64-bit Raspberry Pi OS (aarch64); 32-bit ARM is unsupported." >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), "Requires system Python 3.11: use Raspberry Pi OS Bookworm 64-bit"'
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-picamera2 libportaudio2 libgl1 libglib2.0-0 libxcb-xinerama0 libxkbcommon-x11-0
python3 -m venv --system-site-packages "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install --only-binary=:all: -r "$ROOT/pi/requirements.txt"
"$ROOT/.venv/bin/python" -c 'import cv2, numpy, mediapipe, picamera2; assert hasattr(mediapipe, "solutions"); print("Runtime imports OK")'
echo 'Ready. Run: bash pi/run.sh --source usb --mute'
echo 'CSI camera: bash pi/run.sh --source csi --mute'
