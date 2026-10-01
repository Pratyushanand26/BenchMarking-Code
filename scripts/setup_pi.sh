#!/usr/bin/env bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS / Debian / Ubuntu).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -m)" != "aarch64" ]; then
  echo "[WARN] arch is $(uname -m), not aarch64. MediaPipe/PyTorch wheels need a 64-bit OS on the Pi."
fi

PYV=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYV" -lt 9 ] || [ "$PYV" -gt 12 ]; then
  echo "[ERROR] python3 is 3.$PYV. mediapipe 0.10.14 needs Python 3.9-3.12."
  echo "        Install 3.11 with:  curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.11 .venv"
  echo "        then:               source .venv/bin/activate && uv pip install -r requirements.txt"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 tmux

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python - <<'PY'
import mediapipe as mp, torch, cv2, numpy
assert hasattr(mp, "solutions"), "mediapipe has no mp.solutions - wrong version installed"
print("OK  mediapipe", mp.__version__, "| torch", torch.__version__,
      "| opencv", cv2.__version__, "| numpy", numpy.__version__)
PY
echo
echo "Setup done. Next: copy weights/tsstg-model.pth and videos/*.mp4, then run scripts/run_benchmark.sh"
