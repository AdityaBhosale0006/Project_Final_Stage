#!/usr/bin/env bash
# setup.sh -- Environment setup for BOM-PaddleOCR
#
# Installs everything needed to run the pipeline with the default
# PaddleOCR engine, plus the optional Tesseract fallback engine.
# Verified command-by-command against a real run (GitHub Codespaces,
# Ubuntu 24, Python 3.12).
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -euo pipefail

echo "==> [1/4] System libraries PaddleOCR's opencv-contrib dependency needs"
# paddleocr pulls in opencv-contrib-python (not the headless build) via
# paddlex[ocr-core]. That build links against system OpenGL/GLib, which
# isn't present by default in most containers/Codespaces and fails with:
#   ImportError: libGL.so.1: cannot open shared object file
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0

echo "==> [2/4] Tesseract (optional fallback OCR engine, --engine=tesseract)"
# Not required for the default PaddleOCR engine -- skip this if you only
# ever plan to use --engine=paddle.
sudo apt-get install -y tesseract-ocr

echo "==> [3/4] Python dependencies"
pip install -r requirements.txt --break-system-packages

echo "==> [4/4] Sanity check"
python3 -c "import cv2, numpy, openpyxl, pytesseract; print('core deps OK')"
python3 -c "import paddleocr; print('paddleocr OK', paddleocr.__version__)"

cat <<'EOM'

Setup complete.

Run the pipeline:
  python3 main.py <input_image_path> [output_dir]                 # PaddleOCR (default)
  python3 main.py <input_image_path> [output_dir] --engine=tesseract

Notes:
  - First PaddleOCR run downloads model weights (~150MB) from
    huggingface.co / Baidu BOS / modelscope.cn -- needs outbound network
    access to at least one of those the first time; cached locally after
    that in ~/.paddlex/official_models/.
  - If your network blocks huggingface.co specifically, redirect the
    model source:
      PADDLE_PDX_MODEL_SOURCE=bos python3 main.py <image> [output_dir]
EOM
