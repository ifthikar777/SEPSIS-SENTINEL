"""
Download the offline speech-recognition model used by the voice commands.

The browser's built-in SpeechRecognition is NOT on-device — Chrome streams audio to
Google's servers, so it fails with a "network" error on an offline machine. SepsisGuard
therefore runs recognition locally with Vosk instead: this script fetches the model
once, and from then on voice works with no internet at all.

    python scripts/fetch_voice_model.py

Downloads ~39 MB to models/vosk-model-small-en-us-0.15/ (git-ignored). Re-running is a
no-op if the model is already present; pass --force to re-download.
"""

import io
import os
import shutil
import sys
import urllib.request
import zipfile

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)


def already_present():
    # Vosk models always carry an "am" acoustic-model directory; use it to tell a
    # complete extraction from a half-finished one.
    return os.path.isdir(os.path.join(MODEL_DIR, "am"))


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading {MODEL_NAME} (~39 MB) ...")

    buf = io.BytesIO()
    with urllib.request.urlopen(MODEL_URL, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        while True:
            chunk = resp.read(262144)
            if not chunk:
                break
            buf.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                bar = "#" * (pct * 30 // 100)
                sys.stdout.write(f"\r  [{bar:<30}] {pct:3d}%  {read/1e6:5.1f} MB")
                sys.stdout.flush()
    print()

    print("Extracting ...")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(MODELS_DIR)

    if not already_present():
        raise RuntimeError("Extraction finished but the model directory looks incomplete.")


def main():
    force = "--force" in sys.argv

    if already_present() and not force:
        print(f"Model already present: {MODEL_DIR}")
        print("Voice commands will run fully offline. Use --force to re-download.")
        return 0

    if force and os.path.isdir(MODEL_DIR):
        shutil.rmtree(MODEL_DIR, ignore_errors=True)

    try:
        download()
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(f"You can fetch it manually from {MODEL_URL}")
        print(f"and extract it so that {MODEL_DIR} exists.")
        return 1

    print(f"Done. Model installed at {MODEL_DIR}")
    print("Voice commands will now run fully offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
