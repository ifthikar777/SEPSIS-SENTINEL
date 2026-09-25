"""
Offline speech recognition for the dashboard's voice commands.

The browser's built-in SpeechRecognition is not on-device: Chrome streams microphone
audio to Google's servers, so it returns a "network" error on a machine with no
internet, and it would send anything spoken near the microphone off the box. Neither
is acceptable for a system meant to run air-gapped.

This module runs recognition locally with Vosk instead. The browser captures audio and
streams 16 kHz mono PCM to the server over a WebSocket; nothing leaves the machine.

The model is not committed (~39 MB). Fetch it once with:

    python scripts/fetch_voice_model.py

If the model or the vosk package is missing, is_available() returns False and voice
input is disabled in the dashboard. There is deliberately no fallback to the
browser's own SpeechRecognition API: that would silently reroute microphone audio
to a cloud service, defeating the offline guarantee. Voice either runs locally or
not at all.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_DIR = os.environ.get(
    "SEPSISGUARD_VOICE_MODEL", os.path.join(BASE_DIR, "models", MODEL_NAME)
)

# Vosk expects 16 kHz mono; the browser downsamples to this before sending.
SAMPLE_RATE = 16000

_model = None
_load_error = None


def is_available():
    """True when local recognition can actually run."""
    if _load_error is not None:
        return False
    try:
        import vosk  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(os.path.join(MODEL_DIR, "am"))


def unavailable_reason():
    """Human-readable explanation for the dashboard when local ASR cannot run."""
    try:
        import vosk  # noqa: F401
    except ImportError:
        return "The 'vosk' package is not installed (pip install -r requirements.txt)."
    if not os.path.isdir(os.path.join(MODEL_DIR, "am")):
        return "Speech model not downloaded. Run: python scripts/fetch_voice_model.py"
    if _load_error:
        return f"Speech model failed to load: {_load_error}"
    return "Local speech recognition is unavailable."


def load_model():
    """Load the model once and reuse it across connections."""
    global _model, _load_error
    if _model is not None:
        return _model
    if not is_available():
        return None
    try:
        import vosk
        vosk.SetLogLevel(-1)  # keep Kaldi's own logging out of the server output
        _model = vosk.Model(MODEL_DIR)
        print(f"Offline speech model loaded: {MODEL_NAME}")
    except Exception as exc:            # noqa: BLE001 - surfaced to the client
        _load_error = str(exc)
        print(f"Could not load speech model: {exc}")
        return None
    return _model


class Transcriber:
    """
    Wraps one Vosk recogniser for a single WebSocket connection.

    feed() takes raw 16-bit PCM and returns a finalised transcript when the recogniser
    decides an utterance has ended, otherwise None.
    """

    def __init__(self):
        import vosk
        model = load_model()
        if model is None:
            raise RuntimeError(unavailable_reason())
        self._rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        self._rec.SetWords(False)

    def feed(self, pcm_bytes):
        if self._rec.AcceptWaveform(pcm_bytes):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            return text or None
        return None

    def flush(self):
        """Final transcript for whatever audio is still buffered."""
        text = json.loads(self._rec.FinalResult()).get("text", "").strip()
        return text or None
