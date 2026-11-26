"""Run whisper.cpp on a local 16 kHz mono clip."""

import json
import subprocess
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bodh.audio import Audio, SAMPLE_RATE


def recognize(clip: Audio, model: Path, language: str = "en") -> list[dict]:
    if not model.is_file():
        raise ValueError(f"Whisper model file not found: {model}")
    pcm = np.rint(np.clip(clip.mono, -1, 1) * 32767).astype("<i2").tobytes()
    with TemporaryDirectory(prefix="bodh-whisper-") as folder:
        stem = Path(folder) / "transcript"
        wav = stem.with_suffix(".wav")
        with wave.open(str(wav), "wb") as recording:
            recording.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            recording.writeframes(pcm)
        try:
            subprocess.run(
                ["whisper-cli", "-m", str(model.resolve()), "-f", str(wav), "-l", language,
                 "-oj", "-of", str(stem), "-np", "-ng"],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except FileNotFoundError as error:
            raise ValueError("Install whisper.cpp and make whisper-cli available on PATH") from error
        except subprocess.CalledProcessError as error:
            raise ValueError(error.stderr.strip() or "whisper-cli failed to transcribe the clip") from error
        except subprocess.TimeoutExpired as error:
            raise ValueError("whisper-cli timed out") from error
        report = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return [
        {"text": item["text"].strip(), "start": item["offsets"]["from"] / 1000,
         "end": item["offsets"]["to"] / 1000}
        for item in report["transcription"] if item["text"].strip()
    ]
