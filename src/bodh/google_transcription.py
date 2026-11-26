"""Optional synchronous transcription through Google Cloud Speech-to-Text v1."""

import base64
import json
import subprocess
from contextlib import closing
from http.client import HTTPSConnection

import numpy as np

from bodh.audio import Audio, SAMPLE_RATE


def recognize(clip: Audio, project: str, language: str) -> list[dict]:
    try:
        token = subprocess.run(
            ["gcloud", "auth", "application-default", "print-access-token"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Install the Google Cloud CLI and run gcloud auth application-default login") from error
    if not token:
        raise ValueError("Google Cloud CLI returned no application-default access token")

    pcm = np.rint(np.clip(clip.mono, -1, 1) * 32767).astype("<i2").tobytes()
    request = json.dumps({
        "config": {"encoding": "LINEAR16", "sampleRateHertz": SAMPLE_RATE,
                   "audioChannelCount": 1, "languageCode": language, "model": "default",
                   "enableAutomaticPunctuation": True, "enableWordTimeOffsets": True},
        "audio": {"content": base64.b64encode(pcm).decode("ascii")},
    })
    with closing(HTTPSConnection("speech.googleapis.com", timeout=45)) as connection:
        connection.request("POST", "/v1/speech:recognize", body=request, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/json",
            "x-goog-user-project": project,
        })
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError(f"Google transcription failed (HTTP {response.status}); check project setup, "
                             "billing, API permissions, and quota")
        result = json.load(response)
    segments = []
    for row in result.get("results", []):
        alternatives = row.get("alternatives", [])
        if alternatives and alternatives[0].get("transcript", "").strip():
            best = alternatives[0]
            words = best.get("words", [])
            segments.append({
                "text": best["transcript"].strip(),
                "start": words[0].get("startTime", "0s") if words else None,
                "end": words[-1].get("endTime") if words else None,
            })
    return segments
