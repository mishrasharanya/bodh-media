"""Decode a bounded local clip and extract deterministic audio representations."""

import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
MAX_SECONDS = 30
FFT_SIZE = 512
WINDOW = 400
HOP = 160
MEL_BANDS = 40
MEL_MIN_HZ = 20
POWER_FLOOR = 1e-12
LOW_ENERGY_DBFS = -50


@dataclass(frozen=True)
class Audio:
    mono: np.ndarray
    source_rate: int
    channels: int
    codec: str
    duration: float
    source_peak: float
    source_near_full_scale_fraction: float


def run_media(command: list[str]) -> bytes:
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
    except FileNotFoundError as error:
        raise ValueError("FFmpeg and FFprobe must be available on PATH") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise ValueError(message or f"{command[0]} failed") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"{command[0]} timed out") from error


def load_audio(path: Path) -> Audio:
    if path.suffix.lower() not in {".wav", ".flac", ".mp3"} or not path.is_file():
        raise ValueError("Expected a local WAV, FLAC, or MP3 file")
    path = path.resolve()
    probe = json.loads(run_media([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name", "-of", "json", str(path)
    ]))
    if not probe.get("streams"):
        raise ValueError("No audio stream found")
    stream = probe["streams"][0]
    rate = int(stream.get("sample_rate", 0))
    channels = stream.get("channels", 0)
    if not (1 <= channels <= 8 and 1 <= rate <= 192_000):
        raise ValueError("Unsupported channel count or sample rate")
    decoded = run_media([
        "ffmpeg", "-v", "error", "-xerror", "-nostdin", "-threads", "1",
        "-protocol_whitelist", "file", "-i", str(path), "-map", "0:a:0", "-t",
        str(MAX_SECONDS + 1), "-f", "f32le", "pipe:1"
    ])
    samples = np.frombuffer(decoded, dtype="<f4").reshape(-1, channels)
    if not len(samples) or len(samples) > MAX_SECONDS * rate or not np.isfinite(samples).all():
        raise ValueError(f"Expected finite, nonempty audio up to {MAX_SECONDS} seconds")
    magnitude = np.abs(samples)
    mono = samples.mean(axis=1, dtype=np.float64)
    if rate != SAMPLE_RATE:
        factor = math.gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, rate // factor)
    return Audio(
        mono, rate, channels, stream["codec_name"], len(samples) / rate,
        float(magnitude.max()), float(np.mean(magnitude >= 0.999))
    )


def mel_filters(frequencies: np.ndarray) -> np.ndarray:
    """Return unnormalized triangular filters on the HTK mel scale."""
    mel = 2595 * np.log10(1 + np.array([MEL_MIN_HZ, SAMPLE_RATE / 2]) / 700)
    edges = 700 * (10 ** (np.linspace(*mel, MEL_BANDS + 2) / 2595) - 1)
    rising = (frequencies - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - frequencies) / np.diff(edges)[1:, None]
    return np.maximum(0, np.minimum(rising, falling))


def frame_waveform(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) < WINDOW:
        waveform = np.pad(waveform, (0, WINDOW - len(waveform)))
    return np.lib.stride_tricks.sliding_window_view(waveform, WINDOW)[::HOP]


def active_frames(frames: np.ndarray) -> np.ndarray:
    return np.mean(frames ** 2, axis=1, dtype=np.float64) >= 10 ** (LOW_ENERGY_DBFS / 10)


def trim_quiet_edges(audio: Audio) -> Audio:
    """Keep the waveform between the first and last active analysis windows."""
    indices = np.flatnonzero(active_frames(frame_waveform(audio.mono)))
    if not len(indices):
        return audio
    start, stop = indices[0] * HOP, indices[-1] * HOP + WINDOW
    waveform = audio.mono[start:stop]
    return replace(audio, mono=waveform, duration=len(waveform) / SAMPLE_RATE)


def features(audio: Audio) -> tuple[dict, np.ndarray]:
    """Return clip summaries and frame-by-band log-mel power."""
    frames = frame_waveform(audio.mono)
    window = np.hanning(WINDOW)
    power = np.abs(np.fft.rfft(frames * window, n=FFT_SIZE)) ** 2 / window.sum() ** 2
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    log_mel = 10 * np.log10(np.maximum(power @ mel_filters(frequencies).T, POWER_FLOOR))
    active = active_frames(frames)
    active_power = power[active]
    centroid = None
    if np.any(active):
        totals = active_power.sum(axis=1)
        centroid = float(np.median((active_power @ frequencies) / totals))
    summary = {
        "duration_s": audio.duration,
        "source_sample_rate_hz": audio.source_rate,
        "source_channels": audio.channels,
        "source_codec": audio.codec,
        "source_peak": audio.source_peak,
        "source_near_full_scale_fraction": audio.source_near_full_scale_fraction,
        "mono_rms_dbfs": float(10 * np.log10(max(np.mean(audio.mono ** 2), POWER_FLOOR))),
        "low_energy_frame_fraction": float(np.mean(~active)),
        "active_spectral_centroid_median_hz": centroid,
    }
    summary.update({f"log_mel_mean_{i:02}": float(value)
                    for i, value in enumerate(log_mel.mean(axis=0))})
    return summary, log_mel
