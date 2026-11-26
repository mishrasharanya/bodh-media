"""Diagnose cue drift and test a gain/codec robustness candidate on ASVspoof development."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from bodh import audio, baseline, transfer
from bodh.eda import sha256
from bodh.manifest import read_assets


def mp3_variant(clip: audio.Audio, bitrate: str, gain: float) -> audio.Audio:
    prefix = ["ffmpeg", "-v", "error", "-nostdin"]
    pcm = (clip.mono * gain).astype("<f4").tobytes()
    compressed = subprocess.run(
        [*prefix, "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
         "-c:a", "libmp3lame", "-b:a", bitrate, "-write_xing", "0", "-f", "mp3", "pipe:1"],
        input=pcm, capture_output=True, timeout=30, check=True,
    ).stdout
    decoded = subprocess.run(
        [*prefix, "-f", "mp3", "-i", "pipe:0", "-f", "f32le", "pipe:1"],
        input=compressed, capture_output=True, timeout=30, check=True,
    ).stdout
    samples = np.frombuffer(decoded, dtype="<f4").astype(np.float64)
    return replace(clip, mono=samples, duration=len(samples) / audio.SAMPLE_RATE)


def vectors(clip: audio.Audio) -> tuple[list[float], np.ndarray]:
    trimmed = audio.trim_quiet_edges(clip)
    summary, _ = audio.features(trimmed)
    peak = np.max(np.abs(clip.mono))
    relative = replace(clip, mono=clip.mono / max(peak, 1e-12) * 0.5)
    relative = audio.trim_quiet_edges(relative)
    _, log_mel = audio.features(relative)
    # Center each frame so a uniform level shift does not directly enter the features.
    shape = log_mel - log_mel.mean(axis=1, keepdims=True)
    active = audio.active_frames(audio.frame_waveform(relative.mono))
    active_shape = shape[active] if active.any() else shape
    return [summary[key] for key in baseline.CUES], np.median(active_shape, axis=0)


def features(root: Path, assets: list, hashes: dict, stress: tuple | None) -> tuple:
    def one(asset):
        path = root / asset.path
        if sha256(path) != hashes.get(asset.asset_id):
            raise ValueError(f"Source differs from audited audio: {asset.asset_id}")
        clip = audio.load_audio(path)
        clean = vectors(clip)
        return clean, vectors(mp3_variant(clip, *stress)) if stress else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, assets))
    clean = tuple(np.array(column) for column in zip(*(row[0] for row in rows)))
    changed = tuple(np.array(column) for column in zip(*(row[1] for row in rows))) if stress else None
    return np.array([int(asset.condition == "spoof") for asset in assets]), clean, changed


def cue_summary(y: np.ndarray, x: np.ndarray) -> dict:
    return {name: {
        "bonafide_median": float(np.median(x[y == 0, i])),
        "spoof_median": float(np.median(x[y == 1, i])),
        "spoof_minus_bonafide": float(np.median(x[y == 1, i]) - np.median(x[y == 0, i])),
    } for i, name in enumerate(baseline.CUES)}


def run(out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    audit = Path("runs/asvspoof-preflight")
    root = Path("data/asvspoof2019-pilot")
    data = {}
    for split, stress in (("train", ("64k", 10 ** (-6 / 20))), ("dev", ("48k", 10 ** (-12 / 20)))):
        assets, inventory = transfer.audited_assets(audit, split)
        data[split] = features(root, assets, {key: r["sha256"] for key, r in inventory.items()}, stress)
        print(f"Prepared {len(assets)} {split} clips and their codec variants", flush=True)
    train_y, train, augmented = data["train"]
    dev_y, clean, stressed = data["dev"]
    cue_model = json.loads(Path("runs/asvspoof-baseline/model.json").read_text())["policies"]["trimmed"]["cues"]
    candidate = baseline.fit(np.concatenate([train[1], augmented[1]]), np.tile(train_y, 2))
    candidate["decision_threshold"] = baseline.threshold_on_dev(dev_y, baseline.scores(candidate, clean[1]))
    metrics = {}
    for name, model, col in (("frozen_cues", cue_model, 0), ("spectral_shape_augmented", candidate, 1)):
        metrics[name] = {
            condition: baseline.metrics(dev_y, baseline.scores(model, inputs[col]),
                                        model["decision_threshold"])
            for condition, inputs in (("clean", clean), ("mp3_with_12db_attenuation", stressed))
        }
    external_root = Path("data/in-the-wild-pilot")
    external_y, external, _ = features(
        external_root, read_assets(external_root / "manifest.csv"),
        {row["asset_id"]: row["sha256"] for row in transfer.rows(Path("runs/in-the-wild-pilot/scores.csv"))},
        None,
    )
    out.mkdir(parents=True)
    report = {
        "original_cue_diagnostics": {"asvspoof_dev": cue_summary(dev_y, clean[0]),
                                     "previously_scored_in_the_wild": cue_summary(external_y, external[0])},
        "development_metrics": metrics,
        "candidate_trained_on": "Audited ASVspoof train audio, clean and 64 kbps MP3 at −6 dB",
        "candidate_selection": "Threshold selected on clean development clips",
        "external_usage": "Cue diagnosis only; candidate never scored on these clips",
    }
    transfer.write_json(out / "report.json", report)
    transfer.write_json(out / "candidate.json", candidate)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
