"""Evaluate frozen audio detectors under controlled additive pink noise."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np

from asvspoof_baseline import save_json, table
from cross_corpus_audio import LAGS, paired_auc_interval, report_metrics, temporal_features
from media_check import model_at, score_decoded
from bodh import audio, baseline
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv

SEED = 2026
SNR_LEVELS = (20, 10)


def pink_noise_at_unit_gain(waveform: np.ndarray, asset_id: str) -> np.ndarray:
    """Return pink noise with the source's mean power over clean active frames."""
    frames = audio.frame_waveform(waveform)
    active = audio.active_frames(frames)
    if not active.any():
        raise ValueError(f"No active frames for the SNR reference: {asset_id}")
    key = digest(f"{SEED}:{asset_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
    spectrum = np.fft.rfft(rng.standard_normal(len(waveform)))
    spectrum[0] = 0
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    noise = np.fft.irfft(spectrum, n=len(waveform))
    signal_power = np.mean(frames[active] ** 2)
    noise_power = np.mean(audio.frame_waveform(noise)[active] ** 2)
    if noise_power == 0:
        raise ValueError(f"Cannot calibrate the noise for this waveform: {asset_id}")
    return noise * np.sqrt(signal_power / noise_power)


def run(root: Path, reference: Path, baseline_path: Path, original: Path, out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    model_path = reference / "model.json"
    manifest = root / "manifest.csv"
    combined_report = json.loads((reference / "external.json").read_text())
    fingerprints = {"model_sha256": sha256(model_path), "baseline_sha256": sha256(baseline_path),
                    "external_manifest_sha256": sha256(manifest)}
    if any(combined_report[key] != value for key, value in fingerprints.items()):
        raise ValueError("Model or test manifest differs from the frozen cross-corpus evaluation")
    temporal = json.loads(model_path.read_text())
    if temporal["frame_lags"] != list(LAGS):
        raise ValueError("Unexpected temporal model feature layout")
    cue = model_at(baseline_path)
    assets = read_assets(manifest)
    original_report = json.loads((original / "report.json").read_text())
    if (original_report["manifest_sha256"] != fingerprints["external_manifest_sha256"]
            or original_report["model_sha256"] != fingerprints["baseline_sha256"]):
        raise ValueError("Baseline inventory belongs to a different model or test sample")
    source_rows = {row["asset_id"]: row for row in table(original / "scores.csv")}
    clean_rows = {row["asset_id"]: row for row in table(reference / "external_scores.csv")}

    def score(asset):
        path = root / asset.path
        if sha256(path) != source_rows[asset.asset_id]["sha256"]:
            raise ValueError(f"Recording differs from the original external test: {asset.asset_id}")
        clip = audio.load_audio(path)
        noise = pink_noise_at_unit_gain(clip.mono, asset.asset_id)
        variants = [clip, *(replace(clip, mono=clip.mono + noise * 10 ** (-snr / 20))
                            for snr in SNR_LEVELS)]
        features = np.array([temporal_features(variant) for variant in variants])
        return ([score_decoded(variant, cue)["spoof_logit"] for variant in variants],
                baseline.scores(temporal, features))

    with ThreadPoolExecutor(max_workers=4) as workers:
        cue_scores, temporal_scores = map(np.array, zip(*workers.map(score, assets)))

    labels = np.array([int(asset.condition == "spoof") for asset in assets])
    speakers = np.array([asset.speaker_id for asset in assets])
    results, rows = {}, []
    for name, matrix, model, column in (("cue_baseline", cue_scores, cue, "cue_logit"),
                                        ("temporal", temporal_scores, temporal, "temporal_logit")):
        expected = [float(clean_rows[asset.asset_id][column]) for asset in assets]
        if not np.allclose(matrix[:, 0], expected, atol=1e-10, rtol=0):
            raise ValueError(f"Clean scores differ from frozen evaluation: {name}")
        threshold = model["decision_threshold"]
        clean_decision = matrix[:, 0] >= threshold
        results[name] = {}
        for index, condition in enumerate(("clean", *(f"{snr}_db" for snr in SNR_LEVELS))):
            score_values = matrix[:, index]
            decisions = score_values >= threshold
            flipped = decisions != clean_decision
            corrected = flipped & (decisions == labels)
            metrics = report_metrics(labels, score_values, threshold)
            results[name][condition] = {**metrics, "decision_flips": int(flipped.sum()),
                                        "errors_introduced": int((flipped & ~corrected).sum()),
                                        "errors_corrected": int(corrected.sum())}
            if index:
                results[name][condition]["auc_change_from_clean_95_ci"] = paired_auc_interval(
                    labels, speakers, matrix[:, 0], score_values)
            rows.extend({"asset_id": asset.asset_id, "label": asset.condition, "speaker": asset.speaker_id,
                         "model": name, "condition": condition, "spoof_logit": float(logit),
                         "predicted_spoof": int(decision), "flipped_from_clean": int(flip)}
                        for asset, logit, decision, flip in zip(assets, score_values, decisions, flipped))

    out.mkdir(parents=True)
    report = {**fingerprints, "mix": "stationary pink noise, power matched on original active frames",
              "seed": SEED, "snr_db": list(SNR_LEVELS), "models": results}
    save_json(out / "report.json", report)
    write_csv(out / "scores.csv", tuple(rows[0]), rows)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("in_the_wild", "cross_corpus_run", "baseline_model", "baseline_run", "out"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    run(args.in_the_wild, args.cross_corpus_run, args.baseline_model, args.baseline_run, args.out)
