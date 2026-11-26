"""Fit on the audited ASVspoof pilot and test on the separate In-the-Wild pilot."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256 as digest
from pathlib import Path

import numpy as np

from asvspoof_baseline import BOOTSTRAPS, SEED, save_json, table
from media_check import model_at, score_decoded
from bodh import audio, baseline
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv

LAGS = (1, 4)  # The existing 10 ms hop gives 10 ms and 40 ms differences.


def temporal_features(clip: audio.Audio) -> np.ndarray:
    """Measure median absolute band changes within consecutive active frames."""
    peak = np.max(np.abs(clip.mono))
    normalized = replace(clip, mono=clip.mono * (0.5 / max(peak, 1e-12)))
    _, log_mel = audio.features(normalized)
    centered = log_mel - log_mel.mean(axis=1, keepdims=True)
    active = audio.active_frames(audio.frame_waveform(normalized.mono))
    features = []
    for lag in LAGS:
        if len(active) <= lag:
            features.append(np.zeros(audio.MEL_BANDS))
            continue
        contiguous = np.lib.stride_tricks.sliding_window_view(active, lag + 1).all(axis=1)
        change = np.abs(centered[lag:] - centered[:-lag])[contiguous]
        features.append(np.median(change, axis=0) if len(change) else np.zeros(audio.MEL_BANDS))
    return np.concatenate(features)


def train(root: Path, audit: Path, run: Path) -> None:
    audit_path = audit / "audit.json"
    provenance = json.loads(audit_path.read_text())
    if not provenance["strict_split_overlap_passed"]:
        raise ValueError("Training requires a successful ASVspoof split audit")
    splits = {}
    for split in ("train", "dev"):
        manifest = audit / f"{split}_strict.csv"
        if sha256(manifest) != provenance["strict_splits"][split]["manifest_sha256"]:
            raise ValueError(f"The {split} manifest differs from the audit")
        assets = read_assets(manifest)
        inventory = {row["asset_id"]: row for row in table(audit / f"{split}_integrity.csv")}

        def extract(asset):
            path = root / asset.path
            recorded = inventory[asset.asset_id]
            if recorded["condition"] != asset.condition or sha256(path) != recorded["sha256"]:
                raise ValueError(f"Training source differs from the audit: {asset.asset_id}")
            return temporal_features(audio.load_audio(path))

        with ThreadPoolExecutor(max_workers=4) as workers:
            vectors = np.array(list(workers.map(extract, assets)))
        splits[split] = np.array([int(asset.condition == "spoof") for asset in assets]), vectors
        print(f"Prepared {len(assets)} ASVspoof {split} clips", flush=True)

    y_train, x_train = splits["train"]
    y_dev, x_dev = splits["dev"]
    model = baseline.fit(x_train, y_train)
    dev_score = baseline.scores(model, x_dev)
    threshold = baseline.threshold_on_dev(y_dev, dev_score)
    run.mkdir(parents=True)
    save_json(run / "model.json", {
        **model, "audit_sha256": sha256(audit_path), "frame_lags": list(LAGS),
        "train_clips": len(y_train), "dev_clips": len(y_dev), "l2_penalty": baseline.PENALTY,
        "decision_threshold": threshold, "threshold_rule": "minimum development balanced error",
        "development": report_metrics(y_dev, dev_score, threshold),
    })


def report_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    report = baseline.metrics(y, score, threshold)
    tp, fp, fn = (report["n_spoof"] - report["false_negatives"],
                  report["false_positives"], report["false_negatives"])
    return {**report, "threshold": threshold, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / report["n_spoof"], "f1": 2 * tp / (2 * tp + fp + fn)}


def paired_auc_interval(y: np.ndarray, speakers: np.ndarray, old: np.ndarray, new: np.ndarray) -> list:
    groups = [np.flatnonzero(speakers == speaker) for speaker in np.unique(speakers)]
    rng = np.random.default_rng(SEED)
    delta = []
    for _ in range(BOOTSTRAPS):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(set(y[indices])) < 2:
            continue
        areas = []
        for score in (old, new):
            fpr, tpr, _ = baseline.operating_curve(y[indices], score[indices])
            areas.append(np.trapz(tpr, fpr))
        delta.append(areas[1] - areas[0])
    return np.quantile(delta, [0.025, 0.975]).tolist()


def evaluate(root: Path, audit: Path, baseline_path: Path, run: Path) -> None:
    if (run / "external.json").exists() or (run / "external_scores.csv").exists():
        raise ValueError("External results already exist for this run")
    model_path = run / "model.json"
    model = json.loads(model_path.read_text())
    if model["audit_sha256"] != sha256(audit / "audit.json") or model["frame_lags"] != list(LAGS):
        raise ValueError("The evaluation inputs differ from the frozen training setup")
    frozen = model_at(baseline_path)
    manifest = root / "manifest.csv"
    assets = read_assets(manifest)
    if (any(a.dataset != "in_the_wild" or a.split != "external" or not a.speaker_id for a in assets)
            or {a.condition for a in assets} != {"bonafide", "spoof"}):
        raise ValueError("Expected the labeled In-the-Wild pilot with speaker IDs")
    fingerprints = {(row["sha256"], row["pcm_16k_mono_sha256"])
                    for split in ("train", "dev") for row in table(audit / f"{split}_integrity.csv")}
    old_raw, old_pcm = map(set, zip(*fingerprints))

    def extract(asset):
        path = root / asset.path
        clip = audio.load_audio(path)
        raw = sha256(path)
        pcm = digest(clip.mono.astype("<f4").tobytes()).hexdigest()
        if raw in old_raw or pcm in old_pcm:
            raise ValueError(f"External clip matches audited training or development audio: {asset.asset_id}")
        return raw, pcm, score_decoded(clip, frozen)["spoof_logit"], temporal_features(clip)

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(extract, assets))
    raw, pcm, cues, features = zip(*rows)
    if len(set(raw)) != len(assets) or len(set(pcm)) != len(assets):
        raise ValueError("External sample contains duplicate recordings")
    y = np.array([int(asset.condition == "spoof") for asset in assets])
    old = np.array(cues)
    new = baseline.scores(model, np.array(features))
    speakers = np.array([asset.speaker_id for asset in assets])
    report = {
        "model_sha256": sha256(model_path), "baseline_sha256": sha256(baseline_path),
        "external_manifest_sha256": sha256(manifest), "speakers": len(set(speakers)),
        "cue_baseline": report_metrics(y, old, frozen["decision_threshold"]),
        "temporal": report_metrics(y, new, model["decision_threshold"]),
        "temporal_minus_baseline_auc_95_ci": paired_auc_interval(y, speakers, old, new),
    }
    write_csv(run / "external_scores.csv", ("asset_id", "label", "speaker", "cue_logit", "temporal_logit"), (
        {"asset_id": asset.asset_id, "label": asset.condition, "speaker": asset.speaker_id,
         "cue_logit": cue, "temporal_logit": temporal}
        for asset, cue, temporal in zip(assets, old, new)
    ))
    save_json(run / "external.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("train", "evaluate"):
        command = phases.add_parser(phase)
        for argument in ("root", "audit", "run"):
            command.add_argument(argument, type=Path)
        if phase == "evaluate":
            command.add_argument("baseline", type=Path)
    args = parser.parse_args()
    if args.phase == "train":
        train(args.root, args.audit, args.run)
    else:
        evaluate(args.root, args.audit, args.baseline, args.run)
