"""Evaluate frozen audio embeddings on observed ratings with crossed holdouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bodh import transfer
from bodh.baseline import MEL
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv
from bodh.ssl import EncoderConfig

FOLDS = 5
RIDGE = 1.0
ENDPOINTS = ("realness", "perceived_trustworthiness")


def verify_speakers(assets: list, audit: Path) -> None:
    exposed = set()
    for split in ("train", "dev"):
        membership, inventory = transfer.audited_assets(audit, split)
        exposed.update(speaker for asset in membership
                       for speaker in inventory[asset.asset_id]["vctk_speakers"].split("|"))
    if any(not asset.speaker_id or asset.speaker_id in exposed for asset in assets):
        raise ValueError("Lavan sensitivity set overlaps with mapped ASVspoof train/dev speakers")


def representations(root: Path, assets: list, checkpoint: dict) -> dict:
    source_hashes = {row["path"]: row["sha256"]
                     for row in json.loads((root / "source_files.json").read_text())}
    summaries, clips = zip(*(transfer.spectrogram(root / asset.path, source_hashes[asset.path])
                             for asset in assets))
    features = {
        "average_only": np.empty((len(assets), 0)),
        "provenance_only": np.array([[float(asset.condition == "voice_clone")] for asset in assets]),
        "log_mel": np.array([[summary[key] for key in MEL] for summary in summaries]),
    }
    config = EncoderConfig(**checkpoint["architecture"])
    clips = transfer.normalized(clips, checkpoint)
    for mode in ("random", "pretrained"):
        seeds = transfer.SEEDS if mode == "random" else transfer.SEEDS[:1]
        matrices = []
        for seed in seeds:
            torch.manual_seed(seed)
            encoder = transfer.make_encoder(config, checkpoint, mode == "pretrained").eval()
            with torch.no_grad():
                matrices.append(torch.cat([
                    transfer.pool(encoder, patches, valid, owners, len(take))
                    for take, patches, valid, owners in transfer.batches(clips, config, list(range(len(assets))))
                ]).numpy().astype(np.float64))
        features[mode] = matrices
    return features


def rating_matrices(assets: list, path: Path) -> tuple[dict, list[str]]:
    by_id = {(asset.dataset, asset.asset_id): i for i, asset in enumerate(assets)}
    recorded = [row for row in transfer.rows(path) if (row["dataset"], row["asset_id"]) in by_id
                and row["endpoint"] in ENDPOINTS and row["study_arm"] == "experiment_1b"]
    listeners = sorted({row["participant_id"] for row in recorded})
    by_listener = {key: i for i, key in enumerate(listeners)}
    targets = {endpoint: np.full((len(assets), len(listeners)), np.nan) for endpoint in ENDPOINTS}
    for row in recorded:
        values = targets[row["endpoint"]]
        key = (by_id[row["dataset"], row["asset_id"]], by_listener[row["participant_id"]])
        rating = float(row["value"])
        if np.isfinite(values[key]) or not 0 <= rating <= 100:
            raise ValueError("Invalid or repeated held-out rating")
        values[key] = rating
    return targets, listeners


def ridge_predict(design: np.ndarray, y: np.ndarray, fit: np.ndarray, test: np.ndarray) -> np.ndarray:
    average = y.mean(axis=0)
    if not design.shape[1]:
        return np.broadcast_to(average, (len(test), len(average)))
    fit_x = design[fit]
    mean = fit_x.mean(axis=0)
    scale = np.maximum(fit_x.std(axis=0), 1e-6) * np.sqrt(design.shape[1])
    z = (fit_x - mean) / scale
    weights = z.T @ np.linalg.solve(z @ z.T + RIDGE * np.eye(len(fit)), y - average)
    return np.clip(average + (design[test] - mean) / scale @ weights, 0, 100)


def cross_validate(assets: list, listeners: list, targets: dict, features: dict) -> list[dict]:
    rng = np.random.default_rng(2026)
    speakers = np.array([asset.speaker_id for asset in assets])
    speaker_folds = np.array_split(rng.permutation(np.unique(speakers)), FOLDS)
    listener_folds = np.array_split(rng.permutation(len(listeners)), FOLDS)
    data = []
    for speaker_fold, held_out_speakers in enumerate(speaker_folds, 1):
        held_out = np.isin(speakers, held_out_speakers)
        fit, test = np.flatnonzero(~held_out), np.flatnonzero(held_out)
        for listener_fold, held_out_listeners in enumerate(listener_folds, 1):
            known = np.setdiff1d(np.arange(len(listeners)), held_out_listeners)
            y_fit = np.column_stack([np.nanmean(targets[key][np.ix_(fit, known)], axis=1)
                                     for key in ENDPOINTS])
            y_test = np.column_stack([np.nanmean(targets[key][np.ix_(test, held_out_listeners)], axis=1)
                                      for key in ENDPOINTS])
            for name, representation in features.items():
                matrices = representation if isinstance(representation, list) else [representation]
                predictions = np.mean([ridge_predict(m, y_fit, fit, test) for m in matrices], axis=0)
                for i, asset_index in enumerate(test):
                    for j, endpoint in enumerate(ENDPOINTS):
                        data.append({
                            "asset_id": assets[asset_index].asset_id, "speaker_fold": speaker_fold,
                            "listener_fold": listener_fold, "model": name, "endpoint": endpoint,
                            "observed_mean": float(y_test[i, j]), "predicted_mean": float(predictions[i, j]),
                        })
    return data


def run(root: Path, audit: Path, pretraining: Path, out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    checkpoint = transfer.load_pretraining(pretraining, audit)
    manifest = audit / "lavan_speaker_sensitivity.csv"
    trials = root / "experiment1b" / "trials.csv"
    assets = read_assets(manifest)
    verify_speakers(assets, audit)
    targets, listeners = rating_matrices(assets, trials)
    features = representations(root, assets, checkpoint)
    data = cross_validate(assets, listeners, targets, features)
    report = {
        "pretraining_sha256": sha256(pretraining), "lavan_manifest_sha256": sha256(manifest),
        "trials_sha256": sha256(trials), "speakers": len({asset.speaker_id for asset in assets}),
        "stimuli": len(assets), "listeners": len(listeners), "speaker_and_listener_folds": FOLDS,
        "prediction_unit": "Mean rating for one held-out stimulus from one held-out listener fold",
        "ridge_penalty": RIDGE, "normalizer": "ASVspoof pretraining-fit clips only for both encoders",
        "absolute_error": {},
    }
    for endpoint in ENDPOINTS:
        report["absolute_error"][endpoint] = {}
        for name in features:
            subset = [abs(r["observed_mean"] - r["predicted_mean"]) for r in data
                      if r["endpoint"] == endpoint and r["model"] == name]
            report["absolute_error"][endpoint][name] = {"n_held_out_means": len(subset),
                                                       "mae": float(np.mean(subset))}
    out.mkdir(parents=True)
    write_csv(out / "predictions.csv", tuple(data[0]), data)
    transfer.write_json(out / "results.json", report)
    print(f"Scored observed ratings for {len(assets)} clips across {FOLDS ** 2} crossed folds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("pretraining", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    transfer.configure_cpu()
    run(args.root, args.audit, args.pretraining, args.out)
