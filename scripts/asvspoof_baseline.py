"""Fit CPU-only baselines, then evaluate a saved model on the frozen ASVspoof split."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bodh import audio, baseline
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv

POLICIES = {"whole": lambda clip: clip, "trimmed": audio.trim_quiet_edges}
BOOTSTRAPS = 1000
SEED = 2026


def table(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def split_data(root: Path, audit_root: Path, split: str) -> tuple:
    report = json.loads((audit_root / "audit.json").read_text())
    manifest = audit_root / f"{split}_strict.csv"
    if (not report["strict_split_overlap_passed"]
            or sha256(manifest) != report["strict_splits"][split]["manifest_sha256"]):
        raise ValueError(f"The {split} manifest does not match the successful overlap audit")
    assets = read_assets(manifest)
    inventory = {row["asset_id"]: row["sha256"]
                 for row in table(audit_root / f"{split}_integrity.csv")}
    features = {policy: [] for policy in POLICIES}
    for asset in assets:
        path = root / asset.path
        if sha256(path) != inventory.get(asset.asset_id):
            raise ValueError(f"Audio changed since the audit: {asset.asset_id}")
        clip = audio.load_audio(path)
        for name, prepare in POLICIES.items():
            summary, _ = audio.features(prepare(clip))
            features[name].append(summary)
    labels = np.array([int(asset.condition == "spoof") for asset in assets])
    return assets, labels, features


def matrix(rows: list[dict], columns: tuple[str, ...]) -> np.ndarray:
    return np.array([[row[key] for key in columns] for row in rows], dtype=np.float64)


def save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def train(root: Path, audit: Path, out: Path) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    _, y_train, train_features = split_data(root, audit, "train")
    _, y_dev, dev_features = split_data(root, audit, "dev")
    model = {
        "method": "zero-initialized L2-regularized logistic regression",
        "l2_penalty": baseline.PENALTY,
        "threshold_rule": "minimum development balanced error, tie toward fewer false alarms",
        "audit_sha256": sha256(audit / "audit.json"),
        "quiet_frame_dbfs": audio.LOW_ENERGY_DBFS, "policies": {},
    }
    for policy in POLICIES:
        model["policies"][policy] = {}
        for name, columns in baseline.FEATURES.items():
            fitted = baseline.fit(matrix(train_features[policy], columns), y_train)
            scored = baseline.scores(fitted, matrix(dev_features[policy], columns))
            threshold = baseline.threshold_on_dev(y_dev, scored)
            model["policies"][policy][name] = {
                **fitted, "features": columns, "decision_threshold": threshold,
                "development": baseline.metrics(y_dev, scored, threshold),
            }
    out.mkdir(parents=True)
    save_json(out / "model.json", model)
    print(f"Saved {len(POLICIES) * len(baseline.FEATURES)} frozen models and development thresholds",
          flush=True)


def bootstrap_auc(labels: np.ndarray, groups: np.ndarray, predictions: dict) -> dict:
    by_speaker = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in predictions}
    differences = []
    for _ in range(BOOTSTRAPS):
        draw = rng.integers(len(by_speaker), size=len(by_speaker))
        take = np.concatenate([by_speaker[index] for index in draw])
        if len(set(labels[take])) < 2:
            continue
        for name, score in predictions.items():
            fpr, tpr, _ = baseline.operating_curve(labels[take], score[take])
            samples[name].append(np.trapz(tpr, fpr))
        differences.append(samples["cues_plus_log_mel"][-1] - samples["cues"][-1])
    return {
        "valid_resamples": len(differences),
        "auc_95_ci": {name: np.quantile(values, [0.025, 0.975]).tolist()
                      for name, values in samples.items()},
        "delta_auc_cues_plus_log_mel_minus_cues_95_ci":
            np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def evaluate(root: Path, audit: Path, run: Path) -> None:
    if any((run / name).exists() for name in ("results.json", "evaluation_scores.csv")):
        raise ValueError("Evaluation output already exists in this run")
    model_file = run / "model.json"
    model = json.loads(model_file.read_text())
    if model["audit_sha256"] != sha256(audit / "audit.json"):
        raise ValueError("The saved model and evaluation use different audit snapshots")
    assets, labels, features = split_data(root, audit, "eval")
    groups = np.array([asset.speaker_id for asset in assets])
    strata = {row["asset_id"]: row["stratum"] for row in table(audit / "eval_strata.csv")}
    assignment = np.array([strata[asset.asset_id] for asset in assets])
    if not np.array_equal(assignment == "bonafide", labels == 0):
        raise ValueError("Evaluation strata disagree with manifest labels")
    genuine = assignment == "bonafide"
    partitions = {
        "all": np.ones(len(labels), dtype=bool),
        "seen_algorithm": genuine | (assignment == "seen_algorithm"),
        "unseen_algorithm": genuine | (assignment == "unseen_algorithm"),
    }
    results = {"model_sha256": sha256(model_file), "bootstrap_group": "official protocol speaker",
               "bootstrap_seed": SEED, "policies": {}}
    score_rows = []
    for policy, fitted_models in model["policies"].items():
        summary, predictions = {}, {}
        for name, fitted in fitted_models.items():
            logits = baseline.scores(fitted, matrix(features[policy], fitted["features"]))
            predictions[name] = logits
            threshold = fitted["decision_threshold"]
            summary[name] = {key: baseline.metrics(labels[mask], logits[mask], threshold)
                             for key, mask in partitions.items()}
            for asset, stratum, logit in zip(assets, assignment, logits):
                score_rows.append({"asset_id": asset.asset_id, "stratum": stratum,
                                   "policy": policy, "model": name, "spoof_logit": float(logit),
                                   "predicted_spoof": int(logit >= threshold)})
        results["policies"][policy] = {
            "models": summary, "speaker_bootstrap": bootstrap_auc(labels, groups, predictions),
        }
    save_json(run / "results.json", results)
    write_csv(run / "evaluation_scores.csv", tuple(score_rows[0]), score_rows)
    print(f"Evaluated {len(assets)} held-out clips using frozen weights and thresholds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "evaluate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    action = train if args.phase == "train" else evaluate
    action(args.root, args.audit, args.run)
