"""Compare pretrained and randomly initialized Transformers on ASVspoof."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bodh import baseline, checkpoint as checkpoint_file, transfer
from bodh.eda import sha256
from bodh.manifest import write_csv
from bodh.ssl import EncoderConfig

LEARNING_RATE = 3e-4
MODES = ("random", "pretrained")


def load_split(root: Path, audit: Path, split: str, checkpoint: dict) -> tuple:
    assets, inventory = transfer.audited_assets(audit, split)
    clips = [
        transfer.spectrogram(root / asset.path, inventory[asset.asset_id]["sha256"])[1]
        for asset in assets
    ]
    return (assets, np.array([int(asset.condition == "spoof") for asset in assets]),
            transfer.normalized(clips, checkpoint))


def score(model: transfer.Detector, clips: list, config: EncoderConfig) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([
            model(patches, valid, owners, len(take))
            for take, patches, valid, owners in transfer.batches(clips, config, list(range(len(clips))))
        ]).numpy()


def model_for(mode: str, seed: int, config: EncoderConfig, checkpoint: dict) -> transfer.Detector:
    torch.manual_seed(seed)
    encoder = transfer.make_encoder(config, checkpoint, mode == "pretrained")
    return transfer.Detector(encoder, config.width)


def fit(root: Path, audit: Path, pretraining: Path, out: Path, epochs: int) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    start = time.perf_counter()
    checkpoint = transfer.load_pretraining(pretraining, audit)
    config = EncoderConfig(**checkpoint["architecture"])
    _, y_train, train = load_split(root, audit, "train", checkpoint)
    _, y_dev, dev = load_split(root, audit, "dev", checkpoint)
    targets = torch.as_tensor(y_train, dtype=torch.float32)
    out.mkdir(parents=True)
    metadata = {
        "pretraining_sha256": sha256(pretraining), "audit_sha256": sha256(audit / "audit.json"),
        "epochs": epochs, "learning_rate": LEARNING_RATE, "seed_pairs": list(transfer.SEEDS),
        "whole_utterance": "Quiet edges trimmed; all 256-frame chunks pooled by frame count",
        "selected_epochs": "Highest development AUC with ties favoring the earlier epoch",
        "threshold": "Development threshold minimizes balanced error, with fewer false alarms in ties",
        "n_train": len(train), "n_dev": len(dev), "modes": {},
    }
    history = []
    for mode in MODES:
        dev_logits, checkpoints = [], {}
        for seed in transfer.SEEDS:
            model = model_for(mode, seed, config, checkpoint)
            generator = torch.Generator().manual_seed(seed + 20)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            best_auc = -1.0
            for epoch in range(1, epochs + 1):
                model.train()
                losses = []
                order = torch.randperm(len(train), generator=generator).tolist()
                for take, patches, valid, owners in transfer.batches(train, config, order):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(patches, valid, owners, len(take))
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[take])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item() * len(take))
                prediction = score(model, dev, config)
                fpr, tpr, _ = baseline.operating_curve(y_dev, prediction)
                auc = float(np.trapz(tpr, fpr))
                history.append(dict(mode=mode, seed=seed, epoch=epoch,
                                    train_bce=sum(losses) / len(train), development_auc=auc))
                if auc > best_auc + 1e-12:
                    best_auc, best_epoch, best_prediction = auc, epoch, prediction
                    best_state = {key: weight.detach().clone() for key, weight in model.state_dict().items()}
            name = f"{mode}-{seed}.npz"
            checkpoint_file.save(out / name, best_state)
            checkpoints[str(seed)] = {"epoch": best_epoch, "development_auc": best_auc,
                                      "file": name, "sha256": sha256(out / name)}
            dev_logits.append(best_prediction)
            print(f"{mode} seed {seed} | development AUC {best_auc:.3f} at epoch {best_epoch}", flush=True)
        ensemble = np.mean(dev_logits, axis=0)
        decision = baseline.threshold_on_dev(y_dev, ensemble)
        metadata["modes"][mode] = {"checkpoints": checkpoints, "decision_threshold": decision,
                                   "development": baseline.metrics(y_dev, ensemble, decision)}
    metadata["fit_wall_seconds"] = time.perf_counter() - start
    write_csv(out / "history.csv", tuple(history[0]), history)
    transfer.write_json(out / "frozen.json", metadata)


def error_intervals(y: np.ndarray, speakers: np.ndarray, ensembles: dict) -> dict:
    groups = [np.flatnonzero(speakers == key) for key in np.unique(speakers)]
    rng = np.random.default_rng(2026)
    estimates = []
    for _ in range(1000):
        take = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        if len(set(y[take])) == 2:
            fprs_tprs = [baseline.operating_curve(y[take], ensembles[name][take])[:2] for name in MODES]
            estimates.append([np.trapz(tpr, fpr) for fpr, tpr in fprs_tprs])
    values = np.array(estimates)
    return {
        "valid_speaker_bootstraps": len(values),
        "auc_95_ci": {name: np.quantile(values[:, index], [0.025, 0.975]).tolist()
                      for index, name in enumerate(MODES)},
        "pretrained_minus_random_auc_95_ci": np.quantile(values[:, 1] - values[:, 0], [0.025, 0.975]).tolist(),
    }


def evaluate(root: Path, audit: Path, pretraining: Path, out: Path) -> None:
    if any((out / name).exists() for name in ("results.json", "scores.csv")):
        raise ValueError("This run already has evaluation outputs")
    start = time.perf_counter()
    checkpoint = transfer.load_pretraining(pretraining, audit)
    config = EncoderConfig(**checkpoint["architecture"])
    frozen = json.loads((out / "frozen.json").read_text())
    if (frozen["audit_sha256"] != sha256(audit / "audit.json")
            or frozen["pretraining_sha256"] != sha256(pretraining)):
        raise ValueError("Frozen transfer configuration no longer matches its inputs")
    assets, labels, clips = load_split(root, audit, "eval", checkpoint)
    strata = {row["asset_id"]: row["stratum"] for row in transfer.rows(audit / "eval_strata.csv")}
    conditions = np.array([strata[asset.asset_id] for asset in assets])
    genuine = conditions == "bonafide"
    if not np.array_equal(genuine, labels == 0):
        raise ValueError("Evaluation strata disagree with frozen labels")
    partitions = {"all": np.ones(len(labels), dtype=bool),
                  "seen_algorithm": genuine | (conditions == "seen_algorithm"),
                  "unseen_algorithm": genuine | (conditions == "unseen_algorithm")}
    outcomes, ensembles, score_rows = {}, {}, []
    for mode in MODES:
        settings = frozen["modes"][mode]
        logits, seed_auc = [], {}
        for seed, record in settings["checkpoints"].items():
            model = model_for(mode, int(seed), config, checkpoint)
            path = out / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Frozen checkpoint changed: {path.name}")
            model.load_state_dict(checkpoint_file.load(path)[0])
            result = score(model, clips, config)
            fpr, tpr, _ = baseline.operating_curve(labels, result)
            seed_auc[seed] = float(np.trapz(tpr, fpr))
            logits.append(result)
        ensemble, decision = np.mean(logits, axis=0), settings["decision_threshold"]
        ensembles[mode] = ensemble
        outcomes[mode] = {"seed_auc": seed_auc, "ensemble": {
            key: baseline.metrics(labels[mask], ensemble[mask], decision)
            for key, mask in partitions.items()
        }}
        for asset, category, value in zip(assets, conditions, ensemble):
            score_rows.append({"asset_id": asset.asset_id, "stratum": category, "initialization": mode,
                               "spoof_logit": float(value), "predict_spoof": int(value >= decision)})
    results = {
        "frozen_sha256": sha256(out / "frozen.json"), "n_eval": len(assets), "modes": outcomes,
        "speaker_bootstrap": error_intervals(labels, np.array([a.speaker_id for a in assets]), ensembles),
        "evaluation_wall_seconds": time.perf_counter() - start,
    }
    write_csv(out / "scores.csv", tuple(score_rows[0]), score_rows)
    transfer.write_json(out / "results.json", results)
    print(f"Evaluated {len(assets)} held-out clips", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("pretraining", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    transfer.configure_cpu()
    if args.action == "fit":
        fit(args.root, args.audit, args.pretraining, args.out, args.epochs)
    else:
        evaluate(args.root, args.audit, args.pretraining, args.out)
