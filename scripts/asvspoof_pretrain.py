"""Pretrain a masked spectrogram Transformer on the audited ASVspoof training audio."""

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from bodh import audio, checkpoint
from bodh.eda import sha256
from bodh.manifest import read_assets, write_csv
from bodh.ssl import EncoderConfig, MaskedSpectrogramModel, mask_patches

SEED = 2026
THREADS = 4
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
VALIDATION_SPEAKERS = 4


def audited_training(root: Path, audit: Path) -> tuple[list[dict], dict]:
    report = json.loads((audit / "audit.json").read_text())
    manifest = audit / "train_strict.csv"
    digest = sha256(manifest)
    if (not report["strict_split_overlap_passed"]
            or digest != report["strict_splits"]["train"]["manifest_sha256"]):
        raise ValueError("Pretraining requires the verified strict training split")
    with (audit / "train_integrity.csv").open(newline="", encoding="utf-8") as file:
        integrity = {row["asset_id"]: row for row in csv.DictReader(file)}
    records = []
    original_speakers = set()
    for asset in read_assets(manifest):
        row = integrity[asset.asset_id]
        path = root / asset.path
        if asset.split != "train" or sha256(path) != row["sha256"]:
            raise ValueError(f"Training source failed its audit: {asset.asset_id}")
        mapped = set(filter(None, row["vctk_speakers"].split("|")))
        if not mapped:
            raise ValueError(f"Missing mapped speaker for {asset.asset_id}")
        if len(mapped) == 1:
            original_speakers.update(mapped)
        records.append({"asset_id": asset.asset_id, "path": path,
                        "vctk_speakers": mapped, "sha256": row["sha256"]})
    rng = np.random.default_rng(SEED)
    held_out = set(rng.choice(sorted(original_speakers), VALIDATION_SPEAKERS, replace=False).tolist())
    for row in records:
        speakers = row["vctk_speakers"]
        row["pretraining_role"] = (
            "validation" if speakers <= held_out else
            "excluded" if speakers & held_out else "fit"
        )
    return records, {"audit_sha256": sha256(audit / "audit.json"), "train_manifest_sha256": digest,
                     "internal_validation_speakers": sorted(held_out)}


def spectrograms(records: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    sets = {"fit": [], "validation": []}
    for row in records:
        role = row["pretraining_role"]
        if role in sets:
            _, log_mel = audio.features(audio.load_audio(row["path"]))
            sets[role].append(log_mel)
    fit_frames = np.concatenate(sets["fit"], axis=0)
    center = fit_frames.mean(axis=0)
    scale = np.maximum(fit_frames.std(axis=0), 1e-6)
    return ({role: [torch.tensor((clip - center) / scale, dtype=torch.float32)
                    for clip in clips] for role, clips in sets.items()}, center, scale)


def make_batch(clips: list[torch.Tensor], config: EncoderConfig,
               generator: torch.Generator, random_crop: bool) -> tuple:
    frames = config.max_patches * config.patch_frames
    values = torch.zeros(len(clips), frames, audio.MEL_BANDS)
    valid = torch.zeros(len(clips), frames, dtype=torch.bool)
    for index, clip in enumerate(clips):
        extra = max(0, len(clip) - frames)
        start = (int(torch.randint(extra + 1, (), generator=generator))
                 if random_crop and extra else extra // 2)
        crop = clip[start:start + frames]
        values[index, :len(crop)] = crop
        valid[index, :len(crop)] = True
    return (values.reshape(len(clips), config.max_patches, config.patch_frames, audio.MEL_BANDS),
            valid.reshape(len(clips), config.max_patches, config.patch_frames))


def run_epoch(model: MaskedSpectrogramModel, clips: list[torch.Tensor], config: EncoderConfig,
              generator: torch.Generator, optimizer: torch.optim.Optimizer | None = None) -> tuple:
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(clips), generator=generator).tolist() if training else list(range(len(clips)))
    error_sum, mean_baseline_sum, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for offset in range(0, len(order), BATCH_SIZE):
            batch = [clips[index] for index in order[offset:offset + BATCH_SIZE]]
            patches, valid_frames = make_batch(batch, config, generator, training)
            valid_patches = valid_frames.any(dim=-1)
            hidden = mask_patches(valid_patches, generator)
            target_positions = hidden.unsqueeze(-1) & valid_frames
            target = patches[target_positions]
            predicted = model(patches, valid_patches, hidden)[target_positions]
            loss = (predicted - target).square().mean()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            error_sum += loss.item() * target.numel()
            mean_baseline_sum += target.square().sum().item()
            count += target.numel()
    return error_sum / count, mean_baseline_sum / count


def run(root: Path, audit: Path, out: Path, epochs: int) -> None:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    start = time.perf_counter()
    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    records, provenance = audited_training(root, audit)
    sets, center, scale = spectrograms(records)
    config = EncoderConfig()
    model = MaskedSpectrogramModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    fit_generator = torch.Generator().manual_seed(SEED + 1)
    prepare_seconds = time.perf_counter() - start

    def validate() -> tuple:
        # Repeat the same masks and central crops for every checkpoint comparison.
        return run_epoch(model, sets["validation"], config, torch.Generator().manual_seed(SEED + 2))

    initial, mean_baseline = validate()
    best_loss, best_epoch = initial, 0
    out.mkdir(parents=True)
    weights_path = out / "encoder.npz"
    checkpoint_tmp = out / "encoder.tmp"

    def save_encoder() -> None:
        tensors = {
            **model.encoder.state_dict(),
            "log_mel_center": torch.tensor(center, dtype=torch.float32),
            "log_mel_scale": torch.tensor(scale, dtype=torch.float32),
        }
        checkpoint.save(checkpoint_tmp, tensors, {
            "architecture": asdict(config),
            "epoch": best_epoch, "validation_masked_mse": best_loss,
            "seed": SEED, "provenance": provenance,
        })
        checkpoint_tmp.replace(weights_path)

    save_encoder()
    history = []
    for epoch in range(1, epochs + 1):
        fit_loss, _ = run_epoch(model, sets["fit"], config, fit_generator, optimizer)
        validation_loss, _ = validate()
        if not np.isfinite([fit_loss, validation_loss]).all():
            raise ValueError("Pretraining produced non-finite reconstruction loss")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            save_encoder()
        history.append({"epoch": epoch, "fit_masked_mse": fit_loss,
                        "validation_masked_mse": validation_loss})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d} | fit {fit_loss:.4f} | validation {validation_loss:.4f}", flush=True)
    write_csv(out / "history.csv", tuple(history[0]), history)
    write_csv(out / "pretraining_split.csv", ("asset_id", "pretraining_role", "vctk_speakers", "sha256"), (
        {key: row[key] for key in ("asset_id", "pretraining_role", "sha256")}
        | {"vctk_speakers": "|".join(sorted(row["vctk_speakers"]))} for row in records
    ))
    report = {
        **provenance, "architecture": asdict(config),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "all_parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__), "device": "cpu", "cpu_threads": THREADS,
        "seed": SEED, "epochs": epochs, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "clip_counts": dict(Counter(row["pretraining_role"] for row in records)),
        "validation_mean_prediction_mse": mean_baseline,
        "validation_random_initialization_mse": initial,
        "best_validation_masked_mse": best_loss, "best_epoch": best_epoch,
        "preparation_seconds": prepare_seconds, "elapsed_seconds": time.perf_counter() - start,
        "checkpoint_sha256": sha256(weights_path),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.root, args.audit, args.out, args.epochs)
