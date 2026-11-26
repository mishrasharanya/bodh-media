"""Reuse the masked-patch encoder for supervised clip-level transfer."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bodh import audio, checkpoint as checkpoint_file
from bodh.eda import sha256
from bodh.manifest import read_assets
from bodh.ssl import EncoderConfig, SpectrogramEncoder

SEEDS = (2026, 2027, 2028)
BATCH_SIZE = 16


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def configure_cpu() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def load_pretraining(path: Path, audit: Path) -> dict:
    state, metadata = checkpoint_file.load(path)
    if metadata["provenance"]["audit_sha256"] != sha256(audit / "audit.json"):
        raise ValueError("The pretraining and transfer audit snapshots differ")
    return {**metadata, "log_mel_center": state.pop("log_mel_center"),
            "log_mel_scale": state.pop("log_mel_scale"), "state_dict": state}


def audited_assets(audit: Path, split: str) -> tuple:
    report = json.loads((audit / "audit.json").read_text())
    manifest = audit / f"{split}_strict.csv"
    if (not report["strict_split_overlap_passed"]
            or sha256(manifest) != report["strict_splits"][split]["manifest_sha256"]):
        raise ValueError(f"{split} does not match the overlap audit")
    return read_assets(manifest), {r["asset_id"]: r for r in rows(audit / f"{split}_integrity.csv")}


def spectrogram(path: Path, expected_hash: str) -> tuple[dict, torch.Tensor]:
    if sha256(path) != expected_hash:
        raise ValueError(f"Audio changed since inventory: {path.name}")
    summary, log_mel = audio.features(audio.trim_quiet_edges(audio.load_audio(path)))
    return summary, torch.tensor(log_mel, dtype=torch.float32)


def batches(clips: list[torch.Tensor], config: EncoderConfig, indices: list[int]):
    for start in range(0, len(indices), BATCH_SIZE):
        take = indices[start:start + BATCH_SIZE]
        chunks, owners = [], []
        for owner, index in enumerate(take):
            for chunk in clips[index].split(config.max_patches * config.patch_frames):
                chunks.append(chunk)
                owners.append(owner)
        frames = config.patch_frames * max(
            (len(chunk) + config.patch_frames - 1) // config.patch_frames for chunk in chunks
        )
        data = torch.zeros(len(chunks), frames, audio.MEL_BANDS)
        valid = torch.zeros(len(chunks), frames, dtype=torch.bool)
        for index, chunk in enumerate(chunks):
            data[index, :len(chunk)] = chunk
            valid[index, :len(chunk)] = True
        yield (
            take, data.reshape(len(chunks), -1, config.patch_frames, audio.MEL_BANDS),
            valid.reshape(len(chunks), -1, config.patch_frames), torch.tensor(owners),
        )


def make_encoder(config: EncoderConfig, checkpoint: dict, pretrained: bool) -> SpectrogramEncoder:
    model = SpectrogramEncoder(config)
    if pretrained:
        model.load_state_dict(checkpoint["state_dict"])
    return model


def pool(encoder: SpectrogramEncoder, patches: torch.Tensor, valid: torch.Tensor,
         owners: torch.Tensor, n_clips: int) -> torch.Tensor:
    frame_counts = valid.sum(dim=-1)
    tokens = encoder(patches, frame_counts > 0)
    chunk_sums = (tokens * frame_counts.unsqueeze(-1)).sum(dim=1)
    totals = tokens.new_zeros(n_clips, tokens.shape[-1]).index_add(0, owners, chunk_sums)
    lengths = tokens.new_zeros(n_clips).index_add(0, owners, frame_counts.sum(1).to(tokens.dtype))
    return totals / lengths.unsqueeze(-1)


class Detector(nn.Module):
    def __init__(self, encoder: SpectrogramEncoder, width: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(width, 1)

    def forward(self, patches: torch.Tensor, valid: torch.Tensor, owners: torch.Tensor,
                clips: int) -> torch.Tensor:
        return self.classifier(pool(self.encoder, patches, valid, owners, clips)).squeeze(-1)


def normalized(clips: list[torch.Tensor], checkpoint: dict) -> list[torch.Tensor]:
    return [(clip - checkpoint["log_mel_center"]) / checkpoint["log_mel_scale"] for clip in clips]
