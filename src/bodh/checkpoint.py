"""Store model tensors and JSON metadata without pickle deserialization."""

import json
from pathlib import Path

import numpy as np
import torch


def save(path: Path, state: dict, metadata: dict | None = None) -> None:
    payload = {f"tensor.{name}": value.detach().cpu().numpy() for name, value in state.items()}
    with path.open("wb") as file:
        np.savez(file, metadata=json.dumps(metadata or {}, allow_nan=False), **payload)


def load(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files or any(
            name != "metadata" and not name.startswith("tensor.") for name in archive.files
        ):
            raise ValueError("Unexpected Bodh checkpoint format")
        metadata = json.loads(archive["metadata"].item())
        state = {name.removeprefix("tensor."): torch.from_numpy(archive[name])
                 for name in archive.files if name.startswith("tensor.")}
    return state, metadata
