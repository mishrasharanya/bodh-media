"""File-level provenance shared by downstream research tasks."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Asset:
    dataset: str
    asset_id: str
    modality: str
    path: str
    split: str
    source_group: str
    condition: str
    speaker_id: str = ""
    generator_id: str = ""


COLUMNS = tuple(Asset.__dataclass_fields__)


def read_assets(manifest: Path) -> list[Asset]:
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        if not set(COLUMNS).issubset(rows.fieldnames or ()):
            raise ValueError(f"Asset manifest requires columns {', '.join(COLUMNS)}")
        assets = [Asset(**{key: row[key] or "" for key in COLUMNS}) for row in rows]
    if not assets:
        raise ValueError("Asset manifest is empty")
    keys = set()
    for asset in assets:
        if not all(getattr(asset, key).strip() for key in COLUMNS[:7]):
            raise ValueError(f"Missing required fields for asset {asset.asset_id!r}")
        key = (asset.dataset, asset.asset_id)
        if key in keys:
            raise ValueError(f"Duplicate asset key {key}")
        keys.add(key)
        path = Path(asset.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected a path relative to the audio root for {key}")
    return assets


def write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
