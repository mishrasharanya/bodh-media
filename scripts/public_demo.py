"""Fetch the pinned public demo or make a manifest from a local copy."""

import argparse
import csv
import re
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

from bodh.manifest import Asset, COLUMNS, write_csv

REVISION = "3bb88ab218ea3c3af621572459fd501edfedfc0e"
URL = f"https://huggingface.co/datasets/behavioralsignals/deepfake-detection-demo/resolve/{REVISION}/test/"


def fetch(name: str, path: Path, max_bytes: int) -> None:
    with urlopen(URL + name, timeout=25) as response:
        content = response.read(max_bytes + 1)
    if not content or len(content) > max_bytes:
        raise ValueError(f"Unexpected download size for {name}")
    partial = path.with_suffix(".part")
    partial.write_bytes(content)
    partial.replace(path)


def import_demo(folder: Path, download: bool) -> None:
    manifest = folder / "manifest.csv"
    metadata = folder / "metadata.csv"
    if manifest.exists():
        raise ValueError(f"Manifest already exists: {manifest}")
    if download and not metadata.exists():
        folder.mkdir(parents=True, exist_ok=True)
        fetch("metadata.csv", metadata, 20_000)
    assets = []
    with metadata.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["file_name"]
            language, label = row["language"], row["deepfake"]
            if (not re.fullmatch(r"common_voice_[A-Za-z0-9_-]+\.wav", name)
                    or not re.fullmatch(r"[a-z]{2}", language)
                    or label not in {"bonafide", "spoofed"}):
                raise ValueError(f"Unexpected metadata entry for {name}")
            assets.append(Asset(
                "behavioral_signals_demo", Path(name).stem, "audio", name, "test",
                f"language:{language}", label, generator_id=row["generator"] or "",
            ))
    if len(assets) != 22 or len({asset.asset_id for asset in assets}) != 22:
        raise ValueError("Expected the pinned 22-clip demo metadata")
    for asset in assets:
        path = folder / asset.path
        if download and not path.exists():
            fetch(asset.path, path, 600_000)
        if not path.is_file():
            raise ValueError(f"Missing demo audio {asset.path}")
    write_csv(manifest, COLUMNS, map(asdict, assets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        import_demo(args.folder, args.download)
    except (OSError, ValueError) as error:
        parser.exit(2, f"public_demo: {error}\n")
