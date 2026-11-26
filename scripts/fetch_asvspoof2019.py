"""Fetch a bounded, reproducible subset of the official ASVspoof 2019 LA archive."""

import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from bodh.authenticity import PROTOCOLS, read_asvspoof_protocol
from bodh.manifest import Asset, COLUMNS, write_csv

URL = ("https://datashare.ed.ac.uk/server/api/core/bitstreams/"
       "a9f87c35-f055-4015-80e2-2fdff0d46269/content")
ARCHIVE_SIZE = 7_640_952_520
ETAG = '"30c98f11d8b2bc21f2c257bfd78bb5c5"'
TAIL_BYTES = 16 * 1024**2
COUNTS = {"train": (2580, 22800), "dev": (2548, 22296), "eval": (7355, 63882)}


def download_range(start: int, stop: int) -> bytes:
    command = [
        "curl", "--silent", "--show-error", "--fail", "--suppress-connect-headers",
        "--connect-timeout", "8", "--max-time", "30",
        "--max-filesize", str(stop - start), "--range", f"{start}-{stop - 1}",
        "--dump-header", "-", URL,
    ]
    expected = f"bytes {start}-{stop - 1}/{ARCHIVE_SIZE}"
    for _ in range(3):
        result = subprocess.run(command, capture_output=True, check=False)
        header, separator, content = result.stdout.partition(b"\r\n\r\n")
        fields = {}
        for line in header.decode("iso-8859-1").splitlines()[1:]:
            key, _, value = line.partition(":")
            fields[key.lower()] = value.strip()
        if (not result.returncode and separator and len(content) == stop - start
                and fields.get("content-range") == expected and fields.get("etag") == ETAG):
            return content
    detail = result.stderr.decode(errors="replace").strip()
    raise ValueError(f"Could not verify archive bytes {start}–{stop - 1}. {detail}")


def balanced_sample(labels: dict[str, list[Asset]], limit: int, seed: int) -> list[Asset]:
    rng = random.Random(seed)
    selected = []
    for label in ("bonafide", "spoof"):
        groups = defaultdict(list)
        for asset in labels[label]:
            key = asset.speaker_id if label == "bonafide" else asset.generator_id
            groups[key].append(asset)
        lineup = sorted(groups)
        rng.shuffle(lineup)
        for group in groups.values():
            group.sort(key=lambda asset: asset.asset_id)
            rng.shuffle(group)
        if limit > len(labels[label]):
            raise ValueError(f"Requested more {label} clips than are available")
        # Cycle first through speakers for bona fide clips and through attack IDs for spoofs.
        remaining = limit
        while remaining:
            for key in lineup:
                if remaining and groups[key]:
                    selected.append(groups[key].pop())
                    remaining -= 1
    return sorted(selected, key=lambda asset: asset.asset_id)


def extract(output: Path, limit: int, seed: int) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".asvspoof-", dir=output.parent) as folder:
        stage = Path(folder) / "dataset"
        stage.mkdir()
        with (Path(folder) / "partial.zip").open("w+b", buffering=0) as sparse:
            sparse.truncate(ARCHIVE_SIZE)
            tail = ARCHIVE_SIZE - TAIL_BYTES
            chunk_size = 1024**2

            def cache_range(bounds: tuple[int, int]) -> None:
                start, stop = bounds
                os.pwrite(sparse.fileno(), download_range(start, stop), start)

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(cache_range, (
                    (start, start + chunk_size) for start in range(tail, ARCHIVE_SIZE, chunk_size)
                )))
            print("Verified the official ZIP directory", flush=True)
            with ZipFile(sparse) as archive:
                entries = sorted(archive.infolist(), key=lambda item: item.header_offset)
                following = {a.header_offset: b.header_offset for a, b in zip(entries, entries[1:])}
                following[entries[-1].header_offset] = tail
                protocol_name = PROTOCOLS["train"]
                full = next(info.filename for info in entries if info.filename.endswith(
                    f"ASVspoof2019_LA_cm_protocols/{protocol_name}"
                ) and "__MACOSX" not in PurePosixPath(info.filename).parts)
                prefix = full.removesuffix(f"ASVspoof2019_LA_cm_protocols/{protocol_name}")

                def fetch_member(name: str) -> None:
                    info = archive.getinfo(prefix + name)
                    start = info.header_offset
                    stop = min(following[start], start + info.compress_size + 70_000)
                    if stop <= tail:
                        cache_range((start, stop))
                    elif start < tail:
                        cache_range((start, tail))

                def save_member(name: str) -> bytes:
                    data = archive.read(prefix + name)
                    destination = stage / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                    return data

                selections = {}
                for split, protocol in PROTOCOLS.items():
                    name = f"ASVspoof2019_LA_cm_protocols/{protocol}"
                    fetch_member(name)
                    labels = read_asvspoof_protocol(save_member(name).decode("utf-8"), split)
                    if tuple(len(labels[key]) for key in ("bonafide", "spoof")) != COUNTS[split]:
                        raise ValueError(f"Unexpected published trial counts for {split}")
                    selections[split] = balanced_sample(labels, limit, seed)

                for split, assets in selections.items():
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        for _ in pool.map(fetch_member, (asset.path for asset in assets)):
                            pass
                    for asset in assets:
                        save_member(asset.path)
                    write_csv(stage / f"{split}.csv", COLUMNS, map(asdict, assets))
                    print(f"Verified and extracted {len(assets)} {split} clips", flush=True)

        metadata = {
            "source": "https://doi.org/10.7488/ds/2555", "archive_url": URL,
            "archive_etag": ETAG, "archive_bytes": ARCHIVE_SIZE,
            "seed": seed, "clips_per_class_per_split": limit,
            "sampling": "Cycle through speakers for bona fide and attack IDs for spoof",
        }
        (stage / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-condition", type=int, choices=range(1, 101), default=100,
                        metavar="1–100")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        extract(args.output, args.per_condition, args.seed)
    except (BadZipFile, KeyError, OSError, StopIteration, ValueError) as error:
        parser.exit(2, f"fetch_asvspoof2019: {error}\n")
