"""Download a balanced local sample of the authors' In-the-Wild release."""

import argparse
import csv
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from bodh.manifest import Asset, COLUMNS, write_csv

URL = ("https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/"
       "eee168f92c367f8c82ff2cf42b6f61e362fd6211/release_in_the_wild.zip")
SIZE = 8_161_489_023
ETAG = '"ec530d2076f973d40a8e388e559888f1b4fea467db8126c811d34dd508ee106f"'
TAIL = SIZE - 4 * 1024**2
CLASSES = {"bona-fide": "bonafide", "bonafide": "bonafide", "spoof": "spoof"}


def download(start: int, stop: int, work: Path) -> bytes:
    with TemporaryDirectory(dir=work) as temp:
        data = Path(temp) / "body"
        headers = Path(temp) / "headers"
        for _ in range(3):
            result = subprocess.run([
                "curl", "-fsSL", "--connect-timeout", "8", "--max-time", "35",
                "--max-filesize", str(stop - start), "--range", f"{start}-{stop - 1}",
                "-D", str(headers), "-o", str(data), URL,
            ], capture_output=True)
            if result.returncode == 0:
                response = headers.read_text().strip().split("\n\n")[-1]
                fields = dict(line.lower().split(": ", 1) for line in response.splitlines()[1:]
                              if ": " in line)
                if (fields.get("content-range") == f"bytes {start}-{stop - 1}/{SIZE}"
                        and fields.get("etag") == ETAG and data.stat().st_size == stop - start):
                    return data.read_bytes()
        raise ValueError(f"Publisher did not return the expected bytes {start}–{stop - 1}")


def sample(metadata: bytes, count: int, seed: int) -> tuple[list[dict], str | None]:
    reader = csv.DictReader(metadata.decode("utf-8-sig").splitlines())
    parent_field = next((key for key in ("parent_recording_id", "source_url")
                         if key in (reader.fieldnames or [])), None)
    groups = defaultdict(lambda: defaultdict(list))
    for row in reader:
        if row["label"] not in CLASSES or not re.fullmatch(r"\d+\.wav", row["file"]):
            raise ValueError("Unrecognized In-the-Wild metadata entry")
        groups[row["speaker"]][CLASSES[row["label"]]].append(row)
    speakers = sorted(speaker for speaker, entries in groups.items() if len(entries) == 2)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    for entries in groups.values():
        for label in entries:
            entries[label].sort(key=lambda row: row["file"])
            rng.shuffle(entries[label])
    selected, parents = [], set()
    while len(selected) < count * 2:
        remaining = len(selected)
        for speaker in speakers:
            genuine = groups[speaker]["bonafide"]
            spoof = groups[speaker]["spoof"]
            for label, items in (("bonafide", genuine), ("spoof", spoof)):
                while items and parent_field and items[-1][parent_field] and (
                    label, items[-1][parent_field]
                ) in parents:
                    items.pop()
            if genuine and spoof:
                pair = [genuine.pop(), spoof.pop()]
                selected.extend(pair)
                if parent_field:
                    parents.update((CLASSES[row["label"]], row[parent_field]) for row in pair)
            if len(selected) == count * 2:
                break
        if len(selected) == remaining:
            raise ValueError("Could not fill the matched sample")
    return selected, parent_field


def fetch(out: Path, count: int, seed: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    with TemporaryDirectory(prefix=".itw-", dir=out.parent) as folder:
        work = Path(folder)
        stage = work / "dataset"
        stage.mkdir()
        with (work / "partial.zip").open("w+b", buffering=0) as file:
            file.truncate(SIZE)

            def cache(bounds: tuple[int, int]) -> None:
                start, stop = bounds
                os.pwrite(file.fileno(), download(start, stop, work), start)

            chunk = 1024**2
            with ThreadPoolExecutor(max_workers=10) as pool:
                list(pool.map(cache, ((start, start + chunk) for start in range(TAIL, SIZE, chunk))))
            with ZipFile(file) as archive:
                members = sorted(archive.infolist(), key=lambda info: info.header_offset)
                offsets = {a.filename: b.header_offset for a, b in zip(members, members[1:])}
                offsets[members[-1].filename] = TAIL
                meta = next(info for info in members if PurePosixPath(info.filename).name == "meta.csv")
                prefix = meta.filename.removesuffix("meta.csv")

                def fetch_member(name: str) -> None:
                    start = archive.getinfo(name).header_offset
                    end = min(offsets[name], TAIL)
                    if start < end:
                        cache((start, end))

                fetch_member(meta.filename)
                metadata = archive.read(meta)
                selected, parent = sample(metadata, count, seed)
                paths = [prefix + row["file"] for row in selected]
                print(f"Selected {len(paths)} clips across {len({row['speaker'] for row in selected})} speakers",
                      flush=True)
                with ThreadPoolExecutor(max_workers=10) as pool:
                    list(pool.map(fetch_member, paths))
                assets = []
                for record, name in zip(selected, paths):
                    filename, speaker = record["file"], record["speaker"]
                    (stage / filename).write_bytes(archive.read(name))
                    group = (record[parent] if parent else "") or speaker
                    assets.append(Asset("in_the_wild", filename.removesuffix(".wav"), "audio",
                                        filename, "external", group, CLASSES[record["label"]], speaker))
        write_csv(stage / "manifest.csv", COLUMNS, map(asdict, assets))
        (stage / "source.json").write_text(json.dumps({
            "archive": URL, "etag": ETAG, "seed": seed, "per_class": count,
            "speaker_balanced_across_classes": True, "documented_parent_field": parent,
        }, indent=2) + "\n")
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, choices=range(1, 101), default=100, metavar="1–100")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    fetch(args.output, args.per_class, args.seed)
