"""Import the published Lavan et al. Experiment 1b audio and rating tables."""

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from bodh.manifest import Asset, COLUMNS, write_csv
from bodh.perception import TRIAL_COLUMNS

DATASET = "lavan2025"
ARM = "experiment_1b"
RATINGS = {
    "realness": "ratings/data_exp_191772-v7_task-bjpr.csv",
    "perceived_trustworthiness": "ratings/data_exp_191772-v7_task-5nfg.csv",
}
CONDITIONS = {"p": "human_recording", "c": "voice_clone"}
AUTHOR_LABELS = {"Recording": "human_recording", "Clone": "voice_clone"}
STIMULUS_NAME = r"([pc])(\d{3})[-_]Rainbow_(?:7|9|10|17)\.wav"
OSF_STORAGE = "https://api.osf.io/v2/nodes/mrw5q/files/osfstorage"
OSF_FOLDERS = {
    "human": "6818dfcf88b6f4a3a25ef607", "clone": "6818e00b25be441f889fde89",
    "ratings": "6818ded0c9f3ada26a568cb4",
}
OSF_SNAPSHOT_SHA256 = "da67854578aa8c22614da42bbe122dec1b7af52472f33224196d8e7737931e44"


def fetch(url: str, limit: int, destination: Path | None = None) -> bytes:
    command = ["curl", "-fsSL", "--proto", "=https", "--connect-timeout", "10",
               "--max-time", "180", "--retry", "2", "--max-filesize", str(limit)]
    if destination is not None:
        command.extend(("-o", str(destination)))
    return subprocess.run([*command, url], check=True, stdout=subprocess.PIPE).stdout


def snapshot(listings: dict) -> list[dict]:
    expected_ratings = {Path(path).name for path in RATINGS.values()}
    entries = []
    for category, listing in listings.items():
        if listing["links"]["next"]:
            raise ValueError("Unexpected additional OSF listing pages")
        for item in listing["data"]:
            details = item["attributes"]
            name = details["name"]
            if category == "ratings":
                if name not in expected_ratings:
                    continue
                folder = "ratings"
            else:
                if not re.fullmatch(STIMULUS_NAME, name):
                    raise ValueError(f"Unexpected OSF stimulus name: {name}")
                folder = "audio"
            url = item["links"]["download"]
            if not re.fullmatch(r"https://osf\.io/download/[a-z0-9]+/", url):
                raise ValueError("Unexpected OSF download URL")
            entries.append({"path": f"{folder}/{name}", "url": url, "size": details["size"],
                            "sha256": details["extra"]["hashes"]["sha256"]})
    entries.sort(key=lambda item: item["path"])
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    if sha256(canonical).hexdigest() != OSF_SNAPSHOT_SHA256:
        raise ValueError("The OSF source listing differs from the version used in the Bodh study")
    return entries


def matches(path: Path, entry: dict) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest() == entry["sha256"]


def download(root: Path) -> None:
    entries = snapshot({
        name: json.loads(fetch(f"{OSF_STORAGE}/{identifier}/?page%5Bsize%5D=100", 200_000))
        for name, identifier in OSF_FOLDERS.items()
    })
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=root) as temp:
        for entry in entries:
            path = root / entry["path"]
            if matches(path, entry):
                continue
            if path.exists() or path.is_symlink():
                raise ValueError(f"Existing file differs from the pinned OSF snapshot: {path.name}")
            incoming = Path(temp) / path.name
            fetch(entry["url"], entry["size"], incoming)
            if not matches(incoming, entry):
                raise ValueError(f"Downloaded file differs from the OSF snapshot: {path.name}")
            path.parent.mkdir(exist_ok=True)
            incoming.rename(path)
    (root / "source_files.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def make_assets(root: Path) -> list[Asset]:
    assets = []
    for file in sorted((root / "audio").glob("*.wav")):
        match = re.fullmatch(STIMULUS_NAME, file.name)
        if not match:
            raise ValueError(f"Unexpected Experiment 1b stimulus {file.name}")
        kind, speaker = match.groups()
        speaker = f"vctk_{speaker}"
        assets.append(Asset(
            DATASET, file.name, "audio", f"audio/{file.name}", "unsplit",
            speaker, CONDITIONS[kind], speaker, "elevenlabs" if kind == "c" else "",
        ))
    human = {asset.speaker_id for asset in assets if asset.condition == "human_recording"}
    clone = {asset.speaker_id for asset in assets if asset.condition == "voice_clone"}
    if len(assets) != 80 or len(human) != 40 or human != clone:
        raise ValueError("Expected 40 complete human–clone speaker pairs")
    return assets


def verify_sources(root: Path, expected: set[str]) -> None:
    sources = json.loads((root / "source_files.json").read_text(encoding="utf-8"))
    if len(sources) != len(expected) or {entry["path"] for entry in sources} != expected:
        raise ValueError("Source inventory does not cover exactly the expected files")
    for entry in sources:
        if not matches(root / entry["path"], entry):
            raise ValueError(f"Downloaded file differs from the OSF inventory: {entry['path']}")


def make_trials(root: Path, assets: list[Asset]) -> list[dict]:
    by_name = {asset.asset_id: asset for asset in assets}
    participant_keys = {}
    trials = []
    listeners_by_endpoint = []
    for endpoint, relative in RATINGS.items():
        seen = set()
        with (root / relative).open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                label = row["Spreadsheet: voice_type"]
                response = row["Response"]
                if (row["Response Type"] != "response" or row["Display"] == "attention_check"
                        or not label or not response):
                    continue
                name = row["Spreadsheet: sound"]
                asset = by_name.get(name)
                listener = row["Participant Private ID"]
                key = (listener, name)
                value = float(response)
                if (asset is None or AUTHOR_LABELS.get(label) != asset.condition or not listener
                        or key in seen or not 0 <= value <= 100):
                    raise ValueError(f"Invalid or unmatched {endpoint} trial for {name!r}")
                seen.add(key)
                pseudonym = participant_keys.setdefault(listener, f"listener_{len(participant_keys) + 1:03}")
                trials.append(dict(zip(
                    TRIAL_COLUMNS, (DATASET, name, ARM, pseudonym, endpoint, value)
                )))
        listeners = {listener for listener, _ in seen}
        if len(listeners) != 50 or {name for _, name in seen} != set(by_name):
            raise ValueError(f"Expected ratings from 50 listeners covering all 80 stimuli for {endpoint}")
        missing = len(listeners) * len(assets) - len(seen)
        print(f"{endpoint}: {len(seen)} observed ratings, {missing} missing")
        listeners_by_endpoint.append(listeners)
    if listeners_by_endpoint[0] != listeners_by_endpoint[1]:
        raise ValueError("The endpoints contain different sets of listeners")
    return trials


def run(root: Path, output: Path) -> None:
    assets = make_assets(root)
    verify_sources(root, {asset.path for asset in assets} | set(RATINGS.values()))
    trials = make_trials(root, assets)
    output.mkdir(parents=True)
    write_csv(output / "manifest.csv", COLUMNS, map(asdict, assets))
    write_csv(output / "trials.csv", TRIAL_COLUMNS, trials)
    print(f"Matched {len(trials)} ratings to {len(assets)} local WAV files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--download", action="store_true", help="Acquire the pinned public OSF release")
    args = parser.parse_args()
    try:
        if args.download:
            download(args.root)
        run(args.root, args.output)
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(2, f"lavan2025: {error}\n")
