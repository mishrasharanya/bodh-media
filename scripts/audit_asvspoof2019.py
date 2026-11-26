"""Verify the frozen ASVspoof pilot and check recording overlap with Lavan."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path

from bodh.audio import load_audio
from bodh.authenticity import PROTOCOLS, read_asvspoof_protocol
from bodh.eda import sha256 as hash_file
from bodh.manifest import Asset, COLUMNS, read_assets, write_csv

# The evaluation versions use the same synthesis/conversion algorithms as the train versions.
ATTACK_ALIASES = {"A16": "A04", "A19": "A06"}
SPLIT_KEYS = ("asset_id", "speaker_id", "sha256", "pcm_16k_mono_sha256", "vctk_source_ids",
              "vctk_speakers")


def read_recording_map(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    indexed = {row["ASVspoof_ID"]: row for row in rows}
    if len(rows) != len(indexed):
        raise ValueError("Duplicate utterance key in the organizers' source map")
    return indexed


def vctk_speakers(meta: dict) -> set[str]:
    values = (meta["VCTK_ID"], meta["TTS_VC_target_speaker"], meta["VC_source_VCTK_ID"])
    return {f"vctk_{match[1]}" for value in values
            if (match := re.fullmatch(r"p(\d{3})(?:_.*)?", value))}


def verify_clip(root: Path, asset: Asset, meta: dict) -> dict:
    path = root / asset.path
    speakers = vctk_speakers(meta)
    if not speakers:
        raise ValueError(f"No mapped VCTK speaker for {asset.asset_id}")
    waveform = load_audio(path).mono
    return {
        **asdict(asset), "sha256": hash_file(path),
        "pcm_16k_mono_sha256": sha256(waveform.astype("<f4").tobytes()).hexdigest(),
        "vctk_source_ids": "|".join(sorted({meta["VCTK_ID"], meta["VC_source_VCTK_ID"]} - {"-"})),
        "vctk_speakers": "|".join(sorted(speakers)),
        "attack_family": ATTACK_ALIASES.get(asset.generator_id, asset.generator_id),
    }


def values(rows: list[dict], field: str) -> set[str]:
    return {part for row in rows for part in row[field].split("|") if part}


def overlap_counts(selections: dict[str, list[dict]]) -> dict:
    return {
        f"{a}/{b}": {
            field: len(values(selections[a], field) & values(selections[b], field))
            for field in (*SPLIT_KEYS, "attack_family")
        }
        for a, b in combinations(selections, 2)
    }


def remove_prior_split_overlap(selections: dict[str, list[dict]]) -> tuple[dict, dict]:
    accepted, excluded = {}, {}
    prior = {field: set() for field in SPLIT_KEYS}
    for split, records in selections.items():
        accepted[split], excluded[split] = [], []
        for row in records:
            conflicts = [field for field in SPLIT_KEYS if values([row], field) & prior[field]]
            if conflicts:
                excluded[split].append({**row, "overlap_fields": "|".join(conflicts)})
            else:
                accepted[split].append(row)
        if {row["condition"] for row in accepted[split]} != {"bonafide", "spoof"}:
            raise ValueError(f"{split} has insufficient class coverage after removing overlap")
        for field in SPLIT_KEYS:
            prior[field].update(values(accepted[split], field))
    return accepted, excluded


def audit(root: Path, source_map: Path, lavan_manifest: Path, out: Path) -> dict:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output already exists: {out}")
    sources = read_recording_map(source_map)
    size = json.loads((root / "source.json").read_text(encoding="utf-8"))["clips_per_class_per_split"]
    selections = {}
    inventories = {}
    for split, protocol in PROTOCOLS.items():
        protocol_path = root / "ASVspoof2019_LA_cm_protocols" / protocol
        official = read_asvspoof_protocol(protocol_path.read_text(encoding="utf-8"), split)
        published = {asset.asset_id: asset for rows in official.values() for asset in rows}
        assets = read_assets(root / f"{split}.csv")
        if any(asset != published.get(asset.asset_id) for asset in assets):
            raise ValueError(f"A {split} row does not match its official protocol")
        classes = Counter(asset.condition for asset in assets)
        if classes != {"bonafide": size, "spoof": size}:
            raise ValueError(f"Unexpected class counts in the {split} sample")
        if any(asset.asset_id not in sources for asset in assets):
            raise ValueError(f"A {split} row is missing from the organizers' source map")
        with ThreadPoolExecutor(max_workers=4) as pool:
            selections[split] = list(pool.map(
                lambda asset: verify_clip(root, asset, sources[asset.asset_id]), assets
            ))
        inventories[split] = {
            "classes": dict(classes),
            "official_speakers": len({asset.speaker_id for asset in assets}),
            "verified_pcm_clips": len(assets),
            "spoof_attack_counts": dict(sorted(Counter(
                asset.generator_id for asset in assets if asset.generator_id
            ).items())),
            "manifest_sha256": hash_file(root / f"{split}.csv"),
            "official_protocol_sha256": hash_file(protocol_path),
        }
        print(f"Audited {len(assets)} {split} clips", flush=True)

    strict, excluded = remove_prior_split_overlap(selections)
    overlaps = overlap_counts(strict)
    seen = values(strict["train"] + strict["dev"], "attack_family")
    eval_strata = []
    for row in strict["eval"]:
        condition = ("bonafide" if row["condition"] == "bonafide" else
                     "seen_algorithm" if row["attack_family"] in seen else "unseen_algorithm")
        eval_strata.append({"asset_id": row["asset_id"], "stratum": condition,
                            "attack_family": row["attack_family"]})

    lavan = read_assets(lavan_manifest)
    exposed = values(strict["train"] + strict["dev"], "vctk_speakers")
    overlap = {asset.speaker_id for asset in lavan} & exposed
    sensitivity = [asset for asset in lavan if asset.speaker_id not in exposed]
    report = {
        "official_sample": inventories, "official_overlap_counts": overlap_counts(selections),
        "strict_splits": {
            split: {
                "classes": dict(Counter(row["condition"] for row in records)),
                "exclusions": dict(Counter(
                    field for row in excluded[split] for field in row["overlap_fields"].split("|")
                )),
                "excluded_clips": len(excluded[split]),
            }
            for split, records in strict.items()
        },
        "strict_overlap_counts": overlaps,
        "strict_split_overlap_passed": not any(
            pair[field] for pair in overlaps.values() for field in SPLIT_KEYS
        ),
        "strict_eval_strata": dict(Counter(row["stratum"] for row in eval_strata)),
        "lavan": {
            "overlapping_speakers_with_asvspoof_train_or_dev": len(overlap),
            "total_speakers": len({asset.speaker_id for asset in lavan}),
            "retained_clips_after_speaker_exclusion": len(sensitivity),
            "retained_speakers_after_speaker_exclusion": len({asset.speaker_id for asset in sensitivity}),
        },
        "source_map_sha256": hash_file(source_map),
    }
    out.mkdir(parents=True)
    for split, records in selections.items():
        write_csv(out / f"{split}_integrity.csv", tuple(records[0]), records)
        manifest = out / f"{split}_strict.csv"
        write_csv(manifest, COLUMNS, (
            {key: row[key] for key in COLUMNS} for row in strict[split]
        ))
        report["strict_splits"][split]["manifest_sha256"] = hash_file(manifest)
        if excluded[split]:
            write_csv(out / f"{split}_exclusions.csv", tuple(excluded[split][0]), excluded[split])
    write_csv(out / "eval_strata.csv", ("asset_id", "stratum", "attack_family"), eval_strata)
    if sensitivity:
        write_csv(out / "lavan_speaker_sensitivity.csv", COLUMNS, map(asdict, sensitivity))
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("source_map", type=Path)
    parser.add_argument("lavan_manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.source_map, args.lavan_manifest, args.out)
    if not report["strict_split_overlap_passed"]:
        parser.exit(1, "Cross-split overlap needs review. See audit.json.\n")
