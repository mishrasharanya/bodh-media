"""Translate the official ASVspoof 2019 LA protocol into local asset records."""

import random
import re
from collections import defaultdict
from pathlib import Path

from bodh.manifest import Asset

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def read_asvspoof_protocol(protocol: str, split: str) -> dict[str, list[Asset]]:
    by_condition = defaultdict(list)
    seen = set()
    for number, line in enumerate(protocol.splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five protocol columns on line {number}")
        speaker, clip, _, attack, label = fields
        if (label not in {"bonafide", "spoof"}
                or not re.fullmatch(rf"LA_{split[0].upper()}_\d+", clip)
                or clip in seen or (label == "bonafide" and attack != "-")
                or (label == "spoof" and not re.fullmatch(r"A\d{2}", attack))):
            raise ValueError(f"Invalid or repeated protocol entry on line {number}")
        seen.add(clip)
        by_condition[label].append(Asset(
            "asvspoof2019_la", clip, "audio", f"ASVspoof2019_LA_{split}/flac/{clip}.flac",
            split, speaker, label, speaker, "" if attack == "-" else attack,
        ))
    if set(by_condition) != {"bonafide", "spoof"}:
        raise ValueError("Expected both bona fide and spoof entries in the official protocol")
    return by_condition


def asvspoof_assets(root: Path, split: str, per_condition: int, seed: int) -> list[Asset]:
    protocol = root / "ASVspoof2019_LA_cm_protocols" / PROTOCOLS[split]
    by_condition = read_asvspoof_protocol(protocol.read_text(encoding="utf-8"), split)
    rng = random.Random(seed)
    selected = []
    for _, group in sorted(by_condition.items()):
        selected.extend(rng.sample(sorted(group, key=lambda asset: asset.asset_id),
                                   min(per_condition, len(group))))
    missing = [asset.path for asset in selected if not (root / asset.path).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} protocol audio files, including {missing[0]}")
    return sorted(selected, key=lambda asset: asset.asset_id)
