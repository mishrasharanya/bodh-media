"""Descriptive analysis of observed listener trials linked to stimulus metadata."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from bodh.manifest import read_assets

TRIAL_COLUMNS = (
    "dataset", "asset_id", "study_arm", "participant_id", "endpoint", "value"
)


def describe_trials(manifest: Path, trials: Path) -> dict:
    assets = {(asset.dataset, asset.asset_id): asset for asset in read_assets(manifest)}
    stimuli = defaultdict(lambda: defaultdict(list))
    participants = defaultdict(set)
    with trials.open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        if not set(TRIAL_COLUMNS).issubset(rows.fieldnames or ()):
            raise ValueError(f"Listener trial CSV requires columns {', '.join(TRIAL_COLUMNS)}")
        seen = set()
        for row in rows:
            dataset, clip, arm, listener, endpoint, response = (
                (row[column] or "").strip() for column in TRIAL_COLUMNS
            )
            asset = assets.get((dataset, clip))
            key = (dataset, clip, arm, listener, endpoint)
            if asset is None or not all((arm, listener, endpoint, response)) or key in seen:
                raise ValueError(f"Missing, unmatched or repeated trial at line {rows.line_num}")
            value = float(response)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite trial value at line {rows.line_num}")
            seen.add(key)
            group = (dataset, arm, endpoint, asset.condition)
            stimuli[group][clip].append(value)
            participants[group].add(listener)
    if not stimuli:
        raise ValueError("No listener trials supplied")
    groups = []
    for key, clips in sorted(stimuli.items()):
        dataset, arm, endpoint, condition = key
        groups.append({
            "dataset": dataset, "study_arm": arm, "endpoint": endpoint, "condition": condition,
            "n_stimuli": len(clips), "n_listeners": len(participants[key]),
            "n_trials": sum(map(len, clips.values())),
            "mean_of_stimulus_means": mean(mean(values) for values in clips.values()),
        })
    return {"analysis": "Descriptive observed responses with equal stimulus weights",
            "groups": groups}
