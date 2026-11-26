"""Prepare a historical audience backtest from observed Lavan 1b realness ratings."""

import argparse
from pathlib import Path

import numpy as np

from listener_audience import CONDITIONS, ENDPOINTS, rating_cube
from product_audience import COLUMNS, save
from bodh.eda import sha256
from bodh.manifest import write_csv

SEED = 2026
THRESHOLD = 60


def prepare(root: Path, out: Path) -> None:
    manifest, trials = root / "manifest.csv", root / "trials.csv"
    data, speakers, _ = rating_cube(manifest, trials)
    rng = np.random.default_rng(SEED)
    screening, calibration, evaluation = np.array_split(rng.permutation(len(speakers)), 3)
    realness = data[ENDPOINTS.index("realness")]
    screened = realness[screening, :, CONDITIONS.index("human_recording")]
    if not np.isfinite(screened).any(axis=0).all():
        raise ValueError("Each listener needs an observed screening rating")
    ordered = np.argsort(np.nanmean(screened, axis=0), kind="stable")
    groups = dict(zip(("lower_screening_realness", "higher_screening_realness"), np.array_split(ordered, 2)))
    audiences = {
        "lower_screening_realness": "Listeners below the median rank on a separate set of human recordings",
        "higher_screening_realness": "Listeners above the median rank on a separate set of human recordings",
    }
    outcome = f"Recorded realness rating of at least {THRESHOLD} out of 100"
    out.mkdir(parents=True)
    study = out / "study.json"
    save(study, {"name": "Lavan 1b voice preview", "task": "Rate how real the recording sounds.",
                 "outcome": outcome, "choices": [outcome, f"Recorded realness rating below {THRESHOLD}"],
                 "target_choice": outcome, "audiences": audiences,
                 "variants": {"human_recording": "Human recording", "voice_clone": "Voice clone"}})
    fingerprint = sha256(study)
    output = {"calibration": [], "held_out": []}
    skipped = 0
    for audience, listeners in groups.items():
        halves = np.array_split(rng.permutation(listeners), 2)
        for (stage, speakers_for_stage), members in zip(
                (("calibration", calibration), ("held_out", evaluation)), halves):
            assigned = np.resize(rng.permutation(speakers_for_stage), len(members))
            for i, (listener, speaker) in enumerate(zip(members, assigned)):
                condition = i % len(CONDITIONS)
                observed = realness[speaker, listener, condition]
                if not np.isfinite(observed):
                    skipped += 1
                    continue
                output[stage].append(dict(zip(COLUMNS, (fingerprint, f"study_{listener:03}", audience,
                                                       CONDITIONS[condition], int(observed >= THRESHOLD)))))
    for stage, records in output.items():
        write_csv(out / f"{stage}.csv", COLUMNS, records)
    save(out / "provenance.json", {"manifest_sha256": sha256(manifest), "trials_sha256": sha256(trials),
                                    "seed": SEED, "rating_threshold": THRESHOLD,
                                    "missing_selected_ratings": skipped,
                                    "screening_speakers": [speakers[i] for i in screening],
                                    "calibration_speakers": [speakers[i] for i in calibration],
                                    "held_out_speakers": [speakers[i] for i in evaluation]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    prepare(args.experiment, args.out)
