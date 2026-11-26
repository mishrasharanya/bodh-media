"""Calibrate listener groups and resample observed audio ratings on held-out speakers."""

import argparse
import json
from pathlib import Path

import numpy as np

from bodh.eda import sha256
from bodh.perception import read_listener_trials

ENDPOINTS = ("realness", "perceived_trustworthiness")
CONDITIONS = ("human_recording", "voice_clone")
SEED = 2026
DRAWS = 5000


def rating_cube(manifest: Path, trials: Path) -> tuple[np.ndarray, list[str], dict]:
    records = {}
    by_pair = {}
    for asset, arm, listener, endpoint, value in read_listener_trials(manifest, trials):
        if (asset.dataset, arm) != ("lavan2025", "experiment_1b"):
            raise ValueError("Expected the Lavan Experiment 1b listener panel")
        if (endpoint not in ENDPOINTS or asset.condition not in CONDITIONS
                or not asset.speaker_id or not 0 <= value <= 100):
            raise ValueError(f"Invalid rating or stimulus metadata for {asset.asset_id}")
        pair = (asset.speaker_id, asset.condition)
        if by_pair.setdefault(pair, asset.asset_id) != asset.asset_id:
            raise ValueError(f"Expected one stimulus per voice type for {asset.speaker_id}")
        records[endpoint, asset.speaker_id, listener, asset.condition] = value
    speakers = sorted({speaker for speaker, _ in by_pair})
    listeners = sorted({listener for _, _, listener, _ in records})
    if len(listeners) < 4 or len(speakers) < 4 or len(by_pair) != 2 * len(speakers):
        raise ValueError("Expected complete stimulus pairs and enough listeners to form two groups")
    axes = (ENDPOINTS, speakers, listeners, CONDITIONS)
    positions = [{key: index for index, key in enumerate(axis)} for axis in axes]
    values = np.full(tuple(map(len, axes)), np.nan)
    for key, rating in records.items():
        index = tuple(axis[value] for axis, value in zip(positions, key))
        values[index] = rating
    return values, speakers, by_pair


def paired_summary(differences: np.ndarray, panel_size: int, rng: np.random.Generator) -> dict:
    # Average each listener across their observed pairs, then sample whole listeners.
    # The speaker set stays fixed when resampling panels of the requested size.
    complete = np.isfinite(differences)
    if not complete.any(axis=0).all() or not complete.any(axis=1).all():
        raise ValueError("Each held-out speaker and listener needs at least one complete rating pair")
    listener_means = np.nanmean(differences, axis=0)
    draws = rng.choice(listener_means, size=(DRAWS, panel_size), replace=True).mean(axis=1)
    return {
        "paired_ratings": int(complete.sum()),
        "clone_higher_share": float((differences[complete] > 0).mean()),
        "tied_share": float((differences[complete] == 0).mean()),
        "observed_mean_clone_minus_human": float(listener_means.mean()),
        "resampled_panel_mean_middle_80_percent": np.quantile(draws, [0.10, 0.90]).tolist(),
        "resampled_panel_mean_above_zero_share": float((draws > 0).mean()),
    }


def mean_absolute_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.mean(np.abs(observed - expected)))


def compare_pair(ratings: np.ndarray, panel_size: int, rng: np.random.Generator) -> dict:
    paired = ratings[np.isfinite(ratings).all(axis=1)]
    if not len(paired):
        raise ValueError("The selected pair has no complete ratings in this listener group")
    difference = paired[:, 1] - paired[:, 0]
    return {
        "observed_means": dict(zip(CONDITIONS, paired.mean(axis=0).tolist())),
        "rated_higher": {"human_recording": int((difference < 0).sum()),
                         "voice_clone": int((difference > 0).sum()), "tie": int((difference == 0).sum())},
        "excluded_incomplete_listeners": len(ratings) - len(paired),
        **paired_summary(difference[None, :], panel_size, rng),
    }


def run(root: Path, out: Path, panel_size: int, speaker_pair: str | None = None) -> dict:
    if panel_size < 1:
        raise ValueError("Panel size must be positive")
    manifest, trials = root / "manifest.csv", root / "trials.csv"
    values, speakers, clips = rating_cube(manifest, trials)
    rng = np.random.default_rng(SEED)
    calibration_ids, test_ids = np.array_split(rng.permutation(len(speakers)), 2)
    calibration, test = values[:, calibration_ids], values[:, test_ids]
    held_out_speakers = [speakers[i] for i in test_ids]
    if speaker_pair is not None and speaker_pair not in held_out_speakers:
        raise ValueError(f"Choose a held-out speaker pair from {', '.join(sorted(held_out_speakers))}")
    human_realness = calibration[ENDPOINTS.index("realness"), :, :, CONDITIONS.index("human_recording")]
    if not np.isfinite(human_realness).any(axis=0).all():
        raise ValueError("Every listener needs a human-recording realness rating for calibration")
    calibration_scores = np.nanmean(human_realness, axis=0)
    ordered = np.argsort(calibration_scores, kind="stable")
    lower, higher = np.array_split(ordered, 2)
    cohorts = {"lower_calibration_realness": lower, "higher_calibration_realness": higher}
    pooled = np.nanmean(np.nanmean(calibration, axis=2), axis=1)

    results = {}
    for name, listeners in cohorts.items():
        cohort_calibration = np.nanmean(calibration[:, :, listeners], axis=2)
        cohort_test = np.nanmean(test[:, :, listeners], axis=2)
        estimates = cohort_calibration.mean(axis=1)
        if not np.isfinite(estimates).all() or not np.isfinite(cohort_test).all():
            raise ValueError(f"Missing cohort ratings for a stimulus: {name}")
        results[name] = {
            "listeners": len(listeners),
            "calibration_human_realness": float(calibration_scores[listeners].mean()),
            "endpoints": {},
        }
        for i, endpoint in enumerate(ENDPOINTS):
            group_values = test[i][:, listeners]
            results[name]["endpoints"][endpoint] = {
                "observed_held_out_means": dict(zip(CONDITIONS, cohort_test[i].mean(axis=0).tolist())),
                "held_out_clip_mean_mae": {
                    "cohort_calibration": mean_absolute_error(cohort_test[i], estimates[i]),
                    "pooled_calibration": mean_absolute_error(cohort_test[i], pooled[i]),
                },
                **paired_summary(group_values[:, :, 1] - group_values[:, :, 0], panel_size, rng),
            }
    report = {
        "study": "Lavan et al. (2025), Experiment 1b", "seed": SEED,
        "manifest_sha256": sha256(manifest), "trials_sha256": sha256(trials),
        "calibration_speaker_pairs": [speakers[i] for i in calibration_ids],
        "held_out_speaker_pairs": held_out_speakers,
        "calibration_rule": "Equal groups ranked by mean human-recording realness on calibration speakers",
        "ties_at_cohort_boundary": bool(calibration_scores[lower[-1]] == calibration_scores[higher[0]]),
        "panel_resampling": {"listeners_per_draw": panel_size, "draws": DRAWS, "replacement": True,
                             "fixed_stimuli": "held-out speaker pairs"},
        "cohorts": results,
    }
    if speaker_pair is not None:
        selected = values[:, speakers.index(speaker_pair)]
        scenario_rng = np.random.default_rng(SEED)
        report["voice_comparison"] = {
            "speaker_pair": speaker_pair,
            "stimulus_ids": {condition: clips[speaker_pair, condition] for condition in CONDITIONS},
            "cohorts": {
                name: {endpoint: compare_pair(selected[i, listeners], panel_size, scenario_rng)
                       for i, endpoint in enumerate(ENDPOINTS)}
                for name, listeners in cohorts.items()
            },
        }
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Folder containing the Lavan 1b manifest and trials")
    parser.add_argument("out", type=Path)
    parser.add_argument("--panel-size", type=int, default=10)
    parser.add_argument("--speaker-pair", help="Compare both recordings from one held-out speaker pair")
    args = parser.parse_args()
    run(args.experiment, args.out, args.panel_size, args.speaker_pair)
    print(f"Saved the held-out audience report to {args.out / 'report.json'}")
