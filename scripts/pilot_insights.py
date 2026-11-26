"""Compare observed listener ratings and quiet audio across Bodh's two pilot datasets."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

from bodh.audio import HOP, WINDOW, load_audio

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PILOTS = {
    "Behavioral Signals": ("speech-pilot", "behavioral-signals-demo", "bonafide", "spoofed"),
    "Lavan 1b": ("lavan-1b", "lavan2025", "human_recording", "voice_clone"),
}
ENDPOINTS = ("realness", "perceived_trustworthiness")


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rating_pairs(features: list[dict], trials: list[dict], endpoint: str) -> tuple:
    groups = sorted({row["source_group"] for row in features})
    listeners = sorted({row["participant_id"] for row in trials if row["endpoint"] == endpoint})
    positions = {key: index for index, key in enumerate(listeners)}
    estimates = defaultdict(lambda: np.full(len(listeners), np.nan))
    for row in trials:
        if row["endpoint"] == endpoint:
            estimates[row["asset_id"]][positions[row["participant_id"]]] = float(row["value"])
    by_group = {(row["source_group"], row["condition"]): row["asset_id"] for row in features}
    human, clone = [
        np.stack([estimates[by_group[group, condition]] for group in groups])
        for condition in ("human_recording", "voice_clone")
    ]
    return human, clone


def describe_ratings(human: np.ndarray, clone: np.ndarray) -> tuple[dict, np.ndarray]:
    differences = np.nanmean(clone, axis=1) - np.nanmean(human, axis=1)
    rng = np.random.default_rng(20261125)
    draws = []
    for _ in range(5000):
        speakers = rng.integers(human.shape[0], size=human.shape[0])
        listeners = rng.integers(human.shape[1], size=human.shape[1])
        sample = np.ix_(speakers, listeners)
        draws.append(np.nanmean(clone[sample], axis=1).mean()
                     - np.nanmean(human[sample], axis=1).mean())
    return {
        "pairs": len(differences),
        "mean_clone_minus_human": differences.mean(),
        "bootstrap_95_percent_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "pairs_higher_for_clone": int((differences > 0).sum()),
        "pairs_at_least_10_points_apart": int((np.abs(differences) >= 10).sum()),
        "min_max_difference": [differences.min(), differences.max()],
    }, differences


def quiet_differences(features: list[dict], audio_root: Path, human: str, synthetic: str) -> dict:
    """Compare full recordings with frames between the first and last active frame."""
    paired = defaultdict(dict)
    for row in features:
        sound = load_audio(audio_root / row["path"]).mono
        frames = np.lib.stride_tricks.sliding_window_view(sound, WINDOW)[::HOP]
        quiet = np.mean(frames ** 2, axis=1) < 10 ** (-50 / 10)
        active = np.flatnonzero(~quiet)
        if not len(active):
            raise ValueError(f"No active frames in {row['asset_id']}")
        middle = quiet[active[0]:active[-1] + 1]
        paired[row["source_group"]][row["condition"]] = np.array([quiet.mean(), middle.mean()])
    diffs = 100 * np.stack([pair[human] - pair[synthetic] for pair in paired.values()])
    return {
        label: {
            "n_pairs": len(diffs),
            "median_human_minus_synthetic_pp": np.median(diffs[:, index]),
            "pairs_higher_for_human": int((diffs[:, index] > 0).sum()),
        }
        for index, label in enumerate(("whole_clip", "between_active_frames"))
    }


def run(out: Path) -> None:
    acoustic = {}
    for name, (run_folder, audio_folder, human, synthetic) in PILOTS.items():
        acoustic[name] = quiet_differences(
            rows(Path("runs") / run_folder / "features.csv"),
            Path("data") / audio_folder, human, synthetic
        )
    features = rows(Path("runs/lavan-1b/features.csv"))
    trials = rows(Path("data/lavan2025/experiment1b/trials.csv"))
    summaries, differences = {}, {}
    for endpoint in ENDPOINTS:
        summaries[endpoint], differences[endpoint] = describe_ratings(
            *rating_pairs(features, trials, endpoint)
        )

    out.mkdir(parents=True)
    report = {
        "listener_ratings": summaries,
        "bootstrap": "5,000 resamples of listener IDs and speaker pairs with seed 20261125",
        "quiet_frames": acoustic,
        "quiet_definition": "25 ms frames with RMS below −50 dBFS and a 10 ms hop",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    figure, (ratings, quiet) = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
    colors = ("#00756d", "#ad5429")
    rng = np.random.default_rng(20261125)
    for position, (endpoint, color) in enumerate(zip(ENDPOINTS, colors)):
        values = differences[endpoint]
        ratings.scatter(values, position + rng.uniform(-0.16, 0.16, len(values)),
                        color=color, s=18, alpha=0.72)
        stats = summaries[endpoint]
        low, high = stats["bootstrap_95_percent_interval"]
        mean = stats["mean_clone_minus_human"]
        ratings.errorbar(mean, position + 0.27, xerr=[[mean - low], [high - mean]],
                         color=color, fmt="D", capsize=3, markersize=4)
    ratings.axvline(0, color="#786e66", linewidth=1)
    ratings.set(yticks=[0, 1], yticklabels=["Realness", "Perceived\ntrustworthiness"],
                ylim=(1.48, -0.32), xlabel="Clone minus human (0–100 rating points)",
                title="Lavan 1b · 40 speaker pairs")
    for position, (name, summary) in enumerate(acoustic.items()):
        whole, middle = (
            summary[key]["median_human_minus_synthetic_pp"]
            for key in ("whole_clip", "between_active_frames")
        )
        vertical = (position - 0.04, position + 0.04)
        quiet.plot([whole, middle], vertical, color="#9d948c")
        for value, height, color, marker, label in zip(
            (whole, middle), vertical, colors, ("o", "s"), ("Whole clip", "Between active frames")
        ):
            quiet.scatter(value, height, color=color, marker=marker, s=48,
                          label=label if position == 0 else None, zorder=3)
    quiet.axvline(0, color="#786e66", linewidth=1)
    quiet.set(yticks=[0, 1], yticklabels=["Behavioral\nSignals", "Lavan 1b"],
              ylim=(1.48, -0.40), xlabel="Median human minus synthetic (percentage points)",
              title="Quiet-frame gap")
    quiet.legend(loc="lower right", frameon=False)
    for ax in (ratings, quiet):
        ax.grid(axis="x", alpha=0.2)
    figure.savefig(out / "pilot_insights.png", dpi=175)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    run(parser.parse_args().out)
