"""Summarize a local audio manifest and report acquisition confounds."""

import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np

from bodh import __version__, audio
from bodh.manifest import read_assets, write_csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt


METRICS = {
    "duration_s": "Duration (s)",
    "mono_rms_dbfs": "Mono RMS (dBFS)",
    "low_energy_frame_fraction": "Low-energy frame fraction",
    "active_spectral_centroid_median_hz": "Active spectral centroid (Hz)",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def group_report(groups: dict) -> list[dict]:
    reports = []
    for (dataset, split, condition), rows in sorted(groups.items()):
        report = {"dataset": dataset, "split": split, "condition": condition, "n": len(rows)}
        for field in ("source_group", "speaker_id", "generator_id", "source_sample_rate_hz",
                      "source_channels", "source_codec"):
            report[field + "_counts"] = dict(sorted(Counter(row[field] for row in rows).items()))
        for field in (*METRICS, "source_near_full_scale_fraction"):
            values = [row[field] for row in rows if row[field] is not None]
            quantiles = np.quantile(values, [0.25, 0.5, 0.75]).tolist() if values else [None] * 3
            report[field] = dict(zip(("q25", "median", "q75"), quantiles))
            report[field]["n"] = len(values)
        reports.append(report)
    return reports


def split_audit(rows: list[dict]) -> dict:
    result = {}
    for field in ("source_group", "speaker_id", "generator_id", "sha256"):
        splits = defaultdict(set)
        for row in rows:
            value = row[field]
            if value and row["split"] != "unsplit":
                dataset = None if field == "sha256" else row["dataset"]
                splits[dataset, value].add((row["dataset"], row["split"]))
        result[field + "_across_splits"] = sum(len(values) > 1 for values in splits.values())
    result["repeated_files"] = len(rows) - len({row["sha256"] for row in rows})
    return result


def plot_groups(groups: dict, destination: Path) -> None:
    ordered = sorted(groups.items())
    contexts = {(dataset, split) for dataset, split, _ in groups}
    labels = []
    for (dataset, split, condition), rows in ordered:
        context = f"{dataset.replace('_', ' ')} / {split} / " if len(contexts) > 1 else ""
        name = condition.replace("bonafide", "bona fide").replace("_", " ").capitalize()
        labels.append(f"{context}{name} (n={len(rows)})")
    figure, axes = plt.subplots(
        2, 2, figsize=(10, max(5, 1.1 * len(groups))), layout="constrained", sharey=True
    )
    for ax, (field, title) in zip(axes.flat, METRICS.items()):
        for index, (_, rows) in enumerate(ordered, 1):
            values = [row[field] for row in rows if row[field] is not None]
            if values:
                ax.boxplot([values], positions=[index], vert=False, manage_ticks=False,
                           showfliers=len(values) > 50,
                           medianprops={"color": "#006963", "linewidth": 1.7})
                if len(values) <= 50:
                    jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else [0]
                    ax.scatter(values, index + np.asarray(jitter), s=13, alpha=0.6,
                               color="#006963", zorder=3)
        ax.set_xlabel(title)
        ax.set_yticks(range(1, len(labels) + 1), labels)
        ax.set_ylim(len(labels) + 0.5, 0.5)
        ax.grid(axis="x", alpha=0.25)
    title = "Audio characteristics by dataset, split and condition"
    if len(contexts) == 1:
        dataset, split = next(iter(contexts))
        title = f"{dataset.replace('_', ' ').capitalize()} · {split} split"
    figure.suptitle(title)
    figure.savefig(destination, dpi=145)
    plt.close(figure)


def plot_example(log_mel: np.ndarray, destination: Path) -> None:
    figure, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    seconds = log_mel.shape[0] * audio.HOP / audio.SAMPLE_RATE
    image = ax.imshow(log_mel.T, origin="lower", aspect="auto", interpolation="nearest",
                      extent=(0, seconds, 0, audio.MEL_BANDS), vmin=-100, vmax=0)
    ax.set(xlabel="Frame start time (s)", ylabel="Mel band",
           title="First processed clip")
    figure.colorbar(image, ax=ax, label="Log-mel power (dB)")
    figure.savefig(destination, dpi=145)
    plt.close(figure)


def run_eda(manifest: Path, root: Path, out: Path, *, plot: bool = True) -> dict:
    if out.exists() or out.is_symlink():
        raise ValueError(f"Output directory already exists: {out}")
    assets = read_assets(manifest)
    if any(asset.modality != "audio" for asset in assets):
        raise ValueError("Audio EDA requires a manifest containing only audio assets")
    root = root.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    all_rows = []
    errors = []
    with TemporaryDirectory(prefix=".eda-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        first_clip = None
        for asset in assets:
            path = (root / asset.path).resolve()
            try:
                if not path.is_relative_to(root):
                    raise ValueError("Audio path resolves outside the audio root")
                decoded = audio.load_audio(path)
                summary, log_mel = audio.features(decoded)
                row = {**asdict(asset), "sha256": sha256(path), **summary}
            except (OSError, ValueError) as error:
                errors.append({"dataset": asset.dataset, "asset_id": asset.asset_id,
                               "error": str(error).replace(str(root), "<audio-root>")})
                continue
            groups[asset.dataset, asset.split, asset.condition].append(row)
            all_rows.append(row)
            if first_clip is None:
                first_clip = asset.asset_id
                if plot:
                    plot_example(log_mel, stage / "example_log_mel.png")
        write_csv(stage / "errors.csv", ("dataset", "asset_id", "error"), errors)
        if all_rows:
            write_csv(stage / "features.csv", tuple(all_rows[0]), all_rows)
            if plot:
                plot_groups(groups, stage / "distributions.png")
        report = {
            "manifest_sha256": sha256(manifest), "bodh_version": __version__,
            "versions": {
                "python": platform.python_version(),
                "ffmpeg": audio.run_media(["ffmpeg", "-version"]).decode().splitlines()[0],
                **{name: version(name) for name in ("numpy", "scipy", "matplotlib")},
            },
            "processing": {
                "sample_rate_hz": audio.SAMPLE_RATE, "maximum_clip_seconds": audio.MAX_SECONDS,
                "window_samples": audio.WINDOW, "hop_samples": audio.HOP, "fft_size": audio.FFT_SIZE,
                "mel_bands": audio.MEL_BANDS, "mel_min_hz": audio.MEL_MIN_HZ,
                "low_energy_dbfs": audio.LOW_ENERGY_DBFS, "power_floor": audio.POWER_FLOOR,
            },
            "n_requested": len(assets), "n_processed": len(all_rows), "n_failed": len(errors),
            "requested_by_group": [
                {"dataset": dataset, "split": split, "condition": condition, "n": count}
                for (dataset, split, condition), count in sorted(Counter(
                    (asset.dataset, asset.split, asset.condition) for asset in assets
                ).items())
            ],
            "example_asset_id": first_clip if plot else None,
            "split_audit": split_audit(all_rows), "groups": group_report(groups),
        }
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        stage.rename(out)
    return report
