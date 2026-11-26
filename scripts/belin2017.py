"""Reanalyze the published Belin et al. listener ratings."""

import argparse
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_1samp

from bodh.eda import sha256
from bodh.manifest import Asset, COLUMNS, write_csv

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BLOCKS = ("trustworthiness", "control")


def estimate(values: np.ndarray) -> dict:
    test = ttest_1samp(values, 0, axis=0)
    ci = test.confidence_interval()
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci95_low": ci.low.tolist(), "ci95_high": ci.high.tolist(),
    }


def reanalyze(folder: Path, out: Path) -> None:
    archive = folder / "ratings.zip"
    with ZipFile(archive) as source:
        raw = loadmat(BytesIO(source.read("DATA.mat")))["DATA"]
    if raw.shape != (500, 9, 2) or not np.isfinite(raw).all() or not np.all((0 <= raw) & (raw <= 500)):
        raise ValueError("Unexpected published ratings shape or value range")
    # The first block's S3/S7 means match the published experimental values under block-wise z-scoring.
    within_block = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, ddof=1, keepdims=True)
    axes = (1, 2)
    across_blocks = (raw - raw.mean(axis=axes, keepdims=True)) / raw.std(axis=axes, ddof=1, keepdims=True)
    results = {}
    for label, responses in (("raw_0_to_500", raw), ("within_block_z", within_block),
                             ("across_blocks_z", across_blocks)):
        if not np.isfinite(responses).all():
            raise ValueError("Unexpected constant or missing listener responses")
        results[label] = {
            block: {
                "by_stimulus": estimate(responses[:, :, index]),
                "s7_minus_s3_paired": estimate(responses[:, 6, index] - responses[:, 2, index]),
            }
            for index, block in enumerate(BLOCKS)
        }
    assets = [
        Asset("belin2017", block, "audio", f"{block}.wav", "unsplit", block, f"{block}_montage")
        for block in BLOCKS
    ]
    hashes = {path.name: sha256(path) for path in (archive, *(folder / asset.path for asset in assets))}
    out.mkdir(parents=True)
    report = {
        "source": "https://doi.org/10.1371/journal.pone.0185651",
        "n_listeners": raw.shape[0], "n_stimuli": 18, "n_ratings": raw.size,
        "source_sha256": hashes, "results": results,
        "intervals": "Pointwise 95% t intervals across listeners, conditional on the fixed stimuli",
    }
    (out / "ratings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(out / "montage_manifest.csv", COLUMNS, map(asdict, assets))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for ax, method, ylabel in zip(axes, ("raw_0_to_500", "within_block_z"),
                                  ("Mean rating (0–500)", "Mean standardized rating")):
        for block, color, marker in zip(BLOCKS, ("#006963", "#8E4C79"), ("o", "s")):
            stats = results[method][block]["by_stimulus"]
            positions = np.arange(1, 10)
            ax.plot(positions, stats["mean"], label=block.capitalize(), color=color, marker=marker)
            ax.fill_between(positions, stats["ci95_low"], stats["ci95_high"], alpha=0.18, color=color)
        ax.set(xlabel="Stimulus position in each voice continuum", ylabel=ylabel, xticks=positions)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", frameon=False)
    figure.suptitle("Perceived trustworthiness across two voice continua\n"
                    "500 listeners · Shading shows pointwise 95% confidence intervals")
    figure.savefig(out / "listener_ratings.png", dpi=165)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    reanalyze(args.folder, args.out)
