"""Generate control tones with no speech or listener responses."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

from bodh.manifest import Asset, COLUMNS, write_csv


def generate(folder: Path) -> None:
    folder.mkdir(parents=True)
    assets = []
    for index, rate in enumerate((8_000, 16_000, 22_050, 48_000)):
        seconds = 1 + index / 5
        time = np.arange(round(rate * seconds)) / rate
        for condition, amplitude in (("reference", 0.5), ("reduced_gain", 0.05)):
            name = f"{index}_{condition}"
            wave = amplitude * np.sin(2 * np.pi * (440 + 20 * index) * time)
            write(folder / f"{name}.wav", rate, np.round(wave * 32767).astype(np.int16))
            assets.append(Asset(
                "signal_fixture", name, "audio", f"{name}.wav", "unsplit", str(index), condition
            ))
    write_csv(folder / "manifest.csv", COLUMNS, map(asdict, assets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    generate(parser.parse_args().folder)
