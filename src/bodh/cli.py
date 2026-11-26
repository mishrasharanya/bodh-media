"""Offline commands for the first Bodh research workflows."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bodh.authenticity import PROTOCOLS, asvspoof_assets
from bodh.manifest import COLUMNS, write_csv
from bodh.perception import describe_trials


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bodh", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    asvspoof = commands.add_parser("asvspoof-manifest", help="Sample an official ASVspoof LA split")
    asvspoof.add_argument("root", type=Path)
    asvspoof.add_argument("manifest", type=Path)
    asvspoof.add_argument("--split", choices=PROTOCOLS, default="train")
    asvspoof.add_argument("--per-condition", type=positive, default=100)
    asvspoof.add_argument("--seed", type=int, default=2026)

    eda = commands.add_parser("eda", help="Extract local audio features and summarize groups")
    eda.add_argument("manifest", type=Path)
    eda.add_argument("root", type=Path)
    eda.add_argument("out", type=Path)
    eda.add_argument("--no-plots", action="store_true")

    perception = commands.add_parser("perception", help="Summarize observed listener trials")
    perception.add_argument("manifest", type=Path)
    perception.add_argument("trials", type=Path)

    try:
        args = parser.parse_args(argv)
        if args.command == "asvspoof-manifest":
            if args.manifest.exists() or args.manifest.is_symlink():
                raise ValueError(f"Manifest already exists: {args.manifest}")
            assets = asvspoof_assets(args.root, args.split, args.per_condition, args.seed)
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            write_csv(args.manifest, COLUMNS, map(asdict, assets))
            print(f"Wrote {len(assets)} assets to {args.manifest}")
            return 0
        if args.command == "perception":
            print(json.dumps(describe_trials(args.manifest, args.trials), indent=2))
            return 0
        from bodh.eda import run_eda

        report = run_eda(args.manifest, args.root, args.out, plot=not args.no_plots)
        print(f"Processed {report['n_processed']}/{report['n_requested']} clips. See {args.out}")
        return int(report["n_failed"] > 0)
    except (OSError, ValueError) as error:
        parser.exit(2, f"bodh: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
