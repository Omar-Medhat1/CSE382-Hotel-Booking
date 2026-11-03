"""Command-line entry point for the City Hotel booking project."""

import argparse
import json
import sys
from pathlib import Path

from src.data.validation import validate_source


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show current implementation status.")
    commands.add_parser("download", help="Download and verify the original H2 dataset.")
    commands.add_parser("analyze", help="Generate EDA and the fixed split manifest.")
    commands.add_parser("baseline", help="Run EDA and all Milestone 1 baseline experiments.")
    commands.add_parser("verify", help="Check partitions, artifacts, and reload predictions.")
    prediction = commands.add_parser("predict", help="Load a saved baseline for one booking.")
    prediction.add_argument("--task", choices=["regression", "classification"], required=True)
    prediction.add_argument("--input", type=Path, required=True, help="Booking JSON using original source names.")
    validate = commands.add_parser("validate-data", help="Validate H2 source structure.")
    validate.add_argument("--path", type=Path, help="Alternative CSV path.")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    if args.command == "status":
        print("Milestone 1: data preparation and untuned baselines")
        print("Results:", "available" if (ROOT / "results/run_summary.json").exists() else "run baseline")
        print("Final supervised test evaluation: intentionally unavailable in Milestone 1.")
        print("Team roster: docs/team.md; other GitHub accounts and contributions are pending.")
        print("Repository:", config["repository"])
        return 0

    if args.command in ("download", "analyze", "baseline", "verify", "predict"):
        try:
            if args.command == "download":
                from src.data.loading import download
                print(json.dumps(download(), indent=2))
            elif args.command == "analyze":
                from src.data.loading import load_source
                from src.data.splitting import partition
                from src.analysis.eda import analyze
                frame, _ = load_source()
                labels, manifest, metadata = partition(frame)
                analyze(frame, labels, ROOT / "results")
                manifest.to_csv(ROOT / "results/tables/split_manifest.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
                (ROOT / "results/split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print("Analysis and split manifest written to results/.")
            elif args.command == "baseline":
                from src.experiments import run
                run()
            elif args.command == "verify":
                from src.verification import verify
                print(json.dumps(verify(), indent=2))
            else:
                from src.inference import predict
                print(json.dumps(predict(args.task, args.input), indent=2))
        except (OSError, ValueError, ImportError) as error:
            print(f"Command failed: {error}", file=sys.stderr)
            return 1
        return 0

    source = args.path if args.path is not None else ROOT / config["dataset"]["path"]
    try:
        result = validate_source(source, config["dataset"])
    except (OSError, ValueError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
