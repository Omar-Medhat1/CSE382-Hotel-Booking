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
    validate = commands.add_parser("validate-data", help="Validate H2 source structure.")
    validate.add_argument("--path", type=Path, help="Alternative CSV path.")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    if args.command == "status":
        print("Milestone 1: initial setup")
        print("Available: project structure, configuration, source-data validator.")
        print("Pending: EDA, leakage audit, preprocessing, baseline experiments, report.")
        print("Team details and submitted commit SHA: pending.")
        print("Repository:", config["repository"])
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
