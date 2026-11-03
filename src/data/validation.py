"""Validate the original H2 CSV without modifying its contents."""

import csv
import hashlib
from pathlib import Path


def validate_source(path, dataset_config):
    """Require expected column names, row widths, and record count."""
    path = Path(path)
    expected = dataset_config["columns"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise ValueError("CSV is empty.")
        if len(header) != len(expected) or set(header) != set(expected):
            missing = sorted(set(expected) - set(header))
            extra = sorted(set(header) - set(expected))
            raise ValueError(
                f"Expected {len(expected)} unique H2 columns; got {len(header)}. "
                f"Missing: {missing}. Unexpected: {extra}."
            )
        rows = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"CSV line {line_number} has {len(row)} fields; expected {len(header)}."
                )
            rows += 1
    if rows != dataset_config["expected_rows"]:
        raise ValueError(
            f"Expected {dataset_config['expected_rows']} records; found {rows}."
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if dataset_config.get("sha256") and digest.hexdigest() != dataset_config["sha256"]:
        raise ValueError("Source SHA-256 differs from the verified project snapshot. Inspect the mirror before changing the expected checksum.")
    return {
        "path": str(path.resolve()),
        "rows": rows,
        "columns": len(header),
        "sha256": digest.hexdigest(),
        "structural_validation": "passed",
        "note": "EDA and semantic validation are still required.",
    }
