#!/usr/bin/env python3
"""Convert benchmark tmp_results JSONL files into a single CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, object]]:
    """Yield JSON records from a .jsonl file.

    Empty lines are ignored to make the script resilient to manual edits.
    """

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}\n{raw_line}"
                ) from exc


def collect_records(input_dir: Path) -> List[Dict[str, object]]:
    """Read every *.jsonl file in *input_dir* and return all the records."""

    records: List[Dict[str, object]] = []
    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        for record in iter_jsonl_records(jsonl_path):
            # Keep track of the source file for traceability.
            if "source_file" not in record:
                record = {"source_file": jsonl_path.name, **record}
            records.append(record)
    return records


def determine_fieldnames(records: Iterable[Dict[str, object]]) -> List[str]:
    """Compute ordered CSV fieldnames from the union of all record keys."""

    fieldnames: List[str] = []
    for record in records:
        for key in record.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv(records: List[Dict[str, object]], output_path: Path) -> None:
    """Write *records* into *output_path* as CSV."""

    if not records:
        raise ValueError("No JSONL records found in the input directory.")

    fieldnames = determine_fieldnames(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL benchmark outputs into a single CSV file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the directory that contains .jsonl files (e.g., tmp_results).",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        default=Path("tmp_results.csv"),
        help="Destination CSV file (default: ./tmp_results.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_csv: Path = args.output_csv

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist or is not a directory.")

    records = collect_records(input_dir)
    write_csv(records, output_csv)
    print(f"Wrote {len(records)} rows to {output_csv}")


if __name__ == "__main__":
    main()