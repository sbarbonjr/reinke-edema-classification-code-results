#!/usr/bin/env python3
"""Check the public R1 package inventory, hashes, and aggregate tables."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = {
    "results_r1_nested/aggregate_metrics.csv": {"feature", "model", "n_repeats", "balanced_accuracy_mean"},
    "results_r1_nested/optimizer_search/optimizer_aggregate_metrics.csv": {"feature", "method", "balanced_accuracy_mean"},
    "results_r1_fixed_relu/aggregate_metrics.csv": {"feature", "model", "n_repeats", "balanced_accuracy_mean"},
    "results_r1_fixed_relu_multiseed/aggregate_metrics.csv": {"feature", "model", "n_seeds", "balanced_accuracy_mean_across_seeds"},
    "results_r1_fixed_relu_multiseed/seed_aggregate_metrics.csv": {"feature", "model", "seed", "n_repeats", "balanced_accuracy_mean"},
}
PROHIBITED_COLUMNS = {"group_id", "filename", "file", "label", "probability", "prediction", "segment_id"}


def main() -> None:
    manifest = ROOT / "SHA256SUMS"
    recorded = {}
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or path.resolve().is_relative_to(ROOT.resolve()) is False:
            raise AssertionError(f"Missing or invalid file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {relative}")
        recorded[relative] = expected

    actual_files = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
        and not any(part.startswith(".") and part != ".gitignore" for part in path.relative_to(ROOT).parts)
        and "__pycache__" not in path.parts
        and "reproduced_r1" not in path.parts
        and "figures" not in path.parts
        and path.name != "SHA256SUMS"
    }
    if actual_files != set(recorded):
        raise AssertionError(f"Package inventory mismatch: {sorted(actual_files ^ set(recorded))}")
    for relative in actual_files:
        if Path(relative).suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}:
            raise AssertionError(f"Audio file in package: {relative}")
        if "features_win" in Path(relative).parts or "predictions" in relative or "manifest" in relative:
            raise AssertionError(f"Clinical-level file in package: {relative}")

    for relative, required in EXPECTED_TABLES.items():
        with (ROOT / relative).open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if not required <= columns or PROHIBITED_COLUMNS & columns:
                raise AssertionError(f"Unexpected columns: {relative}")
            rows = list(reader)
            if not rows:
                raise AssertionError(f"Empty table: {relative}")
    print(f"Verified {len(recorded)} public files and {len(EXPECTED_TABLES)} aggregate tables.")


if __name__ == "__main__":
    main()
