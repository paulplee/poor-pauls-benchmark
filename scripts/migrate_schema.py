"""One-off schema migration: regenerate flat CSVs from raw JSONL results.

DO NOT run this until all benchmark machines have finished their current runs.

What it does
------------
Re-flattens every ``results/*.jsonl`` file using the current (updated)
flattener, which now populates the following new columns:

    model_org           — e.g. "unsloth"  (parsed from existing model field)
    model_repo          — e.g. "unsloth/Qwen3.5-2B-GGUF"
    gpu_count           — number of GPUs  (1 for all historical runs)
    gpu_names           — comma-joined GPU names  (same as gpu_name for 1 GPU)
    gpu_total_vram_gb   — sum of VRAM across all GPUs
    split_mode          — null for historical runs (not used)
    tensor_split        — null for historical runs (not used)

The raw ``*.jsonl`` files are the source of truth and are NOT modified.
Only the flat ``*.csv`` files are overwritten.

Safety
------
The script refuses to run unless ``results/backup/`` exists (the user
has already made a backup of the originals).

Usage
-----
    uv run scripts/migrate_schema.py [--dry-run]

After verifying the output looks correct, re-publish to Hugging Face:

    uv run ppb.py publish results/<name>.jsonl --upload

for each results file, or use `ppb all` with the existing suite configs.
Then delete this script and results/backup/ once HF is verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.flattener import COLUMN_ORDER, flatten_benchmark_row  # noqa: E402


def _find_results_dir() -> Path:
    results = _ROOT / "results"
    if not results.is_dir():
        print(f"ERROR: results directory not found at {results}", file=sys.stderr)
        sys.exit(1)
    return results


def _check_backup_exists(results: Path) -> None:
    backup = results / "backup"
    if not backup.is_dir():
        print(
            "ERROR: results/backup/ does not exist.\n"
            "Please create a backup of all result files before running this script:\n"
            "  mkdir results/backup && cp results/*.* results/backup/",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=COLUMN_ORDER,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def migrate(results: Path, dry_run: bool) -> None:
    jsonl_files = sorted(results.glob("*.jsonl"))
    if not jsonl_files:
        print("No *.jsonl files found in results/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL file(s) to migrate.\n")

    total_rows = 0
    for jsonl_path in jsonl_files:
        flat_rows: list[dict] = []
        with jsonl_path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"  WARNING: {jsonl_path.name}:{lineno} — "
                        f"JSON parse error: {exc}",
                        file=sys.stderr,
                    )
                    continue
                flat_rows.extend(flatten_benchmark_row(raw))

        csv_path = jsonl_path.with_suffix(".csv")

        # Spot-check: print first row's new fields for verification
        if flat_rows:
            sample = flat_rows[0]
            new_fields = {
                k: sample.get(k)
                for k in (
                    "model_org",
                    "model_repo",
                    "gpu_count",
                    "gpu_names",
                    "gpu_total_vram_gb",
                    "split_mode",
                    "tensor_split",
                )
            }
            print(f"  {jsonl_path.name}: {len(flat_rows)} row(s)")
            print(f"    new fields: {new_fields}")

        if dry_run:
            print(f"    [dry-run] would write {csv_path.name}")
        else:
            _write_csv(flat_rows, csv_path)
            print(f"    wrote {csv_path.name}")

        total_rows += len(flat_rows)

    print(
        f"\n{'[dry-run] ' if dry_run else ''}"
        f"Migration complete — {total_rows} total flat rows across "
        f"{len(jsonl_files)} file(s)."
    )
    if dry_run:
        print("Re-run without --dry-run to overwrite CSV files.")
    else:
        print(
            "\nNext steps:\n"
            "  1. Spot-check a CSV file to confirm new columns look correct.\n"
            "  2. Delete all files in paulplee/ppb-results/data/ on Hugging Face.\n"
            "  3. Re-upload with:  uv run ppb.py publish results/<name>.jsonl --upload\n"
            "  4. When HF is verified, delete scripts/migrate_schema.py and results/backup/"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing any files.",
    )
    args = parser.parse_args()

    results = _find_results_dir()
    _check_backup_exists(results)
    migrate(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
