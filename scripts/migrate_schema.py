"""One-off schema migration: regenerate flat CSVs from raw JSONL results.

DO NOT run this until all benchmark machines have finished their current runs.

What it does
------------
Re-flattens every ``results/*.jsonl`` file using the current (updated)
flattener.  For each row the script:

  1. Adds missing columns that did not exist in older schema versions
     (model_org, model_repo, gpu_count, gpu_names, gpu_total_vram_gb,
     split_mode, tensor_split, llm_engine_name, llm_engine_version,
     os_distro, os_distro_version, task_type, prompt_dataset,
     num_prompts, n_predict, quality_score, tags).
     Values are derived from existing raw data where possible.
  2. Preserves columns that already exist in newer results — the flattener
     reads them from the raw JSONL record and keeps them as-is.
  3. Resets ``schema_version`` to ``0.1.0`` on every row (the canonical
     version aligned with ``benchmark_version`` / ``pyproject.toml``).

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
import re
import sys
from pathlib import Path

# Ensure project root is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.flattener import COLUMN_ORDER, _SCHEMA_VERSION, flatten_benchmark_row  # noqa: E402


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


# ---------------------------------------------------------------------------
# Backfill helpers — enrich raw JSONL records before flattening
# ---------------------------------------------------------------------------

# Regex to parse OS distro from kernel version string.
# Example: "#8-Ubuntu SMP PREEMPT_DYNAMIC" → "Ubuntu"
_DISTRO_RE = re.compile(
    r"(?i)\b(ubuntu|debian|fedora|centos|rhel|arch|suse|dgx\s*os|pop!?_os|manjaro|alpine)"
)


def _parse_engine_version(raw: dict) -> str | None:
    """Extract llama.cpp version from the runtime info in the raw record.

    Typical value in raw: ``"version: b5063 (58ab80c3)"``
    Returns:              ``"b5063 (58ab80c3)"``
    """
    runtime = (raw.get("hardware") or {}).get("runtime") or {}
    ver_str = runtime.get("llama_bench", "")
    if not ver_str:
        return None
    m = re.search(r"version:\s*(.+)", ver_str, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ver_str.strip() or None


def _parse_os_distro(raw: dict) -> str | None:
    """Best-effort extraction of OS distro from kernel version string."""
    os_info = (raw.get("hardware") or {}).get("os") or {}
    # If the record already has distro info (from newer runs), return as-is
    if os_info.get("distro"):
        return os_info["distro"]
    # Fall back to heuristic on the version string
    version = os_info.get("version", "")
    system = os_info.get("system", "")
    if system == "Darwin":
        return "macOS"
    if system == "Windows":
        return "Windows"
    m = _DISTRO_RE.search(version)
    if m:
        return m.group(1).strip()
    return None


def _backfill_record(raw: dict) -> dict:
    """Inject backfill fields into a raw JSONL record before flattening.

    Only sets keys that are absent — newer records that already contain
    these fields are left untouched.
    """
    runner_type = raw.get("runner_type", "")

    # LLM engine identity
    if "llm_engine_name" not in raw:
        raw["llm_engine_name"] = "llama.cpp"
    if "llm_engine_version" not in raw:
        raw["llm_engine_version"] = _parse_engine_version(raw)

    # OS distro (inject into hardware.os so _extract_hardware picks it up)
    os_info = (raw.get("hardware") or {}).get("os")
    if os_info is not None and "distro" not in os_info:
        distro = _parse_os_distro(raw)
        if distro:
            os_info["distro"] = distro
        # distro_version cannot be recovered from existing data

    # Task type
    if "task_type" not in raw:
        raw["task_type"] = "text-generation"

    # Prompt dataset
    if "prompt_dataset" not in raw:
        if runner_type in ("llama-server", "llama-server-loadtest"):
            raw["prompt_dataset"] = "sharegpt-v3"

    # quality_score and tags default to None (handled by flattener)
    return raw


def migrate(results: Path, dry_run: bool) -> None:
    jsonl_files = sorted(results.glob("*.jsonl"))
    if not jsonl_files:
        print("No *.jsonl files found in results/", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL file(s) to migrate.")
    print(f"Target schema_version: {_SCHEMA_VERSION}\n")

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
                raw = _backfill_record(raw)
                flat_rows.extend(flatten_benchmark_row(raw))

        csv_path = jsonl_path.with_suffix(".csv")

        # Spot-check: print first row's new fields for verification
        if flat_rows:
            sample = flat_rows[0]
            new_fields = {
                k: sample.get(k)
                for k in (
                    "schema_version",
                    "model_org",
                    "model_repo",
                    "gpu_count",
                    "gpu_names",
                    "gpu_total_vram_gb",
                    "split_mode",
                    "tensor_split",
                    "llm_engine_name",
                    "llm_engine_version",
                    "os_distro",
                    "os_distro_version",
                    "task_type",
                    "prompt_dataset",
                    "num_prompts",
                    "n_predict",
                    "quality_score",
                    "tags",
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
