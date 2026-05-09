"""Backfill llm_flags fields into existing PPB result files.

This script adds the new schema v0.10.0 fields to JSONL result files that
were produced before flag-sweep support was added.  It is safe to run on
any PPB result file (local or downloaded from HuggingFace).

Backfilled values
-----------------
* ``llm_flags``      → ``"{}"``      (all llama.cpp defaults)
* ``llm_flags_label`` → ``"default"``
* ``extra_flags_raw`` → ``null``
* ``llm_engine_version`` — captured from the current binary if not already set.
  This is a best-effort approximation; the binary may differ from what was
  used at run time.  A note is added to the ``meta`` field.

Fingerprint preservation
------------------------
``row_id``, ``result_fingerprint``, ``run_fingerprint``, and
``machine_fingerprint`` are NOT modified.  Downstream deduplication stays
intact.

Usage
-----
    # Backfill local files (default: results/*.jsonl)
    uv run scripts/backfill_flags.py

    # Backfill specific files
    uv run scripts/backfill_flags.py results/my_run.jsonl

    # Flatten to CSV as well
    uv run scripts/backfill_flags.py --csv

    # Re-upload to HuggingFace after backfilling
    uv run scripts/backfill_flags.py --upload

    # Dry run (print what would change, don't write)
    uv run scripts/backfill_flags.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.flattener import flatten_benchmark_row

log = logging.getLogger("ppb.backfill")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"version:\s*(.+)", re.IGNORECASE)


def _detect_binary_version(binary: str) -> str | None:
    """Return the version string reported by *binary*, or None."""
    if not shutil.which(binary):
        log.debug("Binary not found on PATH: %s", binary)
        return None
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            m = _VERSION_RE.search(proc.stdout)
            return m.group(1).strip() if m else proc.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _get_engine_version() -> str | None:
    """Probe both llama-bench and llama-server; return the first hit."""
    for binary in ("llama-bench", "llama-server"):
        env_var = "PPB_LLAMA_BENCH" if "bench" in binary else "PPB_LLAMA_SERVER"
        cmd = os.getenv(env_var, binary)
        version = _detect_binary_version(cmd)
        if version:
            log.info("Detected llama.cpp version via %s: %s", cmd, version)
            return version
    log.warning(
        "Could not detect llama.cpp version — llm_engine_version will remain null."
    )
    return None


# ---------------------------------------------------------------------------
# Backfill logic
# ---------------------------------------------------------------------------

_BACKFILL_TIMESTAMP = datetime.now(timezone.utc).isoformat()

# Fields that should NOT be touched (fingerprints / dedup keys)
_IMMUTABLE_FIELDS = frozenset(
    {
        "row_id",
        "result_fingerprint",
        "run_fingerprint",
        "machine_fingerprint",
        "source_file_sha256",
        "submission_id",
    }
)


def _needs_backfill(record: dict[str, Any]) -> bool:
    """Return True if *record* is missing any of the new flag fields."""
    return "llm_flags" not in record or "llm_flags_label" not in record


def _backfill_record(
    record: dict[str, Any],
    engine_version: str | None,
) -> dict[str, Any]:
    """Return a copy of *record* with the new flag fields injected.

    Only modifies fields that are absent or None; immutable fingerprint
    fields are always preserved unchanged.
    """
    r = dict(record)

    # llama.cpp flag fields (new in schema v0.10.0)
    if "llm_flags" not in r:
        r["llm_flags"] = "{}"
    if "llm_flags_label" not in r:
        r["llm_flags_label"] = "default"
    if "extra_flags_raw" not in r:
        r["extra_flags_raw"] = None

    # Best-effort engine version (only fill when currently null)
    if r.get("llm_engine_version") is None and engine_version is not None:
        r["llm_engine_version"] = engine_version

    # Append a note to the meta dict so consumers know this row was backfilled
    existing_meta: dict = {}
    if r.get("meta"):
        try:
            existing_meta = json.loads(r["meta"])
        except (json.JSONDecodeError, TypeError):
            existing_meta = {}

    existing_meta.setdefault("backfilled_flags", True)
    existing_meta.setdefault("backfill_timestamp", _BACKFILL_TIMESTAMP)
    if engine_version and r.get("llm_engine_version") == engine_version:
        existing_meta.setdefault(
            "llm_engine_version_note",
            "best-effort: captured from current binary, may differ from binary used at run time",
        )
    r["meta"] = json.dumps(existing_meta, sort_keys=True)

    return r


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def backfill_file(
    path: Path,
    engine_version: str | None,
    *,
    dry_run: bool = False,
    write_csv: bool = False,
) -> int:
    """Backfill *path* in place.  Returns the count of modified records."""
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, Any]] = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            log.warning("Skipping malformed line %d in %s: %s", i, path.name, exc)

    modified = 0
    out_records: list[dict[str, Any]] = []
    for rec in records:
        if _needs_backfill(rec):
            out_records.append(_backfill_record(rec, engine_version))
            modified += 1
        else:
            out_records.append(rec)

    log.info(
        "%s: %d/%d records need backfill%s",
        path.name,
        modified,
        len(records),
        " (dry run — not writing)" if dry_run else "",
    )

    if not dry_run and modified > 0:
        # Write atomically via a temp file to avoid partial writes
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                for rec in out_records:
                    fh.write(json.dumps(rec) + "\n")
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        if write_csv:
            _write_csv(path, out_records)

    return modified


def _write_csv(jsonl_path: Path, records: list[dict[str, Any]]) -> None:
    """Re-flatten *records* and write a CSV next to the JSONL file."""
    import csv

    from utils.flattener import COLUMN_ORDER

    csv_path = jsonl_path.with_suffix(".csv")
    flat_rows: list[dict] = []
    for rec in records:
        flat_rows.extend(flatten_benchmark_row(rec))

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMN_ORDER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    log.info("Wrote CSV: %s (%d rows)", csv_path.name, len(flat_rows))


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------


def _upload_to_hf(jsonl_path: Path) -> None:
    """Re-flatten the backfilled file and upload to the PPB HF dataset."""
    from utils.publisher import PPB_HF_REPO, publish_to_hf

    records: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    flat_rows: list[dict] = []
    for rec in records:
        flat_rows.extend(flatten_benchmark_row(rec))

    if not flat_rows:
        log.warning("No rows to upload from %s", jsonl_path.name)
        return

    url = publish_to_hf(flat_rows)
    log.info("Uploaded %d rows to %s → %s", len(flat_rows), PPB_HF_REPO, url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill llm_flags fields into PPB result JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="JSONL files to backfill.  Defaults to results/*.jsonl",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Re-flatten and write a CSV alongside each backfilled JSONL.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Re-upload each backfilled file to the PPB HuggingFace dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    parser.add_argument(
        "--engine-version",
        metavar="VERSION",
        help="Override the detected llama.cpp version string (e.g. 'b5063 (58ab80c3)').",
    )
    args = parser.parse_args()

    # Resolve target files
    files: list[Path]
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        files = sorted(results_dir.glob("*.jsonl"))
        if not files:
            log.error("No JSONL files found in %s", results_dir)
            sys.exit(1)

    # Detect engine version once
    if args.engine_version:
        engine_version: str | None = args.engine_version
        log.info("Using supplied engine version: %s", engine_version)
    else:
        engine_version = _get_engine_version()

    total_modified = 0
    for fpath in files:
        if not fpath.is_file():
            log.warning("File not found: %s — skipping", fpath)
            continue
        try:
            n = backfill_file(
                fpath,
                engine_version,
                dry_run=args.dry_run,
                write_csv=args.csv,
            )
            total_modified += n
        except Exception as exc:
            log.error("Failed to backfill %s: %s", fpath.name, exc)
            continue

        if args.upload and not args.dry_run:
            try:
                _upload_to_hf(fpath)
            except Exception as exc:
                log.error("Upload failed for %s: %s", fpath.name, exc)

    if args.dry_run:
        log.info("Dry run complete.  %d record(s) would be modified.", total_modified)
    else:
        log.info("Backfill complete.  %d record(s) modified.", total_modified)


if __name__ == "__main__":
    main()
