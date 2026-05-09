"""Build an aggregated Parquet from the raw ppb-results dataset.

For each unique (gpu_name, model_base, quant, n_ctx, concurrent_users) combination
this script computes:

    mean_throughput_tok_s   — arithmetic mean across repeated runs
    std_throughput_tok_s    — sample standard deviation
    sample_count            — number of rows used
    ci95_low_tok_s          — lower bound of 95 % confidence interval
    ci95_high_tok_s         — upper bound
    mean_avg_ttft_ms        — mean TTFT (when present)
    mean_p50_ttft_ms
    mean_p99_ttft_ms
    mean_avg_itl_ms
    mean_p50_itl_ms
    mean_p99_itl_ms
    mean_avg_power_w        — mean GPU power draw (when present)

The aggregated file is written to ``ppb_results_aggregated.parquet`` and
optionally uploaded to HuggingFace.

Usage
-----
    python scripts/aggregate_results.py                          # local run only
    python scripts/aggregate_results.py --upload                 # upload to HF
    python scripts/aggregate_results.py --input my.parquet       # use local file
    python scripts/aggregate_results.py --dataset user/repo      # override HF repo

Access
------
Without ``--upload``: anyone can run this. It reads the public dataset and
writes a local Parquet file; no credentials required.

With ``--upload``: requires ``HF_TOKEN`` with write access to the dataset
(maintainer-only). Do NOT run this manually if others may be publishing at
the same time — the aggregated file is overwritten by the last writer and
intermediate contributions will be silently dropped from it (though they
remain safe in the raw dataset). The canonical way to run ``--upload`` is
via the scheduled GitHub Actions workflow, which serialises all writes.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "paulplee/ppb-results"
DEFAULT_INPUT_FILENAME = "data/ppb_results_v090.parquet"
DEFAULT_OUTPUT_FILENAME = "data/ppb_results_aggregated.parquet"

GROUP_KEYS = [
    "gpu_name",
    "model_base",
    "quant",
    "n_ctx",
    "concurrent_users",
    "backends",
    "runner_type",
]

# Columns to aggregate — (raw_col, output_col, aggregation)
NUMERIC_AGGS: list[tuple[str, str]] = [
    ("throughput_tok_s", "mean_throughput_tok_s"),
    ("avg_ttft_ms", "mean_avg_ttft_ms"),
    ("p50_ttft_ms", "mean_p50_ttft_ms"),
    ("p99_ttft_ms", "mean_p99_ttft_ms"),
    ("avg_itl_ms", "mean_avg_itl_ms"),
    ("p50_itl_ms", "mean_p50_itl_ms"),
    ("p99_itl_ms", "mean_p99_itl_ms"),
    ("avg_power_w", "mean_avg_power_w"),
    ("avg_gpu_temp_c", "mean_avg_gpu_temp_c"),
]

# ---------------------------------------------------------------------------
# CI calculation helper
# ---------------------------------------------------------------------------

_T95 = {  # t-distribution critical values for 95 % CI (two-tailed, df=n-1)
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
_T95_INF = 1.960  # large-sample approximation


def _t95(df: int) -> float:
    return _T95.get(df, _T95_INF)


def _ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    """Return (low, high) 95 % confidence interval."""
    if n < 2 or std == 0.0:
        return (mean, mean)
    margin = _t95(n - 1) * (std / math.sqrt(n))
    return (round(mean - margin, 4), round(mean + margin, 4))


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def aggregate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return aggregated DataFrame from a raw ppb results DataFrame."""
    available_groups = [k for k in GROUP_KEYS if k in df.columns]
    if not available_groups:
        raise ValueError("DataFrame missing all group-by columns")

    # Build agg dict for all available numeric columns
    agg_dict: dict[str, list[str]] = {}
    available_numerics = []
    for raw_col, _ in NUMERIC_AGGS:
        if raw_col in df.columns:
            agg_dict[raw_col] = ["mean", "std", "count"]
            available_numerics.append(raw_col)
    # Always include throughput count for sample_count
    if "throughput_tok_s" not in agg_dict:
        agg_dict["throughput_tok_s"] = ["count"]

    grouped = df.groupby(available_groups, dropna=False).agg(agg_dict)
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    grouped = grouped.reset_index()

    # Rename mean columns to nicer names
    renames: dict[str, str] = {}
    for raw_col, out_col in NUMERIC_AGGS:
        if f"{raw_col}_mean" in grouped.columns:
            renames[f"{raw_col}_mean"] = out_col
        if f"{raw_col}_std" in grouped.columns:
            renames[f"{raw_col}_std"] = f"std_{raw_col}"
    if "throughput_tok_s_count" in grouped.columns:
        renames["throughput_tok_s_count"] = "sample_count"
    grouped = grouped.rename(columns=renames)

    # Compute 95 % CI for throughput when enough data is present
    if (
        "mean_throughput_tok_s" in grouped.columns
        and "std_throughput_tok_s" in grouped.columns
    ):
        ci_low, ci_high = [], []
        for _, row in grouped.iterrows():
            mean = row["mean_throughput_tok_s"]
            std = row.get("std_throughput_tok_s", float("nan"))
            n = int(row.get("sample_count", 1))
            if pd.isna(mean) or pd.isna(std):
                ci_low.append(float("nan"))
                ci_high.append(float("nan"))
            else:
                lo, hi = _ci95(mean, std, n)
                ci_low.append(lo)
                ci_high.append(hi)
        grouped["ci95_low_tok_s"] = ci_low
        grouped["ci95_high_tok_s"] = ci_high

    # Round numeric columns for readability
    for col in grouped.select_dtypes("float").columns:
        grouped[col] = grouped[col].round(4)

    return grouped


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_from_hf(dataset: str, filename: str) -> pd.DataFrame:
    """Download the parquet file from HuggingFace and return a DataFrame."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
        sys.exit(1)

    logger.info("Downloading %s from %s …", filename, dataset)
    local_path = hf_hub_download(
        repo_id=dataset,
        filename=filename,
        repo_type="dataset",
    )
    return pd.read_parquet(local_path)


def upload_to_hf(local_path: Path, dataset: str, filename: str) -> None:
    """Upload a local file to HuggingFace as a dataset file."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub is not installed.")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN is not set. Set it in your environment or a .env file "
            "before using --upload."
        )
        sys.exit(1)

    api = HfApi(token=token)
    logger.info("Uploading %s to %s/%s …", local_path, dataset, filename)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=dataset,
        repo_type="dataset",
    )
    logger.info("Upload complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        help=f"Local parquet file to use (skips HF download). Default: download {DEFAULT_INPUT_FILENAME}",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HuggingFace dataset repo (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output filename (default: %(default)s)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the aggregated file to HuggingFace after writing it locally",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Drop groups with fewer than this many samples (default: 1 = keep all)",
    )
    args = parser.parse_args(argv)

    # --- Load raw data -------------------------------------------------------
    if args.input:
        input_path = Path(args.input)
        logger.info("Reading %s …", input_path)
        df = pd.read_parquet(input_path)
    else:
        df = load_from_hf(args.dataset, DEFAULT_INPUT_FILENAME)

    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))

    # --- Aggregate -----------------------------------------------------------
    agg = aggregate_df(df)

    if args.min_samples > 1 and "sample_count" in agg.columns:
        before = len(agg)
        agg = agg[agg["sample_count"] >= args.min_samples]
        logger.info(
            "Dropped %d groups with < %d samples", before - len(agg), args.min_samples
        )

    logger.info("Aggregated: %d groups", len(agg))

    # --- Write ---------------------------------------------------------------
    out_path = Path(args.output).name  # local file: just the basename
    agg.to_parquet(out_path, index=False)
    logger.info("Written to %s", out_path)

    if args.upload:
        upload_to_hf(Path(out_path), args.dataset, args.output)

        # Keep the dataset README in sync with the schema source of truth
        readme_src = Path(__file__).parent / "hf_dataset_readme.md"
        if readme_src.exists():
            upload_to_hf(readme_src, args.dataset, "README.md")
        else:
            logger.warning("hf_dataset_readme.md not found — skipping README upload")


if __name__ == "__main__":
    main()
