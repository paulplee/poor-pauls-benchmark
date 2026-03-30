"""Download all files from the PPB results dataset on Hugging Face Hub.

Usage:
    uv run scripts/download_hf_dataset.py
    uv run scripts/download_hf_dataset.py --local-dir results/my-backup
    uv run scripts/download_hf_dataset.py --repo-id other-user/ppb-results

Authentication (any one of):
    - HF_TOKEN environment variable
    - huggingface-cli login (cached credential)
    - --token flag
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all files from the PPB HF dataset for local backup."
    )
    parser.add_argument(
        "--repo-id",
        default="paulplee/ppb-results",
        help="Hugging Face dataset repo ID (default: paulplee/ppb-results)",
    )
    parser.add_argument(
        "--local-dir",
        default="results/hf-backup",
        help="Local directory to download into (default: results/hf-backup)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token (falls back to HF_TOKEN env var or cached login)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    local_dir = Path(args.local_dir)
    print(f"Downloading {args.repo_id} → {local_dir} …")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=args.token,
    )

    data_dir = local_dir / "data"
    if data_dir.exists():
        file_count = sum(1 for _ in data_dir.iterdir())
        print(f"Done. {file_count} file(s) in {data_dir}")
    else:
        print(f"Done. Files saved to {local_dir}")


if __name__ == "__main__":
    main()
