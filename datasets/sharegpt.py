"""
ShareGPT dataset downloader and prompt extractor.

Downloads a cleaned ShareGPT conversation dataset from Hugging Face Hub
and extracts human-turn prompts for use as realistic inference workloads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "data"

# Minimum character length for a prompt to be considered useful.
_MIN_PROMPT_LENGTH = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_sharegpt(dataset_dir: Path | None = None) -> Path:
    """Download the ShareGPT dataset if not already cached.

    Parameters
    ----------
    dataset_dir:
        Directory to store the downloaded file.  Defaults to
        ``datasets/data/`` next to this module.

    Returns
    -------
    Path
        Absolute path to the downloaded JSON file.
    """
    dest = dataset_dir or DEFAULT_DATASET_DIR
    dest.mkdir(parents=True, exist_ok=True)

    local_path = dest / SHAREGPT_FILENAME

    if local_path.exists():
        log.debug("ShareGPT dataset already cached at %s", local_path)
        return local_path

    log.info("Downloading ShareGPT dataset from %s …", SHAREGPT_REPO)

    downloaded: str = hf_hub_download(
        repo_id=SHAREGPT_REPO,
        repo_type="dataset",
        filename=SHAREGPT_FILENAME,
        local_dir=str(dest),
    )

    result = Path(downloaded).resolve()
    log.info("ShareGPT dataset saved to %s", result)
    return result


def load_sharegpt_prompts(
    path: Path,
    max_prompts: int = 100,
) -> list[str]:
    """Extract human-turn prompts from a ShareGPT JSON file.

    The ShareGPT format is a JSON array of conversation objects::

        [
            {
                "id": "...",
                "conversations": [
                    {"from": "human", "value": "Hello, ..."},
                    {"from": "gpt",   "value": "Hi there! ..."},
                    ...
                ]
            },
            ...
        ]

    This function extracts the **first human turn** from each conversation,
    filters out prompts shorter than ``_MIN_PROMPT_LENGTH``, and returns
    up to *max_prompts* results.

    Parameters
    ----------
    path:
        Path to the ShareGPT JSON file.
    max_prompts:
        Maximum number of prompts to return.

    Returns
    -------
    list[str]
        Extracted prompt strings.
    """
    log.debug("Loading ShareGPT prompts from %s (max=%d)", path, max_prompts)

    with path.open("r", encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)

    prompts: list[str] = []

    for conversation in data:
        if len(prompts) >= max_prompts:
            break

        turns = conversation.get("conversations", [])
        if not turns:
            continue

        # Find the first human turn.
        first_human: str | None = None
        for turn in turns:
            if turn.get("from") == "human":
                first_human = turn.get("value", "").strip()
                break

        if first_human and len(first_human) >= _MIN_PROMPT_LENGTH:
            prompts.append(first_human)

    log.info("Loaded %d ShareGPT prompts", len(prompts))
    return prompts
