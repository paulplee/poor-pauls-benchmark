"""
Dataset downloader and prompt extractor for PPB.

Downloads conversational datasets from Hugging Face Hub and extracts
human-turn prompts for use as realistic inference workloads.

The default dataset is ShareGPT — a large collection of real ChatGPT
conversations — but any HF-hosted dataset in the same JSON format can
be used via the ``repo_id`` and ``filename`` parameters.
"""

from __future__ import annotations

import json
import logging
import random
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


def download_dataset(
    repo_id: str = SHAREGPT_REPO,
    filename: str = SHAREGPT_FILENAME,
    dataset_dir: Path | None = None,
) -> Path:
    """Download a conversational dataset from Hugging Face Hub.

    Parameters
    ----------
    repo_id:
        HF Hub dataset repository ID
        (default: ``anon8231489123/ShareGPT_Vicuna_unfiltered``).
    filename:
        File to download from the repository
        (default: ``ShareGPT_V3_unfiltered_cleaned_split.json``).
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

    local_path = dest / filename

    if local_path.exists():
        log.debug("Dataset already cached at %s", local_path)
        return local_path

    log.info("Downloading %s from %s …", filename, repo_id)

    downloaded: str = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(dest),
    )

    result = Path(downloaded).resolve()
    log.info("Dataset saved to %s", result)
    return result


def download_sharegpt(dataset_dir: Path | None = None) -> Path:
    """Download the default ShareGPT dataset.

    Convenience wrapper around :func:`download_dataset` that uses the
    built-in ShareGPT repo and filename.  Kept for backward compatibility.
    """
    return download_dataset(dataset_dir=dataset_dir)


def load_sharegpt_prompts(
    path: Path,
    max_prompts: int = 100,
    shuffle: bool = False,
    seed: int | None = None,
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
    shuffle:
        If ``True``, randomise the order of conversations before
        extracting prompts so repeated runs see different workloads.
    seed:
        Optional RNG seed for reproducible shuffling.

    Returns
    -------
    list[str]
        Extracted prompt strings.
    """
    log.debug("Loading ShareGPT prompts from %s (max=%d)", path, max_prompts)

    with path.open("r", encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)

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
