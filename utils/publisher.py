"""Upload flattened benchmark rows to the central PPB Hugging Face dataset."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

# Central leaderboard repository — all submissions land here.
PPB_HF_REPO = "paulplee/ppb-results"


def check_hf_token(token: str | None = None) -> None:
    """Verify that *token* (or the ambient credential) can upload results.

    Performs an authenticated ``whoami`` call and inspects the token role.
    Classic read-only tokens are caught immediately; fine-grained tokens
    are accepted here (the upload itself will 403 if scoped permissions
    are wrong, with a clear error message).

    Raises
    ------
    PermissionError
        When the token is missing, invalid, or a classic read-only token.
    """
    api = HfApi(token=token)

    # 1. Authentication: token must be valid.
    try:
        user_info = api.whoami()
    except Exception as exc:
        raise PermissionError(
            "Hugging Face token is missing or invalid.\n"
            "  Run  huggingface-cli login  or set the HF_TOKEN environment variable."
        ) from exc

    # 2. Classic tokens carry a "role" field — reject "read".
    role = user_info.get("auth", {}).get("accessToken", {}).get("role")
    if role == "read":
        raise PermissionError(
            "Your Hugging Face token is read-only and cannot upload to the PPB dataset.\n"
            "  Please create a write-access token at https://huggingface.co/settings/tokens\n"
            "  and save it in your .env file as  HF_TOKEN=hf_…"
        )


def publish_to_hf(
    flat_rows: list[dict[str, Any]],
    *,
    token: str | None = None,
) -> str:
    """Write *flat_rows* to a temp JSONL and upload to the PPB dataset repo.

    Parameters
    ----------
    flat_rows:
        List of flat dictionaries (``raw_payload`` should already be
        stripped by the caller).
    token:
        Hugging Face API token.  Falls back to the ``HF_TOKEN`` env-var
        and then to the cached ``huggingface-cli login`` credential.

    Returns
    -------
    str
        URL of the dataset repository.

    Raises
    ------
    PermissionError
        When the user is not authenticated.
    """
    api = HfApi(token=token)

    # -- auth check --------------------------------------------------------
    try:
        api.whoami()
    except Exception as exc:
        raise PermissionError(
            "Not logged in to Hugging Face.\n"
            "  Run  huggingface-cli login  or set the HF_TOKEN environment variable."
        ) from exc

    # -- write temp file ---------------------------------------------------
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    dest_filename = f"results_{ts}_{short_uuid}.jsonl"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        prefix="ppb_normalized_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        for row in flat_rows:
            tmp.write(json.dumps(row, default=str) + "\n")
        tmp_path = tmp.name

    # -- upload ------------------------------------------------------------
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"data/{dest_filename}",
            repo_id=PPB_HF_REPO,
            repo_type="dataset",
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return f"https://huggingface.co/datasets/{PPB_HF_REPO}"


# ---------------------------------------------------------------------------
# Composable-schema lookup helpers
# ---------------------------------------------------------------------------


def _composable_key(model_hf_id: str, hardware: dict[str, Any]) -> tuple[str, str, str]:
    """Build the (gpu_name, model_name, quantization) join key.

    Mirrors the parsing logic used by ``utils.flattener``: the model
    filename is split on the last ``-`` to separate the base model from
    the quantization tag.
    """
    fname = Path(model_hf_id).name if model_hf_id else ""
    stem = fname.rsplit(".gguf", 1)[0]
    if "-" in stem:
        base, quant = stem.rsplit("-", 1)
    else:
        base, quant = stem, ""
    gpus = (hardware or {}).get("gpus") or [{}]
    gpu_name = gpus[0].get("name", "") if gpus else ""
    return gpu_name, base, quant


def fetch_existing_quantitative_for(
    *,
    hf_id: str,
    hardware: dict[str, Any],
    token: str | None = None,
) -> dict[str, Any] | None:
    """Look up the most recent quantitative row for this model+GPU+quant.

    Queries the central ``paulplee/ppb-results`` dataset via the
    ``datasets`` library and returns the latest row matching the
    composable join key, or ``None`` if no match is found / the dataset
    cannot be loaded.

    The returned dict is the raw flattened row from the dataset; callers
    typically read ``vram_cliff_tokens`` from it.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError:
        return None

    gpu_name, model_base, quant = _composable_key(hf_id, hardware)
    if not (gpu_name and model_base):
        return None

    try:
        ds = load_dataset(PPB_HF_REPO, split="train", token=token)
    except Exception:
        return None

    matches: list[dict[str, Any]] = []
    for row in ds:
        if not isinstance(row, dict):
            continue
        # Accept either the new top-level run_type ('quantitative'/'all') or
        # legacy rows where it's absent (treated as quantitative).
        rt = row.get("run_type") or "quantitative"
        if rt not in ("quantitative", "all"):
            continue
        if (
            (row.get("gpu_name") or "") == gpu_name
            and (row.get("model_base") or row.get("model_name") or "") == model_base
            and (row.get("quant") or row.get("quantization") or "") == quant
        ):
            matches.append(row)

    if not matches:
        return None

    # Return the most recent by timestamp (lexicographic ISO-8601 sort).
    matches.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
    return matches[0]
