"""Upload flattened benchmark rows to the central PPB Hugging Face dataset."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

# Central leaderboard repository — all submissions land here.
PPB_HF_REPO = "paulplee/ppb-results"


def check_hf_token(token: str | None = None) -> None:
    """Verify that *token* (or the ambient credential) can write to the PPB dataset.

    Performs an authenticated ``whoami`` call **and** a lightweight write-
    permission probe against the target repo so that write-only vs read-only
    token issues are caught before a long benchmark run starts.

    Raises
    ------
    PermissionError
        When the token is missing, invalid, or read-only.
    """
    api = HfApi(token=token)

    # 1. Authentication: token must be valid.
    try:
        api.whoami()
    except Exception as exc:
        raise PermissionError(
            "Hugging Face token is missing or invalid.\n"
            "  Run  huggingface-cli login  or set the HF_TOKEN environment variable."
        ) from exc

    # 2. Write permission: probe with a tiny metadata-only call.
    # repo_info() with expand=["files"] does not write anything but will
    # raise a 403 if the token is read-only for this repo.
    try:
        api.repo_info(repo_id=PPB_HF_REPO, repo_type="dataset", expand=["files"])
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403:
            raise PermissionError(
                "Your Hugging Face token is read-only and cannot upload to the PPB dataset.\n"
                "  Please create a write-access token at https://huggingface.co/settings/tokens\n"
                "  and save it in your .env file as  HF_TOKEN=hf_…"
            ) from exc
        raise  # re-raise unexpected HF errors unmodified
    except Exception as exc:
        # Network errors etc. — warn but don't abort; the real upload will fail
        # with a clearer message if there is a connectivity problem.
        raise PermissionError(
            f"Could not verify Hugging Face write access: {exc}"
        ) from exc


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
