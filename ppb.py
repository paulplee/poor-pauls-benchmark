"""
Poor Paul's Benchmark (PPB) — CLI entry point.

An automated evaluation framework for local LLM inference
powered by llama.cpp's llama-bench.
"""

import fnmatch
import itertools
import json
import logging
import os
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
import typer
from huggingface_hub import HfApi, RepoFile, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Rich console & logging setup
# ---------------------------------------------------------------------------

ppb_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "hw": "bold magenta",  # hardware specs
    }
)

console = Console(theme=ppb_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# .env & defaults
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env in the current working directory (if present)

DEFAULT_MODELS_DIR = Path(os.getenv("PPB_MODELS_DIR", "./models"))
DEFAULT_RESULTS_FILE = Path(os.getenv("PPB_RESULTS_FILE", "./results.jsonl"))
LLAMA_BENCH_CMD = os.getenv("PPB_LLAMA_BENCH", "llama-bench")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SweepConfig(BaseModel):
    """Validated representation of the ``[sweep]`` block in a sweep TOML file.

    Example TOML::

        [sweep]
        model_path = "./models/model.gguf"
        n_ctx    = [8192, 16384, 32768]   # passed as -p (prompt tokens) to llama-bench
        n_batch  = [512, 1024]            # passed as -b (batch size) to llama-bench
    """

    model_path: Path
    n_ctx: list[int]
    n_batch: list[int]

    @field_validator("model_path", mode="before")
    @classmethod
    def resolve_and_check(cls, v: object) -> Path:
        p = Path(str(v)).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"model_path does not exist: {p}")
        return p

    def combos(self) -> list["BenchCombo"]:
        """Return the full Cartesian product of sweep parameters."""
        return [
            BenchCombo(model_path=self.model_path, n_ctx=ctx, n_batch=batch)
            for ctx, batch in itertools.product(self.n_ctx, self.n_batch)
        ]


@dataclass
class BenchCombo:
    """A single (model, n_ctx, n_batch) parameter combination."""

    model_path: Path
    n_ctx: int
    n_batch: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rich_tqdm(progress: Progress) -> type:
    """Return a tqdm-compatible class that forwards byte progress to *progress*.

    ``hf_hub_download`` accepts a ``tqdm_class`` keyword argument.  By passing
    a class produced here we intercept every ``update(n)`` call the downloader
    makes and forward it straight to a Rich :class:`~rich.progress.Progress`
    task, giving us a live bytes/speed/ETA bar without any polling.
    """

    class _RichTqdm:
        def __init__(
            self,
            iterable=None,
            *,
            desc: str = "",
            total: Optional[int] = None,
            **_kwargs,
        ):
            # Use only the bare filename so the bar label stays short.
            label = Path(desc).name or desc
            self._task = progress.add_task(f"[cyan]{label}[/cyan]", total=total)
            self.iterable = iterable
            self.n = 0

        # -- tqdm interface --------------------------------------------------

        def update(self, n: int = 1) -> None:
            progress.advance(self._task, n)
            self.n += n

        def close(self) -> None:
            # Clamp to 100 % when total was unknown (chunked encoding, etc.).
            progress.update(self._task, completed=self.n, total=self.n)

        # -- context-manager / iterable shims --------------------------------

        def __iter__(self):
            if self.iterable is not None:
                yield from self.iterable

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    return _RichTqdm


def download_model(
    repo_id: str,
    filename_pattern: str,
    token: Optional[str] = None,
    models_dir: Optional[Path] = None,
) -> list[Path]:
    """Download GGUF file(s) matching *filename_pattern* from a HF repo.

    Every file whose name matches the glob is downloaded into *models_dir*
    (defaulting to ``./models`` or the ``PPB_MODELS_DIR`` env-var).

    Parameters
    ----------
    repo_id:
        Hugging Face repository ID, e.g. ``"bartowski/Llama-3-8B-Instruct-GGUF"``.
    filename_pattern:
        Glob pattern matched against the repo's file listing,
        e.g. ``"*Q4_K_M.gguf"`` or ``"*.gguf"``.
    token:
        Optional Hugging Face API token.  When *None* the token saved by
        ``huggingface-cli login`` (or the ``HF_TOKEN`` env-var) is used.
    models_dir:
        Destination directory.  Falls back to ``DEFAULT_MODELS_DIR``.

    Returns
    -------
    list[Path]
        Absolute paths to every downloaded (or already-cached) model file.

    Raises
    ------
    FileNotFoundError
        If no file in the repository matches *filename_pattern*.
    RepositoryNotFoundError
        If the repository does not exist or authentication failed.
    """
    dest = models_dir or DEFAULT_MODELS_DIR
    dest.mkdir(parents=True, exist_ok=True)

    # --- resolve the glob pattern to exact filenames -------------------------
    api = HfApi(token=token)

    try:
        repo_files: list[str] = [
            f.rfilename
            for f in api.list_repo_tree(repo_id, repo_type="model")
            if isinstance(f, RepoFile)
        ]
    except RepositoryNotFoundError:
        console.print(
            "\n[error]Repository not found or access denied.[/error]\n"
            f"  Repo: [bold]{repo_id}[/bold]\n\n"
            "  This usually means one of:\n"
            "    1. The repo ID is misspelled.\n"
            "    2. The repo is private/gated and you are not authenticated.\n\n"
            "  To log in, run:\n"
            "    [bold cyan]uv run huggingface-cli login[/bold cyan]\n"
            "  or set the [bold]HF_TOKEN[/bold] environment variable.\n"
        )
        raise

    matches = fnmatch.filter(repo_files, filename_pattern)

    if not matches:
        raise FileNotFoundError(
            f"No file in '{repo_id}' matches pattern '{filename_pattern}'.\n"
            f"Available files: {repo_files}"
        )

    console.print(
        f"[info]Matched {len(matches)} file(s) in[/info] [bold]{repo_id}[/bold]:"
    )
    for m in matches:
        console.print(f"  • {m}")

    # --- download each match with a Rich progress bar ------------------------
    downloaded_paths: list[Path] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        RichTqdm = _make_rich_tqdm(progress)
        overall = progress.add_task("[bold]Total", total=len(matches))

        for filename in matches:
            # Fast-path: skip the network round-trip when the file is already
            # present in the target directory.
            local_file = dest / filename
            if local_file.exists():
                console.print(f"  [info]↩ Cached[/info]  {filename}")
                downloaded_paths.append(local_file.resolve())
                progress.advance(overall)
                continue

            downloaded: str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest),
                token=token,
                tqdm_class=RichTqdm,  # ← live byte progress
            )

            progress.advance(overall)
            model_path = Path(downloaded).resolve()
            downloaded_paths.append(model_path)
            console.print(f"  [success]✓[/success] {filename}")

    console.print(
        f"\n[success]Done![/success] {len(downloaded_paths)} file(s) saved to "
        f"[bold]{dest.resolve()}[/bold]"
    )
    return downloaded_paths


def run_llama_bench(combo: BenchCombo, results_file: Path) -> dict | None:
    """Run ``llama-bench`` for one parameter combination and record the result.

    Parameters
    ----------
    combo:
        The model/context/batch combination to benchmark.
    results_file:
        JSONL file to append the enriched result record to.

    Returns
    -------
    dict | None
        The enriched result dict, or *None* when the run failed.
    """
    cmd: list[str] = [
        LLAMA_BENCH_CMD,
        "-m",
        str(combo.model_path),
        "-p",  # n_ctx → prompt token count → drives KV cache allocation
        str(combo.n_ctx),
        "-b",
        str(combo.n_batch),
        "-o",
        "json",
    ]

    log.debug("Running: %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        log.error(
            "llama-bench exited with code %d\n%s",
            proc.returncode,
            proc.stderr.strip(),
        )
        return None

    try:
        bench_data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        log.error(
            "Failed to parse llama-bench output: %s\nRaw stdout:\n%s",
            exc,
            proc.stdout[:500],
        )
        return None

    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(combo.model_path),
        "n_ctx": combo.n_ctx,
        "n_batch": combo.n_batch,
        "results": bench_data,
    }

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return record


def execute_sweep(config_path: Path, results_file: Path) -> None:
    """Parse *config_path*, enumerate all combos, and run llama-bench for each.

    Parameters
    ----------
    config_path:
        Path to the sweep TOML file.
    results_file:
        Destination JSONL file for results.
    """
    if not config_path.exists():
        console.print(f"[error]Config file not found:[/error] {config_path}")
        raise typer.Exit(code=1)

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    if "sweep" not in raw:
        console.print(
            "[error]Missing [sweep] section in config file.[/error]\n"
            f"  File: {config_path}"
        )
        raise typer.Exit(code=1)

    try:
        cfg = SweepConfig(**raw["sweep"])
    except Exception as exc:  # pydantic ValidationError
        console.print(f"[error]Invalid sweep config:[/error] {exc}")
        raise typer.Exit(code=1) from exc

    combos = cfg.combos()
    total = len(combos)

    console.print(
        f"[info]Sweep:[/info] [bold]{total}[/bold] combination(s) "
        f"→ model [hw]{cfg.model_path.name}[/hw]\n"
        f"  n_ctx   : {cfg.n_ctx}\n"
        f"  n_batch : {cfg.n_batch}\n"
        f"  Results : [bold]{results_file.resolve()}[/bold]"
    )

    passed = failed = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• [cyan]{task.fields[combo]}[/cyan]"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Sweep", total=total, combo="starting…")

        for i, combo in enumerate(combos, start=1):
            label = f"ctx={combo.n_ctx} batch={combo.n_batch}"
            progress.update(task, combo=label)

            record = run_llama_bench(combo, results_file=results_file)

            if record is None:
                console.print(f"  [error]✗[/error] [{i}/{total}] {label} — FAILED")
                failed += 1
            else:
                # Pull out the tok/s figure if llama-bench emits it
                tps: str = ""
                try:
                    tps_val = record["results"][0]["avg_ts"]
                    tps = f"  {tps_val:.1f} tok/s"
                except (KeyError, IndexError, TypeError):
                    pass
                console.print(f"  [success]✓[/success] [{i}/{total}] {label}{tps}")
                passed += 1

            progress.advance(task)

    status = "success" if failed == 0 else "warning"
    console.print(
        f"\n[{status}]Sweep complete.[/{status}] "
        f"{passed} passed, {failed} failed — "
        f"results in [bold]{results_file.resolve()}[/bold]"
    )


# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ppb",
    help="Poor Paul's Benchmark — find the absolute limit of your local AI hardware.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def download(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repo ID (e.g. QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)",
    ),
    filename: str = typer.Argument(
        ..., help='Glob pattern for the GGUF file (e.g. "*Q4_K_M.gguf" or "*.gguf")'
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="HF_TOKEN",
        help="Hugging Face API token (or set HF_TOKEN env-var)",
    ),
    models_dir: Optional[Path] = typer.Option(
        None,
        "--models-dir",
        "-d",
        envvar="PPB_MODELS_DIR",
        help="Destination directory for models (default: ./models)",
    ),
) -> None:
    """Download GGUF model(s) from Hugging Face Hub."""
    console.print(
        f"[info]Downloading[/info] [bold]{filename}[/bold] from [bold]{repo_id}[/bold] …"
    )
    try:
        paths = download_model(repo_id, filename, token=token, models_dir=models_dir)
        for p in paths:
            log.info("Downloaded → %s", p)
    except FileNotFoundError as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc
    except RepositoryNotFoundError:
        # Detailed message already printed by download_model.
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[error]Download failed:[/error] {exc}")
        log.exception("Unexpected error during download")
        raise typer.Exit(code=1) from exc


@app.command()
def sweep(
    config: Path = typer.Argument(
        ...,
        help="Path to a TOML sweep configuration file (must contain a [sweep] section)",
        exists=True,
        readable=True,
    ),
    results_file: Path = typer.Option(
        DEFAULT_RESULTS_FILE,
        "--results",
        "-r",
        envvar="PPB_RESULTS_FILE",
        help="JSONL file to append benchmark results to (default: ./results.jsonl)",
    ),
) -> None:
    """Run a declarative parameter sweep using llama-bench."""
    console.print(f"[info]Starting sweep[/info] with config [bold]{config}[/bold] …")
    execute_sweep(config_path=Path(config), results_file=results_file)


@app.command(name="auto-limit")
def auto_limit(
    model: str = typer.Option(..., "--model", "-m", help="Path to the GGUF model file"),
    min_ctx: int = typer.Option(
        2048, "--min-ctx", help="Minimum context length to probe"
    ),
    max_ctx: int = typer.Option(
        128000, "--max-ctx", help="Maximum context length to probe"
    ),
) -> None:
    """Binary-search for the maximum context window before OOM."""
    console.print(
        f"[info]Auto-limit[/info] probing [hw]{model}[/hw] "
        f"between [bold]{min_ctx}[/bold] and [bold]{max_ctx}[/bold] …"
    )
    log.info("auto-limit command — not yet implemented")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
