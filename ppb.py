"""
Poor Paul's Benchmark (PPB) — CLI entry point.

An automated evaluation framework for local LLM inference
powered by llama.cpp's llama-bench.
"""

import contextlib
import fnmatch
import itertools
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
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

from runners import get_runner

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
# Hardware fingerprinting
# ---------------------------------------------------------------------------


class HardwareSniffer:
    """Detect and report the host's hardware profile.

    The resulting dictionary is designed to be injected into every result
    record written to ``results.jsonl``, giving full hardware context to
    every benchmark entry.
    """

    def snapshot(self) -> dict:
        """Return a hardware-profile dictionary."""
        info: dict = {
            "os": self._detect_os(),
            "cpu": self._detect_cpu(),
            "ram": self._detect_ram(),
            "gpus": self._detect_gpus(),
            "runtime": self._detect_runtime(),
        }
        return info

    # -- OS ----------------------------------------------------------------

    @staticmethod
    def _detect_os() -> dict:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        }

    # -- CPU ---------------------------------------------------------------

    @staticmethod
    def _detect_cpu() -> dict:
        data: dict = {
            "arch": platform.machine(),
            "processor": platform.processor() or "unknown",
        }

        system = platform.system()
        if system == "Darwin":
            # Apple Silicon / Intel Mac — parse system_profiler
            try:
                out = subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType"],
                    text=True,
                    timeout=10,
                )
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("Chip:") or line.startswith("Processor Name:"):
                        data["model"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Total Number of Cores:"):
                        data["cores"] = line.split(":", 1)[1].strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        elif system == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            data["model"] = line.split(":", 1)[1].strip()
                            break
                cores = os.cpu_count()
                if cores:
                    data["cores"] = str(cores)
            except OSError:
                pass

        return data

    # -- RAM ---------------------------------------------------------------

    @staticmethod
    def _detect_ram() -> dict:
        data: dict = {}
        system = platform.system()

        if system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            data["total_bytes"] = kb * 1024
                            data["total_gb"] = round(kb / (1024 ** 2), 1)
                            break
            except OSError:
                pass

        elif system == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    timeout=5,
                )
                total = int(out.strip())
                data["total_bytes"] = total
                data["total_gb"] = round(total / (1024 ** 3), 1)
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                pass

        elif system == "Windows":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                data["total_bytes"] = stat.ullTotalPhys
                data["total_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 1)
            except Exception:  # noqa: BLE001
                pass

        return data

    # -- GPU ---------------------------------------------------------------

    @staticmethod
    def _detect_gpus() -> list[dict]:
        system = platform.system()
        gpus: list[dict] = []

        # --- NVIDIA via pynvml -------------------------------------------
        if system in ("Linux", "Windows"):
            try:
                import pynvml  # type: ignore[import-untyped]  # nvidia-ml-py exposes this namespace

                pynvml.nvmlInit()
                driver = pynvml.nvmlSystemGetDriverVersion()
                driver_str = driver if isinstance(driver, str) else driver.decode()

                # CUDA version: int like 12080 → "12.8"
                cuda_ver: str = ""
                try:
                    cuda_int = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_ver = f"{cuda_int // 1000}.{(cuda_int % 1000) // 10}"
                except pynvml.NVMLError:
                    pass

                count = pynvml.nvmlDeviceGetCount()
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode()
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpu: dict = {
                        "index": i,
                        "name": name,
                        "driver": driver_str,
                        "vram_total_bytes": mem.total,
                        "vram_total_gb": round(float(mem.total) / (1024 ** 3), 1),
                    }

                    if cuda_ver:
                        gpu["cuda_version"] = cuda_ver

                    # Compute capability — determines available CUDA kernels
                    try:
                        cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        gpu["compute_capability"] = f"{cc[0]}.{cc[1]}"
                    except pynvml.NVMLError:
                        pass

                    # Power limit in watts (useful for tok/W benchmarks)
                    try:
                        power_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                        gpu["power_limit_w"] = round(power_mw / 1000)
                    except pynvml.NVMLError:
                        pass

                    # PCIe gen + width (bottleneck for large model loading)
                    try:
                        gpu["pcie_gen"] = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                        gpu["pcie_width"] = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                    except pynvml.NVMLError:
                        pass

                    gpus.append(gpu)
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001 — library or hardware missing
                pass

            # Fallback: try nvidia-smi if pynvml unavailable and we got nothing
            if not gpus and shutil.which("nvidia-smi"):
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,driver_version,memory.total,"
                            "compute_cap,power.limit,pcie.link.gen.max,pcie.link.width.max",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=10,
                    )
                    for line in out.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            vram_mb = float(parts[3])
                            gpu = {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "driver": parts[2],
                                "vram_total_bytes": int(vram_mb * 1024 ** 2),
                                "vram_total_gb": round(vram_mb / 1024, 1),
                            }
                            if len(parts) > 4 and parts[4]:
                                gpu["compute_capability"] = parts[4]
                            if len(parts) > 5 and parts[5]:
                                with contextlib.suppress(ValueError):
                                    gpu["power_limit_w"] = round(float(parts[5]))
                            if len(parts) > 6 and parts[6]:
                                with contextlib.suppress(ValueError):
                                    gpu["pcie_gen"] = int(parts[6])
                            if len(parts) > 7 and parts[7]:
                                with contextlib.suppress(ValueError):
                                    gpu["pcie_width"] = int(parts[7])
                            gpus.append(gpu)
                except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                    pass

        # --- macOS: Apple Silicon / discrete GPU -------------------------
        elif system == "Darwin":
            try:
                out = subprocess.check_output(
                    ["system_profiler", "SPDisplaysDataType"],
                    text=True,
                    timeout=10,
                )
                current: dict = {}
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("Chipset Model:"):
                        if current:
                            gpus.append(current)
                        current = {"name": line.split(":", 1)[1].strip()}
                    elif line.startswith("VRAM") and current:
                        current["vram"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Metal Support:") and current:
                        current["metal_version"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Total Number of Cores:") and current:
                        # GPU cores on Apple Silicon
                        current["gpu_cores"] = line.split(":", 1)[1].strip()
                if current:
                    gpus.append(current)
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

        return gpus

    # -- Runtime -----------------------------------------------------------

    @staticmethod
    def _detect_runtime() -> dict:
        """Detect software versions relevant to benchmark reproducibility."""
        data: dict = {
            "python_version": platform.python_version(),
        }

        # llama-bench version / build hash
        if shutil.which(LLAMA_BENCH_CMD):
            try:
                proc = subprocess.run(
                    [LLAMA_BENCH_CMD, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # Only parse stdout; version info from stderr risks matching error messages.
                # Require a successful exit so "error: invalid param --version" is ignored.
                if proc.returncode == 0:
                    _ver_re = re.compile(r"\bversion\b|\bbuild\b", re.IGNORECASE)
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        if line and _ver_re.search(line) and not line.lower().startswith("usage:"):
                            data["llama_bench"] = line
                            break
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass

        return data


# Module-level singleton
_hw_sniffer = HardwareSniffer()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SweepConfig(BaseModel):
    """Validated representation of the ``[sweep]`` block in a sweep TOML file.

    ``model_path`` may be:

    * A path to a single ``.gguf`` file.
    * A path to a **directory** — every ``.gguf`` file inside is tested.
    * A **glob pattern** (e.g. ``~/models/*IQ4*.gguf``) — all matching files
      are tested.

    Example TOML::

        [sweep]
        model_path = "./models/model.gguf"          # single file
        model_path = "~/models/"                    # whole directory
        model_path = "~/models/*Q4_K_M*.gguf"       # glob
        n_ctx    = [8192, 16384, 32768]
        n_batch  = [512, 1024]
    """

    model_path: str  # raw value: single file, directory, or glob pattern
    n_ctx: list[int]
    n_batch: list[int]
    runner_type: str = "llama-bench"
    runner_params: dict[str, Any] = Field(default_factory=dict)

    # Populated by the model_validator below; not read from TOML.
    model_paths: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolve_model_paths(self) -> "SweepConfig":
        """Expand *model_path* into a concrete list of ``.gguf`` files."""
        raw = self.model_path
        expanded = Path(raw).expanduser()
        resolved = expanded.resolve()

        if resolved.is_dir():
            # Directory: collect every .gguf inside (non-recursive).
            paths = sorted(resolved.glob("*.gguf"))
            if not paths:
                raise ValueError(f"No .gguf files found in directory: {resolved}")
        elif resolved.is_file():
            paths = [resolved]
        else:
            # Treat as a glob pattern: split into parent directory + name pattern.
            parent = expanded.parent.resolve()
            pattern = expanded.name
            if not parent.exists():
                raise ValueError(f"model_path does not exist: {resolved}")
            paths = sorted(p.resolve() for p in parent.glob(pattern))
            if not paths:
                raise ValueError(f"No files match pattern: {raw}")

        self.model_paths = paths
        return self

    def combos(self) -> list["BenchCombo"]:
        """Return the full Cartesian product of models × n_ctx × n_batch."""
        return [
            BenchCombo(model_path=model, n_ctx=ctx, n_batch=batch)
            for model, ctx, batch in itertools.product(
                self.model_paths, self.n_ctx, self.n_batch
            )
        ]


@dataclass
class BenchCombo:
    """A single (model, n_ctx, n_batch) parameter combination."""

    model_path: Path
    n_ctx: int
    n_batch: int


class AutoLimitConfig(BaseModel):
    """Validated representation of the ``[auto-limit]`` block in a suite TOML.

    Example TOML::

        [auto-limit]
        model_path = "~/models/model.gguf"
        min_ctx    = 2048
        max_ctx    = 131072
        tolerance  = 1024
    """

    model_path: str
    min_ctx: int = 2048
    max_ctx: int = 131072
    tolerance: int = 1024
    runner_type: str = "llama-bench"
    runner_params: dict[str, Any] = Field(default_factory=dict)

    # Populated by the model_validator below; not read from TOML.
    resolved_model_path: Path = Field(default=Path())

    @model_validator(mode="after")
    def resolve_model(self) -> "AutoLimitConfig":
        """Expand and resolve *model_path* to an absolute file path."""
        expanded = Path(self.model_path).expanduser().resolve()
        if not expanded.exists():
            raise ValueError(f"Model file not found: {expanded}")
        if not expanded.is_file():
            raise ValueError(f"model_path must be a single file, got: {expanded}")
        self.resolved_model_path = expanded
        return self


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


# ---------------------------------------------------------------------------
# Suite config helpers
# ---------------------------------------------------------------------------


def _resolve_results_file(
    config_path: Path | None,
    cli_override: Path | None,
    toml_results: str | None,
) -> Path:
    """Determine the results JSONL path.

    Precedence (highest → lowest):
        1. ``--results`` CLI flag (*cli_override*)
        2. ``results`` field in the TOML root
        3. Auto-generated: ``<toml_stem>_<YYYYMMDD_HHMM>.jsonl``
    """
    if cli_override is not None:
        return cli_override
    if toml_results:
        return Path(toml_results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    stem = config_path.stem if config_path else "results"
    return Path("results") / f"{stem}_{ts}.jsonl"


def load_suite_config(
    config_path: Path,
) -> tuple[dict, Path]:
    """Parse a benchmark suite TOML and return ``(raw_dict, default_results)``.

    The returned *default_results* uses :func:`_resolve_results_file` with
    no CLI override — callers should prefer calling ``_resolve_results_file``
    directly when a CLI ``--results`` flag is available.
    """
    if not config_path.exists():
        console.print(f"[error]Config file not found:[/error] {config_path}")
        raise typer.Exit(code=1)

    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    return raw, _resolve_results_file(config_path, None, raw.get("results"))


# Keys that may appear at the TOML root and are inherited by sections.
_SHARED_TOML_KEYS = {"model_path", "runner_type", "runner_params"}


def _merge_shared_params(raw: dict, section: str) -> dict:
    """Merge root-level shared params into a section dict.

    Section-level values take precedence over root-level values.
    This lets users declare ``model_path`` once at the top of the TOML
    and have it inherited by both ``[auto-limit]`` and ``[sweep]``.
    """
    shared = {k: v for k, v in raw.items() if k in _SHARED_TOML_KEYS}
    section_data = dict(raw.get(section, {}))
    return {**shared, **section_data}


# ---------------------------------------------------------------------------
# JSONL result writer
# ---------------------------------------------------------------------------


def _write_result(record: dict, results_file: Path) -> None:
    """Append one JSON record as a line to *results_file*."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def execute_auto_limit(
    model_path: Path,
    min_ctx: int,
    max_ctx: int,
    tolerance: int = 1024,
    runner_type: str = "llama-bench",
    runner_params: dict[str, Any] | None = None,
) -> int:
    """Binary-search for the largest safe context window.

    Parameters
    ----------
    model_path:
        GGUF file to probe.
    min_ctx / max_ctx:
        Search bounds (inclusive).
    tolerance:
        Stop when ``hi - lo < tolerance``; the safe value is ``lo``.
    runner_params:
        Extra parameters forwarded to ``runner.setup()``.

    Returns
    -------
    int
        Largest *n_ctx* that did **not** trigger OOM.
    """
    runner = get_runner(runner_type)
    runner.setup(runner_params or {})

    lo, hi = min_ctx, max_ctx
    last_good: int = 0
    iteration = 0

    model_name = model_path.name
    console.print()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.fields[detail]}[/cyan]"),
        console=console,
        transient=False,
    ) as progress:
        # Approximate upper bound on iterations: log2((max_ctx - min_ctx) / tolerance)
        est_iters = max(1, math.ceil(math.log2(max(1, (hi - lo) / tolerance))))
        task = progress.add_task(
            f"Probing [hw]{model_name}[/hw]",
            total=est_iters,
            detail=f"lo={lo}  hi={hi}",
        )

        while hi - lo >= tolerance:
            mid = (lo + hi) // 2
            iteration += 1
            progress.update(
                task,
                detail=f"try n_ctx={mid:,}  (lo={lo:,}  hi={hi:,})",
            )
            t0 = time.monotonic()
            ok = runner.probe_ctx(model_path, mid)
            elapsed = time.monotonic() - t0
            if ok:
                last_good = mid
                lo = mid + 1
                status = "[success]✓ pass[/success]"
            else:
                hi = mid - 1
                status = "[error]✗ OOM[/error]"
            console.print(
                f"  iter {iteration:>2}: n_ctx={mid:>7,}  {status}  "
                f"→ window [{lo:,}, {hi:,}]  ({elapsed:.1f}s)"
            )
            progress.advance(task)

        progress.update(task, detail="done", completed=est_iters)

    safe = last_good if last_good else lo

    runner.teardown()
    return safe


def execute_sweep(
    results_file: Path,
    config_path: Path | None = None,
    sweep_config: SweepConfig | None = None,
    max_ctx_cap: int | None = None,
) -> None:
    """Run a parameter sweep.

    Exactly one of *config_path* or *sweep_config* must be supplied.

    Parameters
    ----------
    results_file:
        Destination JSONL file for results.
    config_path:
        Path to a TOML file (must contain a ``[sweep]`` section).
    sweep_config:
        Pre-built :class:`SweepConfig` — used by the CLI-only path.
    max_ctx_cap:
        If set, combos whose ``n_ctx`` exceeds this value are skipped.
    """
    if config_path is not None and sweep_config is not None:
        raise ValueError("Provide config_path or sweep_config, not both.")

    if sweep_config is not None:
        cfg = sweep_config
    elif config_path is not None:
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
            cfg = SweepConfig(**_merge_shared_params(raw, "sweep"))
        except Exception as exc:  # pydantic ValidationError
            console.print(f"[error]Invalid sweep config:[/error] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        raise ValueError("One of config_path or sweep_config is required.")

    combos = cfg.combos()

    # Apply max-ctx cap (from auto-limit or CLI override)
    if max_ctx_cap is not None:
        original = len(combos)
        combos = [c for c in combos if c.n_ctx <= max_ctx_cap]
        skipped = original - len(combos)
        if skipped:
            console.print(
                f"  [warning]Skipping {skipped} combo(s) with n_ctx > {max_ctx_cap:,}[/warning]"
            )

    total = len(combos)

    model_names = ", ".join(f"[hw]{m.name}[/hw]" for m in cfg.model_paths)
    console.print(
        f"[info]Sweep:[/info] [bold]{total}[/bold] combination(s) "
        f"across [bold]{len(cfg.model_paths)}[/bold] model(s): {model_names}\n"
        f"  Runner  : {cfg.runner_type}\n"
        f"  n_ctx   : {cfg.n_ctx}\n"
        f"  n_batch : {cfg.n_batch}\n"
        f"  Results : [bold]{results_file.resolve()}[/bold]"
    )

    runner = get_runner(cfg.runner_type)
    runner.setup(cfg.runner_params)

    passed = failed = 0

    try:
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
                label = f"{combo.model_path.name} ctx={combo.n_ctx} batch={combo.n_batch}"
                progress.update(task, combo=label)

                run_config: dict[str, Any] = {
                    "model_path": str(combo.model_path),
                    "n_ctx": combo.n_ctx,
                    "n_batch": combo.n_batch,
                }
                t0 = time.monotonic()
                raw_result = runner.run(run_config)
                elapsed = time.monotonic() - t0

                if raw_result is None:
                    record = None
                else:
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "runner_type": cfg.runner_type,
                        "model_path": str(combo.model_path),
                        "n_ctx": combo.n_ctx,
                        "n_batch": combo.n_batch,
                        "hardware": _hw_sniffer.snapshot(),
                        "results": raw_result["results"],
                    }
                    _write_result(record, results_file)

                dur = f"  ({elapsed:.1f}s)"
                if record is None:
                    console.print(f"  [error]✗[/error] [{i}/{total}] {label} — FAILED{dur}")
                    failed += 1
                else:
                    # Pull out the tok/s figure if llama-bench emits it
                    tps: str = ""
                    try:
                        tps_val = record["results"][0]["avg_ts"]
                        tps = f"  {tps_val:.1f} tok/s"
                    except (KeyError, IndexError, TypeError):
                        pass
                    console.print(f"  [success]✓[/success] [{i}/{total}] {label}{tps}{dur}")
                    passed += 1

                progress.advance(task)

    finally:
        runner.teardown()

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
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to a TOML suite file containing a [sweep] section",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to GGUF model file/dir/glob (overrides TOML)"
    ),
    n_ctx: Optional[str] = typer.Option(
        None, "--n-ctx", help="Comma-separated context sizes, e.g. '8192,16384,32768'"
    ),
    n_batch: Optional[str] = typer.Option(
        None, "--n-batch", help="Comma-separated batch sizes, e.g. '512,1024'"
    ),
    runner: Optional[str] = typer.Option(
        None, "--runner", help="Runner backend (default: llama-bench)"
    ),
    results_file: Optional[Path] = typer.Option(
        None,
        "--results",
        "-r",
        help="JSONL results file (default: auto-generated from config name + timestamp)",
    ),
) -> None:
    """Run a declarative parameter sweep.

    Can be driven by a TOML config (``ppb sweep suite.toml``) or purely
    by CLI flags (``ppb sweep --model ./m.gguf --n-ctx 8192,16384 --n-batch 512``).
    CLI flags override TOML values when both are provided.
    """

    def _parse_int_list(raw: str, name: str) -> list[int]:
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError as exc:
            console.print(f"[error]Invalid --{name}:[/error] {exc}")
            raise typer.Exit(code=1) from exc

    cfg: SweepConfig
    resolved_results: Path

    if config is not None:
        # ---- TOML-driven (with optional CLI overrides) --------------------
        cfg_path = Path(config)
        raw, default_results = load_suite_config(cfg_path)
        if "sweep" not in raw:
            console.print(
                "[error]Missing [sweep] section in config file.[/error]\n"
                f"  File: {cfg_path}"
            )
            raise typer.Exit(code=1)

        sweep_raw: dict[str, Any] = _merge_shared_params(raw, "sweep")
        # CLI overrides
        if model is not None:
            sweep_raw["model_path"] = model
        if n_ctx is not None:
            sweep_raw["n_ctx"] = _parse_int_list(n_ctx, "n-ctx")
        if n_batch is not None:
            sweep_raw["n_batch"] = _parse_int_list(n_batch, "n-batch")
        if runner is not None:
            sweep_raw["runner_type"] = runner

        try:
            cfg = SweepConfig(**sweep_raw)
        except Exception as exc:
            console.print(f"[error]Invalid sweep config:[/error] {exc}")
            raise typer.Exit(code=1) from exc

        resolved_results = _resolve_results_file(
            cfg_path, results_file, raw.get("results")
        )
    else:
        # ---- Pure-CLI mode ------------------------------------------------
        if not model:
            console.print(
                "[error]Provide a TOML config or --model flag.[/error]\n"
                "  Usage: ppb sweep suite.toml\n"
                "     or: ppb sweep --model ./m.gguf --n-ctx 8192,16384 --n-batch 512"
            )
            raise typer.Exit(code=1)
        if not n_ctx or not n_batch:
            console.print(
                "[error]--n-ctx and --n-batch are required in CLI-only mode.[/error]"
            )
            raise typer.Exit(code=1)

        try:
            cfg = SweepConfig(
                model_path=model,
                n_ctx=_parse_int_list(n_ctx, "n-ctx"),
                n_batch=_parse_int_list(n_batch, "n-batch"),
                runner_type=runner or "llama-bench",
            )
        except Exception as exc:
            console.print(f"[error]Invalid sweep config:[/error] {exc}")
            raise typer.Exit(code=1) from exc

        resolved_results = _resolve_results_file(None, results_file, None)

    console.print(f"[info]Starting sweep[/info] …")
    execute_sweep(results_file=resolved_results, sweep_config=cfg)


@app.command(name="hw-info")
def hw_info() -> None:
    """Print a snapshot of the detected hardware profile."""
    hw = _hw_sniffer.snapshot()

    console.print("\n[info]Hardware Profile[/info]")
    console.print(f"  OS          : [bold]{hw['os']['system']} {hw['os']['release']}[/bold]  ({hw['os']['machine']})")

    cpu = hw["cpu"]
    cpu_label = cpu.get("model", cpu.get("processor", "unknown"))
    cores = cpu.get("cores", "?")
    console.print(f"  CPU         : [bold]{cpu_label}[/bold]  ({cores} cores)")

    ram = hw["ram"]
    ram_gb = ram.get("total_gb", "?")
    console.print(f"  RAM         : [bold]{ram_gb} GB[/bold]")

    gpus = hw["gpus"]
    if gpus:
        for gpu in gpus:
            vram = gpu.get("vram_total_gb", gpu.get("vram", "?"))
            parts: list[str] = [f"[hw]{gpu['name']}[/hw]"]
            parts.append(f"{vram} GB VRAM")
            if "compute_capability" in gpu:
                parts.append(f"sm_{gpu['compute_capability'].replace('.', '')}")
            if "cuda_version" in gpu:
                parts.append(f"CUDA {gpu['cuda_version']}")
            if "metal_version" in gpu:
                parts.append(gpu["metal_version"])
            if "gpu_cores" in gpu:
                parts.append(f"{gpu['gpu_cores']} GPU cores")
            if "driver" in gpu:
                parts.append(f"driver {gpu['driver']}")
            if "power_limit_w" in gpu:
                parts.append(f"{gpu['power_limit_w']} W TDP")
            if "pcie_gen" in gpu:
                parts.append(f"PCIe {gpu['pcie_gen']}.0 x{gpu.get('pcie_width', '?')}")
            idx = gpu.get("index", "?")
            console.print(f"  GPU [{idx}]     : " + "  ".join(parts))
    else:
        console.print("  GPU         : [warning]none detected[/warning]")

    rt = hw.get("runtime", {})
    console.print(f"  Python      : [bold]{rt.get('python_version', '?')}[/bold]")
    if "llama_bench" in rt:
        console.print(f"  llama-bench : [bold]{rt['llama_bench']}[/bold]")

    console.print()

    # Also dump the raw dict for piping / debugging
    log.debug("Hardware snapshot:\n%s", json.dumps(hw, indent=2))


@app.command(name="auto-limit")
def auto_limit(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to a TOML suite file containing an [auto-limit] section",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Path to the GGUF model file (overrides TOML)"
    ),
    min_ctx: Optional[int] = typer.Option(
        None, "--min-ctx", help="Minimum context length to probe (default: 2048)"
    ),
    max_ctx: Optional[int] = typer.Option(
        None, "--max-ctx", help="Maximum context length to probe (default: 131072)"
    ),
    tolerance: Optional[int] = typer.Option(
        None,
        "--tolerance",
        "-t",
        help="Stop searching when hi - lo < this value (default: 1024)",
    ),
    runner: Optional[str] = typer.Option(
        None,
        "--runner",
        help="Runner backend to use for probing (default: llama-bench)",
    ),
) -> None:
    """Binary-search for the maximum context window before OOM.

    Can be driven by a TOML config (``ppb auto-limit suite.toml``) or purely
    by CLI flags (``ppb auto-limit --model ./m.gguf``).
    CLI flags override TOML values.
    """
    # --- Load TOML defaults if a config was provided -------------------------
    al_model: str | None = model
    al_min: int = 2048
    al_max: int = 131072
    al_tol: int = 1024
    al_runner: str = "llama-bench"
    al_runner_params: dict[str, Any] = {}

    if config is not None:
        cfg_path = Path(config)
        if not cfg_path.exists():
            console.print(f"[error]Config file not found:[/error] {cfg_path}")
            raise typer.Exit(code=1)
        raw, _ = load_suite_config(cfg_path)
        if "auto-limit" in raw:
            try:
                al_cfg = AutoLimitConfig(**_merge_shared_params(raw, "auto-limit"))
            except Exception as exc:
                console.print(f"[error]Invalid [auto-limit] config:[/error] {exc}")
                raise typer.Exit(code=1) from exc
            al_model = al_model or str(al_cfg.resolved_model_path)
            al_min = al_cfg.min_ctx
            al_max = al_cfg.max_ctx
            al_tol = al_cfg.tolerance
            al_runner = al_cfg.runner_type
            al_runner_params = al_cfg.runner_params
        elif model is None:
            console.print(
                "[error]No [auto-limit] section in config and no --model flag.[/error]"
            )
            raise typer.Exit(code=1)
    elif model is None:
        console.print("[error]Provide a TOML config or --model flag.[/error]")
        raise typer.Exit(code=1)

    # CLI overrides beat TOML
    if min_ctx is not None:
        al_min = min_ctx
    if max_ctx is not None:
        al_max = max_ctx
    if tolerance is not None:
        al_tol = tolerance
    if runner is not None:
        al_runner = runner

    model_path = Path(al_model).expanduser().resolve()  # type: ignore[arg-type]
    if not model_path.exists():
        console.print(f"[error]Model file not found:[/error] {model_path}")
        raise typer.Exit(code=1)

    console.print(
        f"[info]Auto-limit[/info] probing [hw]{model_path.name}[/hw]\n"
        f"  Range     : [bold]{al_min:,}[/bold] → [bold]{al_max:,}[/bold] tokens\n"
        f"  Tolerance : [bold]{al_tol:,}[/bold] tokens"
    )

    safe = execute_auto_limit(
        model_path=model_path,
        min_ctx=al_min,
        max_ctx=al_max,
        tolerance=al_tol,
        runner_type=al_runner,
        runner_params=al_runner_params,
    )

    if safe == 0:
        console.print(
            "\n[error]Could not find a working context size.[/error]\n"
            f"  Even n_ctx={al_min:,} failed — check that the model loads at all."
        )
        raise typer.Exit(code=1)

    console.print(
        f"\n[success]✓ Maximum safe context for[/success] [hw]{model_path.name}[/hw]\n"
        f"\n    [bold green]{safe:,} tokens[/bold green]\n"
    )


@app.command(name="all")
def run_all(
    config: Path = typer.Argument(
        ...,
        help="Path to a TOML benchmark suite file",
        exists=True,
        readable=True,
    ),
    results_file: Optional[Path] = typer.Option(
        None,
        "--results",
        "-r",
        help="JSONL results file (default: auto-generated from config name + timestamp)",
    ),
) -> None:
    """Run the full benchmark suite: auto-limit → sweep.

    1. **auto-limit** — discover the maximum safe context window.
    2. **sweep** — run the parameter sweep, skipping any combo whose
       ``n_ctx`` exceeds the limit found in step 1.

    Both steps read their configuration from the same TOML file.
    If the TOML has no ``[auto-limit]`` section, the auto-limit step
    is skipped and the sweep runs unmodified.
    """
    raw, default_results = load_suite_config(config)
    resolved_results = results_file if results_file is not None else default_results

    console.print(
        f"[info]Full benchmark suite[/info] from [bold]{config}[/bold]\n"
        f"  Results → [bold]{resolved_results.resolve()}[/bold]\n"
    )

    # -- Phase 1: auto-limit (optional) -----------------------------------
    max_ctx_cap: int | None = None

    if "auto-limit" in raw:
        console.print("[info]Phase 1 / 2:[/info] auto-limit")
        try:
            al_cfg = AutoLimitConfig(**_merge_shared_params(raw, "auto-limit"))
        except Exception as exc:
            console.print(f"[error]Invalid [auto-limit] config:[/error] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(
            f"  Probing [hw]{al_cfg.resolved_model_path.name}[/hw]\n"
            f"  Range     : [bold]{al_cfg.min_ctx:,}[/bold] → "
            f"[bold]{al_cfg.max_ctx:,}[/bold] tokens\n"
            f"  Tolerance : [bold]{al_cfg.tolerance:,}[/bold] tokens"
        )

        safe = execute_auto_limit(
            model_path=al_cfg.resolved_model_path,
            min_ctx=al_cfg.min_ctx,
            max_ctx=al_cfg.max_ctx,
            tolerance=al_cfg.tolerance,
            runner_type=al_cfg.runner_type,
            runner_params=al_cfg.runner_params,
        )

        if safe == 0:
            console.print(
                "\n[error]auto-limit failed — could not find a working context "
                "size. Skipping sweep.[/error]"
            )
            raise typer.Exit(code=1)

        max_ctx_cap = safe
        console.print(
            f"\n[success]✓ Max safe context:[/success] "
            f"[bold green]{safe:,} tokens[/bold green]\n"
        )
    else:
        console.print("[info]No [auto-limit] section — skipping to sweep.[/info]\n")

    # -- Phase 2: sweep ---------------------------------------------------
    if "sweep" not in raw:
        console.print("[warning]No [sweep] section — nothing more to do.[/warning]")
        return

    console.print("[info]Phase 2 / 2:[/info] sweep")
    execute_sweep(
        config_path=config,
        results_file=resolved_results,
        max_ctx_cap=max_ctx_cap,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
