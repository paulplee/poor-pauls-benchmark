"""
Poor Paul's Benchmark (PPB) — CLI entry point.

An automated evaluation framework for local LLM inference with a
pluggable runner architecture.  Built-in runners: ``llama-bench``
(raw throughput) and ``llama-server`` (TTFT / ITL latency).
"""

import contextlib
import csv
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
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
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

from datasets import download_dataset
from datasets.sharegpt import SHAREGPT_FILENAME, SHAREGPT_REPO
from runners import get_runner
from utils.flattener import compute_file_sha256, flatten_benchmark_row
from utils.publisher import check_hf_token, publish_to_hf

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
# httpx logs every request at INFO level; suppress it globally — per-request
# lines (health polls, completion POSTs) are noise for an end-user tool.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# .env & defaults
# ---------------------------------------------------------------------------

load_dotenv(override=True)  # .env always wins over pre-set shell env vars

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
                            data["total_gb"] = round(kb / (1024**2), 1)
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
                data["total_gb"] = round(total / (1024**3), 1)
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
                data["total_gb"] = round(stat.ullTotalPhys / (1024**3), 1)
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
                        "vram_total_gb": round(float(mem.total) / (1024**3), 1),
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
                        gpu["pcie_gen"] = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(
                            handle
                        )
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
                        if len(parts) < 4:
                            continue
                        gpu: dict = {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "driver": parts[2],
                        }
                        # memory.total is "[N/A]" on unified-memory GPUs (e.g.
                        # NVIDIA GB10 Grace Blackwell) — treat system RAM as VRAM.
                        vram_str = parts[3]
                        if vram_str and vram_str not in ("[N/A]", "N/A"):
                            with contextlib.suppress(ValueError):
                                vram_mb = float(vram_str)
                                gpu["vram_total_bytes"] = int(vram_mb * 1024**2)
                                gpu["vram_total_gb"] = round(vram_mb / 1024, 1)
                        if len(parts) > 4 and parts[4] not in ("[N/A]", "N/A", ""):
                            gpu["compute_capability"] = parts[4]
                        if len(parts) > 5 and parts[5] not in ("[N/A]", "N/A", ""):
                            with contextlib.suppress(ValueError):
                                gpu["power_limit_w"] = round(float(parts[5]))
                        if len(parts) > 6 and parts[6] not in ("[N/A]", "N/A", ""):
                            with contextlib.suppress(ValueError):
                                gpu["pcie_gen"] = int(parts[6])
                        if len(parts) > 7 and parts[7] not in ("[N/A]", "N/A", ""):
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
                        vram_str = line.split(":", 1)[1].strip()  # e.g. "8 GB"
                        m = re.match(r"([\d.]+)\s*GB", vram_str, re.IGNORECASE)
                        if m:
                            current["vram_total_gb"] = float(m.group(1))
                    elif line.startswith("Metal Support:") and current:
                        # Use Metal version as the driver identifier on macOS
                        current["driver"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Total Number of Cores:") and current:
                        # GPU cores on Apple Silicon
                        current["gpu_cores"] = line.split(":", 1)[1].strip()
                if current:
                    # Apple Silicon uses unified memory — no dedicated VRAM line.
                    # Fall back to total system RAM as the GPU memory budget.
                    if "vram_total_gb" not in current:
                        try:
                            mem_out = subprocess.check_output(
                                ["sysctl", "-n", "hw.memsize"],
                                text=True,
                                timeout=5,
                            )
                            current["vram_total_gb"] = round(
                                int(mem_out.strip()) / (1024**3), 1
                            )
                        except (
                            subprocess.SubprocessError,
                            FileNotFoundError,
                            ValueError,
                        ):
                            pass
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
                        if (
                            line
                            and _ver_re.search(line)
                            and not line.lower().startswith("usage:")
                        ):
                            data["llama_bench"] = line
                            break
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                pass

        return data


# Module-level singleton
_hw_sniffer = HardwareSniffer()


# ---------------------------------------------------------------------------
# GPU power sampler
# ---------------------------------------------------------------------------


class PowerSampler:
    """Sample power draw in a background thread during a benchmark run.

    Source priority (first one that succeeds wins):

    1. **NVIDIA NVML** — GPU board power on Linux / Windows.
    2. **Linux RAPL** — Intel / AMD CPU *package* power via the kernel
       ``powercap`` interface (``/sys/class/powercap/intel-rapl:*/``).  No
       root required on most distributions (Ubuntu, Fedora, …).
    3. **macOS IOReport** — total SoC power (CPU + GPU + DRAM + fabric) on
       Apple Silicon via ``libIOReport.dylib``.  No root required.
    4. **macOS powermetrics** — total SoC power (CPU + GPU + ANE) on Apple
       Silicon.  Attempted without ``sudo``; only succeeds when the process
       already has the required privilege (e.g. running as root or with an
       appropriate entitlement).  Silently skipped otherwise.

    Usage::

        sampler = PowerSampler()
        sampler.start()
        runner.run(config)          # ← benchmark happens here
        avg_w, max_w = sampler.stop()

    Returns ``(None, None)`` when no power source is available.
    Polls every 0.5 s; fine-grained enough for multi-second runs with
    negligible overhead.
    """

    _POLL_INTERVAL_S = 0.5

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a daemon thread."""
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float | None, float | None]:
        """Stop sampling and return ``(avg_watts, max_watts)``.

        Safe to call even if ``start()`` was never called or no power
        source was found.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if not self._samples:
            return None, None
        return round(sum(self._samples) / len(self._samples), 1), round(
            max(self._samples), 1
        )

    def _poll(self) -> None:
        if self._try_nvml():
            return
        if platform.system() == "Linux" and self._try_rapl():
            return
        if platform.system() == "Darwin":
            if self._try_ioreport():
                return
            self._try_powermetrics()

    # -- source: NVIDIA NVML -----------------------------------------------

    def _try_nvml(self) -> bool:
        """Return True and run sampling loop if NVML / NVIDIA GPU is available."""
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                pynvml.nvmlShutdown()
                return False
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Verify the power query is supported before entering the loop.
            pynvml.nvmlDeviceGetPowerUsage(handle)
            while not self._stop_event.is_set():
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    self._samples.append(mw / 1000.0)
                except pynvml.NVMLError:
                    pass
                self._stop_event.wait(self._POLL_INTERVAL_S)
            pynvml.nvmlShutdown()
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- source: Linux RAPL (Intel / AMD CPU package) ----------------------

    def _try_rapl(self) -> bool:
        """Return True and run sampling loop using Linux powercap RAPL.

        Reads CPU *package* power (the whole socket — cores + uncore + iGPU)
        from the kernel ``powercap`` sysfs interface.  No root required on
        most modern Linux distributions.
        """
        # The path layout differs slightly between kernel versions / distros.
        candidates = [
            Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
            Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
        ]
        energy_path: Path | None = None
        for p in candidates:
            try:
                p.read_text()
                energy_path = p
                break
            except (FileNotFoundError, PermissionError):
                continue
        if energy_path is None:
            return False

        # The counter wraps at max_energy_range_uj; handle that gracefully.
        try:
            max_range = int(
                (energy_path.parent / "max_energy_range_uj").read_text().strip()
            )
        except Exception:  # noqa: BLE001
            max_range = 2**32

        try:
            prev_uj = int(energy_path.read_text().strip())
            prev_t = time.monotonic()
            self._stop_event.wait(self._POLL_INTERVAL_S)
            while not self._stop_event.is_set():
                curr_uj = int(energy_path.read_text().strip())
                curr_t = time.monotonic()
                dt = curr_t - prev_t
                d_uj = (curr_uj - prev_uj) % max_range  # wraparound-safe
                if dt > 0:
                    self._samples.append(round(d_uj / (dt * 1e6), 1))  # µJ → W
                prev_uj = curr_uj
                prev_t = curr_t
                self._stop_event.wait(self._POLL_INTERVAL_S)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- source: macOS IOReport (Apple Silicon SoC, no root) ----------------

    # Channels whose names match these patterns are sub-components already
    # accounted for by a higher-level aggregate (e.g. per-core CPU entries
    # are summed in "CPU Energy").  We also skip "* Energy" channels whose
    # values use nJ instead of mJ, duplicating a top-level channel.
    _IOREPORT_SKIP_RE = re.compile(
        r"^EACC_CPU\d"  # per efficiency-core
        r"|^PACC\d+_CPU\d"  # per performance-core
        r"|^EACC_CPU$"  # E-cluster aggregate (in "CPU Energy")
        r"|^PACC\d+_CPU$"  # P-cluster aggregate (in "CPU Energy")
        r"|DTL"  # per-domain technology level detail
        r"| Energy$"  # nJ-scale aggregate duplicates ("GPU Energy")
    )

    def _try_ioreport(self) -> bool:
        """Sample SoC power via the macOS IOReport framework (no root needed).

        Reads the "Energy Model" channel group through ``libIOReport.dylib``,
        which exposes per-block energy counters (CPU, GPU, DRAM, AMCC, …)
        without requiring ``sudo``.  Works on Apple Silicon M1–M4.
        """
        try:
            import ctypes
            import ctypes.util
            import plistlib
            import struct

            cf_path = ctypes.util.find_library("CoreFoundation")
            if cf_path is None:
                return False
            cf = ctypes.cdll.LoadLibrary(cf_path)
            iorep = ctypes.cdll.LoadLibrary("/usr/lib/libIOReport.dylib")
        except (OSError, TypeError):
            return False

        CFTypeRef = ctypes.c_void_p
        CFStringRef = ctypes.c_void_p
        CFDictionaryRef = ctypes.c_void_p
        CFAllocatorRef = ctypes.c_void_p
        CFDataRef = ctypes.c_void_p
        CFIndex = ctypes.c_long

        kCFStringEncodingUTF8 = 0x08000100
        kCFPropertyListXMLFormat_v1_0 = 100

        # CoreFoundation signatures
        cf.CFStringCreateWithCString.restype = CFStringRef
        cf.CFStringCreateWithCString.argtypes = [
            CFAllocatorRef,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [CFTypeRef]
        cf.CFPropertyListCreateData.restype = CFDataRef
        cf.CFPropertyListCreateData.argtypes = [
            CFAllocatorRef,
            CFTypeRef,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        cf.CFDataGetLength.restype = CFIndex
        cf.CFDataGetLength.argtypes = [CFDataRef]
        cf.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_uint8)
        cf.CFDataGetBytePtr.argtypes = [CFDataRef]

        # IOReport signatures
        iorep.IOReportCopyChannelsInGroup.restype = CFDictionaryRef
        iorep.IOReportCopyChannelsInGroup.argtypes = [
            CFStringRef,
            CFStringRef,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        iorep.IOReportCreateSubscription.restype = CFTypeRef
        iorep.IOReportCreateSubscription.argtypes = [
            CFTypeRef,
            CFDictionaryRef,
            ctypes.POINTER(CFDictionaryRef),
            ctypes.c_uint64,
            CFTypeRef,
        ]
        iorep.IOReportCreateSamples.restype = CFDictionaryRef
        iorep.IOReportCreateSamples.argtypes = [CFTypeRef, CFDictionaryRef, CFTypeRef]
        iorep.IOReportCreateSamplesDelta.restype = CFDictionaryRef
        iorep.IOReportCreateSamplesDelta.argtypes = [
            CFDictionaryRef,
            CFDictionaryRef,
            CFTypeRef,
        ]

        def _cfstr(s: bytes):  # -> CFStringRef
            return cf.CFStringCreateWithCString(None, s, kCFStringEncodingUTF8)

        def _cfdict_to_plist(ref: int) -> dict | list | None:
            """Convert a CFPropertyList to a Python object via XML plist."""
            if not ref:
                return None
            xml = cf.CFPropertyListCreateData(
                None,
                ref,
                kCFPropertyListXMLFormat_v1_0,
                0,
                None,
            )
            if not xml:
                return None
            length = cf.CFDataGetLength(xml)
            buf = cf.CFDataGetBytePtr(xml)
            raw = bytes(
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8 * length)).contents
            )
            cf.CFRelease(xml)
            return plistlib.loads(raw)

        def _watts_from_delta(delta_ref: int, elapsed_s: float) -> float | None:
            """Parse a delta sample and return total SoC watts."""
            data = _cfdict_to_plist(delta_ref)
            if not isinstance(data, dict):
                return None
            channels = data.get("IOReportChannels")
            if not channels:
                return None
            total_mj = 0.0
            for ch in channels:
                legend = ch.get("LegendChannel", [])
                name = legend[2] if len(legend) > 2 else ""
                if not isinstance(name, str) or self._IOREPORT_SKIP_RE.search(name):
                    continue
                raw = ch.get("RawElements")
                if not isinstance(raw, bytes) or len(raw) < 40:
                    continue
                # Energy delta (mJ) sits at int64 index 4 (byte offset 32).
                mj = struct.unpack_from("<q", raw, 32)[0]
                if 0 < mj < 1_000_000:  # sanity: < 1 MJ per interval
                    total_mj += mj
            if total_mj <= 0 or elapsed_s <= 0:
                return None
            return round(total_mj / 1000.0 / elapsed_s, 1)

        # --- Set up subscription -----------------------------------------
        try:
            group = _cfstr(b"Energy Model")
            channels = iorep.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
            if not channels:
                return False

            sub_dict = CFDictionaryRef()
            sub = iorep.IOReportCreateSubscription(
                None,
                channels,
                ctypes.byref(sub_dict),
                0,
                None,
            )
            if not sub:
                return False

            # Probe: take one delta to verify the API returns valid data.
            s1 = iorep.IOReportCreateSamples(sub, sub_dict, None)
            if not s1:
                return False
            self._stop_event.wait(self._POLL_INTERVAL_S)
            s2 = iorep.IOReportCreateSamples(sub, sub_dict, None)
            if not s2:
                cf.CFRelease(s1)
                return False
            delta = iorep.IOReportCreateSamplesDelta(s1, s2, None)
            watts = _watts_from_delta(delta, self._POLL_INTERVAL_S) if delta else None
            if delta:
                cf.CFRelease(delta)
            cf.CFRelease(s1)

            if watts is None:
                cf.CFRelease(s2)
                return False
            self._samples.append(watts)

            # Confirmed working — keep sampling until stop requested.
            prev = s2
            while not self._stop_event.is_set():
                self._stop_event.wait(self._POLL_INTERVAL_S)
                cur = iorep.IOReportCreateSamples(sub, sub_dict, None)
                if not cur:
                    continue
                delta = iorep.IOReportCreateSamplesDelta(prev, cur, None)
                if delta:
                    w = _watts_from_delta(delta, self._POLL_INTERVAL_S)
                    if w is not None:
                        self._samples.append(w)
                    cf.CFRelease(delta)
                cf.CFRelease(prev)
                prev = cur
            cf.CFRelease(prev)
            return True
        except Exception:  # noqa: BLE001
            return False

    # -- source: macOS powermetrics (Apple Silicon SoC) --------------------

    def _try_powermetrics(self) -> bool:
        """Return True and run sampling loop using macOS ``powermetrics``.

        Samples total SoC *package* energy (CPU + GPU + ANE) via one
        ``powermetrics`` subprocess per poll interval.  Requires that the
        calling process already has root / the necessary entitlement;
        silently returns False when permission is denied.
        """
        if not shutil.which("powermetrics"):
            return False
        interval_ms = int(self._POLL_INTERVAL_S * 1000)
        cmd = [
            "powermetrics",
            "--samplers",
            "cpu_power",
            "-n",
            "1",
            "-i",
            str(interval_ms),
            "-f",
            "json",
        ]
        # Probe: one sample to confirm we have permission and can parse.
        try:
            probe = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._POLL_INTERVAL_S + 2,
            )
            if probe.returncode != 0:
                return False
            watts = self._parse_powermetrics_json(probe.stdout, interval_ms)
            if watts is None:
                return False
            self._samples.append(watts)
        except Exception:  # noqa: BLE001
            return False

        # Confirmed working — continue sampling until stop requested.
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._POLL_INTERVAL_S + 2,
                )
                if result.returncode == 0:
                    watts = self._parse_powermetrics_json(result.stdout, interval_ms)
                    if watts is not None:
                        self._samples.append(watts)
            except Exception:  # noqa: BLE001
                pass
        return True

    @staticmethod
    def _parse_powermetrics_json(output: str, interval_ms: int) -> float | None:
        """Extract total package watts from a ``powermetrics -f json`` snapshot.

        ``powermetrics`` reports energy in mJ consumed during the sample
        interval; divide by elapsed seconds to get average watts.
        """
        try:
            data = json.loads(output)
            elapsed_ns = data.get("elapsed_ns") or (interval_ms * 1_000_000)
            elapsed_s = elapsed_ns / 1e9
            processor = data.get("processor") or {}
            # Apple Silicon layout: processor.packages[0].package_energy (mJ)
            packages = processor.get("packages") or []
            if packages:
                energy_mj = packages[0].get("package_energy")
                if energy_mj is not None and elapsed_s > 0:
                    return round(energy_mj / elapsed_s / 1000, 1)  # mJ/s → W
            # Older / Intel layout: processor.package_energy (mJ)
            energy_mj = processor.get("package_energy")
            if energy_mj is not None and elapsed_s > 0:
                return round(energy_mj / elapsed_s / 1000, 1)
        except Exception:  # noqa: BLE001
            pass
        return None


# ---------------------------------------------------------------------------
# Thermal sampler (GPU temp, CPU temp, fan speed)
# ---------------------------------------------------------------------------


class ThermalSampler:
    """Sample GPU temperature, CPU temperature, and fan speed during a run.

    Source priority per metric (first one that succeeds wins):

    **GPU temperature / GPU fan speed**

    1. NVIDIA NVML — Linux / Windows.
    2. Not available on macOS Apple Silicon (SoC has no discrete GPU sensor).

    **CPU temperature**

    1. Linux ``/sys/class/hwmon`` or ``/sys/class/thermal``.
    2. macOS Apple Silicon SMC via IOKit (``TC0P`` / ``Tp09`` keys).
    3. Windows WMI ``MSAcpi_ThermalZoneTemperature`` (best-effort).

    **Fan speed**

    1. NVML GPU fan (Linux / Windows).
    2. Linux ``/sys/class/hwmon/*/fan*_input``.
    3. macOS SMC via IOKit (``F0Ac`` key).

    Usage::

        sampler = ThermalSampler()
        sampler.start()
        runner.run(config)
        stats = sampler.stop()
        # stats = {
        #     "avg_gpu_temp_c": 72.3, "max_gpu_temp_c": 78.0,
        #     "avg_cpu_temp_c": 65.1, "max_cpu_temp_c": 71.0,
        #     "avg_fan_speed_rpm": 1200, "max_fan_speed_rpm": 1500,
        # }

    Returns ``{}`` when no thermal source is available.
    Polls every 0.5 s matching :class:`PowerSampler`.
    """

    _POLL_INTERVAL_S = 0.5

    def __init__(self) -> None:
        self._gpu_temps: list[float] = []
        self._cpu_temps: list[float] = []
        self._fan_speeds: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin sampling in a daemon thread."""
        self._gpu_temps = []
        self._cpu_temps = []
        self._fan_speeds = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Stop sampling and return aggregated stats.

        Keys present only when at least one valid sample was collected.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        stats: dict[str, float] = {}
        if self._gpu_temps:
            stats["avg_gpu_temp_c"] = round(
                sum(self._gpu_temps) / len(self._gpu_temps), 1
            )
            stats["max_gpu_temp_c"] = round(max(self._gpu_temps), 1)
        if self._cpu_temps:
            stats["avg_cpu_temp_c"] = round(
                sum(self._cpu_temps) / len(self._cpu_temps), 1
            )
            stats["max_cpu_temp_c"] = round(max(self._cpu_temps), 1)
        if self._fan_speeds:
            stats["avg_fan_speed_rpm"] = round(
                sum(self._fan_speeds) / len(self._fan_speeds)
            )
            stats["max_fan_speed_rpm"] = round(max(self._fan_speeds))
        return stats

    # -- orchestration ------------------------------------------------------

    def _poll(self) -> None:
        """Detect available sources, then loop until stop."""
        system = platform.system()
        gpu_reader = self._make_gpu_temp_reader(system)
        cpu_reader = self._make_cpu_temp_reader(system)
        fan_reader = self._make_fan_reader(system)

        if not any([gpu_reader, cpu_reader, fan_reader]):
            return  # nothing available on this box

        while not self._stop_event.is_set():
            if gpu_reader:
                val = gpu_reader()
                if val is not None:
                    self._gpu_temps.append(val)
            if cpu_reader:
                val = cpu_reader()
                if val is not None:
                    self._cpu_temps.append(val)
            if fan_reader:
                val = fan_reader()
                if val is not None:
                    self._fan_speeds.append(val)
            self._stop_event.wait(self._POLL_INTERVAL_S)

    # -- GPU temperature readers -------------------------------------------

    def _make_gpu_temp_reader(self, system: str) -> Callable[[], float | None] | None:
        """Return a callable that reads GPU temp in °C, or None."""
        if system in ("Linux", "Windows"):
            return self._try_nvml_gpu_temp_reader()
        return None

    @staticmethod
    def _try_nvml_gpu_temp_reader() -> Callable[[], float | None] | None:
        """Build an NVML-based GPU temperature reader (GPU index 0)."""
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                pynvml.nvmlShutdown()
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Verify the query works before committing.
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            def _read() -> float | None:
                try:
                    return float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except Exception:  # noqa: BLE001
                    return None

            return _read
        except Exception:  # noqa: BLE001
            return None

    # -- CPU temperature readers -------------------------------------------

    def _make_cpu_temp_reader(self, system: str) -> Callable[[], float | None] | None:
        """Return a callable that reads CPU temp in °C, or None."""
        if system == "Linux":
            return self._try_linux_cpu_temp_reader()
        if system == "Darwin":
            return self._try_macos_smc_cpu_temp_reader()
        if system == "Windows":
            return self._try_windows_cpu_temp_reader()
        return None

    @staticmethod
    def _try_linux_cpu_temp_reader() -> Callable[[], float | None] | None:
        """Read CPU temp from sysfs hwmon or thermal_zone."""
        # Strategy: find the first readable temperature source.
        # hwmon is preferred (labeled, more reliable).
        hwmon_base = Path("/sys/class/hwmon")
        if hwmon_base.exists():
            for hwmon in sorted(hwmon_base.iterdir()):
                # Check if this hwmon is a CPU/SoC sensor.
                name_file = hwmon / "name"
                if name_file.exists():
                    name = name_file.read_text().strip().lower()
                    # Common CPU sensor names across vendors
                    if any(
                        k in name
                        for k in (
                            "coretemp",
                            "k10temp",
                            "zenpower",
                            "cpu_thermal",
                            "soc_thermal",
                        )
                    ):
                        # Find the first temp*_input file
                        for temp_file in sorted(hwmon.glob("temp*_input")):
                            try:
                                val = int(temp_file.read_text().strip())
                                if 0 < val < 150_000:  # millidegrees sanity
                                    path = temp_file

                                    def _read(p: Path = path) -> float | None:
                                        try:
                                            return int(p.read_text().strip()) / 1000.0
                                        except (OSError, ValueError):
                                            return None

                                    return _read
                            except (OSError, ValueError):
                                continue

        # Fallback: thermal_zone
        tz_base = Path("/sys/class/thermal")
        if tz_base.exists():
            for tz in sorted(tz_base.glob("thermal_zone*")):
                tz_type = tz / "type"
                if tz_type.exists():
                    t = tz_type.read_text().strip().lower()
                    if any(k in t for k in ("cpu", "soc", "x86_pkg", "acpitz")):
                        temp_file = tz / "temp"
                        if temp_file.exists():
                            try:
                                val = int(temp_file.read_text().strip())
                                if 0 < val < 150_000:
                                    path = temp_file

                                    def _read(p: Path = path) -> float | None:
                                        try:
                                            return int(p.read_text().strip()) / 1000.0
                                        except (OSError, ValueError):
                                            return None

                                    return _read
                            except (OSError, ValueError):
                                continue
        return None

    @staticmethod
    def _try_macos_smc_cpu_temp_reader() -> Callable[[], float | None] | None:
        """Read CPU temperature from Apple SMC via IOKit.

        Uses the ``AppleSMC`` IOService to read SMC keys.  The CPU
        die/proximity temperature is exposed under keys like ``TC0P``
        (Intel), ``Tp09`` (Apple Silicon M1), or ``Tp05`` (M2/M3/M4).
        No root required on macOS.
        """
        try:
            import ctypes
            import struct

            iokit = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/IOKit.framework/IOKit"
            )
            cf = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )

            kIOMasterPortDefault = ctypes.c_uint(0)

            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            cf.CFRelease.restype = None
            cf.CFRelease.argtypes = [ctypes.c_void_p]

            iokit.IOServiceMatching.restype = ctypes.c_void_p
            iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
            iokit.IOServiceGetMatchingService.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            iokit.IOServiceOpen.restype = ctypes.c_int
            iokit.IOServiceOpen.argtypes = [
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
            ]
            iokit.IOServiceClose.restype = ctypes.c_int
            iokit.IOServiceClose.argtypes = [ctypes.c_uint]

            # SMC structs
            class SMCKeyData_vers_t(ctypes.Structure):
                _fields_ = [
                    ("major", ctypes.c_uint8),
                    ("minor", ctypes.c_uint8),
                    ("build", ctypes.c_uint8),
                    ("reserved", ctypes.c_uint8),
                    ("release", ctypes.c_uint16),
                ]

            class SMCKeyData_pLimitData_t(ctypes.Structure):
                _fields_ = [
                    ("version", ctypes.c_uint16),
                    ("length", ctypes.c_uint16),
                    ("cpuPLimit", ctypes.c_uint32),
                    ("gpuPLimit", ctypes.c_uint32),
                    ("memPLimit", ctypes.c_uint32),
                ]

            class SMCKeyData_keyInfo_t(ctypes.Structure):
                _fields_ = [
                    ("dataSize", ctypes.c_uint32),
                    ("dataType", ctypes.c_uint32),
                    ("dataAttributes", ctypes.c_uint8),
                ]

            class SMCKeyData_t(ctypes.Structure):
                _fields_ = [
                    ("key", ctypes.c_uint32),
                    ("vers", SMCKeyData_vers_t),
                    ("pLimitData", SMCKeyData_pLimitData_t),
                    ("keyInfo", SMCKeyData_keyInfo_t),
                    ("result", ctypes.c_uint8),
                    ("status", ctypes.c_uint8),
                    ("data8", ctypes.c_uint8),
                    ("data32", ctypes.c_uint32),
                    ("bytes", ctypes.c_uint8 * 32),
                ]

            KERNEL_INDEX_SMC = 2
            SMC_CMD_READ_KEYINFO = 9
            SMC_CMD_READ_BYTES = 5

            # Open SMC connection
            service = iokit.IOServiceGetMatchingService(
                kIOMasterPortDefault,
                iokit.IOServiceMatching(b"AppleSMC"),
            )
            if not service:
                return None

            conn = ctypes.c_uint()
            # mach_task_self() — ctypes gives us the current task port
            import ctypes.util

            libc_path = ctypes.util.find_library("c")
            if libc_path is None:
                return None
            libc = ctypes.cdll.LoadLibrary(libc_path)
            libc.mach_task_self.restype = ctypes.c_uint
            task = libc.mach_task_self()

            result = iokit.IOServiceOpen(service, task, 0, ctypes.byref(conn))
            if result != 0:
                return None

            def _smc_key_to_uint32(key: str) -> int:
                return struct.unpack(">I", key.encode("ascii"))[0]

            def _read_smc_key(key: str) -> float | None:
                """Read a single SMC key and interpret as sp78 temperature."""
                input_struct = SMCKeyData_t()
                output_struct = SMCKeyData_t()

                key_int = _smc_key_to_uint32(key)

                # Step 1: get key info (data size + type)
                input_struct.key = key_int
                input_struct.data8 = SMC_CMD_READ_KEYINFO
                ret = iokit.IOConnectCallStructMethod(
                    conn,
                    KERNEL_INDEX_SMC,
                    ctypes.byref(input_struct),
                    ctypes.sizeof(SMCKeyData_t),
                    ctypes.byref(output_struct),
                    ctypes.byref(ctypes.c_size_t(ctypes.sizeof(SMCKeyData_t))),
                )
                if ret != 0:
                    return None

                # Step 2: read the value bytes
                input_struct.keyInfo.dataSize = output_struct.keyInfo.dataSize
                input_struct.data8 = SMC_CMD_READ_BYTES
                ret = iokit.IOConnectCallStructMethod(
                    conn,
                    KERNEL_INDEX_SMC,
                    ctypes.byref(input_struct),
                    ctypes.sizeof(SMCKeyData_t),
                    ctypes.byref(output_struct),
                    ctypes.byref(ctypes.c_size_t(ctypes.sizeof(SMCKeyData_t))),
                )
                if ret != 0:
                    return None

                # Interpret as sp78 (signed 7.8 fixed-point)
                data_size = output_struct.keyInfo.dataSize
                if data_size >= 2:
                    raw = (output_struct.bytes[0] << 8) | output_struct.bytes[1]
                    # sp78: high byte is integer part, low byte is fractional
                    temp = raw / 256.0
                    if 0 < temp < 150:  # sanity check
                        return temp
                return None

            # Probe: try known CPU temperature keys
            cpu_temp_keys = ["TC0P", "Tp09", "Tp05", "Tp01", "Tp0D"]
            working_key: str | None = None
            for k in cpu_temp_keys:
                val = _read_smc_key(k)
                if val is not None:
                    working_key = k
                    break

            if working_key is None:
                iokit.IOServiceClose(conn)
                return None

            def _read(key: str = working_key) -> float | None:
                return _read_smc_key(key)

            return _read

        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _try_windows_cpu_temp_reader() -> Callable[[], float | None] | None:
        """Read CPU temp from Windows WMI (best-effort)."""
        try:
            import subprocess

            # WMI thermal zone — works on some but not all Windows machines.
            probe = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi "
                    "| Select-Object -First 1 -ExpandProperty CurrentTemperature",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if probe.returncode != 0 or not probe.stdout.strip():
                return None
            # Value is in tenths of Kelvin
            raw = int(probe.stdout.strip())
            temp_c = (raw / 10.0) - 273.15
            if not (0 < temp_c < 150):
                return None

            def _read() -> float | None:
                try:
                    result = subprocess.run(
                        [
                            "powershell",
                            "-NoProfile",
                            "-Command",
                            "Get-CimInstance MSAcpi_ThermalZoneTemperature "
                            "-Namespace root/wmi "
                            "| Select-Object -First 1 -ExpandProperty CurrentTemperature",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode != 0 or not result.stdout.strip():
                        return None
                    raw = int(result.stdout.strip())
                    temp = (raw / 10.0) - 273.15
                    return round(temp, 1) if 0 < temp < 150 else None
                except Exception:  # noqa: BLE001
                    return None

            return _read
        except Exception:  # noqa: BLE001
            return None

    # -- Fan speed readers -------------------------------------------------

    def _make_fan_reader(self, system: str) -> Callable[[], float | None] | None:
        """Return a callable that reads fan speed in RPM, or None."""
        if system in ("Linux", "Windows"):
            # Try NVML GPU fan first (reports % → we convert using max RPM est)
            reader = self._try_nvml_fan_reader()
            if reader:
                return reader
            if system == "Linux":
                return self._try_linux_fan_reader()
            return None
        if system == "Darwin":
            return self._try_macos_smc_fan_reader()
        return None

    @staticmethod
    def _try_nvml_fan_reader() -> Callable[[], float | None] | None:
        """Read GPU fan speed via NVML (percentage → approximate RPM)."""
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                pynvml.nvmlShutdown()
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Probe: verify fan query works (blower-less cards raise an error).
            pct = pynvml.nvmlDeviceGetFanSpeed(handle)
            if pct is None:
                return None

            def _read() -> float | None:
                try:
                    pct = pynvml.nvmlDeviceGetFanSpeed(handle)
                    # NVML returns percentage (0..100).
                    # Return as-is in % since true RPM isn't available.
                    return float(pct) if pct is not None else None
                except Exception:  # noqa: BLE001
                    return None

            return _read
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _try_linux_fan_reader() -> Callable[[], float | None] | None:
        """Read fan RPM from sysfs hwmon."""
        hwmon_base = Path("/sys/class/hwmon")
        if not hwmon_base.exists():
            return None
        for hwmon in sorted(hwmon_base.iterdir()):
            for fan_file in sorted(hwmon.glob("fan*_input")):
                try:
                    val = int(fan_file.read_text().strip())
                    if val > 0:  # spinning fan found
                        path = fan_file

                        def _read(p: Path = path) -> float | None:
                            try:
                                v = int(p.read_text().strip())
                                return float(v) if v >= 0 else None
                            except (OSError, ValueError):
                                return None

                        return _read
                except (OSError, ValueError):
                    continue
        return None

    @staticmethod
    def _try_macos_smc_fan_reader() -> Callable[[], float | None] | None:
        """Read fan speed from Apple SMC via IOKit (``F0Ac`` key).

        Returns actual RPM on Macs with fans.  Fanless MacBooks will
        have the SMC key missing or returning 0 — we return None for those.
        """
        try:
            import ctypes
            import struct

            iokit = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/IOKit.framework/IOKit"
            )

            kIOMasterPortDefault = ctypes.c_uint(0)

            iokit.IOServiceMatching.restype = ctypes.c_void_p
            iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
            iokit.IOServiceGetMatchingService.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            iokit.IOServiceOpen.restype = ctypes.c_int
            iokit.IOServiceOpen.argtypes = [
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.POINTER(ctypes.c_uint),
            ]
            iokit.IOServiceClose.restype = ctypes.c_int
            iokit.IOServiceClose.argtypes = [ctypes.c_uint]

            class SMCKeyData_vers_t(ctypes.Structure):
                _fields_ = [
                    ("major", ctypes.c_uint8),
                    ("minor", ctypes.c_uint8),
                    ("build", ctypes.c_uint8),
                    ("reserved", ctypes.c_uint8),
                    ("release", ctypes.c_uint16),
                ]

            class SMCKeyData_pLimitData_t(ctypes.Structure):
                _fields_ = [
                    ("version", ctypes.c_uint16),
                    ("length", ctypes.c_uint16),
                    ("cpuPLimit", ctypes.c_uint32),
                    ("gpuPLimit", ctypes.c_uint32),
                    ("memPLimit", ctypes.c_uint32),
                ]

            class SMCKeyData_keyInfo_t(ctypes.Structure):
                _fields_ = [
                    ("dataSize", ctypes.c_uint32),
                    ("dataType", ctypes.c_uint32),
                    ("dataAttributes", ctypes.c_uint8),
                ]

            class SMCKeyData_t(ctypes.Structure):
                _fields_ = [
                    ("key", ctypes.c_uint32),
                    ("vers", SMCKeyData_vers_t),
                    ("pLimitData", SMCKeyData_pLimitData_t),
                    ("keyInfo", SMCKeyData_keyInfo_t),
                    ("result", ctypes.c_uint8),
                    ("status", ctypes.c_uint8),
                    ("data8", ctypes.c_uint8),
                    ("data32", ctypes.c_uint32),
                    ("bytes", ctypes.c_uint8 * 32),
                ]

            KERNEL_INDEX_SMC = 2
            SMC_CMD_READ_KEYINFO = 9
            SMC_CMD_READ_BYTES = 5

            service = iokit.IOServiceGetMatchingService(
                kIOMasterPortDefault,
                iokit.IOServiceMatching(b"AppleSMC"),
            )
            if not service:
                return None

            conn = ctypes.c_uint()
            import ctypes.util

            libc_path = ctypes.util.find_library("c")
            if libc_path is None:
                return None
            libc = ctypes.cdll.LoadLibrary(libc_path)
            libc.mach_task_self.restype = ctypes.c_uint
            task = libc.mach_task_self()

            result = iokit.IOServiceOpen(service, task, 0, ctypes.byref(conn))
            if result != 0:
                return None

            def _smc_key_to_uint32(key: str) -> int:
                return struct.unpack(">I", key.encode("ascii"))[0]

            def _read_smc_fpe2(key: str) -> float | None:
                """Read an SMC key interpreted as fpe2 (fan speed in RPM)."""
                input_struct = SMCKeyData_t()
                output_struct = SMCKeyData_t()

                key_int = _smc_key_to_uint32(key)

                input_struct.key = key_int
                input_struct.data8 = SMC_CMD_READ_KEYINFO
                ret = iokit.IOConnectCallStructMethod(
                    conn,
                    KERNEL_INDEX_SMC,
                    ctypes.byref(input_struct),
                    ctypes.sizeof(SMCKeyData_t),
                    ctypes.byref(output_struct),
                    ctypes.byref(ctypes.c_size_t(ctypes.sizeof(SMCKeyData_t))),
                )
                if ret != 0:
                    return None

                input_struct.keyInfo.dataSize = output_struct.keyInfo.dataSize
                input_struct.data8 = SMC_CMD_READ_BYTES
                ret = iokit.IOConnectCallStructMethod(
                    conn,
                    KERNEL_INDEX_SMC,
                    ctypes.byref(input_struct),
                    ctypes.sizeof(SMCKeyData_t),
                    ctypes.byref(output_struct),
                    ctypes.byref(ctypes.c_size_t(ctypes.sizeof(SMCKeyData_t))),
                )
                if ret != 0:
                    return None

                data_size = output_struct.keyInfo.dataSize
                if data_size >= 2:
                    # fpe2: unsigned 14.2 fixed-point
                    raw = (output_struct.bytes[0] << 8) | output_struct.bytes[1]
                    rpm = raw / 4.0
                    return rpm if rpm > 0 else None
                return None

            # Probe: try F0Ac (fan 0 actual speed)
            probe_val = _read_smc_fpe2("F0Ac")
            if probe_val is None:
                iokit.IOServiceClose(conn)
                return None

            def _read() -> float | None:
                return _read_smc_fpe2("F0Ac")

            return _read

        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def _ensure_models(
    repo_id: str,
    filename: str,
    models_dir: str | Path = "./models",
    token: str | None = None,
) -> list[tuple[Path, str]]:
    """Download model(s) from HF and return ``(local_path, hf_identifier)`` pairs.

    *hf_identifier* has the form ``repo_id/actual_filename`` and is stored
    in the JSONL envelope as the ``model`` field.

    mmproj sidecar files are excluded by :func:`download_model` before
    anything is downloaded.
    """
    paths = download_model(
        repo_id,
        filename,
        models_dir=Path(models_dir).expanduser(),
        token=token,
    )
    return [(p, f"{repo_id}/{p.name}") for p in paths]


class SweepConfig(BaseModel):
    """Validated representation of the ``[sweep]`` block in a sweep TOML file.

    ``repo_id`` / ``filename`` identify a Hugging Face GGUF model.
    ``filename`` may be a glob pattern (e.g. ``"*Q4_K_M.gguf"``).

    Example TOML::

        [sweep]
        repo_id    = "unsloth/Qwen3.5-9B-GGUF"
        filename   = "Qwen3.5-9B-Q8_0.gguf"
        models_dir = "./models"
        n_ctx      = [8192, 16384, 32768]
        n_batch    = [512, 1024]
    """

    repo_id: str
    filename: str
    exclude_filename: list[str] = Field(default_factory=list)
    models_dir: str = "./models"
    n_ctx: list[int]
    n_batch: list[int]
    concurrent_users: list[int] = Field(default_factory=lambda: [1])
    runner_type: str = "llama-bench"
    runner_params: dict[str, Any] = Field(default_factory=dict)

    # Populated externally by _ensure_models(); not read from TOML.
    resolved_models: list[tuple[Path, str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def combos(self) -> list["BenchCombo"]:
        """Return the full Cartesian product of models × n_ctx × n_batch × concurrent_users."""
        return [
            BenchCombo(
                model_path=local,
                model=hf_id,
                n_ctx=ctx,
                n_batch=batch,
                concurrent_users=users,
            )
            for (local, hf_id), ctx, batch, users in itertools.product(
                self.resolved_models,
                self.n_ctx,
                self.n_batch,
                self.concurrent_users,
            )
        ]


@dataclass
class BenchCombo:
    """A single (model, n_ctx, n_batch, concurrent_users) parameter combination."""

    model_path: Path
    model: str  # HF identifier: repo_id/filename
    n_ctx: int
    n_batch: int
    concurrent_users: int = 1


class VramCliffConfig(BaseModel):
    """Validated representation of the ``[vram-cliff]`` block in a suite TOML.

    ``repo_id`` / ``filename`` use the same HF coordinates as
    :class:`SweepConfig`.  When the glob matches multiple models,
    vram-cliff probes each one independently.

    Example TOML::

        [vram-cliff]
        repo_id    = "unsloth/Qwen3.5-9B-GGUF"
        filename   = "*.gguf"
        models_dir = "./models"
        min_ctx    = 2048
        max_ctx    = 131072
        tolerance  = 1024
    """

    repo_id: str
    filename: str
    models_dir: str = "./models"
    min_ctx: int = 2048
    max_ctx: int = 131072
    tolerance: int = 1024
    runner_type: str = "llama-bench"
    runner_params: dict[str, Any] = Field(default_factory=dict)

    # Populated externally by _ensure_models(); not read from TOML.
    resolved_models: list[tuple[Path, str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


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
        Hugging Face repository ID, e.g. ``"unsloth/Qwen3.5-0.8B-GGUF"``.
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
    dest = (models_dir or DEFAULT_MODELS_DIR).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    # --- resolve the glob pattern to exact filenames -------------------------
    api = HfApi(token=token)

    try:
        all_repo_files: list[RepoFile] = [
            f
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

    repo_files_by_name: dict[str, RepoFile] = {
        f.rfilename: f for f in all_repo_files
    }
    matches = fnmatch.filter(repo_files_by_name.keys(), filename_pattern)

    # Silently exclude mmproj sidecar files — they are multimodal-projector
    # weights that cannot be benchmarked as standalone models.  Users
    # routinely use "*.gguf" globs that unintentionally pick them up.
    matches = [m for m in matches if not Path(m).name.startswith("mmproj-")]

    if not matches:
        raise FileNotFoundError(
            f"No file in '{repo_id}' matches pattern '{filename_pattern}'.\n"
            f"Available files: {list(repo_files_by_name.keys())}"
        )

    console.print(
        f"[info]Matched {len(matches)} file(s) in[/info] [bold]{repo_id}[/bold]:"
    )
    for m in matches:
        console.print(f"  • {m}")

    # --- pre-flight: skip files that already exist with matching size ---------
    need_download: list[str] = []
    downloaded_paths: list[Path] = []

    for filename in matches:
        local_path = (dest / filename).resolve()
        expected_size = repo_files_by_name[filename].size
        if local_path.is_file() and expected_size is not None:
            actual_size = local_path.stat().st_size
            if actual_size == expected_size:
                downloaded_paths.append(local_path)
                console.print(f"  [success]✓[/success] {filename} [dim](cached)[/dim]")
                continue
        need_download.append(filename)

    if not need_download:
        console.print(
            f"\n[success]All {len(matches)} file(s) already present in[/success] "
            f"[bold]{dest.resolve()}[/bold]"
        )
        return downloaded_paths

    console.print(
        f"[info]Downloading {len(need_download)} of {len(matches)} file(s)…[/info]"
    )

    # --- download only the missing/incomplete files --------------------------
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
        overall = progress.add_task("[bold]Total", total=len(need_download))

        for filename in need_download:
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
        f"\n[success]Done![/success] {len(downloaded_paths)} file(s) in "
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
_SHARED_TOML_KEYS = {
    "repo_id",
    "filename",
    "exclude_filename",
    "models_dir",
    "runner_type",
    "runner_params",
}


def _merge_shared_params(raw: dict, section: str) -> dict:
    """Merge root-level shared params into a section dict.

    Section-level values take precedence over root-level values.
    This lets users declare ``model_path`` once at the top of the TOML
    and have it inherited by both ``[vram-cliff]`` and ``[sweep]``.
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


def execute_vram_cliff(
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

    safe = last_good

    runner.teardown()
    return safe


def _count_lines(path: Path) -> int:
    """Return the number of lines in *path*, or 0 if it doesn't exist."""
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _read_lines_from(path: Path, start: int) -> list[str]:
    """Return lines from *path* starting at 0-based line *start*."""
    lines: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= start:
                lines.append(line)
    return lines


def execute_sweep(
    results_file: Path,
    config_path: Path | None = None,
    sweep_config: SweepConfig | None = None,
    max_ctx_caps: dict[Path, int] | None = None,
    *,
    suite_run_id: str | None = None,
    completed_models: set[str] | None = None,
    on_model_done: "Callable[[str, Path, int], None] | None" = None,
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
    max_ctx_caps:
        Per-model context caps (from ``vram-cliff``).  Combos whose
        ``n_ctx`` exceeds the cap for their model are skipped.
    suite_run_id:
        Optional identifier for the suite run.  When set, every JSONL
        record includes a ``suite_run_id`` field that ties all results
        from the same ``ppb all`` invocation together.
    completed_models:
        Set of HF model identifiers (``repo_id/filename``) that have
        already been benchmarked.  All combos for these models are
        skipped, enabling resume of interrupted runs.
    on_model_done:
        Optional callback invoked after all combos for a single model
        complete.  Signature: ``(model_hf_id, results_file, line_offset)``
        where *line_offset* is the JSONL line index where this model's
        results start (for incremental reads).
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

        # Download / cache models if not already resolved
        if not cfg.resolved_models:
            cfg.resolved_models = _ensure_models(
                cfg.repo_id,
                cfg.filename,
                cfg.models_dir,
            )
    else:
        raise ValueError("One of config_path or sweep_config is required.")

    # Apply exclude_filename filter (works regardless of how models were resolved)
    if cfg.exclude_filename:
        excluded = [
            (p, hf_id)
            for p, hf_id in cfg.resolved_models
            if any(fnmatch.fnmatch(p.name, pat) for pat in cfg.exclude_filename)
        ]
        if excluded:
            for _, hf_id in excluded:
                console.print(
                    f"  [info]Excluding[/info] [hw]{hf_id}[/hw] "
                    f"(matched exclude_filename)"
                )
        cfg.resolved_models = [
            (p, hf_id)
            for p, hf_id in cfg.resolved_models
            if not any(fnmatch.fnmatch(p.name, pat) for pat in cfg.exclude_filename)
        ]
        if not cfg.resolved_models:
            console.print("[error]All models were excluded — nothing to benchmark.[/error]")
            raise typer.Exit(code=1)

    combos = cfg.combos()

    # Apply per-model max_ctx caps (from vram-cliff)
    if max_ctx_caps:
        # Inject each model's discovered max context into the sweep's n_ctx
        # list so there's always at least one runnable size per model, even
        # when every user-requested n_ctx exceeds the hardware limit.
        cap_values = {v for v in max_ctx_caps.values() if v > 0}
        new_ctx = sorted(set(cfg.n_ctx) | cap_values)
        if new_ctx != sorted(cfg.n_ctx):
            added = cap_values - set(cfg.n_ctx)
            console.print(
                f"  [info]Injecting vram-cliff cap(s) into sweep n_ctx:[/info] "
                f"{', '.join(f'{v:,}' for v in sorted(added))}"
            )
            cfg.n_ctx = new_ctx
            combos = cfg.combos()

        original = len(combos)
        combos = [
            c
            for c in combos
            if c.model_path not in max_ctx_caps or c.n_ctx <= max_ctx_caps[c.model_path]
        ]
        skipped = original - len(combos)
        if skipped:
            console.print(
                f"  [warning]Skipping {skipped} combo(s) exceeding per-model ctx cap[/warning]"
            )

    # --- Group combos by model (models are already the outermost axis) ----
    model_groups: list[tuple[str, list[BenchCombo]]] = []
    for model_hf_id, group in itertools.groupby(combos, key=lambda c: c.model):
        model_groups.append((model_hf_id, list(group)))

    # --- Apply completed_models filter ------------------------------------
    if completed_models:
        active_groups: list[tuple[str, list[BenchCombo]]] = []
        skipped_models = 0
        for hf_id, grp in model_groups:
            if hf_id in completed_models:
                skipped_models += 1
                console.print(
                    f"  [info]⏭  Skipping[/info] [hw]{hf_id}[/hw] — already completed"
                )
            else:
                active_groups.append((hf_id, grp))
        if skipped_models:
            console.print(
                f"  [info]Resuming — {skipped_models} model(s) already done[/info]"
            )
        model_groups = active_groups

    total = sum(len(grp) for _, grp in model_groups)

    model_names = ", ".join(f"[hw]{hf_id}[/hw]" for _, hf_id in cfg.resolved_models)
    sweep_info = (
        f"[info]Sweep:[/info] [bold]{total}[/bold] combination(s) "
        f"across [bold]{len(cfg.resolved_models)}[/bold] model(s): {model_names}\n"
        f"  Runner  : {cfg.runner_type}\n"
        f"  n_ctx   : {cfg.n_ctx}\n"
        f"  n_batch : {cfg.n_batch}\n"
    )
    if cfg.concurrent_users != [1]:
        sweep_info += f"  Users   : {cfg.concurrent_users}\n"
    sweep_info += f"  Results : [bold]{results_file.resolve()}[/bold]"
    console.print(sweep_info)

    runner = get_runner(cfg.runner_type)
    runner.setup(cfg.runner_params)

    passed = failed = 0
    skipped = 0
    i = 0  # global combo counter

    # After this many *consecutive* failures for the same model, skip the
    # rest of its combos.  Prevents burning hours waiting for a model that
    # can't start (e.g. GPU resource exhaustion after SIGKILL).
    _MAX_CONSECUTIVE_FAILURES = 5

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

            for model_hf_id, model_combos in model_groups:
                line_offset = _count_lines(results_file)
                consecutive_failures = 0

                for combo in model_combos:
                    i += 1

                    # Skip remaining combos for this model if too many
                    # consecutive failures (likely a persistent issue like
                    # GPU resource exhaustion or corrupt model file).
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        remaining = len(model_combos) - (
                            model_combos.index(combo)
                        )
                        console.print(
                            f"  [warning]⏭  Skipping {remaining} remaining combo(s) for "
                            f"{combo.model_path.name} after "
                            f"{_MAX_CONSECUTIVE_FAILURES} consecutive failures[/warning]"
                        )
                        skipped += remaining
                        progress.advance(task, remaining)
                        break

                    label = f"{combo.model_path.name} ctx={combo.n_ctx} batch={combo.n_batch}"
                    if combo.concurrent_users > 1:
                        label += f" users={combo.concurrent_users}"
                    progress.update(task, combo=label)

                    run_config: dict[str, Any] = {
                        "model_path": str(combo.model_path),
                        "n_ctx": combo.n_ctx,
                        "n_batch": combo.n_batch,
                        "concurrent_users": combo.concurrent_users,
                    }
                    t0 = time.monotonic()
                    _power_sampler = PowerSampler()
                    _thermal_sampler = ThermalSampler()
                    _power_sampler.start()
                    _thermal_sampler.start()
                    raw_result = runner.run(run_config)
                    avg_power_w, max_power_w = _power_sampler.stop()
                    thermal_stats = _thermal_sampler.stop()
                    elapsed = time.monotonic() - t0

                    record: dict[str, Any] | None
                    if raw_result is None:
                        record = None
                    else:
                        record = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "runner_type": cfg.runner_type,
                            "model": combo.model,
                            "n_ctx": combo.n_ctx,
                            "n_batch": combo.n_batch,
                            "concurrent_users": combo.concurrent_users,
                            "hardware": _hw_sniffer.snapshot(),
                            "results": raw_result["results"],
                        }
                        if suite_run_id is not None:
                            record["suite_run_id"] = suite_run_id
                        if avg_power_w is not None:
                            record["avg_power_w"] = avg_power_w
                            record["max_power_w"] = max_power_w
                        if thermal_stats:
                            record.update(thermal_stats)
                        _write_result(record, results_file)

                    dur = f"  ({elapsed:.1f}s)"
                    if record is None:
                        consecutive_failures += 1
                        console.print(
                            f"  [error]✗[/error] [{i}/{total}] {label} — FAILED{dur}"
                        )
                        failed += 1
                    else:
                        consecutive_failures = 0
                        # Pull out the tok/s figure if llama-bench emits it
                        tps: str = ""
                        try:
                            tps_val = record["results"][0]["avg_ts"]
                            tps = f"  {tps_val:.1f} tok/s"
                        except (KeyError, IndexError, TypeError):
                            pass
                        pwr = (
                            f"  {avg_power_w:.0f}W avg"
                            if avg_power_w is not None
                            else ""
                        )
                        therm = ""
                        if thermal_stats.get("avg_gpu_temp_c") is not None:
                            therm += f"  GPU {thermal_stats['avg_gpu_temp_c']:.0f}°C"
                        elif thermal_stats.get("avg_cpu_temp_c") is not None:
                            therm += f"  CPU {thermal_stats['avg_cpu_temp_c']:.0f}°C"
                        console.print(
                            f"  [success]✓[/success] [{i}/{total}] {label}{tps}{pwr}{therm}{dur}"
                        )
                        passed += 1

                    progress.advance(task)

                # -- model done: fire callback --------------------------------
                if on_model_done is not None:
                    on_model_done(model_hf_id, results_file, line_offset)

    finally:
        runner.teardown()

    status = "success" if failed == 0 and skipped == 0 else "warning"
    skip_msg = f", {skipped} skipped" if skipped else ""
    console.print(
        f"\n[{status}]Sweep complete.[/{status}] "
        f"{passed} passed, {failed} failed{skip_msg} — "
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


@app.command(name="download-dataset")
def download_dataset_cmd(
    dataset_dir: Optional[Path] = typer.Option(
        None,
        "--dataset-dir",
        "-d",
        help="Directory for cached dataset files (default: datasets/data/)",
    ),
    repo: str = typer.Option(
        SHAREGPT_REPO,
        "--repo",
        "-R",
        help="Hugging Face Hub dataset repository ID.",
    ),
    filename: str = typer.Option(
        SHAREGPT_FILENAME,
        "--filename",
        "-f",
        help="Filename to download from the repository.",
    ),
) -> None:
    """Download a conversational dataset for llama-server benchmarks.

    By default downloads the ShareGPT dataset (~700 MB).  Use --repo and
    --filename to fetch a different HF-hosted dataset instead.

    The dataset is also downloaded automatically on the first
    ``llama-server`` benchmark run, but this command lets you pre-fetch it
    explicitly — useful for offline or air-gapped environments.
    """
    console.print(
        f"[info]Downloading[/info] [bold]{filename}[/bold] from [bold]{repo}[/bold] …"
    )
    try:
        path = download_dataset(
            repo_id=repo,
            filename=filename,
            dataset_dir=dataset_dir,
        )
        console.print(f"[success]✓ Dataset ready:[/success] [bold]{path}[/bold]")
    except Exception as exc:
        console.print(f"[error]Dataset download failed:[/error] {exc}")
        log.exception("Unexpected error during dataset download")
        raise typer.Exit(code=1) from exc


@app.command()
def sweep(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to a TOML suite file containing a [sweep] section",
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", "-R", help="HF repo ID (e.g. unsloth/Qwen3.5-9B-GGUF)"
    ),
    filename_pattern: Optional[str] = typer.Option(
        None, "--filename", "-f", help='GGUF filename or glob (e.g. "*Q4_K_M.gguf")'
    ),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", "-d", help="Local directory for downloaded models"
    ),
    n_ctx: Optional[str] = typer.Option(
        None, "--n-ctx", help="Comma-separated context sizes, e.g. '8192,16384,32768'"
    ),
    n_batch: Optional[str] = typer.Option(
        None, "--n-batch", help="Comma-separated batch sizes, e.g. '512,1024'"
    ),
    runner: Optional[str] = typer.Option(
        None,
        "--runner",
        help="Runner backend: llama-bench, llama-server (default: llama-bench)",
    ),
    concurrent_users: Optional[str] = typer.Option(
        None,
        "--concurrent-users",
        help="Comma-separated concurrent user counts, e.g. '1,2,4,8' (default: 1)",
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
    by CLI flags (``ppb sweep --repo-id org/repo --filename '*.gguf' --n-ctx 8192 --n-batch 512``).
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
        if repo_id is not None:
            sweep_raw["repo_id"] = repo_id
        if filename_pattern is not None:
            sweep_raw["filename"] = filename_pattern
        if models_dir is not None:
            sweep_raw["models_dir"] = models_dir
        if n_ctx is not None:
            sweep_raw["n_ctx"] = _parse_int_list(n_ctx, "n-ctx")
        if n_batch is not None:
            sweep_raw["n_batch"] = _parse_int_list(n_batch, "n-batch")
        if runner is not None:
            sweep_raw["runner_type"] = runner
        if concurrent_users is not None:
            sweep_raw["concurrent_users"] = _parse_int_list(
                concurrent_users, "concurrent-users"
            )

        try:
            cfg = SweepConfig(**sweep_raw)
        except Exception as exc:
            console.print(f"[error]Invalid sweep config:[/error] {exc}")
            raise typer.Exit(code=1) from exc

        # Download / cache models
        try:
            cfg.resolved_models = _ensure_models(
                cfg.repo_id,
                cfg.filename,
                cfg.models_dir,
            )
        except (FileNotFoundError, RepositoryNotFoundError) as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(code=1) from exc

        resolved_results = _resolve_results_file(
            cfg_path, results_file, raw.get("results")
        )
    else:
        # ---- Pure-CLI mode ------------------------------------------------
        if not repo_id or not filename_pattern:
            console.print(
                "[error]Provide a TOML config or --repo-id + --filename flags.[/error]\n"
                "  Usage: ppb sweep suite.toml\n"
                "     or: ppb sweep --repo-id org/repo --filename '*.gguf' "
                "--n-ctx 8192,16384 --n-batch 512"
            )
            raise typer.Exit(code=1)
        if not n_ctx or not n_batch:
            console.print(
                "[error]--n-ctx and --n-batch are required in CLI-only mode.[/error]"
            )
            raise typer.Exit(code=1)

        try:
            cu = (
                _parse_int_list(concurrent_users, "concurrent-users")
                if concurrent_users
                else [1]
            )
            cfg = SweepConfig(
                repo_id=repo_id,
                filename=filename_pattern,
                models_dir=models_dir or "./models",
                n_ctx=_parse_int_list(n_ctx, "n-ctx"),
                n_batch=_parse_int_list(n_batch, "n-batch"),
                runner_type=runner or "llama-bench",
                concurrent_users=cu,
            )
        except Exception as exc:
            console.print(f"[error]Invalid sweep config:[/error] {exc}")
            raise typer.Exit(code=1) from exc

        # Download / cache models
        try:
            cfg.resolved_models = _ensure_models(
                cfg.repo_id,
                cfg.filename,
                cfg.models_dir,
            )
        except (FileNotFoundError, RepositoryNotFoundError) as exc:
            console.print(f"[error]{exc}[/error]")
            raise typer.Exit(code=1) from exc

        resolved_results = _resolve_results_file(None, results_file, None)

    console.print("[info]Starting sweep[/info] …")
    execute_sweep(results_file=resolved_results, sweep_config=cfg)


@app.command(name="hw-info")
def hw_info() -> None:
    """Print a snapshot of the detected hardware profile."""
    hw = _hw_sniffer.snapshot()

    console.print("\n[info]Hardware Profile[/info]")
    console.print(
        f"  OS          : [bold]{hw['os']['system']} {hw['os']['release']}[/bold]  ({hw['os']['machine']})"
    )

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


@app.command(name="vram-cliff")
def vram_cliff(
    config: Optional[Path] = typer.Argument(
        None,
        help="Path to a TOML suite file containing a [vram-cliff] section",
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", "-R", help="HF repo ID (e.g. unsloth/Qwen3.5-9B-GGUF)"
    ),
    filename_pattern: Optional[str] = typer.Option(
        None, "--filename", "-f", help='GGUF filename or glob (e.g. "*Q4_K_M.gguf")'
    ),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", "-d", help="Local directory for downloaded models"
    ),
    min_ctx: Optional[int] = typer.Option(
        None, "--min_ctx", help="Minimum context length to probe (default: 2048)"
    ),
    max_ctx: Optional[int] = typer.Option(
        None, "--max_ctx", help="Maximum context length to probe (default: 131072)"
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
        help="Runner backend to use for probing: llama-bench, llama-server (default: llama-bench)",
    ),
) -> None:
    """Binary-search for the maximum context window before OOM.

    Downloads models from Hugging Face if not already cached, then runs
    the search independently for each matched model.

    Can be driven by a TOML config (``ppb vram-cliff suite.toml``) or purely
    by CLI flags (``ppb vram-cliff --repo-id org/repo --filename '*.gguf'``).
    CLI flags override TOML values.
    """
    # --- Load TOML defaults if a config was provided -------------------------
    al_repo_id: str | None = repo_id
    al_filename: str | None = filename_pattern
    al_models_dir: str = models_dir or "./models"
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
        if "vram-cliff" in raw:
            try:
                al_cfg = VramCliffConfig(**_merge_shared_params(raw, "vram-cliff"))
            except Exception as exc:
                console.print(f"[error]Invalid [vram-cliff] config:[/error] {exc}")
                raise typer.Exit(code=1) from exc
            al_repo_id = al_repo_id or al_cfg.repo_id
            al_filename = al_filename or al_cfg.filename
            al_models_dir = models_dir or al_cfg.models_dir
            al_min = al_cfg.min_ctx
            al_max = al_cfg.max_ctx
            al_tol = al_cfg.tolerance
            al_runner = al_cfg.runner_type
            al_runner_params = al_cfg.runner_params
        elif repo_id is None:
            console.print(
                "[error]No [vram-cliff] section in config and no --repo-id flag.[/error]"
            )
            raise typer.Exit(code=1)
    elif repo_id is None:
        console.print(
            "[error]Provide a TOML config or --repo-id + --filename flags.[/error]"
        )
        raise typer.Exit(code=1)

    if not al_repo_id or not al_filename:
        console.print("[error]Both repo_id and filename are required.[/error]")
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

    # Download / cache models
    try:
        resolved = _ensure_models(al_repo_id, al_filename, al_models_dir)
    except (FileNotFoundError, RepositoryNotFoundError, Exception) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(code=1) from exc

    model_names = ", ".join(f"[hw]{hf_id}[/hw]" for _, hf_id in resolved)
    console.print(
        f"[info]VRAM Cliff[/info] probing {len(resolved)} model(s): {model_names}\n"
        f"  Range     : [bold]{al_min:,}[/bold] → [bold]{al_max:,}[/bold] tokens\n"
        f"  Tolerance : [bold]{al_tol:,}[/bold] tokens"
    )

    all_passed = True

    for mp, hf_id in resolved:
        safe = execute_vram_cliff(
            model_path=mp,
            min_ctx=al_min,
            max_ctx=al_max,
            tolerance=al_tol,
            runner_type=al_runner,
            runner_params=al_runner_params,
        )

        if safe == 0:
            console.print(
                f"\n[error]Could not find a working context size for[/error] "
                f"[hw]{hf_id}[/hw]\n"
                f"  Even n_ctx={al_min:,} failed — check that the model loads at all."
            )
            all_passed = False
        else:
            console.print(
                f"\n[success]✓ Maximum safe context for[/success] [hw]{hf_id}[/hw]\n"
                f"\n    [bold green]{safe:,} tokens[/bold green]\n"
            )

    if not all_passed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _find_resumable_results(config_path: Path) -> Path | None:
    """Scan ``results/`` for the most recent JSONL from a prior run of *config_path*.

    Filenames follow the pattern ``<stem>_YYYYMMDD_HHMM.jsonl`` (produced by
    :func:`_resolve_results_file`).  We match against the TOML stem to avoid
    picking up files from a different suite.

    Returns the most recent matching path, or *None* if none exists.
    """
    results_dir = Path("results")
    if not results_dir.is_dir():
        return None
    stem = config_path.stem  # e.g. "Qwen3.5-0.8B"
    candidates = sorted(
        results_dir.glob(f"{stem}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _detect_completed_models(
    results_path: Path,
    sweep_cfg: "SweepConfig",
    resolved_models: list[tuple[Path, str]],
    max_ctx_caps: dict[Path, int] | None,
) -> tuple[set[str] | None, str | None]:
    """Read an existing results file and determine which models are done.

    A model is "completed" when the number of records with its HF id
    equals the expected combo count (n_ctx × n_batch × concurrent_users,
    after applying *max_ctx_caps* filtering).

    Returns ``(completed_model_ids, suite_run_id)`` extracted from the
    file.  *suite_run_id* is read from the first record that has one;
    *None* if the file pre-dates this feature.
    """
    if not results_path.exists():
        return None, None

    # Count records per model HF id in the file.
    model_counts: dict[str, int] = {}
    existing_run_id: str | None = None
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hf_id = row.get("model", "")
            model_counts[hf_id] = model_counts.get(hf_id, 0) + 1
            if existing_run_id is None:
                existing_run_id = row.get("suite_run_id")

    if not model_counts:
        return None, existing_run_id

    # Compute expected combos per model.
    completed: set[str] = set()
    for model_path, hf_id in resolved_models:
        expected = (
            len(sweep_cfg.n_ctx)
            * len(sweep_cfg.n_batch)
            * len(sweep_cfg.concurrent_users)
        )
        # Apply vram-cliff caps — subtract combos that would be filtered.
        if max_ctx_caps and model_path in max_ctx_caps:
            cap = max_ctx_caps[model_path]
            valid_ctx = [c for c in sweep_cfg.n_ctx if c <= cap]
            expected = (
                len(valid_ctx)
                * len(sweep_cfg.n_batch)
                * len(sweep_cfg.concurrent_users)
            )
        actual = model_counts.get(hf_id, 0)
        if actual >= expected > 0:
            completed.add(hf_id)

    return (completed or None), existing_run_id


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
    """Run the full benchmark suite: vram-cliff → sweep → publish.

    1. **Download** — fetch models from Hugging Face (if not already cached).
    2. **vram-cliff** — discover the maximum safe context window for each model.
    3. **sweep** — run the parameter sweep, skipping any combo whose
       ``n_ctx`` exceeds the per-model limit found in step 2.
    4. **publish** — after *each* model completes, its results are
       published incrementally (CSV + optional HF upload).

    Both steps read their configuration from the same TOML file.
    If the TOML has no ``[vram-cliff]`` section, the vram-cliff step
    is skipped and the sweep runs unmodified.

    **Auto-resume:** if a previous run for the same suite was interrupted,
    PPB detects the existing results file and automatically skips models
    that have already been fully benchmarked.
    """
    raw, default_results = load_suite_config(config)
    resolved_results = results_file if results_file is not None else default_results

    console.print(
        f"[info]Full benchmark suite[/info] from [bold]{config}[/bold]\n"
        f"  Results → [bold]{resolved_results.resolve()}[/bold]\n"
    )

    # -- Download models (shared repo_id/filename from root or sections) ----
    # Determine HF coordinates — prefer root-level shared keys.
    shared = {k: v for k, v in raw.items() if k in _SHARED_TOML_KEYS}
    r_id = (
        shared.get("repo_id")
        or (raw.get("sweep", {}).get("repo_id"))
        or (raw.get("vram-cliff", {}).get("repo_id"))
    )
    r_fn = (
        shared.get("filename")
        or (raw.get("sweep", {}).get("filename"))
        or (raw.get("vram-cliff", {}).get("filename"))
    )
    r_dir = (
        shared.get("models_dir")
        or (raw.get("sweep", {}).get("models_dir"))
        or (raw.get("vram-cliff", {}).get("models_dir"))
        or "./models"
    )

    if not r_id or not r_fn:
        console.print(
            "[error]repo_id and filename are required (in TOML root or sections).[/error]"
        )
        raise typer.Exit(code=1)

    console.print("[info]Downloading / verifying models…[/info]")
    try:
        resolved_models = _ensure_models(r_id, r_fn, r_dir)
    except (FileNotFoundError, RepositoryNotFoundError, Exception) as exc:
        console.print(f"[error]Model download failed:[/error] {exc}")
        raise typer.Exit(code=1) from exc

    # Apply exclude_filename filter from TOML (shared or sweep section).
    raw_exclude = (
        raw.get("exclude_filename")
        or shared.get("exclude_filename")
        or raw.get("sweep", {}).get("exclude_filename")
        or []
    )
    if raw_exclude:
        excluded = [
            (p, hf_id)
            for p, hf_id in resolved_models
            if any(fnmatch.fnmatch(p.name, pat) for pat in raw_exclude)
        ]
        if excluded:
            for _, hf_id in excluded:
                console.print(
                    f"  [info]Excluding[/info] [hw]{hf_id}[/hw] "
                    f"(matched exclude_filename)"
                )
        resolved_models = [
            (p, hf_id)
            for p, hf_id in resolved_models
            if not any(fnmatch.fnmatch(p.name, pat) for pat in raw_exclude)
        ]
        if not resolved_models:
            console.print("[error]All models were excluded — nothing to benchmark.[/error]")
            raise typer.Exit(code=1)

    # -- Pre-flight: HF write-token check (before the long benchmark) ------
    if raw.get("publish", {}).get("upload", False):
        pub_token = raw["publish"].get("token") or None
        console.print("[info]Checking Hugging Face upload permissions…[/info]")
        try:
            check_hf_token(pub_token)
            console.print(
                "  [success]✅ HF token has write access — upload will succeed.[/success]\n"
            )
        except PermissionError as exc:
            console.print(f"\n[error]HF token check failed:[/error] {exc}")
            console.print(
                "[warning]Fix your token now to avoid losing results at the end "
                "of a long benchmark run.[/warning]\n"
                "  Continuing anyway — results will still be saved locally."
            )

    # -- Auto-resume detection ---------------------------------------------
    suite_run_id: str = uuid.uuid4().hex
    completed_models: set[str] | None = None

    # Build the expected combo count per model *before* running sweep.
    # We'll need this after we know the sweep config.
    # (Deferred until sweep_cfg is available below.)

    # Resume only activates when the *exact* results file already exists:
    #   • User passed --results pointing at an existing file, OR
    #   • The auto-generated path for *this* run already exists (same minute restart).
    # We intentionally do NOT scan for old files from prior runs — those are
    # separate benchmark sessions, not interrupted work.
    resume_path: Path | None = None
    if resolved_results.exists():
        resume_path = resolved_results

    # -- Phase 1: vram-cliff (optional) -----------------------------------
    max_ctx_caps: dict[Path, int] | None = None

    if "vram-cliff" in raw:
        console.print("[info]Phase 1 / 2:[/info] vram-cliff")
        try:
            al_cfg = VramCliffConfig(**_merge_shared_params(raw, "vram-cliff"))
        except Exception as exc:
            console.print(f"[error]Invalid [vram-cliff] config:[/error] {exc}")
            raise typer.Exit(code=1) from exc
        al_cfg.resolved_models = resolved_models

        model_names = ", ".join(f"[hw]{hf_id}[/hw]" for _, hf_id in resolved_models)
        console.print(
            f"  Probing {len(resolved_models)} model(s): {model_names}\n"
            f"  Range     : [bold]{al_cfg.min_ctx:,}[/bold] → "
            f"[bold]{al_cfg.max_ctx:,}[/bold] tokens\n"
            f"  Tolerance : [bold]{al_cfg.tolerance:,}[/bold] tokens"
        )

        caps: dict[Path, int] = {}
        any_failed = False

        for mp, hf_id in resolved_models:
            safe = execute_vram_cliff(
                model_path=mp,
                min_ctx=al_cfg.min_ctx,
                max_ctx=al_cfg.max_ctx,
                tolerance=al_cfg.tolerance,
                runner_type=al_cfg.runner_type,
                runner_params=al_cfg.runner_params,
            )

            if safe == 0:
                console.print(
                    f"\n[error]vram-cliff failed for[/error] [hw]{hf_id}[/hw] "
                    f"— could not find a working context size."
                )
                caps[mp] = 0  # sentinel: exclude all combos for this model
                any_failed = True
            else:
                caps[mp] = safe
                console.print(
                    f"\n[success]✓ Max safe context for[/success] "
                    f"[hw]{hf_id}[/hw]: "
                    f"[bold green]{safe:,} tokens[/bold green]"
                )

        if not any(v > 0 for v in caps.values()):
            console.print(
                "\n[error]vram-cliff failed for all models. Skipping sweep.[/error]"
            )
            raise typer.Exit(code=1)

        if any_failed:
            console.print(
                "\n[warning]Some models failed vram-cliff and will be skipped "
                "in the sweep.[/warning]"
            )

        max_ctx_caps = caps
        console.print()  # blank line before phase 2
    else:
        console.print("[info]No [vram-cliff] section — skipping to sweep.[/info]\n")

    # -- Phase 2: sweep ---------------------------------------------------
    if "sweep" not in raw:
        console.print("[warning]No [sweep] section — nothing more to do.[/warning]")
        return

    console.print("[info]Phase 2 / 2:[/info] sweep")
    try:
        sweep_cfg = SweepConfig(**_merge_shared_params(raw, "sweep"))
    except Exception as exc:
        console.print(f"[error]Invalid sweep config:[/error] {exc}")
        raise typer.Exit(code=1) from exc
    sweep_cfg.resolved_models = resolved_models

    # -- Resume: compute expected combos per model and detect completed -----
    if resume_path is not None:
        completed_models, existing_run_id = _detect_completed_models(
            resume_path,
            sweep_cfg,
            resolved_models,
            max_ctx_caps,
        )
        if existing_run_id:
            suite_run_id = existing_run_id
        if completed_models:
            remaining = len(resolved_models) - len(completed_models)
            console.print(
                f"\n[bold cyan]🔄 RESUMING[/bold cyan] previous run from "
                f"[bold]{resume_path.name}[/bold]\n"
                f"  Suite run  : [bold]{suite_run_id[:12]}…[/bold]\n"
                f"  Completed  : [bold]{len(completed_models)}[/bold] model(s)\n"
                f"  Remaining  : [bold]{remaining}[/bold] model(s)\n"
            )
        else:
            console.print(
                f"[info]Found existing results file [bold]{resume_path.name}[/bold] "
                f"but no models are fully completed — starting fresh.[/info]\n"
            )

    # -- Incremental publish callback --------------------------------------
    pub_cfg = raw.get("publish", {})
    submitter = pub_cfg.get("submitter", "")
    pub_token = pub_cfg.get("token") or None
    do_upload = pub_cfg.get("upload", True) if "publish" in raw else False

    def _on_model_done(model_hf_id: str, rfile: Path, line_offset: int) -> None:
        """Publish results incrementally after each model completes."""
        if "publish" not in raw:
            return

        new_lines = _read_lines_from(rfile, line_offset)
        if not new_lines:
            return

        # Flatten only the new rows for this model
        new_flat: list[dict[str, Any]] = []
        for raw_line in new_lines:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            for flat in flatten_benchmark_row(row):
                flat.pop("raw_payload", None)
                if submitter:
                    flat["submitter"] = submitter
                flat["submission_id"] = suite_run_id
                flat["submitted_at"] = datetime.now(timezone.utc).isoformat()
                flat["source_file_sha256"] = compute_file_sha256(rfile)
                new_flat.append(flat)

        if not new_flat:
            return

        # Always rewrite the full CSV from the entire results file
        all_flat = _flatten_results_file(rfile, submitter=submitter)
        if all_flat:
            csv_path = rfile.with_suffix(".csv")
            _write_csv(all_flat, csv_path)

        model_name = (
            Path(model_hf_id.split("/")[-1]).stem if "/" in model_hf_id else model_hf_id
        )
        console.print(
            f"\n  [info]📦 Publishing[/info] [hw]{model_name}[/hw] — "
            f"[bold]{len(new_flat)}[/bold] row(s)"
        )

        if do_upload:
            try:
                publish_to_hf(new_flat, token=pub_token)
                console.print("  [success]✅ Uploaded to Hugging Face[/success]")
            except Exception as exc:
                console.print(
                    f"  [error]Upload failed for {model_name}:[/error] {exc}\n"
                    f"  [warning]Results saved locally — you can retry with "
                    f"[bold]ppb publish {rfile} --upload[/bold][/warning]"
                )

    execute_sweep(
        sweep_config=sweep_cfg,
        results_file=resolved_results,
        max_ctx_caps=max_ctx_caps,
        suite_run_id=suite_run_id,
        completed_models=completed_models,
        on_model_done=_on_model_done,
    )

    # -- Final summary (no monolithic publish — done incrementally) --------
    if "publish" in raw and resolved_results.exists():
        # Write final CSV from full results file (in case callback missed any)
        all_flat = _flatten_results_file(resolved_results, submitter=submitter)
        if all_flat:
            csv_path = resolved_results.with_suffix(".csv")
            _write_csv(all_flat, csv_path)
            console.print(
                f"\n  [info]Final CSV:[/info] [bold]{len(all_flat)}[/bold] total row(s) → "
                f"[bold]{csv_path.resolve()}[/bold]"
            )
        console.print(
            f"\n[success]✅ All done![/success]  Suite run: [bold]{suite_run_id[:12]}…[/bold]\n"
            f"  View the global leaderboard at [bold]https://poorpaul.dev/leaderboard[/bold]"
        )


# ---------------------------------------------------------------------------
# Flatten / Export / Publish helpers
# ---------------------------------------------------------------------------


def _flatten_results_file(
    results_path: Path,
    *,
    submitter: str = "",
) -> list[dict[str, Any]]:
    """Read a raw JSONL file and return flattened rows (without raw_payload)."""
    submission_id = uuid.uuid4().hex
    submitted_at = datetime.now(timezone.utc).isoformat()
    source_sha = compute_file_sha256(results_path)

    flat_rows: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for flat in flatten_benchmark_row(row):
                # Strip raw_payload (too large for HF / Arrow)
                flat.pop("raw_payload", None)
                if submitter:
                    flat["submitter"] = submitter
                flat["submission_id"] = submission_id
                flat["submitted_at"] = submitted_at
                flat["source_file_sha256"] = source_sha
                flat_rows.append(flat)
    return flat_rows


def _write_csv(flat_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write flat rows to a CSV file using the canonical COLUMN_ORDER."""
    from utils.flattener import COLUMN_ORDER

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMN_ORDER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)


@app.command(name="export")
def export_cmd(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a raw JSONL results file",
        exists=True,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Destination path (.csv or .jsonl)",
    ),
) -> None:
    """Export raw JSONL results to a flat CSV or JSONL file."""
    flat_rows: list[dict[str, Any]] = []
    with input_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flat_rows.extend(flatten_benchmark_row(row))

    if not flat_rows:
        console.print("[warning]No rows found in input file.[/warning]")
        raise typer.Exit(code=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()

    if suffix == ".csv":
        _write_csv(flat_rows, output_file)
    elif suffix == ".jsonl":
        with output_file.open("w", encoding="utf-8") as fh:
            for r in flat_rows:
                fh.write(json.dumps(r, default=str) + "\n")
    else:
        console.print(
            f"[error]Unsupported output format:[/error] {suffix}\n  Use .csv or .jsonl"
        )
        raise typer.Exit(code=1)

    console.print(
        f"\n[success]✅ Exported {len(flat_rows)} row(s)[/success] → "
        f"[bold]{output_file.resolve()}[/bold]\n"
        f"  {'📊 Excel-ready!' if suffix == '.csv' else '📄 Flat JSONL — Arrow-friendly!'}"
    )


@app.command(name="publish")
def publish_cmd(
    results: list[Path] = typer.Argument(
        ...,
        help="One or more raw JSONL results files (shell globs supported)",
        exists=True,
        readable=True,
    ),
    upload: bool = typer.Option(
        False,
        "--upload/--no-upload",
        help="Upload to the PPB Hugging Face leaderboard (default: local CSV only)",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="HF_TOKEN",
        help="Hugging Face API token (or set HF_TOKEN / run huggingface-cli login)",
    ),
) -> None:
    """Flatten results to a local CSV and optionally upload to the PPB leaderboard.

    Accepts one or more JSONL result files (shell globs like ``results/*.jsonl``
    are expanded by the shell).  A CSV is written alongside each input file.
    Add ``--upload`` to push all rows to Hugging Face in one batch.
    """
    submitter = typer.prompt(
        "Display name for the leaderboard (leave blank to skip)",
        default="",
        show_default=False,
    )

    all_flat_rows: list[dict[str, Any]] = []

    for results_file in results:
        console.print(f"[info]Reading and flattening {results_file.name}…[/info]")
        flat_rows = _flatten_results_file(results_file, submitter=submitter)

        if not flat_rows:
            console.print(
                f"[warning]No rows found in {results_file.name} — skipping.[/warning]"
            )
            continue

        # -- Always write a local CSV per file -----------------------------
        csv_path = results_file.with_suffix(".csv")
        _write_csv(flat_rows, csv_path)
        console.print(
            f"  [success]✅ {len(flat_rows)} row(s) → CSV[/success] → "
            f"[bold]{csv_path.resolve()}[/bold]"
        )

        all_flat_rows.extend(flat_rows)

    if not all_flat_rows:
        console.print("[warning]No rows found in any results file.[/warning]")
        raise typer.Exit(code=1)

    console.print(
        f"\n  📊 {len(all_flat_rows)} total row(s) across {len(results)} file(s) — Excel-ready!"
    )

    # -- Upload to HF (opt-in) ---------------------------------------------
    if not upload:
        console.print(
            "\n[info]To upload to the leaderboard, re-run with [bold]--upload[/bold].[/info]"
        )
        return

    console.print("  Uploading to Hugging Face…")
    try:
        publish_to_hf(all_flat_rows, token=token)
    except Exception as exc:
        console.print(f"\n[error]Publish failed:[/error] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        "\n[success]✅ Published to Hugging Face![/success]\n"
        "  View the global leaderboard at [bold]https://poorpaul.dev/leaderboard[/bold]"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
