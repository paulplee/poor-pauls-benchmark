"""
Lightweight, pure-Python GGUF header parser.

Reads only the metadata key-value section from a GGUF file (no tensor
data is touched).  This keeps dependencies at zero and is fast enough
for pre-flight checks — a typical model's header is < 10 MB even for
models with 262K-token vocabularies embedded in the metadata.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# GGUF value-type enumeration
# ---------------------------------------------------------------------------

_TYPE_UINT8 = 0
_TYPE_INT8 = 1
_TYPE_UINT16 = 2
_TYPE_INT16 = 3
_TYPE_UINT32 = 4
_TYPE_INT32 = 5
_TYPE_FLOAT32 = 6
_TYPE_BOOL = 7
_TYPE_STRING = 8
_TYPE_ARRAY = 9
_TYPE_UINT64 = 10
_TYPE_INT64 = 11
_TYPE_FLOAT64 = 12

# struct format strings (little-endian) for scalar types
_SCALAR_FMT: dict[int, str] = {
    _TYPE_UINT8: "<B",
    _TYPE_INT8: "<b",
    _TYPE_UINT16: "<H",
    _TYPE_INT16: "<h",
    _TYPE_UINT32: "<I",
    _TYPE_INT32: "<i",
    _TYPE_FLOAT32: "<f",
    _TYPE_BOOL: "<B",
    _TYPE_UINT64: "<Q",
    _TYPE_INT64: "<q",
    _TYPE_FLOAT64: "<d",
}

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian


# ---------------------------------------------------------------------------
# Low-level binary readers
# ---------------------------------------------------------------------------


def _read_bytes(f, n: int) -> bytes:
    """Read exactly *n* bytes or raise."""
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Expected {n} bytes, got {len(data)}")
    return data


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_bytes(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_bytes(f, 8))[0]


def _read_string(f) -> str:
    length = _read_u64(f)
    return _read_bytes(f, length).decode("utf-8")


def _read_value(f, vtype: int) -> Any:
    """Read a single GGUF metadata value of the given type."""
    if vtype == _TYPE_STRING:
        return _read_string(f)
    if vtype == _TYPE_ARRAY:
        elem_type = _read_u32(f)
        count = _read_u64(f)
        return [_read_value(f, elem_type) for _ in range(count)]
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"Unknown GGUF value type: {vtype}")
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_bytes(f, size))[0]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class GGUFMetadata:
    """Parsed metadata from a GGUF file header.

    Only the fields useful for VRAM estimation are extracted; the full
    key-value dict is available as ``raw`` for debugging.
    """

    file_size_bytes: int = 0
    architecture: str | None = None
    size_label: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    expert_count: int | None = None
    expert_used_count: int | None = None
    vocab_size: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_gguf_metadata(path: Path) -> GGUFMetadata:
    """Read GGUF header metadata from *path*.

    Only the header + metadata KV pairs are read; tensor data is not
    touched.  Raises :exc:`ValueError` for non-GGUF files and
    :exc:`EOFError` for truncated headers.
    """
    file_size = path.stat().st_size

    with path.open("rb") as f:
        magic = _read_u32(f)
        if magic != GGUF_MAGIC:
            raise ValueError(
                f"Not a GGUF file (magic 0x{magic:08X}, expected 0x{GGUF_MAGIC:08X})"
            )

        version = _read_u32(f)
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version} (expected 2 or 3)")

        _tensor_count = _read_u64(f)
        metadata_kv_count = _read_u64(f)

        raw: dict[str, Any] = {}
        for _ in range(metadata_kv_count):
            key = _read_string(f)
            vtype = _read_u32(f)
            value = _read_value(f, vtype)
            raw[key] = value

    arch = raw.get("general.architecture")

    # Extract vocab size from tokenizer tokens array length.
    vocab_size: int | None = None
    tokens = raw.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list):
        vocab_size = len(tokens)

    # Build the arch-prefixed key lookup helper.
    def _arch_val(suffix: str) -> Any:
        if arch is None:
            return None
        return raw.get(f"{arch}.{suffix}")

    return GGUFMetadata(
        file_size_bytes=file_size,
        architecture=arch,
        size_label=raw.get("general.size_label"),
        context_length=_arch_val("context_length"),
        embedding_length=_arch_val("embedding_length"),
        block_count=_arch_val("block_count"),
        head_count=_arch_val("attention.head_count"),
        head_count_kv=_arch_val("attention.head_count_kv"),
        expert_count=_arch_val("expert_count"),
        expert_used_count=_arch_val("expert_used_count"),
        vocab_size=vocab_size,
        raw=raw,
    )


def estimate_kv_cache_bytes(
    meta: GGUFMetadata,
    n_ctx: int,
    n_parallel: int = 1,
    kv_dtype_bytes: int = 2,
) -> int | None:
    """Estimate total KV cache size in bytes for the given parameters.

    Returns ``None`` when model metadata lacks the fields needed for
    the calculation.

    Parameters
    ----------
    meta:
        Parsed GGUF metadata (needs ``head_count_kv`` and either
        ``embedding_length`` + ``head_count`` for deriving head dim,
        or the raw attention key/value length fields).
    n_ctx:
        Total context length (tokens) across all parallel slots.
    n_parallel:
        Number of parallel KV-cache slots (``--parallel`` flag).
        llama.cpp pre-allocates ``n_ctx`` tokens *total*, divided
        equally among slots — so the total KV size depends only on
        ``n_ctx``, not ``n_ctx * n_parallel``.
    kv_dtype_bytes:
        Bytes per KV element.  Default 2 (FP16), which is what
        llama.cpp uses for the KV cache by default.
    """
    # Need at minimum head dimensions to compute KV size.
    if meta.embedding_length is None or meta.head_count is None:
        return None

    n_kv_heads = meta.head_count_kv if meta.head_count_kv is not None else meta.head_count
    head_dim = meta.embedding_length // meta.head_count
    n_layers = meta.block_count if meta.block_count is not None else 1

    # KV cache = n_ctx × n_layers × 2(K+V) × n_kv_heads × head_dim × dtype
    # llama.cpp allocates n_ctx tokens total (shared across parallel slots).
    return n_ctx * n_layers * 2 * n_kv_heads * head_dim * kv_dtype_bytes


def estimate_total_vram_bytes(
    meta: GGUFMetadata,
    n_ctx: int,
    n_parallel: int = 1,
    overhead_bytes: int = 512 * 1024 * 1024,
) -> int | None:
    """Estimate total VRAM needed: model weights + KV cache + overhead.

    Returns ``None`` if KV cache cannot be estimated.

    Parameters
    ----------
    overhead_bytes:
        Fixed overhead for compute buffers, activations, and llama.cpp
        internals.  Default 512 MiB — deliberately conservative.
    """
    kv = estimate_kv_cache_bytes(meta, n_ctx, n_parallel)
    if kv is None:
        return None
    return meta.file_size_bytes + kv + overhead_bytes
