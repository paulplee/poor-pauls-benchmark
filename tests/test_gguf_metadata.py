"""Tests for the GGUF metadata reader (utils/gguf_metadata)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from utils.gguf_metadata import (
    GGUF_MAGIC,
    GGUFMetadata,
    estimate_kv_cache_bytes,
    estimate_total_vram_bytes,
    read_gguf_metadata,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic GGUF files
# ---------------------------------------------------------------------------

# GGUF value-type constants (mirror the module's private ones)
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12


def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_scalar(vtype: int, value) -> bytes:
    fmt_map = {
        _T_UINT8: "<B",
        _T_INT8: "<b",
        _T_UINT16: "<H",
        _T_INT16: "<h",
        _T_UINT32: "<I",
        _T_INT32: "<i",
        _T_FLOAT32: "<f",
        _T_BOOL: "<B",
        _T_UINT64: "<Q",
        _T_INT64: "<q",
        _T_FLOAT64: "<d",
    }
    return struct.pack(fmt_map[vtype], value)


def _pack_kv(key: str, vtype: int, value) -> bytes:
    """Pack a single GGUF metadata key-value pair."""
    buf = _pack_string(key) + struct.pack("<I", vtype)
    if vtype == _T_STRING:
        buf += _pack_string(value)
    elif vtype == _T_ARRAY:
        elem_type, elements = value
        buf += struct.pack("<I", elem_type)
        buf += struct.pack("<Q", len(elements))
        for elem in elements:
            if elem_type == _T_STRING:
                buf += _pack_string(elem)
            else:
                buf += _pack_scalar(elem_type, elem)
    else:
        buf += _pack_scalar(vtype, value)
    return buf


def write_synthetic_gguf(
    path: Path,
    kvs: list[tuple[str, int, object]],
    *,
    version: int = 3,
    tensor_count: int = 0,
) -> None:
    """Write a minimal GGUF file with the given metadata KV pairs.

    No actual tensor data is written — only the header + metadata,
    which is all ``read_gguf_metadata`` touches.
    """
    kv_data = b""
    for key, vtype, value in kvs:
        kv_data += _pack_kv(key, vtype, value)

    header = struct.pack(
        "<IIQQ",
        GGUF_MAGIC,
        version,
        tensor_count,
        len(kvs),
    )

    path.write_bytes(header + kv_data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def llama_gguf(tmp_path: Path) -> Path:
    """Synthetic GGUF modelling a LLaMA-like 2B model."""
    p = tmp_path / "model.gguf"
    write_synthetic_gguf(
        p,
        [
            ("general.architecture", _T_STRING, "llama"),
            ("general.size_label", _T_STRING, "2B"),
            ("llama.context_length", _T_UINT64, 8192),
            ("llama.embedding_length", _T_UINT64, 2048),
            ("llama.block_count", _T_UINT64, 24),
            ("llama.attention.head_count", _T_UINT64, 16),
            ("llama.attention.head_count_kv", _T_UINT64, 4),
            # Minimal 3-token vocab to exercise array parsing
            ("tokenizer.ggml.tokens", _T_ARRAY, (_T_STRING, ["<s>", "</s>", "hello"])),
        ],
    )
    return p


@pytest.fixture()
def gemma4_gguf(tmp_path: Path) -> Path:
    """Synthetic GGUF modelling a Gemma 4 E2B-like model (5.1B total)."""
    p = tmp_path / "gemma4-E2B.gguf"
    write_synthetic_gguf(
        p,
        [
            ("general.architecture", _T_STRING, "gemma4"),
            ("general.size_label", _T_STRING, "2.3B"),
            ("gemma4.context_length", _T_UINT64, 131072),
            ("gemma4.embedding_length", _T_UINT64, 2304),
            ("gemma4.block_count", _T_UINT64, 35),
            ("gemma4.attention.head_count", _T_UINT64, 8),
            ("gemma4.attention.head_count_kv", _T_UINT64, 4),
            (
                "tokenizer.ggml.tokens",
                _T_ARRAY,
                (_T_STRING, [f"tok_{i}" for i in range(262144)]),
            ),
        ],
    )
    return p


@pytest.fixture()
def moe_gguf(tmp_path: Path) -> Path:
    """Synthetic GGUF for a Mixture-of-Experts model."""
    p = tmp_path / "moe.gguf"
    write_synthetic_gguf(
        p,
        [
            ("general.architecture", _T_STRING, "llama"),
            ("general.size_label", _T_STRING, "26B-A4B"),
            ("llama.context_length", _T_UINT64, 262144),
            ("llama.embedding_length", _T_UINT64, 3072),
            ("llama.block_count", _T_UINT64, 30),
            ("llama.attention.head_count", _T_UINT64, 24),
            ("llama.attention.head_count_kv", _T_UINT64, 8),
            ("llama.expert_count", _T_UINT32, 128),
            ("llama.expert_used_count", _T_UINT32, 8),
        ],
    )
    return p


# ---------------------------------------------------------------------------
# Tests — parsing
# ---------------------------------------------------------------------------


class TestReadMetadata:
    def test_basic_fields(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        assert meta.architecture == "llama"
        assert meta.size_label == "2B"
        assert meta.context_length == 8192
        assert meta.embedding_length == 2048
        assert meta.block_count == 24
        assert meta.head_count == 16
        assert meta.head_count_kv == 4

    def test_vocab_size_from_tokens_array(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        assert meta.vocab_size == 3

    def test_gemma4_architecture(self, gemma4_gguf: Path) -> None:
        meta = read_gguf_metadata(gemma4_gguf)
        assert meta.architecture == "gemma4"
        assert meta.size_label == "2.3B"
        assert meta.context_length == 131072
        assert meta.vocab_size == 262144

    def test_moe_fields(self, moe_gguf: Path) -> None:
        meta = read_gguf_metadata(moe_gguf)
        assert meta.expert_count == 128
        assert meta.expert_used_count == 8

    def test_file_size_bytes(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        assert meta.file_size_bytes == llama_gguf.stat().st_size
        assert meta.file_size_bytes > 0

    def test_raw_dict_populated(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        assert "general.architecture" in meta.raw
        assert meta.raw["general.architecture"] == "llama"

    def test_version_2_accepted(self, tmp_path: Path) -> None:
        p = tmp_path / "v2.gguf"
        write_synthetic_gguf(
            p,
            [("general.architecture", _T_STRING, "llama")],
            version=2,
        )
        meta = read_gguf_metadata(p)
        assert meta.architecture == "llama"

    def test_invalid_magic_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.gguf"
        p.write_bytes(b"\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Not a GGUF file"):
            read_gguf_metadata(p)

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "v99.gguf"
        header = struct.pack("<IIQQ", GGUF_MAGIC, 99, 0, 0)
        p.write_bytes(header)
        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            read_gguf_metadata(p)

    def test_truncated_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "truncated.gguf"
        p.write_bytes(struct.pack("<I", GGUF_MAGIC))  # only magic, no version
        with pytest.raises(EOFError):
            read_gguf_metadata(p)

    def test_missing_arch_fields_return_none(self, tmp_path: Path) -> None:
        """A file with only general.architecture but no llm.* keys."""
        p = tmp_path / "minimal.gguf"
        write_synthetic_gguf(
            p,
            [("general.architecture", _T_STRING, "some_unknown_arch")],
        )
        meta = read_gguf_metadata(p)
        assert meta.architecture == "some_unknown_arch"
        assert meta.context_length is None
        assert meta.embedding_length is None
        assert meta.head_count is None


class TestScalarTypes:
    """Verify parsing of every GGUF scalar type."""

    @pytest.mark.parametrize(
        "vtype, value",
        [
            (_T_UINT8, 255),
            (_T_INT8, -1),
            (_T_UINT16, 65535),
            (_T_INT16, -32768),
            (_T_UINT32, 42),
            (_T_INT32, -42),
            (_T_FLOAT32, 3.14),
            (_T_BOOL, 1),
            (_T_UINT64, 2**40),
            (_T_INT64, -(2**40)),
            (_T_FLOAT64, 2.718281828),
        ],
    )
    def test_scalar_roundtrip(self, tmp_path: Path, vtype: int, value) -> None:
        p = tmp_path / "scalar.gguf"
        write_synthetic_gguf(p, [("test.key", vtype, value)])
        meta = read_gguf_metadata(p)
        result = meta.raw["test.key"]
        if vtype == _T_FLOAT32:
            assert abs(result - value) < 1e-5
        elif vtype == _T_FLOAT64:
            assert abs(result - value) < 1e-9
        else:
            assert result == value


# ---------------------------------------------------------------------------
# Tests — KV cache estimation
# ---------------------------------------------------------------------------


class TestEstimateKvCache:
    def test_basic_estimate(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        kv = estimate_kv_cache_bytes(meta, n_ctx=8192)
        # n_ctx * n_layers * 2(K+V) * n_kv_heads * head_dim * 2(FP16)
        # 8192 * 24 * 2 * 4 * 128 * 2 = 805_306_368
        assert kv == 8192 * 24 * 2 * 4 * 128 * 2

    def test_returns_none_without_metadata(self) -> None:
        meta = GGUFMetadata()  # all None
        assert estimate_kv_cache_bytes(meta, n_ctx=4096) is None

    def test_uses_head_count_when_kv_missing(self, tmp_path: Path) -> None:
        """Without head_count_kv, falls back to head_count (MHA)."""
        p = tmp_path / "mha.gguf"
        write_synthetic_gguf(
            p,
            [
                ("general.architecture", _T_STRING, "llama"),
                ("llama.embedding_length", _T_UINT64, 2048),
                ("llama.block_count", _T_UINT64, 24),
                ("llama.attention.head_count", _T_UINT64, 16),
                # No head_count_kv — MHA model
            ],
        )
        meta = read_gguf_metadata(p)
        kv = estimate_kv_cache_bytes(meta, n_ctx=4096)
        # Should use head_count=16: 4096 * 24 * 2 * 16 * 128 * 2
        assert kv == 4096 * 24 * 2 * 16 * 128 * 2


class TestEstimateTotalVram:
    def test_includes_model_size_and_overhead(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        total = estimate_total_vram_bytes(meta, n_ctx=8192)
        kv = estimate_kv_cache_bytes(meta, n_ctx=8192)
        assert total is not None and kv is not None
        assert total == meta.file_size_bytes + kv + 512 * 1024 * 1024

    def test_custom_overhead(self, llama_gguf: Path) -> None:
        meta = read_gguf_metadata(llama_gguf)
        total = estimate_total_vram_bytes(meta, n_ctx=4096, overhead_bytes=0)
        kv = estimate_kv_cache_bytes(meta, n_ctx=4096)
        assert total is not None and kv is not None
        assert total == meta.file_size_bytes + kv

    def test_returns_none_without_metadata(self) -> None:
        meta = GGUFMetadata()
        assert estimate_total_vram_bytes(meta, n_ctx=4096) is None
