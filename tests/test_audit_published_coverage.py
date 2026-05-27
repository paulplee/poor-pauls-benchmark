"""Unit tests for the published-data audit script.

These tests focus on the pure-function pieces (HF row → expectation
mapping, snapshot indexing, report rendering) so they run fast and offline.
The end-to-end script is exercised manually or via the scheduled CI job
that actually hits the network — running it under pytest would require
~100 MB of HF downloads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_published_coverage import (
    AuditResult,
    Finding,
    Pair,
    PairExpectation,
    check_static_pair,
    enumerate_pairs,
    index_snapshot,
    write_report,
)


def test_enumerate_pairs_splits_quant_and_qual() -> None:
    df = pd.DataFrame(
        [
            {  # plain quantitative row
                "model_base": "qwen3.6-27b",
                "gpu_name": "RTX 4060 Ti 16GB",
                "run_type": "quantitative",
                "throughput_tok_s": 42.0,
            },
            {  # explicit qualitative row
                "model_base": "qwen3.6-27b",
                "gpu_name": "RTX 4060 Ti 16GB",
                "run_type": "qualitative",
                "context_rot_score": 0.71,
            },
            {  # qualitative-by-metric even though run_type is missing
                "model_base": "qwen3.6-27b",
                "gpu_name": "RTX 4060 Ti 16GB",
                "run_type": None,
                "mt_bench_score": 7.4,
            },
            {  # second pair, quantitative only
                "model_base": "gemma4-e4b",
                "gpu_name": "M4 Pro",
                "run_type": "quantitative",
                "throughput_tok_s": 18.0,
            },
            {  # noise — should be filtered out
                "model_base": None,
                "gpu_name": "RTX 4060 Ti 16GB",
                "run_type": "quantitative",
            },
        ]
    )

    pairs = enumerate_pairs(df)
    qwen = Pair(model="qwen3.6-27b", gpu="RTX 4060 Ti 16GB")
    gemma = Pair(model="gemma4-e4b", gpu="M4 Pro")
    assert set(pairs) == {qwen, gemma}
    # The mt_bench-only row counts toward both quant (run_type defaults to
    # quantitative) and qual (because the metric is populated).
    assert pairs[qwen].quantitative_rows == 2
    assert pairs[qwen].qualitative_rows == 2
    assert pairs[gemma].quantitative_rows == 1
    assert pairs[gemma].qualitative_rows == 0


def test_index_snapshot_counts_qual_by_metric() -> None:
    snap = [
        {
            "model_base": "qwen3.6-27b",
            "gpu_name": "RTX 4060 Ti 16GB",
            "run_type": "quantitative",
        },
        {
            "model_base": "qwen3.6-27b",
            "gpu_name": "RTX 4060 Ti 16GB",
            "run_type": "qualitative",
            "overall_tool_accuracy": 0.83,
        },
    ]
    idx = index_snapshot(snap)
    pair = Pair("qwen3.6-27b", "RTX 4060 Ti 16GB")
    assert idx[pair]["quant"] == 1
    assert idx[pair]["qual"] == 1


def test_check_static_pair_flags_missing_qualitative() -> None:
    pair = Pair("ghost", "phantom-gpu")
    exp = PairExpectation(quantitative_rows=4, qualitative_rows=2)
    result = AuditResult()

    # Snapshot has the quant rows but no qual rows.
    snapshot_idx = {pair: {"quant": 4, "qual": 0}}
    check_static_pair(snapshot_idx, pair, exp, result)
    assert len(result.findings) == 1
    assert result.findings[0].surface == "static-snapshot"
    assert "qualitative" in result.findings[0].detail


def test_check_static_pair_flags_completely_missing() -> None:
    pair = Pair("ghost", "phantom-gpu")
    exp = PairExpectation(quantitative_rows=4, qualitative_rows=2)
    result = AuditResult()
    check_static_pair({}, pair, exp, result)
    assert len(result.findings) == 1
    assert result.findings[0].severity == "missing"


def test_write_report_groups_findings(tmp_path: Path) -> None:
    result = AuditResult(total_pairs=3, pairs_with_quant=3, pairs_with_qual=1)
    pair = Pair("qwen3.6-27b", "RTX 4060 Ti 16GB")
    result.add(Finding(pair, "mcp-qualitative", "missing", "no qual rows"))
    result.add(Finding(pair, "static-snapshot", "missing", "absent from snapshot"))

    out = tmp_path / "report.md"
    write_report(result, out)
    text = out.read_text()
    assert "Pairs audited: **3**" in text
    assert "## mcp-qualitative (1)" in text
    assert "## static-snapshot (1)" in text
    assert "`qwen3.6-27b`" in text


def test_write_report_clean_run(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_report(AuditResult(total_pairs=5, pairs_with_quant=5, pairs_with_qual=2), out)
    text = out.read_text()
    assert "surfaced correctly" in text
