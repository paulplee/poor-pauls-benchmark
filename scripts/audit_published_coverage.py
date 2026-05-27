"""Audit the published PPB data pipeline end-to-end.

For every distinct (model_base, gpu_name) pair present in the HuggingFace
dataset ``paulplee/ppb-results`` this script verifies that the pair is also
exposed by:

    1. The MCP REST API at ``https://mcp.poorpaul.dev/api/v1`` — both the
       quantitative (``/results``) and qualitative (``/qualitative``)
       endpoints, as appropriate.
    2. The static results snapshot served by the website at
       ``https://poorpaul.dev/data/results.json`` (the build-time artifact
       consumed by /insights when the live API is unreachable).
    3. (Optional) The live /insights page — rendered via Playwright — so we
       can confirm the charts actually populate for the pair.

A Markdown report listing every missing or mismatched pair is written to
``audit_report.md`` (or ``--report PATH``). The script exits with a non-zero
status code when discrepancies are found, making it suitable as a scheduled
CI check.

Usage
-----
    # Full audit, data-layer only (fast, no browser dep)
    uv run python scripts/audit_published_coverage.py

    # Limit to N (model, GPU) pairs — useful for local smoke tests
    uv run python scripts/audit_published_coverage.py --limit-pairs 20

    # Include the headless-browser /insights check (requires playwright)
    uv run python scripts/audit_published_coverage.py --with-browser

    # Override endpoints (e.g. point at a local MCP during development)
    uv run python scripts/audit_published_coverage.py \\
        --mcp-base http://localhost:8000/api/v1 \\
        --site-base http://localhost:3000

Exit codes
----------
    0 — every pair surfaced correctly
    1 — at least one discrepancy was reported
    2 — fatal error before the audit could complete
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger("ppb.audit")

DEFAULT_DATASET = "paulplee/ppb-results"
DEFAULT_MCP_BASE = "https://mcp.poorpaul.dev/api/v1"
DEFAULT_SITE_BASE = "https://poorpaul.dev"
DEFAULT_REPORT_PATH = Path("audit_report.md")

QUALITATIVE_COLUMNS = (
    "context_rot_score",
    "overall_tool_accuracy",
    "quality_composite_score",
    "mt_bench_score",
)


# ── Data classes ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pair:
    model: str
    gpu: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.model!r} on {self.gpu!r}"


@dataclass
class PairExpectation:
    """What we expect a (model, gpu) pair to show in each surface."""

    quantitative_rows: int = 0
    qualitative_rows: int = 0


@dataclass
class Finding:
    pair: Pair
    surface: str  # "mcp-results" | "mcp-qualitative" | "static-snapshot" | "insights-page"
    severity: str  # "missing" | "undercount" | "error"
    detail: str


@dataclass
class AuditResult:
    total_pairs: int = 0
    pairs_with_quant: int = 0
    pairs_with_qual: int = 0
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        logger.warning("[%s/%s] %s — %s", finding.surface, finding.severity, finding.pair, finding.detail)


# ── HuggingFace download ───────────────────────────────────────────────────


def download_hf_dataset(dataset: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Download every raw-rows shard from the HF dataset and return a DataFrame.

    The dataset historically shipped JSONL shards but is now a small set of
    Parquet files (one raw, one aggregated). We prefer the raw file because
    the audit needs per-row granularity (and the qualitative columns) — the
    aggregated file collapses repeated runs and drops qualitative metrics.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=dataset, repo_type="dataset")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    jsonl_files = [f for f in files if f.endswith(".jsonl")]

    if parquet_files:
        # Prefer the raw, non-aggregated file. Anything without "aggregated"
        # in the name is treated as raw — fall back to the largest file if
        # only aggregated names remain.
        raw = [f for f in parquet_files if "aggregated" not in f.lower()]
        chosen = raw if raw else parquet_files
        logger.info("Loading %d Parquet file(s) from %s: %s", len(chosen), dataset, chosen)
        frames: list[pd.DataFrame] = []
        for fname in chosen:
            local = hf_hub_download(
                repo_id=dataset,
                filename=fname,
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            frames.append(pd.read_parquet(local))
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        logger.info("Loaded %d rows from Parquet", len(df))
        return df

    if not jsonl_files:
        raise RuntimeError(
            f"No Parquet or JSONL shards found in {dataset}; saw {len(files)} files"
        )

    logger.info("Loading %d JSONL shard(s) from %s", len(jsonl_files), dataset)
    rows: list[dict[str, Any]] = []
    for fname in jsonl_files:
        local = hf_hub_download(
            repo_id=dataset,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        with open(local, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping bad line in %s: %s", fname, exc)
    logger.info("Loaded %d rows from JSONL", len(rows))
    return pd.DataFrame(rows)


def _is_quantitative(row: pd.Series) -> bool:
    rt = row.get("run_type")
    if isinstance(rt, str) and rt.strip().lower() == "qualitative":
        return False
    return True


def enumerate_pairs(df: pd.DataFrame) -> dict[Pair, PairExpectation]:
    """Group the raw dataset into expectations per (model_base, gpu_name) pair."""
    if df.empty:
        return {}
    if "model_base" not in df.columns or "gpu_name" not in df.columns:
        raise RuntimeError("Dataset is missing model_base/gpu_name columns")

    expectations: dict[Pair, PairExpectation] = {}
    # Drop rows with no model or GPU — they cannot be queried meaningfully.
    sub = df.dropna(subset=["model_base", "gpu_name"]).copy()
    sub["model_base"] = sub["model_base"].astype(str)
    sub["gpu_name"] = sub["gpu_name"].astype(str)

    run_type = sub.get("run_type")
    if run_type is None:
        sub["run_type"] = "quantitative"
    else:
        sub["run_type"] = run_type.fillna("quantitative").astype(str)

    for (model, gpu), grp in sub.groupby(["model_base", "gpu_name"], dropna=True):
        if not model or not gpu or model == "nan" or gpu == "nan":
            continue
        exp = expectations.setdefault(Pair(model=model, gpu=gpu), PairExpectation())
        quant_mask = grp["run_type"].str.lower() != "qualitative"
        exp.quantitative_rows = int(quant_mask.sum())
        # A row is considered qualitative either by run_type or by having any
        # qualitative metric populated. Some legacy submissions kept
        # run_type=quantitative even while filling MT-Bench scores.
        has_qual_metric = pd.Series(False, index=grp.index)
        for col in QUALITATIVE_COLUMNS:
            if col in grp.columns:
                has_qual_metric = has_qual_metric | grp[col].notna()
        qual_mask = (grp["run_type"].str.lower() == "qualitative") | has_qual_metric
        exp.qualitative_rows = int(qual_mask.sum())
    return expectations


# ── REST API checks ────────────────────────────────────────────────────────


def _safe_get(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    *,
    min_interval_s: float = 0.0,
    max_retries: int = 3,
) -> tuple[int, Any]:
    """GET with simple rate-limit throttle and 429 backoff.

    The MCP server enforces a 60-req/minute limit by default. ``min_interval_s``
    paces requests; on 429 we sleep for ``Retry-After`` (or an exponential
    backoff) and retry up to ``max_retries`` times.
    """
    if min_interval_s > 0:
        last = getattr(_safe_get, "_last_call", 0.0)
        wait = min_interval_s - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    backoff = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params, timeout=30.0)
        except httpx.HTTPError as exc:
            _safe_get._last_call = time.monotonic()  # type: ignore[attr-defined]
            return -1, {"error": str(exc)}
        _safe_get._last_call = time.monotonic()  # type: ignore[attr-defined]
        if resp.status_code == 429 and attempt < max_retries:
            retry_after = resp.headers.get("retry-after")
            sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else backoff
            logger.info("429 from %s — sleeping %.1fs before retry", url, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2
            continue
        if resp.status_code != 200:
            return resp.status_code, {"error": resp.text[:200]}
        try:
            return resp.status_code, resp.json()
        except ValueError as exc:
            return resp.status_code, {"error": f"non-JSON response: {exc}"}
    return 429, {"error": "rate-limited after retries"}


def check_mcp_pair(
    client: httpx.Client,
    mcp_base: str,
    pair: Pair,
    expectation: PairExpectation,
    result: AuditResult,
    *,
    min_interval_s: float = 0.0,
) -> None:
    """Hit /results and /qualitative; record findings for missing data."""

    if expectation.quantitative_rows > 0:
        status, body = _safe_get(
            client,
            f"{mcp_base.rstrip('/')}/results",
            {"gpu": pair.gpu, "model": pair.model, "limit": 5000},
            min_interval_s=min_interval_s,
        )
        if status != 200:
            result.add(
                Finding(
                    pair=pair,
                    surface="mcp-results",
                    severity="error",
                    detail=f"HTTP {status}: {body.get('error') if isinstance(body, dict) else body}",
                )
            )
        else:
            rows = body.get("rows") if isinstance(body, dict) else None
            if not rows:
                result.add(
                    Finding(
                        pair=pair,
                        surface="mcp-results",
                        severity="missing",
                        detail=f"expected ≥{expectation.quantitative_rows} quant rows, got 0",
                    )
                )

    if expectation.qualitative_rows > 0:
        status, body = _safe_get(
            client,
            f"{mcp_base.rstrip('/')}/qualitative",
            {"gpu": pair.gpu, "model": pair.model, "limit": 200},
            min_interval_s=min_interval_s,
        )
        if status != 200:
            result.add(
                Finding(
                    pair=pair,
                    surface="mcp-qualitative",
                    severity="error",
                    detail=f"HTTP {status}: {body.get('error') if isinstance(body, dict) else body}",
                )
            )
        else:
            rows = body.get("rows") if isinstance(body, dict) else None
            if not rows:
                result.add(
                    Finding(
                        pair=pair,
                        surface="mcp-qualitative",
                        severity="missing",
                        detail=(
                            f"expected ≥{expectation.qualitative_rows} qual rows, got 0 — "
                            "this is the most likely cause of an empty insights page"
                        ),
                    )
                )


# ── Static snapshot check ─────────────────────────────────────────────────


def fetch_static_snapshot(site_base: str) -> list[dict[str, Any]]:
    url = f"{site_base.rstrip('/')}/data/results.json"
    logger.info("Fetching static snapshot %s", url)
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"{url} did not return a JSON list")
    logger.info("Static snapshot contains %d rows", len(data))
    return data


def index_snapshot(rows: Iterable[dict[str, Any]]) -> dict[Pair, dict[str, int]]:
    out: dict[Pair, dict[str, int]] = {}
    for r in rows:
        model = r.get("model_base") or r.get("model")
        gpu = r.get("gpu_name")
        if not model or not gpu:
            continue
        pair = Pair(model=str(model), gpu=str(gpu))
        counts = out.setdefault(pair, {"quant": 0, "qual": 0})
        if any(r.get(col) not in (None, "") for col in QUALITATIVE_COLUMNS):
            counts["qual"] += 1
        run_type = (r.get("run_type") or "").lower()
        if run_type != "qualitative":
            counts["quant"] += 1
    return out


def check_static_pair(
    snapshot_index: dict[Pair, dict[str, int]],
    pair: Pair,
    expectation: PairExpectation,
    result: AuditResult,
) -> None:
    counts = snapshot_index.get(pair)
    if counts is None:
        if expectation.quantitative_rows or expectation.qualitative_rows:
            result.add(
                Finding(
                    pair=pair,
                    surface="static-snapshot",
                    severity="missing",
                    detail="pair absent from /data/results.json — site needs a redeploy / fetch-data run",
                )
            )
        return
    if expectation.quantitative_rows and counts["quant"] == 0:
        result.add(
            Finding(
                pair=pair,
                surface="static-snapshot",
                severity="missing",
                detail="no quantitative rows in /data/results.json",
            )
        )
    if expectation.qualitative_rows and counts["qual"] == 0:
        result.add(
            Finding(
                pair=pair,
                surface="static-snapshot",
                severity="missing",
                detail="no qualitative rows in /data/results.json",
            )
        )


# ── Optional Playwright /insights check ───────────────────────────────────


def check_insights_browser(
    site_base: str,
    sample_pairs: list[tuple[Pair, PairExpectation]],
    result: AuditResult,
) -> None:
    """For a small sample of pairs, load /insights and assert charts render."""
    try:
        from playwright.sync_api import (  # type: ignore[import-not-found]
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )
    except ImportError:
        logger.warning(
            "playwright is not installed — skipping browser checks. "
            "Install with `uv pip install playwright && playwright install chromium` to enable.",
        )
        return

    base = site_base.rstrip("/")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        for pair, expectation in sample_pairs:
            params = httpx.QueryParams({"gpu": pair.gpu, "model": pair.model})
            url = f"{base}/insights?{params}"
            try:
                page.goto(url, wait_until="networkidle", timeout=30_000)
            except PlaywrightTimeoutError as exc:
                result.add(
                    Finding(
                        pair=pair,
                        surface="insights-page",
                        severity="error",
                        detail=f"timeout loading {url}: {exc}",
                    )
                )
                continue

            # Heuristic: every chart renders an <svg> (Recharts) or <canvas>.
            try:
                chart_count = page.evaluate(
                    "() => document.querySelectorAll('svg.recharts-surface, canvas').length"
                )
            except Exception as exc:  # pragma: no cover - defensive
                result.add(
                    Finding(
                        pair=pair,
                        surface="insights-page",
                        severity="error",
                        detail=f"page eval failed: {exc}",
                    )
                )
                continue

            if chart_count == 0:
                result.add(
                    Finding(
                        pair=pair,
                        surface="insights-page",
                        severity="missing",
                        detail=f"no chart elements rendered at {url}",
                    )
                )
                continue

            # Probe each qualitative tab when the pair has qualitative data.
            if expectation.qualitative_rows:
                for tab_label in ("Context Rot", "Tools", "Quality", "Multi-Turn"):
                    try:
                        page.get_by_role("button", name=tab_label).first.click(timeout=2_000)
                        page.wait_for_timeout(750)
                        tab_chart_count = page.evaluate(
                            "() => document.querySelectorAll('svg.recharts-surface, canvas').length"
                        )
                    except Exception as exc:
                        result.add(
                            Finding(
                                pair=pair,
                                surface="insights-page",
                                severity="error",
                                detail=f"failed to open tab {tab_label}: {exc}",
                            )
                        )
                        continue
                    if tab_chart_count == 0:
                        result.add(
                            Finding(
                                pair=pair,
                                surface="insights-page",
                                severity="missing",
                                detail=f"tab {tab_label} rendered no charts at {url}",
                            )
                        )
        browser.close()


# ── Report ─────────────────────────────────────────────────────────────────


def write_report(result: AuditResult, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Poor Paul's Benchmark — Published Data Audit")
    lines.append("")
    lines.append(f"- Pairs audited: **{result.total_pairs}**")
    lines.append(f"- Pairs with quantitative data: **{result.pairs_with_quant}**")
    lines.append(f"- Pairs with qualitative data: **{result.pairs_with_qual}**")
    lines.append(f"- Findings: **{len(result.findings)}**")
    lines.append("")

    if not result.findings:
        lines.append("All published (model, GPU) pairs surfaced correctly. ✓")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    grouped: dict[str, list[Finding]] = {}
    for f in result.findings:
        grouped.setdefault(f.surface, []).append(f)

    for surface in sorted(grouped):
        findings = grouped[surface]
        lines.append(f"## {surface} ({len(findings)})")
        lines.append("")
        lines.append("| Severity | Model | GPU | Detail |")
        lines.append("| --- | --- | --- | --- |")
        for f in sorted(findings, key=lambda x: (x.severity, x.pair.model, x.pair.gpu)):
            detail = f.detail.replace("|", "\\|")
            lines.append(f"| {f.severity} | `{f.pair.model}` | `{f.pair.gpu}` | {detail} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset id")
    p.add_argument("--mcp-base", default=DEFAULT_MCP_BASE, help="MCP REST API base URL")
    p.add_argument("--site-base", default=DEFAULT_SITE_BASE, help="poorpaul.dev base URL")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Markdown report output path")
    p.add_argument("--limit-pairs", type=int, default=0, help="If >0, audit at most N random pairs (smoke test)")
    p.add_argument(
        "--with-browser",
        action="store_true",
        help="Also drive a headless Chromium against /insights for a sample of pairs",
    )
    p.add_argument(
        "--browser-sample",
        type=int,
        default=10,
        help="Number of pairs to spot-check via Playwright when --with-browser is set",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (defaults to ~/.cache/huggingface)",
    )
    p.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip the static snapshot check (/data/results.json)",
    )
    p.add_argument(
        "--mcp-rps",
        type=float,
        default=0.9,
        help=(
            "Max requests/sec to the MCP REST API. Default 0.9 stays under the "
            "deployed 60/min rate limit. Set to 0 to disable throttling."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    try:
        df = download_hf_dataset(args.dataset, cache_dir=args.cache_dir)
    except Exception as exc:  # pragma: no cover - network
        logger.exception("Failed to download dataset %s", args.dataset)
        print(f"FATAL: could not download {args.dataset}: {exc}", file=sys.stderr)
        return 2

    expectations = enumerate_pairs(df)
    if not expectations:
        print("FATAL: no (model, GPU) pairs found in dataset", file=sys.stderr)
        return 2

    pairs = sorted(expectations.items(), key=lambda kv: (kv[0].model, kv[0].gpu))
    if args.limit_pairs and args.limit_pairs < len(pairs):
        rng = random.Random(42)
        pairs = rng.sample(pairs, args.limit_pairs)

    result = AuditResult(
        total_pairs=len(pairs),
        pairs_with_quant=sum(1 for _, e in pairs if e.quantitative_rows),
        pairs_with_qual=sum(1 for _, e in pairs if e.qualitative_rows),
    )

    # 1. MCP REST checks
    logger.info("Checking MCP REST API at %s …", args.mcp_base)
    min_interval = (1.0 / args.mcp_rps) if args.mcp_rps and args.mcp_rps > 0 else 0.0
    with httpx.Client(timeout=30.0, headers={"user-agent": "ppb-audit/0.1"}) as client:
        for i, (pair, exp) in enumerate(pairs, start=1):
            if i % 25 == 0:
                logger.info("  … %d/%d pairs queried", i, len(pairs))
            check_mcp_pair(client, args.mcp_base, pair, exp, result, min_interval_s=min_interval)

    # 2. Static snapshot
    if not args.skip_static:
        try:
            snapshot = fetch_static_snapshot(args.site_base)
            snapshot_index = index_snapshot(snapshot)
            for pair, exp in pairs:
                check_static_pair(snapshot_index, pair, exp, result)
        except Exception as exc:
            logger.exception("Static snapshot fetch failed")
            result.add(
                Finding(
                    pair=Pair("*", "*"),
                    surface="static-snapshot",
                    severity="error",
                    detail=f"could not fetch snapshot: {exc}",
                )
            )

    # 3. Browser checks (sample only — full sweep would be hours)
    if args.with_browser:
        sample_size = min(args.browser_sample, len(pairs))
        # Prefer pairs with qualitative data — those are the ones the user noticed missing.
        qual_pairs = [(p, e) for p, e in pairs if e.qualitative_rows]
        non_qual = [(p, e) for p, e in pairs if not e.qualitative_rows]
        rng = random.Random(7)
        sample: list[tuple[Pair, PairExpectation]] = []
        sample.extend(rng.sample(qual_pairs, min(len(qual_pairs), sample_size)))
        remaining = sample_size - len(sample)
        if remaining > 0:
            sample.extend(rng.sample(non_qual, min(len(non_qual), remaining)))
        logger.info("Driving headless browser for %d sample pair(s)", len(sample))
        check_insights_browser(args.site_base, sample, result)

    write_report(result, args.report)
    print(
        f"Audit complete: {len(result.findings)} finding(s) across {result.total_pairs} pair(s). "
        f"Report → {args.report}",
    )
    return 1 if result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
