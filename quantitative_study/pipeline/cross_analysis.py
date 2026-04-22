"""
Cross-analysis: combines Harbor matrix data with doc metadata to answer:
  1. What benchmarks are hardest? (lowest best-model scores)
  2. How do scores break down across domains (coding vs non-coding)?
  3. How much progress did models make over time? (from doc temporal data)
  4. Harness vs model importance (from habor-analyze outputs)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_INFO_DIR, DOMAIN_MAP, MATRIX_COLUMN_TO_STEM,
    SKIP_METRICS, transform_doc_score,
)

_STEM_TO_MATRIX: dict[str, str] = {}
for _mc, _st in MATRIX_COLUMN_TO_STEM.items():
    if _st is not None:
        _STEM_TO_MATRIX[_st] = _mc


def _domain_for(name: str) -> tuple[str, str]:
    """Look up domain by matrix column name, JSON stem, or normalized form."""
    if name in DOMAIN_MAP:
        return DOMAIN_MAP[name]
    hyphen = name.replace("_", "-")
    if hyphen in DOMAIN_MAP:
        return DOMAIN_MAP[hyphen]
    if name in _STEM_TO_MATRIX and _STEM_TO_MATRIX[name] in DOMAIN_MAP:
        return DOMAIN_MAP[_STEM_TO_MATRIX[name]]
    return ("Other", "Non-Coding")


def _is_well_scaled(series: pd.Series, lo: float = -0.01, hi: float = 1.01) -> bool:
    """True when all non-NaN values fall within [lo, hi]."""
    vals = series.dropna()
    if vals.empty:
        return False
    return bool((vals >= lo).all() and (vals <= hi).all())


# ===================================================================
# 1. Hardest benchmarks (lowest best-model score)
# ===================================================================

def benchmark_difficulty_table(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    rows: list[dict[str, Any]] = []
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty:
            continue
        best_idx = vals.idxmax()
        domain, superdomain = _domain_for(col)
        rows.append(dict(
            benchmark=col,
            domain=domain,
            superdomain=superdomain,
            best_score=vals.max(),
            best_model=harbor_df.loc[best_idx, "model"],
            best_agent=harbor_df.loc[best_idx, "agent"],
            median_score=vals.median(),
            worst_score=vals.min(),
            n_entries=len(vals),
            spread=vals.max() - vals.min(),
        ))
    return pd.DataFrame(rows).sort_values("best_score", ascending=True)


# ===================================================================
# 2. Domain-level aggregation
# ===================================================================

def domain_summary(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    domain_scores: dict[str, list[float]] = {}
    domain_bests: dict[str, list[float]] = {}
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty or not _is_well_scaled(vals):
            continue
        domain, _ = _domain_for(col)
        domain_scores.setdefault(domain, []).extend(vals.tolist())
        domain_bests.setdefault(domain, []).append(vals.max())

    rows = []
    for domain in sorted(domain_scores):
        all_vals = domain_scores[domain]
        bests = domain_bests[domain]
        rows.append(dict(
            domain=domain,
            n_benchmarks=len(bests),
            mean_best_score=np.mean(bests),
            mean_all_scores=np.mean(all_vals),
            median_all_scores=np.median(all_vals),
            std_all_scores=np.std(all_vals),
            min_best_score=min(bests),
        ))
    return pd.DataFrame(rows).sort_values("mean_best_score", ascending=False)


def superdomain_summary(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    sd_scores: dict[str, list[float]] = {}
    sd_bests: dict[str, list[float]] = {}
    for col in bench_cols:
        vals = harbor_df[col].dropna()
        if vals.empty or not _is_well_scaled(vals):
            continue
        _, superdomain = _domain_for(col)
        sd_scores.setdefault(superdomain, []).extend(vals.tolist())
        sd_bests.setdefault(superdomain, []).append(vals.max())

    rows = []
    for sd in sorted(sd_scores):
        all_vals = sd_scores[sd]
        bests = sd_bests[sd]
        rows.append(dict(
            superdomain=sd,
            n_benchmarks=len(bests),
            mean_best_score=np.mean(bests),
            mean_all_scores=np.mean(all_vals),
            median_all_scores=np.median(all_vals),
            std_all_scores=np.std(all_vals),
        ))
    return pd.DataFrame(rows)


def per_model_domain_scores(harbor_df: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (model, domain) — averaged across benchmarks in that domain."""
    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]

    well_scaled = {c for c in bench_cols if _is_well_scaled(harbor_df[c])}
    records = []
    for _, row in harbor_df.iterrows():
        model, agent = row["model"], row["agent"]
        for col in bench_cols:
            if col not in well_scaled:
                continue
            v = row[col]
            if pd.isna(v):
                continue
            domain, superdomain = _domain_for(col)
            records.append(dict(
                model=model, agent=agent, benchmark=col,
                domain=domain, superdomain=superdomain,
                score=float(v),
            ))
    long = pd.DataFrame(records)
    if long.empty:
        return pd.DataFrame()

    grouped = long.groupby(["model", "domain"]).agg(
        mean_score=("score", "mean"),
        n_benchmarks=("benchmark", "nunique"),
    ).reset_index()
    return grouped.sort_values(["domain", "mean_score"], ascending=[True, False])


# ===================================================================
# 3. Progress over time from doc data
# ===================================================================

def _parse_date(s: str) -> str | None:
    """Normalize date strings to 'YYYY-MM' for grouping."""
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{1,2})", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    return None


def _extract_best_score(
    results: list[dict], primary_metric: str, stem: str,
) -> float | None:
    """From a list of model result entries, extract the best score.
    Prioritizes the primary_metric; falls back to first numeric non-skip metric.
    Applies score transforms and discards values outside [0, 1] post-transform."""
    pm_lower = primary_metric.lower().strip() if primary_metric else ""
    best_primary: float | None = None
    best_fallback: float | None = None

    for entry in results:
        scores = entry.get("scores", [])
        for s in scores:
            metric = str(s.get("metric", "")).lower().strip()
            if metric in SKIP_METRICS:
                continue
            val = s.get("value")
            if val is None:
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            v = transform_doc_score(v, stem)
            if v < -0.01 or v > 1.01:
                continue
            is_primary = pm_lower and (pm_lower in metric or metric in pm_lower)
            if is_primary:
                if best_primary is None or v > best_primary:
                    best_primary = v
            else:
                if best_fallback is None or v > best_fallback:
                    best_fallback = v

    return best_primary if best_primary is not None else best_fallback


def progress_over_time(info_dir: Path | None = None) -> pd.DataFrame:
    """For benchmarks with multiple temporal snapshots, extract the best score
    at each time point to show progress."""
    info_dir = info_dir or BENCHMARK_INFO_DIR
    rows: list[dict[str, Any]] = []
    for fp in sorted(info_dir.glob("*.json")):
        with open(fp) as fh:
            doc = json.load(fh)
        rot = doc.get("results_over_time", [])
        if len(rot) < 2:
            continue
        stem = fp.stem
        primary = doc.get("evaluation", {}).get("primary_metric", "")
        name = doc.get("name", stem)
        for snapshot in rot:
            date_str = snapshot.get("date", "")
            ym = _parse_date(date_str)
            if not ym:
                continue
            results = snapshot.get("results", [])
            if not results:
                continue
            best = _extract_best_score(results, primary, stem)
            if best is not None:
                rows.append(dict(
                    benchmark=stem,
                    benchmark_name=name,
                    date_ym=ym,
                    best_score=best,
                    n_models=len(results),
                ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["benchmark", "date_ym"])
    deduped = df.groupby(["benchmark", "date_ym"]).agg(
        benchmark_name=("benchmark_name", "first"),
        best_score=("best_score", "max"),
        n_models=("n_models", "max"),
    ).reset_index()
    return deduped.sort_values(["benchmark", "date_ym"])


def progress_summary(progress_df: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark: earliest score, latest score, delta, months spanned."""
    if progress_df.empty:
        return pd.DataFrame()
    rows = []
    for bench, g in progress_df.groupby("benchmark"):
        g = g.sort_values("date_ym")
        earliest = g.iloc[0]
        latest = g.iloc[-1]
        delta = latest["best_score"] - earliest["best_score"]
        domain, superdomain = _domain_for(bench)
        rows.append(dict(
            benchmark=bench,
            domain=domain,
            superdomain=superdomain,
            earliest_date=earliest["date_ym"],
            latest_date=latest["date_ym"],
            earliest_best=earliest["best_score"],
            latest_best=latest["best_score"],
            absolute_progress=delta,
            n_snapshots=len(g),
        ))
    return pd.DataFrame(rows).sort_values("absolute_progress", ascending=False)


def domain_progress_summary(prog_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate progress by domain and superdomain."""
    if prog_summary.empty:
        return pd.DataFrame()
    rows = []
    for domain, g in prog_summary.groupby("domain"):
        rows.append(dict(
            domain=domain,
            superdomain=g.iloc[0]["superdomain"],
            n_benchmarks=len(g),
            mean_absolute_progress=g["absolute_progress"].mean(),
            median_absolute_progress=g["absolute_progress"].median(),
            mean_latest_best=g["latest_best"].mean(),
        ))
    return pd.DataFrame(rows).sort_values("mean_absolute_progress", ascending=False)
