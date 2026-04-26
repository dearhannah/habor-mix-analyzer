"""
Core analyses:
  A. Harbor vs Doc comparison (long table + summary)
  B. Harbor data anomaly detection
  C. Agent capability analysis (agent lift, per-benchmark direction)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .alignment import find_doc_match, resolve_stem
from .loading import DocSlice, harbor_to_long


# ===================================================================
# A. Harbor vs Doc comparison
# ===================================================================

def build_comparison_long(
    harbor_df: pd.DataFrame,
    docs: dict[str, DocSlice],
) -> pd.DataFrame:
    stems = set(docs.keys())
    long = harbor_to_long(harbor_df)
    rows: list[dict[str, Any]] = []

    for _, r in long.iterrows():
        col = str(r["matrix_column"])
        model, agent = str(r["model"]), str(r["agent"])
        stem = resolve_stem(col, stems)
        sh = r["score_harbor"]
        harbor_val = None if pd.isna(sh) else float(sh)

        if stem is None:
            rows.append(dict(
                matrix_column=col, stem="", benchmark_name="",
                model=model, agent=agent,
                score_harbor=harbor_val, score_doc=None, delta=None,
                doc_model_matched="", doc_agent_matched="",
                doc_metric_used="", primary_metric_json="",
                doc_slice_date="", doc_source_url="",
                match_status="skipped_no_json",
            ))
            continue

        doc = docs[stem]
        sd, dm, da, mu, st = find_doc_match(model, agent, doc)
        delta = (harbor_val - sd) if (sd is not None and harbor_val is not None and st.startswith("matched")) else None

        rows.append(dict(
            matrix_column=col, stem=stem, benchmark_name=doc.benchmark_name,
            model=model, agent=agent,
            score_harbor=harbor_val, score_doc=sd, delta=delta,
            doc_model_matched=dm, doc_agent_matched=da,
            doc_metric_used=mu, primary_metric_json=doc.primary_metric,
            doc_slice_date=doc.slice_date, doc_source_url=doc.source_url,
            match_status=st,
        ))

    return pd.DataFrame(rows)


def build_comparison_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    matched = long_df[long_df["match_status"].str.startswith("matched")]
    has_delta = matched.dropna(subset=["delta"])
    if has_delta.empty:
        return pd.DataFrame()

    rows = []
    for (stem, model, agent), g in has_delta.groupby(["stem", "model", "agent"]):
        r = g.iloc[0]
        rows.append(dict(
            benchmark=stem, model=model, harbor_agent=agent,
            score_harbor=r["score_harbor"], score_doc=r["score_doc"],
            delta=r["delta"], abs_delta=abs(r["delta"]),
            doc_model_matched=r["doc_model_matched"],
            doc_agent_matched=r["doc_agent_matched"],
            match_status=r["match_status"],
            doc_metric=r["doc_metric_used"],
            doc_slice_date=r["doc_slice_date"],
        ))
    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False)


# ===================================================================
# B. Anomaly detection
# ===================================================================

def detect_anomalies(harbor_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["model", "agent"]
    long = harbor_df.melt(id_vars=id_cols, var_name="benchmark", value_name="score")
    long = long.dropna(subset=["score"])

    flags: list[dict] = []
    for _, r in long.iterrows():
        s = float(r["score"])
        bench = r["benchmark"]
        issues = []
        if s < -0.01:
            issues.append("negative")
        if s > 1.01:
            issues.append("above_1")
        if s == 0.0:
            issues.append("exact_zero")
        if abs(s) > 100:
            issues.append("extreme_magnitude")
        if issues:
            flags.append(dict(
                benchmark=bench, model=r["model"], agent=r["agent"],
                score=s, issues="|".join(issues),
            ))
    return pd.DataFrame(flags)


def anomaly_summary_by_benchmark(anomalies: pd.DataFrame) -> pd.DataFrame:
    if anomalies.empty:
        return pd.DataFrame()
    rows = []
    for bench, g in anomalies.groupby("benchmark"):
        rows.append(dict(
            benchmark=bench,
            n_anomalous=len(g),
            issues=", ".join(sorted(set("|".join(g["issues"]).split("|")))),
            example_score=g["score"].iloc[0],
        ))
    return pd.DataFrame(rows).sort_values("n_anomalous", ascending=False)


# ===================================================================
# C. Agent analysis
# ===================================================================

def agent_effect_by_status(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean/median delta grouped by match_status."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for st, g in summary.groupby("match_status"):
        rows.append(dict(
            match_status=st, n=len(g),
            mean_delta=g["delta"].mean(),
            median_delta=g["delta"].median(),
            std_delta=g["delta"].std(),
        ))
    return pd.DataFrame(rows)


def agent_effect_by_agent(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean/median delta grouped by harbor_agent."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for agent, g in summary.groupby("harbor_agent"):
        rows.append(dict(
            harbor_agent=agent, n=len(g),
            mean_delta=g["delta"].mean(),
            median_delta=g["delta"].median(),
            std_delta=g["delta"].std(),
        ))
    return pd.DataFrame(rows)


def benchmark_direction(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark: is Harbor systematically higher/lower/similar?"""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for bench, g in summary.groupby("benchmark"):
        md = g["delta"].mean()
        direction = "HARBOR_HIGHER" if md > 0.05 else ("HARBOR_LOWER" if md < -0.05 else "SIMILAR")
        rows.append(dict(
            benchmark=bench, n=len(g),
            mean_delta=md,
            median_delta=g["delta"].median(),
            direction=direction,
        ))
    return pd.DataFrame(rows).sort_values("mean_delta")


def agent_lift_table(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (benchmark, model) with both terminus-2 and another agent,
    compute agent_lift = delta_agent - delta_terminus2."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (bench, model), g in summary.groupby(["benchmark", "model"]):
        t2 = g[g["harbor_agent"] == "terminus-2"]
        others = g[g["harbor_agent"] != "terminus-2"]
        if t2.empty or others.empty:
            continue
        t2_delta = float(t2.iloc[0]["delta"])
        for _, other in others.iterrows():
            lift = float(other["delta"]) - t2_delta
            rows.append(dict(
                benchmark=bench, model=model,
                harbor_agent=other["harbor_agent"],
                delta_terminus2=t2_delta,
                delta_agent=float(other["delta"]),
                agent_lift=lift,
            ))
    return pd.DataFrame(rows).sort_values("agent_lift", ascending=False)


def rank_correlation_by_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between Harbor and Doc rankings within each benchmark."""
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for bench, g in summary.groupby("benchmark"):
        g = g.dropna(subset=["score_harbor", "score_doc"])
        if len(g) < 3:
            continue
        rho, pval = sp_stats.spearmanr(g["score_harbor"], g["score_doc"])
        rows.append(dict(benchmark=bench, n=len(g), spearman_rho=rho, p_value=pval))
    return pd.DataFrame(rows).sort_values("spearman_rho")
