"""
Model vs Agent: Apples-to-Apples Effect Comparison

Methodology — both effects use the same statistic (mean absolute delta)
on benchmark-normalized scores, so they are directly comparable:

  Model effect per benchmark:
    Mean of all pairwise |score_i - score_j| across terminus-2 rows.
    Captures: how much does choosing a different model typically change performance?

  Agent effect per benchmark:
    Mean of |score(model, company_agent) - score(model, terminus-2)| across models.
    Captures: how much does switching from terminus-2 to a company agent typically change performance?

Both are in normalized score units (mean 0, ~unit variance per benchmark column),
so aggregation across benchmarks is meaningful.

Usage (from habor-mix-analyzer/):
  uv run python quantitative_study/scripts/final_analysis/model_vs_agent_apples_to_apples.py
"""

from __future__ import annotations
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NORMALIZED_PATH = (
    ROOT / "data" / "processed" / "intermediate"
    / "benchmark_from_task_aggregate_normalized_matrix.csv"
)
OUTPUT_DIR = ROOT / "output" / "final_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TERMINUS_MODELS = 3
MIN_AGENT_PAIRS = 2


def compute_per_benchmark_effects(df: pd.DataFrame) -> pd.DataFrame:
    benchmarks = [c for c in df.columns if c not in ("model", "agent")]
    results = []

    for bench in benchmarks:
        bench_data = df[["model", "agent", bench]].dropna()

        # Model effect: mean pairwise |delta| within terminus-2 condition
        terminus_rows = bench_data[bench_data["agent"] == "terminus-2"]
        if len(terminus_rows) < MIN_TERMINUS_MODELS:
            continue
        terminus_scores = terminus_rows[bench].values
        model_mean_abs_delta = float(np.mean(
            [abs(a - b) for a, b in combinations(terminus_scores, 2)]
        ))

        # Agent effect: |company_agent - terminus-2| within each model
        agent_deltas = []
        for model, group in bench_data.groupby("model"):
            if "terminus-2" not in group["agent"].values:
                continue
            t2_score = group.loc[group["agent"] == "terminus-2", bench].values[0]
            for _, row in group[group["agent"] != "terminus-2"].iterrows():
                agent_deltas.append(abs(row[bench] - t2_score))

        if len(agent_deltas) < MIN_AGENT_PAIRS:
            continue

        agent_mean_abs_delta = float(np.mean(agent_deltas))
        ratio = model_mean_abs_delta / agent_mean_abs_delta if agent_mean_abs_delta > 0 else np.nan

        results.append({
            "benchmark": bench,
            "model_effect_mean_abs_delta": round(model_mean_abs_delta, 6),
            "agent_effect_mean_abs_delta": round(agent_mean_abs_delta, 6),
            "model_n_terminus_rows": len(terminus_rows),
            "agent_n_within_model_pairs": len(agent_deltas),
            "model_vs_agent_ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
            "dominant": "model" if model_mean_abs_delta > agent_mean_abs_delta else "agent",
        })

    return pd.DataFrame(results).sort_values("model_vs_agent_ratio", ascending=False)


def compute_summary(per_bench: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for effect, col in [
        ("model", "model_effect_mean_abs_delta"),
        ("agent", "agent_effect_mean_abs_delta"),
    ]:
        s = per_bench[col]
        rows.append({
            "effect": effect,
            "mean_across_benchmarks": round(s.mean(), 6),
            "median_across_benchmarks": round(s.median(), 6),
            "p25_across_benchmarks": round(s.quantile(0.25), 6),
            "p75_across_benchmarks": round(s.quantile(0.75), 6),
            "n_benchmarks": len(s),
        })
    summary = pd.DataFrame(rows)

    model_mean   = summary.loc[summary["effect"] == "model", "mean_across_benchmarks"].values[0]
    agent_mean   = summary.loc[summary["effect"] == "agent", "mean_across_benchmarks"].values[0]
    model_median = summary.loc[summary["effect"] == "model", "median_across_benchmarks"].values[0]
    agent_median = summary.loc[summary["effect"] == "agent", "median_across_benchmarks"].values[0]

    print("\n=== Apples-to-Apples: Model vs Agent Effect ===")
    print("Statistic: mean(|delta|) on benchmark-normalized scores\n")
    print(f"Model effect — mean: {model_mean:.4f},  median: {model_median:.4f}")
    print(f"Agent effect — mean: {agent_mean:.4f},  median: {agent_median:.4f}")
    print(f"\nModel/Agent ratio — mean: {model_mean/agent_mean:.2f}x,  median: {model_median/agent_median:.2f}x")
    print(f"\nBenchmarks model > agent : {(per_bench['dominant']=='model').sum()} / {len(per_bench)}")
    print(f"Benchmarks agent > model : {(per_bench['dominant']=='agent').sum()} / {len(per_bench)}")
    return summary


def create_figure(per_bench: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter per benchmark
    ax = axes[0]
    colors = ["#e74c3c" if d == "agent" else "#3498db" for d in per_bench["dominant"]]
    ax.scatter(
        per_bench["model_effect_mean_abs_delta"],
        per_bench["agent_effect_mean_abs_delta"],
        c=colors, s=60, alpha=0.75, edgecolor="white",
    )
    max_val = max(
        per_bench["model_effect_mean_abs_delta"].max(),
        per_bench["agent_effect_mean_abs_delta"].max(),
    ) * 1.08
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.45, linewidth=1.2, label="Equal effects")
    ax.set_xlabel("Model Effect  [ mean|Δ| across model pairs, terminus-2 only ]", fontsize=9)
    ax.set_ylabel("Agent Effect  [ mean|Δ| company vs terminus-2, per model ]", fontsize=9)
    ax.set_title(
        "Per-Benchmark: Model vs Agent Effect\n(blue = model dominates, red = agent dominates)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    for _, row in per_bench.nlargest(4, "model_vs_agent_ratio").iterrows():
        ax.annotate(
            row["benchmark"],
            (row["model_effect_mean_abs_delta"], row["agent_effect_mean_abs_delta"]),
            fontsize=7, alpha=0.8,
        )
    for _, row in per_bench.nsmallest(3, "model_vs_agent_ratio").iterrows():
        ax.annotate(
            row["benchmark"],
            (row["model_effect_mean_abs_delta"], row["agent_effect_mean_abs_delta"]),
            fontsize=7, alpha=0.8,
        )

    # Right: summary bar chart
    ax2 = axes[1]
    stats = {
        "Mean\nacross benchmarks": (
            per_bench["model_effect_mean_abs_delta"].mean(),
            per_bench["agent_effect_mean_abs_delta"].mean(),
        ),
        "Median\nacross benchmarks": (
            per_bench["model_effect_mean_abs_delta"].median(),
            per_bench["agent_effect_mean_abs_delta"].median(),
        ),
    }
    x = np.arange(len(stats))
    width = 0.32
    model_vals = [v[0] for v in stats.values()]
    agent_vals  = [v[1] for v in stats.values()]
    bars1 = ax2.bar(x - width / 2, model_vals, width, label="Model effect", color="#3498db", alpha=0.85)
    bars2 = ax2.bar(x + width / 2, agent_vals,  width, label="Agent effect", color="#e74c3c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(stats.keys()), fontsize=9)
    ax2.set_ylabel("mean|Δ| in normalized score units", fontsize=9)
    ax2.set_title(
        "Overall Summary: Model vs Agent\n(same statistic — apples-to-apples)", fontsize=10
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    for bar in [*bars1, *bars2]:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {output_path}")


def main() -> None:
    df = pd.read_csv(NORMALIZED_PATH)
    print(f"Loaded {len(df)} model+agent combinations, {len(df.columns)-2} benchmarks")

    per_bench = compute_per_benchmark_effects(df)
    summary = compute_summary(per_bench)

    per_bench_path = OUTPUT_DIR / "model_vs_agent_per_benchmark.csv"
    summary_path   = OUTPUT_DIR / "model_vs_agent_summary.csv"
    per_bench.to_csv(per_bench_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nPer-benchmark table: {per_bench_path}")
    print(f"Summary table:       {summary_path}")

    create_figure(per_bench, OUTPUT_DIR / "model_vs_agent_apples_to_apples.png")


if __name__ == "__main__":
    main()
