"""
Model vs Agent Effect Analysis

This script compares the impact of switching models vs switching agents
on benchmark scores using a straightforward effect size comparison.

Method:
- Model Effect: Fix agent to terminus-2, calculate score range across models
- Agent Effect: For each model, calculate score change when switching from terminus-2 to company agent

This avoids the confounding issues of Partial R² when model-agent pairings are not fully crossed.

Usage (from habor-mix-analyzer/):
  uv run python quantitative_study/scripts/model_vs_agent_effect_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "benchmark_level_matrix.csv"
OUTPUT_DIR = ROOT / "output" / "key_analyses" / "figures" / "benchmark_level"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_model_vs_agent_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model effect and agent effect for each benchmark."""
    benchmarks = [c for c in df.columns[2:] if df[c].notna().sum() >= 10]
    results = []

    for bench in benchmarks:
        bench_data = df[["model", "agent", bench]].dropna()

        # 1. Model effect: Fix terminus-2, measure score range across models
        terminus_data = bench_data[bench_data["agent"] == "terminus-2"]
        if len(terminus_data) >= 2:
            model_range = terminus_data[bench].max() - terminus_data[bench].min()
            model_best = terminus_data.loc[terminus_data[bench].idxmax(), "model"]
            model_worst = terminus_data.loc[terminus_data[bench].idxmin(), "model"]
        else:
            model_range = np.nan
            model_best = model_worst = ""

        # 2. Agent effect: For each model, calculate (company_agent - terminus-2)
        agent_deltas = []
        for model in bench_data["model"].unique():
            model_data = bench_data[bench_data["model"] == model]
            if "terminus-2" in model_data["agent"].values:
                terminus_score = model_data[model_data["agent"] == "terminus-2"][bench].values[0]
                other_agents = model_data[model_data["agent"] != "terminus-2"]
                for _, row in other_agents.iterrows():
                    agent_deltas.append(row[bench] - terminus_score)

        if agent_deltas:
            agent_effect_mean = np.mean(np.abs(agent_deltas))
            agent_effect_max = np.max(np.abs(agent_deltas))
        else:
            agent_effect_mean = agent_effect_max = np.nan

        results.append({
            "benchmark": bench,
            "model_range": model_range,
            "agent_effect_mean": agent_effect_mean,
            "agent_effect_max": agent_effect_max,
            "model_vs_agent_ratio": model_range / agent_effect_mean if agent_effect_mean > 0 else np.nan,
            "best_model": model_best,
            "worst_model": model_worst,
        })

    return pd.DataFrame(results).dropna().sort_values("model_vs_agent_ratio", ascending=False)


def print_summary(result_df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("=== Model Effect vs Agent Effect Comparison ===\n")
    print("Model Range: Score range across models, fixing terminus-2")
    print("Agent Effect: Avg score change from switching agent (within same model)\n")

    print(result_df[["benchmark", "model_range", "agent_effect_mean", "model_vs_agent_ratio"]]
          .head(20).to_string(index=False))

    print("\n=== Summary Statistics ===")
    print(f"Model Range Mean: {result_df['model_range'].mean():.4f}")
    print(f"Agent Effect Mean: {result_df['agent_effect_mean'].mean():.4f}")
    print(f"Model/Agent Ratio Mean: {result_df['model_vs_agent_ratio'].mean():.2f}x")
    print(f"Model/Agent Ratio Median: {result_df['model_vs_agent_ratio'].median():.2f}x")
    print()
    print(f"Benchmarks where Model > Agent: {(result_df['model_vs_agent_ratio'] > 1).sum()}")
    print(f"Benchmarks where Agent > Model: {(result_df['model_vs_agent_ratio'] < 1).sum()}")


def create_visualization(result_df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison visualization."""
    result_df_sorted = result_df.sort_values("model_vs_agent_ratio", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Plot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(
        result_df["model_range"],
        result_df["agent_effect_mean"],
        s=80, alpha=0.7, c="#3498db", edgecolor="white"
    )
    max_val = max(result_df["model_range"].max(), result_df["agent_effect_mean"].max())
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Model = Agent")
    ax1.set_xlabel("Model Effect (score range across models, fixing terminus-2)", fontsize=11)
    ax1.set_ylabel("Agent Effect (avg score change from switching agent)", fontsize=11)
    ax1.set_title("Model Effect vs Agent Effect by Benchmark")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Annotate outliers
    for _, row in result_df.nlargest(5, "model_range").iterrows():
        ax1.annotate(row["benchmark"], (row["model_range"], row["agent_effect_mean"]),
                     fontsize=8, alpha=0.8)

    # Plot 2: Ratio bar chart
    ax2 = axes[1]
    top_df = result_df_sorted.tail(25)
    colors = ["#27ae60" if r > 1 else "#e74c3c" for r in top_df["model_vs_agent_ratio"]]
    ax2.barh(top_df["benchmark"], top_df["model_vs_agent_ratio"], color=colors, edgecolor="white")
    ax2.axvline(1, color="black", linestyle="--", linewidth=1.5, label="Model = Agent")
    ax2.set_xlabel("Model Effect / Agent Effect Ratio")
    ax2.set_title("How Much More Does Model Matter Than Agent?\n(Green = Model matters more)")
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} model+agent combinations")

    # Compute effects
    result_df = compute_model_vs_agent_effects(df)

    # Print summary
    print_summary(result_df)

    # Save results
    output_csv = ROOT / "output" / "model_vs_agent_effect_comparison.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    # Create visualization
    create_visualization(result_df, OUTPUT_DIR / "model_vs_agent_effect_comparison.png")


if __name__ == "__main__":
    main()
