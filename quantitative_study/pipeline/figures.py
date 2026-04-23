"""
Paper-ready figures for the quantitative comparison pipeline.

All figures are saved to FIGURE_DIR (output/quantitative/figures/).
Call `generate_all_figures(...)` from the pipeline runner.

Visual style aligned with habor-analyze (src/habor_mix_analyzer/core/plotting.py).
"""

from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    COLOR_BLUE, COLOR_GRAY, COLOR_GREEN, COLOR_GRID, COLOR_AXIS,
    COLOR_PURPLE, COLOR_RED,
    DPI, DOMAIN_COLORS, FIGSIZE_SINGLE, FIGSIZE_WIDE,
    FIGURE_DIR, MATCH_STATUS_COLORS, MATCH_STATUS_LABELS,
    PLOT_STYLE, SUPERDOMAIN_COLORS,
)


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


def _wrap(value: object, width: int = 24) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=DPI, pad_inches=0.08)
    plt.close(fig)
    print(f"  [fig] {path}")


# ===================================================================
# Fig 1: Scatter — Harbor score vs Doc score
# ===================================================================

def fig_harbor_vs_doc_scatter(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for status, color in MATCH_STATUS_COLORS.items():
        sub = summary[summary["match_status"] == status]
        if sub.empty:
            continue
        ax.scatter(
            sub["score_doc"], sub["score_harbor"],
            c=color, label=MATCH_STATUS_LABELS.get(status, status),
            s=95, alpha=0.9, edgecolors="white", linewidths=0.5,
        )

    lims = [0, 1.05]
    ax.plot(lims, lims, color=COLOR_AXIS, linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Original benchmark score (doc)")
    ax.set_ylabel("Harbor score")
    ax.set_title("Harbor vs Original Benchmark Scores")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, "harbor_vs_doc_scatter.png")


# ===================================================================
# Fig 2: Bar chart — mean delta by benchmark
# ===================================================================

def fig_delta_by_benchmark(direction_df: pd.DataFrame) -> None:
    if direction_df.empty:
        return
    _apply_style()
    df = direction_df.sort_values("mean_delta")
    labels = [_wrap(b, 28) for b in df["benchmark"]]
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.42 * len(df))))

    colors = []
    for d in df["direction"]:
        if d == "HARBOR_HIGHER":
            colors.append(COLOR_GREEN)
        elif d == "HARBOR_LOWER":
            colors.append(COLOR_RED)
        else:
            colors.append(COLOR_GRAY)

    ax.barh(labels, df["mean_delta"], color=colors, edgecolor="white")
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.axvline(-0.05, color=COLOR_GRAY, linewidth=0.5, linestyle=":")
    ax.axvline(0.05, color=COLOR_GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("Mean Δ  (Harbor − Original)")
    ax.set_title("Score Difference by Benchmark")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["mean_delta"] + (0.005 if row["mean_delta"] >= 0 else -0.005),
            i, f'n={int(row["n"])}',
            va="center", ha="left" if row["mean_delta"] >= 0 else "right",
            fontsize=9, color="gray",
        )

    fig.tight_layout()
    _save(fig, "delta_by_benchmark.png")


# ===================================================================
# Fig 3: Agent lift heatmap
# ===================================================================

def fig_agent_lift_heatmap(lift_df: pd.DataFrame) -> None:
    if lift_df.empty:
        return
    _apply_style()
    pivot = lift_df.pivot_table(
        index="model", columns="benchmark", values="agent_lift", aggfunc="mean",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(3.8, 1.0 + 0.55 * pivot.shape[0])))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.1)
    image = ax.imshow(pivot.values, cmap="BrBG", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([_wrap(col, 14) for col in pivot.columns], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([_wrap(v, 20) for v in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.6 else "black")

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Agent lift over terminus-2")
    ax.set_title("Agent Lift vs Terminus-2  (Δ_agent − Δ_terminus)")
    fig.tight_layout()
    _save(fig, "agent_lift_heatmap.png")


# ===================================================================
# Fig 4: Delta distribution by match status
# ===================================================================

def fig_delta_distribution(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    statuses = [s for s in MATCH_STATUS_COLORS if s in summary["match_status"].values]
    data = [summary[summary["match_status"] == s]["delta"].dropna().values for s in statuses]
    labels = [_wrap(MATCH_STATUS_LABELS.get(s, s), 22) for s in statuses]
    colors = [MATCH_STATUS_COLORS[s] for s in statuses]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, vert=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color=COLOR_AXIS, linewidth=1, linestyle="--")
    ax.set_ylabel("Δ  (Harbor − Original)")
    ax.set_title("Delta Distribution by Match Type")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, "delta_distribution_boxplot.png")


# ===================================================================
# Fig 5: Hardest benchmarks (lowest best-model score, colored by domain)
# ===================================================================

def fig_hardest_benchmarks(difficulty_df: pd.DataFrame) -> None:
    if difficulty_df.empty:
        return
    _apply_style()
    df = difficulty_df.sort_values("best_score", ascending=True).head(40)
    labels = [_wrap(b, 24) for b in df["benchmark"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]
    fig, ax = plt.subplots(figsize=(11.5, max(8.2, 0.38 * len(df))))
    ax.barh(labels, df["best_score"], color=colors, edgecolor="white")
    ax.set_xlabel("Best score across all model+agent pairs")
    ax.set_title("Benchmark Difficulty: Room for Improvement")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.set_xlim(0, 1.05)

    for i, (_, row) in enumerate(df.iterrows()):
        label = _wrap(row["best_model"], 18)
        ax.text(row["best_score"] + 0.01, i, label,
                va="center", fontsize=8, color="gray")

    handles = []
    for domain, color in DOMAIN_COLORS.items():
        if domain in df["domain"].values:
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="white", label=domain))
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    _save(fig, "hardest_benchmarks.png")


# ===================================================================
# Fig 6: Domain score comparison (grouped bar)
# ===================================================================

def fig_domain_scores(domain_df: pd.DataFrame) -> None:
    if domain_df.empty:
        return
    _apply_style()
    df = domain_df.sort_values("mean_best_score")
    labels = [_wrap(d, 20) for d in df["domain"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]

    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.8 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.15, df["mean_best_score"], height=0.3, color=colors,
            edgecolor="white", label="Mean best score")
    ax.barh(y + 0.15, df["mean_all_scores"], height=0.3,
            color=[c + "88" for c in colors], edgecolor="white",
            label="Mean all entries")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title("Score Distribution by Domain")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_best_score"] + 0.01, i - 0.15,
                f'n={int(row["n_benchmarks"])}',
                va="center", fontsize=9, color="gray")

    fig.tight_layout()
    _save(fig, "domain_scores.png")


# ===================================================================
# Fig 7: Coding vs Non-Coding comparison (superdomain)
# ===================================================================

def fig_coding_vs_noncoding(superdomain_df: pd.DataFrame) -> None:
    if superdomain_df.empty:
        return
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(superdomain_df))
    colors = [SUPERDOMAIN_COLORS.get(sd, COLOR_GRAY) for sd in superdomain_df["superdomain"]]
    width = 0.3
    ax.bar(x - width / 2, superdomain_df["mean_best_score"], width,
           color=colors, edgecolor="white", label="Mean best score")
    ax.bar(x + width / 2, superdomain_df["mean_all_scores"], width,
           color=[c + "88" for c in colors], edgecolor="white",
           label="Mean all entries")

    ax.set_xticks(x)
    ax.set_xticklabels(superdomain_df["superdomain"])
    ax.set_ylabel("Score")
    ax.set_title("Coding vs Non-Coding: Score Comparison")
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)

    for i, (_, row) in enumerate(superdomain_df.iterrows()):
        ax.text(i, row["mean_best_score"] + 0.02,
                f'n={int(row["n_benchmarks"])}',
                ha="center", fontsize=11, color=COLOR_AXIS)

    fig.tight_layout()
    _save(fig, "coding_vs_noncoding.png")


# ===================================================================
# Fig 8: Progress over time (best score trajectory)
# ===================================================================

def fig_progress_over_time(progress_df: pd.DataFrame) -> None:
    if progress_df.empty:
        return
    _apply_style()
    benchmarks = progress_df["benchmark"].unique()
    top_n = min(12, len(benchmarks))
    coverage = progress_df.groupby("benchmark").size().nlargest(top_n).index
    plot_df = progress_df[progress_df["benchmark"].isin(coverage)].copy()

    all_dates = sorted(plot_df["date_ym"].unique())
    date_to_x = {d: i for i, d in enumerate(all_dates)}
    plot_df["_x"] = plot_df["date_ym"].map(date_to_x)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    palette = list(DOMAIN_COLORS.values()) + [COLOR_RED, COLOR_GRAY, COLOR_PURPLE]
    for i, (bench, g) in enumerate(plot_df.groupby("benchmark")):
        g = g.sort_values("_x")
        color = palette[i % len(palette)]
        ax.plot(g["_x"], g["best_score"], "o-", label=_wrap(bench, 18),
                color=color, markersize=5, linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Date (YYYY-MM)")
    ax.set_ylabel("Best score at that snapshot")
    ax.set_title("Score Progress Over Time (Doc Snapshots)")
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=9)

    step = max(1, len(all_dates) // 12)
    tick_positions = list(range(0, len(all_dates), step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([all_dates[i] for i in tick_positions], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    _save(fig, "progress_over_time.png")


# ===================================================================
# Fig 9: Domain progress bar chart
# ===================================================================

def fig_domain_progress(domain_prog_df: pd.DataFrame) -> None:
    if domain_prog_df.empty:
        return
    _apply_style()
    df = domain_prog_df.sort_values("mean_absolute_progress")
    labels = [_wrap(d, 20) for d in df["domain"]]
    colors = [DOMAIN_COLORS.get(d, COLOR_GRAY) for d in df["domain"]]

    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.8 * len(df))))
    ax.barh(labels, df["mean_absolute_progress"], color=colors, edgecolor="white")
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Mean absolute score improvement\n(latest snapshot − earliest snapshot)")
    ax.set_title("Progress by Domain (Doc Historical Data)")
    ax.set_ylabel("")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["mean_absolute_progress"] + 0.005, i,
                f'n={int(row["n_benchmarks"])}',
                va="center", fontsize=9, color="gray")

    fig.tight_layout()
    _save(fig, "domain_progress.png")


# ===================================================================
# Fig 10: Overall model ranking (mean score across all benchmarks)
# ===================================================================

def fig_model_ranking(harbor_df: pd.DataFrame) -> None:
    if harbor_df.empty:
        return
    _apply_style()
    from .cross_analysis import _is_well_scaled

    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    well_scaled = [c for c in bench_cols if _is_well_scaled(harbor_df[c])]

    model_means = (
        harbor_df.groupby("model")[well_scaled]
        .mean()
        .mean(axis=1)
        .sort_values(ascending=True)
    )
    if model_means.empty:
        return

    labels = [_wrap(m, 22) for m in model_means.index]
    fig, ax = plt.subplots(figsize=(11.5, max(4, 0.55 * len(model_means))))
    bars = ax.barh(labels, model_means.values, color=COLOR_BLUE, edgecolor="white")
    ax.set_xlabel("Mean score (across well-scaled benchmarks)")
    ax.set_title("Overall Model Ranking")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)

    for i, v in enumerate(model_means.values):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=9, color=COLOR_AXIS)

    fig.tight_layout()
    _save(fig, "model_ranking.png")


# ===================================================================
# Fig 11: Per-domain agent effect (terminus-2 vs others, Coding vs Non-Coding)
# ===================================================================

def fig_agent_domain_effect(harbor_df: pd.DataFrame) -> None:
    if harbor_df.empty or "agent" not in harbor_df.columns:
        return
    _apply_style()
    from .cross_analysis import _domain_for, _is_well_scaled

    id_cols = ["model", "agent"]
    bench_cols = [c for c in harbor_df.columns if c not in id_cols]
    well_scaled = {c for c in bench_cols if _is_well_scaled(harbor_df[c])}

    records = []
    for model, group in harbor_df.groupby("model"):
        t2 = group[group["agent"] == "terminus-2"]
        others = group[group["agent"] != "terminus-2"]
        if t2.empty or others.empty:
            continue
        t2_row = t2.iloc[0]
        for _, other_row in others.iterrows():
            for col in bench_cols:
                if col not in well_scaled:
                    continue
                v_t2 = t2_row[col]
                v_other = other_row[col]
                if pd.isna(v_t2) or pd.isna(v_other):
                    continue
                domain, superdomain = _domain_for(col)
                records.append(dict(
                    model=model,
                    agent=other_row["agent"],
                    benchmark=col,
                    domain=domain,
                    superdomain=superdomain,
                    delta=float(v_other) - float(v_t2),
                ))

    if not records:
        return
    df = pd.DataFrame(records)

    agg = df.groupby(["agent", "superdomain"]).agg(
        mean_delta=("delta", "mean"),
        n=("delta", "size"),
    ).reset_index()

    agents = sorted(agg["agent"].unique())
    superdomains = sorted(agg["superdomain"].unique())

    fig, ax = plt.subplots(figsize=(10, max(4, 0.8 * len(agents))))
    y = np.arange(len(agents))
    bar_height = 0.35
    sd_colors = [SUPERDOMAIN_COLORS.get(sd, COLOR_GRAY) for sd in superdomains]

    for j, sd in enumerate(superdomains):
        vals = []
        ns = []
        for agent in agents:
            row = agg[(agg["agent"] == agent) & (agg["superdomain"] == sd)]
            vals.append(float(row["mean_delta"].iloc[0]) if not row.empty else 0)
            ns.append(int(row["n"].iloc[0]) if not row.empty else 0)
        offset = (j - (len(superdomains) - 1) / 2) * bar_height
        bars = ax.barh(y + offset, vals, height=bar_height,
                       color=sd_colors[j], edgecolor="white", label=sd)
        for i, (v, n) in enumerate(zip(vals, ns)):
            if n > 0:
                ax.text(v + (0.003 if v >= 0 else -0.003), y[i] + offset,
                        f"n={n}", va="center",
                        ha="left" if v >= 0 else "right",
                        fontsize=8, color="gray")

    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(a, 20) for a in agents])
    ax.axvline(0, color=COLOR_AXIS, linewidth=1)
    ax.set_xlabel("Mean score change vs terminus-2")
    ax.set_title("Agent Effect by Domain (vs Terminus-2)")
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "agent_domain_effect.png")


# ===================================================================
# Public entry point
# ===================================================================

def generate_all_figures(
    summary: pd.DataFrame,
    direction: pd.DataFrame,
    lift: pd.DataFrame,
) -> None:
    print("\nGenerating comparison figures...")
    fig_harbor_vs_doc_scatter(summary)
    fig_delta_by_benchmark(direction)
    fig_agent_lift_heatmap(lift)
    fig_delta_distribution(summary)


def generate_cross_figures(
    difficulty: pd.DataFrame,
    domain_df: pd.DataFrame,
    superdomain_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    domain_prog_df: pd.DataFrame,
    harbor_df: pd.DataFrame | None = None,
) -> None:
    print("\nGenerating cross-analysis figures...")
    fig_hardest_benchmarks(difficulty)
    fig_domain_scores(domain_df)
    fig_coding_vs_noncoding(superdomain_df)
    fig_progress_over_time(progress_df)
    fig_domain_progress(domain_prog_df)
    if harbor_df is not None:
        fig_model_ranking(harbor_df)
        fig_agent_domain_effect(harbor_df)
    print(f"All figures saved to {FIGURE_DIR}/")
