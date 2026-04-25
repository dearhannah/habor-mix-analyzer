from __future__ import annotations

from itertools import combinations

from ..core import *

MIN_TERMINUS_MODELS = 3
MIN_AGENT_PAIRS = 2


def model_vs_agent_apples_to_apples(normalized_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apples-to-apples comparison of model vs agent effect sizes.

    Both effects use mean(|delta|) on benchmark-normalized scores:
    - Model effect: mean pairwise |score_i - score_j| across terminus-2 rows.
    - Agent effect: mean |score(model, company_agent) - score(model, terminus-2)| across models.
    """
    benchmarks = [c for c in normalized_df.columns if c not in KEY_COLUMNS]
    per_bench_rows = []

    for bench in benchmarks:
        bench_data = normalized_df[KEY_COLUMNS + [bench]].dropna()

        terminus_rows = bench_data[bench_data["agent"] == "terminus-2"]
        if len(terminus_rows) < MIN_TERMINUS_MODELS:
            continue
        model_mean_abs_delta = float(np.mean(
            [abs(a - b) for a, b in combinations(terminus_rows[bench].values, 2)]
        ))

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

        per_bench_rows.append({
            "benchmark": bench,
            "model_effect_mean_abs_delta": round(model_mean_abs_delta, 6),
            "agent_effect_mean_abs_delta": round(agent_mean_abs_delta, 6),
            "model_n_terminus_rows": len(terminus_rows),
            "agent_n_within_model_pairs": len(agent_deltas),
            "model_vs_agent_ratio": round(ratio, 4) if not np.isnan(ratio) else np.nan,
            "dominant": "model" if model_mean_abs_delta > agent_mean_abs_delta else "agent",
        })

    per_bench = pd.DataFrame(per_bench_rows).sort_values("model_vs_agent_ratio", ascending=False)

    summary_rows = []
    for effect, col in [
        ("model", "model_effect_mean_abs_delta"),
        ("agent", "agent_effect_mean_abs_delta"),
    ]:
        s = per_bench[col]
        summary_rows.append({
            "effect": effect,
            "mean_across_benchmarks": round(s.mean(), 6),
            "median_across_benchmarks": round(s.median(), 6),
            "p25_across_benchmarks": round(s.quantile(0.25), 6),
            "p75_across_benchmarks": round(s.quantile(0.75), 6),
            "n_benchmarks": len(s),
        })
    summary = pd.DataFrame(summary_rows)

    return per_bench, summary
