from __future__ import annotations

from ..core import *

from .intermediate_tables import (
    _adjusted_r2,
    _permutation_partial_r2,
    design_matrix,
    fit_r2,
    fit_r2_full,
    variance_decomposition,
)


def adjusted_group_effects(long_df: pd.DataFrame, group_col: str, controls: list[str]) -> pd.DataFrame:
    df = long_df.dropna(subset=["normalized_score"]).copy()
    y = df["normalized_score"].astype(float)
    x = design_matrix(df, controls)
    if x.empty:
        df["adjusted_score"] = y
    else:
        model = LinearRegression()
        model.fit(x, y)
        df["adjusted_score"] = y - model.predict(x) + float(y.mean())
    return (
        df.groupby(group_col)
        .agg(
            adjusted_mean=("adjusted_score", "mean"),
            adjusted_std=("adjusted_score", "std"),
            unadjusted_benchmark_relative_mean=("normalized_score", "mean"),
            observations=("normalized_score", "size"),
        )
        .reset_index()
        .sort_values("adjusted_mean", ascending=False)
    )


def filtered_variance_decomposition(benchmark_long_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return variance_decomposition(benchmark_long_df[benchmark_long_df["benchmark"].isin(cols)].copy())


def benchmark_model_agent_role_by_benchmark(long_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for benchmark in cols:
        df = long_df[long_df["benchmark"] == benchmark].copy()
        y = df["normalized_score"].astype(float)
        n = len(y)
        model_info = fit_r2_full(df, y, ["model"])
        agent_info = fit_r2_full(df, y, ["agent"])
        full_info = fit_r2_full(df, y, ["model", "agent"])

        model_partial = full_info["r2"] - agent_info["r2"]
        agent_partial = full_info["r2"] - model_info["r2"]
        model_adj_partial = full_info["adj_r2"] - agent_info["adj_r2"]
        agent_adj_partial = full_info["adj_r2"] - model_info["adj_r2"]

        rows.append(
            {
                "benchmark": benchmark,
                "model_only_r2": model_info["r2"],
                "model_only_adj_r2": model_info["adj_r2"],
                "model_n_predictors": model_info["n_predictors"],
                "agent_only_r2": agent_info["r2"],
                "agent_only_adj_r2": agent_info["adj_r2"],
                "agent_n_predictors": agent_info["n_predictors"],
                "model_partial_r2_over_agent": model_partial,
                "agent_partial_r2_over_model": agent_partial,
                "model_adj_partial_r2_over_agent": model_adj_partial,
                "agent_adj_partial_r2_over_model": agent_adj_partial,
                "full_model_plus_agent_r2": full_info["r2"],
                "full_model_plus_agent_adj_r2": full_info["adj_r2"],
                "dominant_dimension": "model"
                if model_adj_partial > agent_adj_partial
                else "agent",
            }
        )
    return pd.DataFrame(rows).sort_values("model_adj_partial_r2_over_agent", ascending=False)
