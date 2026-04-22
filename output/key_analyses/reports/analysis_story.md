# Key Analysis Cross-Benchmark Analysis

This report is intended to be read directly. Figures and compact table previews are embedded inline; CSV paths are listed for exact numbers and reproducibility.

## Directory Contract

- Key analysis tables: `output/key_analyses/tables/`
- Key analysis figures: `output/key_analyses/figures/`
- Layered table groups: `benchmark_level/`, `leaderboards/`, `task_level/`, `harbormix/`, and `provenance/`.
- Layered figure groups: `benchmark_level/`, `leaderboards/`, `task_level/`, and `harbormix/`.
- Leaderboard figures are further layered into `leaderboards/per_benchmark/` and `leaderboards/clustered/`; the report embeds only the clustered pages to keep reading compact.
- Intermediate study outputs: `output/intermediate_studies/`
- Intermediate imputed matrices and diagnostics: `data/processed/intermediate/`

The current preprocessing contract is task-first: task scores are filled first, then benchmark scores are aggregated from the filled task matrix. The original benchmark matrix is retained as metadata and as a sanity check, but benchmark scores are not imputed directly.

BenchPress mapping note: Dimitris's BenchPress repo treats benchmark prediction as an explicit analysis problem, compares benchmark-regression and SVD families under held-out validation, and uses a blend because regression can be more accurate while SVD gives broader coverage. Our schema is task-rich rather than model-benchmark-only, so we map that lesson by validating several task-fill families on held-out observed cells, using benchmark/task predictability tables for difficulty ranking, and keeping task aggregate alignment as a diagnostic rather than blindly trusting every aggregate.

Data provenance for the main studies:

| analysis | primary_matrix | processed_output |
| --- | --- | --- |
| coverage filtering | task-observed benchmark aggregate metadata | data/processed/intermediate/benchmark_from_task_aggregate_column_quality.csv |
| agent+model aggregate leaderboard | benchmark scores aggregated from filled task matrix | data/processed/intermediate/benchmark_from_task_aggregate_matrix.csv |
| per-benchmark mini-leaderboards | benchmark scores aggregated from filled task matrix | data/processed/intermediate/benchmark_from_task_aggregate_matrix.csv |
| model vs agent roles | benchmark-relative matrix aggregated from filled tasks | data/processed/intermediate/benchmark_from_task_aggregate_normalized_matrix.csv |
| benchmark predictability and similarity | benchmark-relative matrix aggregated from filled tasks | data/processed/intermediate/benchmark_from_task_aggregate_normalized_matrix.csv |
| terminus harness deltas | benchmark-relative matrix aggregated from filled tasks | data/processed/intermediate/benchmark_from_task_aggregate_normalized_matrix.csv |
| task similarity and representatives | filled task benchmark-relative matrix plus task quality metadata | data/processed/intermediate/task_imputed_normalized_matrix.csv |
| HaborMix candidate selection | processed task item statistics | data/processed/intermediate/task_item_stats.csv |

Preprocessing diagnostics:

Task imputation method: each score column is robustly centered and scaled after `log1p` for nonnegative unbounded columns. The pipeline compares column-median, row-mean shrinkage, two-way shrinkage, and low-rank iterative SVD candidates on held-out observed cells, then uses the lowest-MAE method. For this run the selected task imputer is `column_median` with rank 0. Observed task cells are restored exactly; filled task scores are inverse-transformed and clipped to the observed range of that task. Benchmark scores are then calculated as per-benchmark means across those task scores. Task-level imputation remains less stable than dense benchmark tables because the task matrix is much wider and sparser, so task conclusions are restricted to reliable bounded non-degenerate tasks.

| matrix | preprocessing_method | selected_imputation_method | missing_fraction_before_processing | selected_imputation_rank | task_imputation_method_used_for_benchmark_aggregation | task_imputation_rank_used_for_benchmark_aggregation | holdout_cells | holdout_rmse_scaled_score_space | holdout_mae_scaled_score_space |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | task_imputation_then_benchmark_aggregate |  | 0.108 |  | column_median | 0.000 | 0 |  |  |
| task | column_median | column_median | 0.323 | 0.000 |  |  | 9520 | 24.428 | 0.756 |

Held-out task imputation comparison:

| method | rank | holdout_cells | rmse | mae |
| --- | --- | --- | --- | --- |
| column_median | 0 | 9520 | 24.428 | 0.756 |
| iterative_svd | 2 | 9520 | 24.422 | 0.757 |
| row_mean_shrunk | 0 | 9520 | 117.389 | 2.001 |
| two_way_shrunk | 0 | 9520 | 117.542 | 2.303 |

Interpretation: the task matrix is sparse and very wide, so this is a validation-backed fill, not ground truth. The aggregate benchmark table is more stable than individual filled cells because it averages over many task columns, but sparse benchmarks should still be read with their task-cell missingness fields.

Reliability conclusion: SVD is not automatically the right fill here. In this run, low-rank SVD improves RMSE because it reduces a few large scaled errors, but it loses on MAE, which is the better primary criterion for this mixed-scale sparse matrix because many robust-normalized task columns have outlier-sensitive tails. The selected column-median fill is therefore intentionally conservative: it preserves each task's observed center and avoids hallucinating row-level structure where the held-out cells do not support it.

## Research Question Coverage Checklist

| Question | Status | Main artifacts |
| --- | --- | --- |
| Agent vs model role overall and per benchmark | covered | `tables/benchmark_level/benchmark_variance_decomposition_filtered.csv`, `tables/benchmark_level/benchmark_model_agent_role_by_benchmark.csv`, `figures/benchmark_level/benchmark_model_vs_agent_role.png` |
| BenchPress-style benchmark predictability and hard-to-predict benchmarks/tasks | covered | `tables/benchmark_level/benchmark_uniqueness_filtered.csv`, `tables/task_level/task_predictability_ranked.csv`, benchmark/task predictability figures |
| Benchmark/task similarity and clustering | covered | `tables/benchmark_level/benchmark_similarity_clusters.csv`, `tables/task_level/task_cross_benchmark_similarity.csv`, clustered heatmaps |
| Representative tasks per benchmark | covered | `tables/task_level/task_representative_tasks.csv`, `figures/task_level/task_best_representatives.png` |
| Mini-leaderboards grouped by similar benchmarks | covered | `tables/leaderboards/benchmark_mini_leaderboards.csv`, `figures/leaderboards/clustered/mini_leaderboards_cluster_*.png` |
| Agent harness improvements over Terminus | covered | `tables/benchmark_level/benchmark_agent_lift_vs_terminus.csv`, `tables/benchmark_level/terminus_delta_by_model.csv`, Terminus heatmaps |
| Quantitative HaborMix task selection | covered | `tables/harbormix/harbormix_selected_tasks.csv`, `tables/harbormix/harbormix_selection_by_benchmark.csv`, `figures/harbormix/harbormix_selection_diagnostics.png` |

## Study 1: Coverage Filtering

**Method:** Benchmark-level claims use benchmark aggregates derived from the filled task matrix, but coverage filtering still uses evidence metadata: at least 15 agent+model rows need some observed task evidence for that benchmark, and the missingness fields describe coverage before task filling. This filter keeps the key analysis story from leaning too heavily on filled task values.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/benchmark_level/benchmark_filtering.csv`

**Result overview and analysis:**
- Included 51 of 56 benchmarks.
- Excluded sparse benchmarks: swebench-multilingual, devopsgym, quixbugs, multi-swe-bench, qwen-coder.
- Benchmark scores are task aggregates, not direct benchmark-imputation outputs; pre-aggregation benchmark missing fraction was 0.108.

| benchmark | include_in_key_analysis | observed_count | missing_fraction | task_cell_missing_fraction |
| --- | --- | --- | --- | --- |
| aider-polyglot | True | 28 | 0.000 | 0.039 |
| aime | True | 28 | 0.000 | 0.064 |
| arc-agi-2 | True | 28 | 0.000 | 0.045 |
| bigcodebench | True | 28 | 0.000 | 0.005 |
| bixbench | True | 28 | 0.000 | 0.018 |
| codepde | True | 28 | 0.000 | 0.000 |
| compilebench | True | 28 | 0.000 | 0.155 |
| deepsynth | True | 28 | 0.000 | 0.072 |
| gpqa-diamond | True | 28 | 0.000 | 0.060 |
| humanevalfix | True | 28 | 0.000 | 0.016 |
| ineqmath | True | 28 | 0.000 | 0.065 |
| kumo | True | 28 | 0.000 | 0.385 |

**Insight and findings:** Sparse columns should stay in appendix/provisional analysis until more experiments land. The main key analysis story should use the coverage-filtered benchmark set.

## Study 2: Model vs Agent Roles

**Method:** I fit fixed-effect regressions in two views. The overall view decomposes benchmark-relative score into model, agent, benchmark, and interaction terms. The per-benchmark view fits each benchmark separately and compares partial R2 from model after controlling for agent against partial R2 from agent after controlling for model.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/benchmark_level/benchmark_variance_decomposition_filtered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_model_agent_role_by_benchmark.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_model_adjusted_effects.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_agent_adjusted_effects.csv`

![Benchmark-level variance attribution](../figures/benchmark_level/benchmark_variance_attribution.png)

![Per-benchmark model vs agent explanatory power](../figures/benchmark_level/benchmark_model_vs_agent_role.png)

![Model effects adjusted for agent and benchmark](../figures/benchmark_level/benchmark_model_adjusted_effects.png)

![Agent effects adjusted for model and benchmark](../figures/benchmark_level/benchmark_agent_adjusted_effects.png)

**Result overview and analysis:**
| component | partial_r2_over_other_main_effects | r2 | type |
| --- | --- | --- | --- |
| model:benchmark | 0.456 | 0.501 | interaction_increment |
| agent:benchmark | 0.088 | 0.133 | interaction_increment |
| all_main_effects | 0.046 | 0.046 | combined |
| benchmark | 0.035 | 0.035 | main_effect |
| model | 0.009 | 0.009 | main_effect |
| model:agent | 0.008 | 0.054 | interaction_increment |
| agent | 0.001 | 0.002 | main_effect |

Benchmarks with the largest model-vs-agent role imbalance:
| benchmark | model_partial_r2_over_agent | agent_partial_r2_over_model | dominant_dimension |
| --- | --- | --- | --- |
| kumo | 0.885 | 0.039 | model |
| crustbench | 0.847 | 0.008 | model |
| terminal-bench | 0.844 | 0.028 | model |
| aider-polyglot | 0.814 | 0.025 | model |
| strongreject | 0.870 | 0.082 | model |
| qcircuitbench | 0.816 | 0.030 | model |
| algotune | 0.774 | 0.016 | model |
| sldbench | 0.757 | 0.003 | model |
| gpqa-diamond | 0.820 | 0.070 | model |
| swebench-verified | 0.754 | 0.019 | model |

**Insight and findings:** Model identity explains much more overall variation than agent identity, but the role varies by benchmark. Agent effects are more useful as benchmark-specific harnessing effects than as a universal main effect.

## Study 3: Agent+Model Leaderboards

**Method:** I keep `agent+model` rankings as descriptive mini-leaderboards. Per-benchmark mini-leaderboards use benchmark scores aggregated from filled tasks on each benchmark's original metric scale. Each mini-leaderboard shows all available agent+model rows, grouped by model with colored bars for agents, so the same plot makes model differences and agent harness differences visible. The aggregate top-agent plot uses mean within-benchmark score percentile, because averaging scores across benchmarks with different scales would be misleading.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/leaderboards/benchmark_agent_model_scores.csv`
- `output/key_analyses/tables/leaderboards/benchmark_scores_long.csv`
- `output/key_analyses/tables/leaderboards/benchmark_mini_leaderboards.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_similarity_clusters.csv`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_*.png`
- `output/key_analyses/figures/leaderboards/per_benchmark/mini_leaderboard_*.png`

![Top agent+model pairs on included benchmarks](../figures/leaderboards/benchmark_agent_model_top_scores.png)

![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_1_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_1_page_1.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_2_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_2_page_1.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_1.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_2.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_2.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_3.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_3.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_4.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_4.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_5.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_5.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_6.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_6.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_7.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_7.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_8.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_8.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_9.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_9.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_10.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_10.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_3_page_11.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_11.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_4_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_4_page_1.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_5_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_5_page_1.png)
![Mini-leaderboards leaderboards/clustered/mini_leaderboards_cluster_6_page_1.png](../figures/leaderboards/clustered/mini_leaderboards_cluster_6_page_1.png)

**Result overview and analysis:**
| rank | agent_model | mean_score_percentile_across_benchmarks | original_benchmark_table_coverage |
| --- | --- | --- | --- |
| 1 | codex + gpt-5.4 | 0.884 | 0.804 |
| 2 | gemini-cli + gemini-3.1-pro-preview | 0.850 | 0.765 |
| 3 | terminus-2 + gemini-3.1-pro-preview | 0.800 | 0.765 |
| 4 | claude-code + claude-opus-4-6 | 0.736 | 0.706 |
| 5 | terminus-2 + claude-opus-4-6 | 0.734 | 0.627 |
| 6 | claude-code + claude-sonnet-4-6 | 0.679 | 0.686 |
| 7 | gemini-cli + gemini-3-flash-preview | 0.651 | 0.725 |
| 8 | terminus-2 + claude-sonnet-4-6 | 0.648 | 0.667 |
| 9 | terminus-2 + gemini-3-flash-preview | 0.640 | 0.784 |
| 10 | terminus-2 + kimi-k2.5 | 0.578 | 0.490 |

**Insight and findings:** Benchmark scores should be read benchmark by benchmark. The percentile aggregate is a compact descriptive ranking only; it is not a causal agent claim because model and agent are entangled in the row identity.

## Study 4: Benchmark Predictability and Similarity

**Method:** Following the BenchPress idea, each included benchmark is predicted from the other included benchmarks using ridge regression with cross-validation over agent+model rows. Low or negative R2 means the benchmark is hard to reconstruct from the rest and likely contributes distinct information. Benchmark similarity uses same-dimensional vectors: every benchmark is represented by its score profile across the same agent+model rows, and clustering is run on `1 - |Spearman correlation|` so benchmarks with similar ranking behavior sit together even if one is directionally inverted.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/benchmark_level/benchmark_uniqueness_filtered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_redundancy_pairs_filtered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_correlation_clustered.csv`

![Benchmark predictability ranking](../figures/benchmark_level/benchmark_uniqueness_vs_coverage.png)

![Clustered benchmark similarity heatmap](../figures/benchmark_level/benchmark_similarity_clustered_heatmap.png)

**Result overview and analysis:**
Hardest-to-predict benchmarks:
| benchmark | cv_r2_from_other_included_benchmarks | cv_rmse |
| --- | --- | --- |
| gaia2 | -7328869232983.916 | 1367629.244 |
| mmmlu | -6294264180510.100 | 1490833.498 |
| seal0 | -6027652281331.326 | 1660166.450 |
| omnimath | -5132215651331.958 | 1431221.147 |
| spreadsheetbench | -4709106010170.977 | 2776662.397 |
| medagentbench | -3080444906170.212 | 1456575.203 |
| usaco | -3038326129567.491 | 1584033.348 |
| skillsbench | -1417353578363.764 | 901302.996 |
| arc-agi-2 | -1273356797026.026 | 1077311.689 |
| sldbench | -1243554648327.123 | 3030588.874 |

Most similar benchmark pairs:
| left | right | spearman |
| --- | --- | --- |
| spreadsheetbench | terminal-bench | 0.913 |
| deepsynth | replicationbench | 0.903 |
| mmmlu | research-code-bench | 0.900 |
| aime | omnimath | 0.891 |
| deepsynth | terminal-bench | 0.887 |
| spreadsheetbench | gaia2 | 0.869 |
| deepsynth | spider2 | 0.867 |
| aider-polyglot | algotune | 0.859 |
| aider-polyglot | deepsynth | 0.855 |
| replicationbench | skillsbench | 0.855 |

**Insight and findings:** Predictable benchmarks are candidates for compression; hard-to-predict benchmarks should be preserved when the goal is behavioral breadth. Similarity clusters are also the basis for the grouped mini-leaderboards.

Paper-facing read: the least reconstructable benchmark set starts with gaia2, mmmlu, seal0, omnimath, spreadsheetbench. These are the strongest candidates to preserve when reducing the suite because other benchmark scores do not explain them well. In contrast, the most predictable benchmarks start with swegym, mlgym, swtbench, algotune, kumo; these are not useless, but they are where compression or cluster-level reporting is easiest to justify.

## Study 5: Task Similarity, Predictability, and Representatives

**Method:** Task-level analysis uses reliable, bounded, non-degenerate tasks only. Task similarity also uses same-dimensional vectors: every task is represented by its filled, standardized score profile across the same agent+model rows. A task is hard to predict when its maximum absolute profile correlation to peer tasks in the same benchmark is low. Representativeness is no longer pure correlation with the benchmark aggregate: I compute a leave-one-out aggregate correlation and multiply it by observed cross-agent/model variance, so redundant but low-discrimination tasks no longer dominate. Within- and cross-benchmark task similarity use median absolute task-profile correlations, with at most the most discriminative 40 tasks sampled per benchmark for cross-benchmark pair summaries. Difficulty tiers use mean task score thresholds: frontier <5%, hard 5-30%, medium 30-70%, easy 70-95%, saturated >95%.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/task_level/task_predictability_ranked.csv`
- `output/key_analyses/tables/task_level/task_representative_tasks.csv`
- `output/key_analyses/tables/task_level/task_within_benchmark_similarity.csv`
- `output/key_analyses/tables/task_level/task_cross_benchmark_similarity.csv`

![Hard-to-predict reliable tasks](../figures/task_level/task_hard_to_predict_ranked.png)

![Best representative task per benchmark](../figures/task_level/task_best_representatives.png)

![Task similarity across benchmark pairs](../figures/task_level/task_similarity_benchmark_pair_heatmap.png)

**Result overview and analysis:**
Hardest-to-predict reliable tasks:
| benchmark | task_id | task_unpredictability_score | difficulty_tier | task_score |
| --- | --- | --- | --- | --- |
| swebenchpro | instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5 | 0.846 | frontier | 0.014 |
| medagentbench | task7_12 | 0.696 | frontier | 0.036 |
| swebench-verified | sympy__sympy-13852 | 0.691 | frontier | 0.014 |
| codepde | codepde-advection | 0.681 | easy | 0.921 |
| swebench-multilingual | laravel__framework-46234 | 0.674 | hard | 0.062 |
| bixbench | bix-29-q2 | 0.666 | frontier | 0.050 |
| replicationbench | astm3__spectral_similarity_search | 0.664 | frontier | 0.021 |
| hle | hle/hle__67190e8172e53012645b0124 | 0.659 | frontier | 0.025 |
| bixbench | bix-10-q5 | 0.658 | hard | 0.079 |
| gaia2 | gaia2-cli/0643_750ea4udiytzv2211crwkewkhydnq5lx | 0.656 | frontier | 0.012 |
| mmau | adcc613e-3c79-4478-8f08-d408984265a6 | 0.650 | medium | 0.336 |
| bfcl | bfcl-live-multiple-83-38-0 | 0.647 | easy | 0.764 |

Most representative tasks:
| benchmark | task_id | useful_representativeness_score | representativeness_score | difficulty_tier | task_score |
| --- | --- | --- | --- | --- | --- |
| research-code-bench | eomt_extract_query_key_value | 0.445 | 0.912 | medium | 0.414 |
| research-code-bench | len_split_input_and_compute_norm | 0.443 | 0.901 | medium | 0.471 |
| research-code-bench | llm-sci-use_confidence_interval_calculation | 0.443 | 0.901 | medium | 0.471 |
| research-code-bench | grid-cell-conformal-isometry__dx_to_theta_id_dr | 0.442 | 0.921 | medium | 0.436 |
| research-code-bench | eomt_generate_class_logits | 0.441 | 0.896 | medium | 0.471 |
| research-code-bench | llm-sci-use_mixture_log_likelihood_calculation | 0.441 | 0.918 | medium | 0.436 |
| research-code-bench | tabdiff_make_sure_learnable_parameter_ks_for_categorical_features_are_positive | 0.440 | 0.908 | medium | 0.464 |
| research-code-bench | fractalgen_unpatchify | 0.439 | 0.907 | medium | 0.464 |
| research-code-bench | eomt_store_parameters | 0.439 | 0.914 | medium | 0.436 |
| research-code-bench | minp_scale_min_p_threshold | 0.438 | 0.891 | medium | 0.471 |
| research-code-bench | eomt_normalization | 0.438 | 0.891 | medium | 0.471 |
| research-code-bench | eomt_scale_block_forward | 0.438 | 0.921 | medium | 0.429 |

Benchmarks with strongest within-benchmark task similarity:
| benchmark | n_reliable_tasks | median_abs_task_similarity_within_benchmark |
| --- | --- | --- |
| humanevalfix | 145 | 0.870 |
| ineqmath | 100 | 0.793 |
| mmmlu | 135 | 0.706 |
| research-code-bench | 191 | 0.682 |
| arc-agi-2 | 92 | 0.680 |
| lawbench | 174 | 0.636 |
| mlgym | 2 | 0.631 |
| usaco | 99 | 0.616 |
| kumo | 181 | 0.599 |
| sldbench | 8 | 0.533 |

**Insight and findings:** Task predictability and useful representativeness are different objectives. Representative tasks are the base set for predicting benchmark aggregates; hard-to-predict and difficult tasks are additional stress tests for broad coverage.

Paper-facing read: the hardest-to-predict task examples begin with swebenchpro/instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5, medagentbench/task7_12, swebench-verified/sympy__sympy-13852. The most useful representative task examples begin with research-code-bench/eomt_extract_query_key_value, research-code-bench/len_split_input_and_compute_norm, research-code-bench/llm-sci-use_confidence_interval_calculation. That split is the main reason HaborMix should not be selected from one scalar alone: a task can be representative without being unique, and a unique task can be too idiosyncratic to stand in for its benchmark.

## Study 6: Terminus Harnessing Effects

**Method:** Terminus is treated as the fair baseline across models. For every model with both `terminus-2` and another agent row, I compute paired benchmark-relative score deltas while holding the model fixed.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/benchmark_level/benchmark_agent_lift_vs_terminus.csv`
- `output/key_analyses/tables/benchmark_level/terminus_delta_by_model.csv`
- `output/intermediate_studies/benchmark_level/benchmark_agent_lift_by_benchmark.csv`

![Agent lift vs terminus by benchmark](../figures/benchmark_level/benchmark_agent_lift_heatmap.png)

![Agent lift vs terminus by model](../figures/benchmark_level/terminus_delta_by_model_heatmap.png)

**Result overview and analysis:**
| agent | mean_delta_vs_terminus | win_rate_vs_terminus | compared_models |
| --- | --- | --- | --- |
| qwen-coder | 1.793 | 0.569 | 1 |
| codex | 0.090 | 0.660 | 3 |
| gemini-cli | 0.025 | 0.618 | 2 |
| claude-code | -1953093.878 | 0.490 | 7 |

| model | agent | mean_delta_vs_terminus | win_rate_vs_terminus |
| --- | --- | --- | --- |
| qwen3-max | qwen-coder | 1.793 | 0.569 |
| gpt-5.4 | codex | 0.995 | 0.902 |
| gpt-5-mini | codex | 0.453 | 0.765 |
| gemini-3.1-pro-preview | gemini-cli | 0.235 | 0.667 |
| claude-sonnet-4-6 | claude-code | 0.023 | 0.569 |
| claude-opus-4-6 | claude-code | -0.043 | 0.353 |
| MiniMax-M2.5 | claude-code | -0.080 | 0.588 |
| glm-5 | claude-code | -0.115 | 0.431 |
| gemini-3-flash-preview | gemini-cli | -0.185 | 0.569 |
| mimo-v2-pro | claude-code | -0.290 | 0.490 |
| kimi-k2.5 | claude-code | -0.385 | 0.333 |
| gpt-5-nano | codex | -1.177 | 0.314 |

**Insight and findings:** Paired deltas are the best current evidence for whether an agent harness improves over Terminus. The deltas vary by model and benchmark, so claims should avoid saying one harness universally dominates.

## Study 7: HaborMix Selection

**Method:** Candidate tasks must be reliable and bounded. The final HaborMix selection targets a compact 100-200 task set, currently 160 tasks. It first includes a small base set of useful representative tasks per benchmark, then fills the remaining slots with a diversity-aware ranking over difficult, frontier-with-variance, unique/unpredictable, and high-composite tasks. The composite score combines useful representativeness, difficulty, unique/unpredictable signal, and cross-agent/model discrimination; it no longer excludes frontier or saturated items by construction. The broader scored pool is retained under intermediate studies for auditability.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/harbormix/harbormix_selected_tasks.csv`
- `output/key_analyses/tables/harbormix/harbormix_selection_by_benchmark.csv`
- `output/intermediate_studies/task_level/harbormix_scored_task_pool.csv`
- `output/intermediate_studies/task_level/task_frontier_or_saturated_watchlist.csv`

![HaborMix selection diagnostics](../figures/harbormix/harbormix_selection_diagnostics.png)

![Reliable bounded task difficulty composition](../figures/task_level/task_reliable_difficulty_composition.png)

![Reliable bounded task difficulty composition by percentage](../figures/task_level/task_reliable_difficulty_composition_percent.png)

**Result overview and analysis:**
| benchmark | difficulty_tier | selected_tasks | mean_selection_score | mean_representative_signal | mean_unique_unpredictable_signal | mean_difficulty_signal |
| --- | --- | --- | --- | --- | --- | --- |
| featurebench-modal | hard | 4 | 0.769 | 0.916 | 0.478 | 0.774 |
| reasoning-gym | medium | 4 | 0.741 | 0.963 | 0.609 | 0.434 |
| financeagent | frontier | 3 | 0.793 | 0.667 | 0.667 | 0.983 |
| hle | medium | 3 | 0.772 | 0.979 | 0.436 | 0.683 |
| gso | hard | 3 | 0.763 | 0.929 | 0.447 | 0.745 |
| gpqa-diamond | medium | 3 | 0.742 | 0.961 | 0.671 | 0.388 |
| replicationbench | hard | 3 | 0.740 | 0.946 | 0.356 | 0.789 |
| arc-agi-2 | hard | 3 | 0.739 | 0.870 | 0.460 | 0.760 |
| scicode | medium | 3 | 0.738 | 0.920 | 0.667 | 0.412 |
| crustbench | medium | 3 | 0.737 | 0.924 | 0.547 | 0.513 |
| gaia | medium | 3 | 0.718 | 0.966 | 0.402 | 0.576 |
| labbench | hard | 3 | 0.716 | 0.968 | 0.202 | 0.733 |
| aider-polyglot | medium | 3 | 0.706 | 0.931 | 0.431 | 0.529 |
| mmmlu | medium | 3 | 0.703 | 0.923 | 0.519 | 0.444 |

- Selected 160 final HaborMix tasks from the broader scored candidate pool.

**Insight and findings:** HaborMix selection is quantitative and auditable: representative tasks anchor the minimal benchmark-prediction base, while difficult and unique/unpredictable tasks add breadth.

Paper-facing read: the final 160-task set is intentionally not just a hard-task list. Its difficulty composition is medium: 79, easy: 37, hard: 36, frontier: 7, saturated: 1. The base representative set keeps each included benchmark anchored to its aggregate behavior, while the diversity-aware fill adds difficult, frontier, and uniquely informative items.

## Study 8: Task Aggregate vs Benchmark-Level Score Alignment

**Method:** For each benchmark, I average benchmark-relative task scores and correlate that aggregate with the benchmark-level benchmark-relative score across agent+model rows. This is intentionally diagnostic rather than a hard gate: the new benchmark score is already the task aggregate, so this table mainly identifies benchmarks where reliable bounded tasks alone do or do not track the full task-derived benchmark aggregate.

**Code files:**
- `src/habor_mix_analyzer/core/`
- `src/habor_mix_analyzer/preprocessing/svd_imputation.py`
- `src/habor_mix_analyzer/studies/coverage_filtering.py`
- `src/habor_mix_analyzer/studies/intermediate_tables.py`
- `src/habor_mix_analyzer/studies/model_agent_roles.py`
- `src/habor_mix_analyzer/studies/benchmark_predictability.py`
- `src/habor_mix_analyzer/studies/benchmark_similarity.py`
- `src/habor_mix_analyzer/studies/leaderboards.py`
- `src/habor_mix_analyzer/studies/terminus_comparison.py`
- `src/habor_mix_analyzer/studies/task_alignment.py`
- `src/habor_mix_analyzer/studies/task_selection.py`
- `src/habor_mix_analyzer/studies/task_similarity.py`
- `src/habor_mix_analyzer/studies/provenance.py`
- `src/habor_mix_analyzer/visualization/`
- `src/habor_mix_analyzer/reporting/key_analysis_report.py`
- `src/habor_mix_analyzer/cli.py`

**Result paths:**
- `output/key_analyses/tables/task_level/task_to_benchmark_alignment.csv`

![Task aggregate vs benchmark score alignment](../figures/task_level/task_to_benchmark_alignment.png)

**Result overview and analysis:**
| benchmark | n_reliable_bounded_tasks | spearman_agent_model_correlation | alignment_quality |
| --- | --- | --- | --- |
| financeagent | 5 | 1.000 | strong |
| swtbench | 50 | 0.999 | strong |
| terminal-bench | 89 | 0.992 | strong |
| usaco | 100 | 0.992 | strong |
| research-code-bench | 212 | 0.992 | strong |
| gaia | 165 | 0.991 | strong |
| scicode | 80 | 0.991 | strong |
| livecodebench | 100 | 0.990 | strong |
| featurebench-modal | 185 | 0.990 | strong |
| compilebench | 15 | 0.990 | strong |
| labbench | 181 | 0.989 | strong |
| aider-polyglot | 225 | 0.984 | strong |

**Insight and findings:** Strong alignment means the reliable bounded subset is a good proxy for the task-derived benchmark score. Weak alignment is not used to remove benchmarks automatically; it flags cases for manual benchmark/task inspection.

## Cross-Study Story

The emerging story is that benchmark diversity matters more than a single aggregate leaderboard. Coverage filtering removes sparse columns from the main benchmark-level claims, and the imputation diagnostic makes the same point from the preprocessing side: the data are dense only after filling, and the selected fill is conservative because held-out validation did not justify stronger SVD structure by MAE.

Within the retained benchmarks, model identity is usually more explanatory than agent identity, but the per-benchmark partial-R2 view prevents overstatement: some benchmark slices are much more sensitive to harness choice than the overall decomposition suggests. That is why the report keeps descriptive `agent+model` leaderboards for browsing, but uses paired Terminus deltas when making harness claims.

The BenchPress-style predictability layer identifies a preservation/compression axis. Benchmarks such as gaia2, mmmlu, seal0, omnimath, spreadsheetbench are hard to reconstruct and therefore carry distinctive signal. Benchmarks such as swegym, mlgym, swtbench, algotune, kumo are easier to reconstruct and can be grouped more aggressively. The clustered heatmaps and clustered mini-leaderboards give the visual version of the same argument.

The task layer answers a different selection problem. Representative tasks are useful as small proxies for benchmark aggregates; unpredictable and difficult tasks are useful as stress tests. HaborMix combines those roles by taking representative base tasks first, then filling with difficult, unique, and discriminative tasks until the final compact set reaches the target size range. That is the clearest story for why HaborMix is not merely a random subset, not merely a hard subset, and not merely a redundant set of benchmark prototypes.

## Artifact Index

All key analysis tables:
- `output/key_analyses/tables/benchmark_level/benchmark_agent_adjusted_effects.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_agent_lift_vs_terminus.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_correlation_clustered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_filtering.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_model_adjusted_effects.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_model_agent_role_by_benchmark.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_redundancy_pairs_filtered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_similarity_clusters.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_uniqueness_filtered.csv`
- `output/key_analyses/tables/benchmark_level/benchmark_variance_decomposition_filtered.csv`
- `output/key_analyses/tables/benchmark_level/terminus_delta_by_model.csv`
- `output/key_analyses/tables/harbormix/harbormix_selected_tasks.csv`
- `output/key_analyses/tables/harbormix/harbormix_selection_by_benchmark.csv`
- `output/key_analyses/tables/leaderboards/benchmark_agent_model_scores.csv`
- `output/key_analyses/tables/leaderboards/benchmark_mini_leaderboards.csv`
- `output/key_analyses/tables/leaderboards/benchmark_scores_long.csv`
- `output/key_analyses/tables/provenance/analysis_data_provenance.csv`
- `output/key_analyses/tables/provenance/imputation_diagnostics_summary.csv`
- `output/key_analyses/tables/task_level/task_benchmark_reliable_summary.csv`
- `output/key_analyses/tables/task_level/task_cross_benchmark_similarity.csv`
- `output/key_analyses/tables/task_level/task_predictability_ranked.csv`
- `output/key_analyses/tables/task_level/task_representative_tasks.csv`
- `output/key_analyses/tables/task_level/task_to_benchmark_alignment.csv`
- `output/key_analyses/tables/task_level/task_within_benchmark_similarity.csv`

All key analysis figures:
- `output/key_analyses/figures/benchmark_level/benchmark_agent_adjusted_effects.png`
- `output/key_analyses/figures/benchmark_level/benchmark_agent_lift_heatmap.png`
- `output/key_analyses/figures/benchmark_level/benchmark_model_adjusted_effects.png`
- `output/key_analyses/figures/benchmark_level/benchmark_model_vs_agent_role.png`
- `output/key_analyses/figures/benchmark_level/benchmark_similarity_clustered_heatmap.png`
- `output/key_analyses/figures/benchmark_level/benchmark_uniqueness_vs_coverage.png`
- `output/key_analyses/figures/benchmark_level/benchmark_variance_attribution.png`
- `output/key_analyses/figures/benchmark_level/terminus_delta_by_model_heatmap.png`
- `output/key_analyses/figures/harbormix/harbormix_selection_diagnostics.png`
- `output/key_analyses/figures/leaderboards/benchmark_agent_model_top_scores.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_1_page_1.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_2_page_1.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_1.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_10.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_11.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_2.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_3.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_4.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_5.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_6.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_7.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_8.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_3_page_9.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_4_page_1.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_5_page_1.png`
- `output/key_analyses/figures/leaderboards/clustered/mini_leaderboards_cluster_6_page_1.png`
- `output/key_analyses/figures/task_level/task_best_representatives.png`
- `output/key_analyses/figures/task_level/task_hard_to_predict_ranked.png`
- `output/key_analyses/figures/task_level/task_reliable_difficulty_composition.png`
- `output/key_analyses/figures/task_level/task_reliable_difficulty_composition_percent.png`
- `output/key_analyses/figures/task_level/task_similarity_benchmark_pair_heatmap.png`
- `output/key_analyses/figures/task_level/task_to_benchmark_alignment.png`
- `output/key_analyses/figures/leaderboards/per_benchmark/` (51 per-benchmark mini-leaderboard files, listed by directory rather than expanded here)

## Not Completed Yet

- Trial reliability, pass@k, efficiency curves, token/tool cost analysis, and trajectory failure taxonomy still require per-trial run records.
- Full IRT/DIF still requires repeated binary/calibrated task outcomes or enough dense task observations to fit stable item-response models.
- Provider scaling analysis still requires external model metadata such as provider family, parameter scale, release date, and inference budget.
