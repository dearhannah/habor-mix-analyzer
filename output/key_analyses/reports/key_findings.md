# Key Findings for Key Analysis Drafting

0. The processed benchmark matrix is task-first and uses `row_mean_shrunk` task filling, not direct benchmark imputation.

Benchmark scores are simple means over the filled task matrix. The imputation reliability check compares candidate task fillers on held-out observed cells; SVD has lower RMSE here but higher MAE, so the selected fill is conservative rather than low-rank.

| method | rank | holdout_cells | rmse | mae |
| --- | --- | --- | --- | --- |
| row_mean_shrunk | 0 | 5192 | 12.445 | 0.724 |
| column_median | 0 | 5192 | 12.460 | 0.749 |
| iterative_svd | 2 | 5192 | 12.852 | 0.823 |
| two_way_shrunk | 0 | 5192 | 12.249 | 1.004 |

1. Use 50 coverage-filtered benchmarks for benchmark-level claims; keep sparse benchmarks in appendix/provisional analysis.

The filtering table is now evidence-based on the task-first pipeline: benchmark scores come from filled-task aggregates, while the missingness columns describe how much original task evidence supported each aggregate before filling.

![Benchmark predictability ranking](../figures/benchmark_level/benchmark_uniqueness_vs_coverage.png)

| benchmark | include_in_key_analysis | observed_count | task_cell_missing_fraction |
| --- | --- | --- | --- |
| aider-polyglot | True | 16 | 0.000 |
| aime | True | 16 | 0.000 |
| algotune | True | 16 | 0.000 |
| arc-agi-2 | True | 16 | 0.000 |
| bfcl | True | 16 | 0.002 |
| bigcodebench | True | 16 | 0.000 |
| bixbench | True | 16 | 0.018 |
| codepde | True | 16 | 0.000 |

2. Model identity is the larger overall factor, but the model-vs-agent balance varies by benchmark; use the per-benchmark role plot for qualified claims.

This is the clean answer to the agent-vs-model question: make the broad statement from the overall fixed-effect decomposition, then qualify it with per-benchmark partial R2 rather than collapsing everything into a single agent+model row label.

![Per-benchmark model vs agent explanatory power](../figures/benchmark_level/benchmark_model_vs_agent_role.png)

| benchmark | model_adj_partial_r2_over_agent | agent_adj_partial_r2_over_model | dominant_dimension |
| --- | --- | --- | --- |
| research-code-bench | -0.016 | 1.672 | agent |
| mmmlu | 0.084 | 1.732 | agent |
| kumo | 1.094 | 0.099 | model |
| swtbench | 1.063 | 0.077 | model |
| crustbench | 0.930 | -0.053 | model |
| terminal-bench | 0.984 | 0.005 | model |
| quixbugs | 0.961 | -0.013 | model |
| pixiu | 1.038 | 0.069 | model |

3. Separate model and agent dimensions. The useful agent evidence is paired lift over `terminus-2` for the same model, not an unqualified agent+model leaderboard.

The Terminus table should be read as a harnessing-effect estimate: the paired comparison holds model fixed where the same model appears under Terminus and another agent.

![Agent lift vs terminus by model](../figures/benchmark_level/terminus_delta_by_model_heatmap.png)

| agent | mean_delta_vs_terminus | win_rate_vs_terminus | compared_models |
| --- | --- | --- | --- |
| claude-code | 0.292 | 0.627 | 3 |
| gemini-cli | -0.040 | 0.650 | 2 |
| codex | -0.211 | 0.707 | 3 |

4. BenchPress-style predictability applies here: redundant benchmarks can be compressed; least-predictable benchmarks should be preserved for behavioral breadth.

The benchmark-predictability result is deliberately separate from clustering: regression asks whether other benchmarks reconstruct a target, while the heatmap shows score-profile similarity. Use both when deciding whether two benchmarks are redundant.

![Clustered benchmark similarity heatmap](../figures/benchmark_level/benchmark_similarity_clustered_heatmap.png)

| benchmark | cv_r2_from_other_included_benchmarks | cv_rmse |
| --- | --- | --- |
| aime | -9.718 | 2.183 |
| gpqa-diamond | -2.261 | 1.253 |
| simpleqa | -2.091 | 2.095 |
| gso | -1.928 | 0.842 |
| financeagent_terminal | -1.918 | 1.103 |
| gaia2 | -1.806 | 1.145 |
| mmau | -1.386 | 2.321 |
| featbench | -1.291 | 0.733 |

5. Task predictability and task representativeness are distinct: hard-to-predict tasks are stress tests, while representative tasks are compact proxies for a benchmark.

The representative-task score now uses leave-one-out aggregate correlation times task variance, so tasks that are merely typical but non-discriminative are less likely to dominate the selected base set.

![Hard-to-predict reliable tasks](../figures/task_level/task_hard_to_predict_ranked.png)

![Best representative task per benchmark](../figures/task_level/task_best_representatives.png)

| benchmark | task_id | task_unpredictability_score | difficulty_tier |
| --- | --- | --- | --- |
| swebench-multilingual | fastlane__fastlane-19207 | 0.743 | hard |
| mmmlu | mmmlu-bn-bd-00015 | 0.676 | hard |
| strongreject | strongreject_hate_harassment_and_discrimination_0000_pap_logical_appeal | 0.616 | easy |
| bfcl | bfcl-live-multiple-83-38-0 | 0.613 | medium |
| swebench-multilingual | jordansissel__fpm-1850 | 0.604 | easy |
| swebench-multilingual | hashicorp__terraform-34814 | 0.604 | medium |
| crustbench | crustbench-fs-c | 0.600 | hard |
| bixbench | bix-29-q2 | 0.594 | hard |

| benchmark | task_id | useful_representativeness_score | difficulty_tier |
| --- | --- | --- | --- |
| mmmlu | mmmlu-ko-kr-00106 | 0.470 | medium |
| mmmlu | mmmlu-bn-bd-00012 | 0.470 | medium |
| mmmlu | mmmlu-en-us-00032 | 0.470 | medium |
| mmmlu | mmmlu-ja-jp-00099 | 0.464 | medium |
| mmmlu | mmmlu-yo-ng-00142 | 0.464 | medium |
| research-code-bench | minp_convert_logits_to_probabilities | 0.459 | medium |
| research-code-bench | len_split_input_and_compute_norm | 0.459 | medium |
| research-code-bench | fractalgen_chunk_mask_to_pred | 0.459 | medium |

6. The current HaborMix final set contains 160 diversified tasks.

The HaborMix scorer is no longer centered on moderate difficulty. It first takes useful representative base tasks, then fills to a compact target size with difficult, frontier-with-variance, unique/unpredictable, and high-composite tasks.

![HaborMix selection diagnostics](../figures/harbormix/harbormix_selection_diagnostics.png)

| benchmark | difficulty_tier | selected_tasks | mean_selection_score |
| --- | --- | --- | --- |
| bfcl | medium | 3 | 0.842 |
| pixiu | medium | 3 | 0.770 |
| reasoning-gym | medium | 3 | 0.762 |
| seal0 | medium | 3 | 0.747 |
| arc-agi-2 | medium | 3 | 0.737 |
| featurebench-modal | medium | 3 | 0.728 |
| strongreject | medium | 3 | 0.717 |
| aider-polyglot | medium | 3 | 0.707 |
| skillsbench | medium | 3 | 0.705 |
| spider2 | medium | 3 | 0.705 |

7. Task-to-benchmark alignment should be used as a sanity check before interpreting benchmark-level scores from task-level tables.

This table is diagnostic rather than a gate. Weak alignment means the reliable bounded subset may not proxy the full task-derived aggregate well; it does not automatically remove the benchmark from the benchmark-level analysis.

![Task aggregate vs benchmark score alignment](../figures/task_level/task_to_benchmark_alignment.png)

| benchmark | n_reliable_bounded_tasks | spearman_agent_model_correlation | alignment_quality |
| --- | --- | --- | --- |
| gso | 101 | 0.997 | strong |
| kumo | 3 | 0.994 | strong |
| featurebench-modal | 185 | 0.994 | strong |
| arc-agi-2 | 100 | 0.994 | strong |
| aider-polyglot | 225 | 0.991 | strong |
| hle | 236 | 0.991 | strong |
| replicationbench | 90 | 0.988 | strong |
| crustbench | 100 | 0.988 | strong |

Primary reference file: `output/key_analyses/reports/analysis_story.md`.