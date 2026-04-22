# Quantitative study plan

This folder holds the **design** for analyses that compare:

1. **Curated benchmark metadata** — `benchmark_info_jobs/*.json` (filled from official sources per `benchmark_info_template.md`).
2. **Published / leaderboard numbers** — embedded in each JSON as `results_over_time`, or scraped snapshots.
3. **Harbor experiment outputs** — scores from your evaluation runs (must be aligned to the same `(benchmark, model, agent/system)` units).

The core **analysis pipeline** in `habor-mix-analyzer` consumes **wide CSV matrices** (`data/raw/benchmark_level_matrix.csv`, etc.: rows = `model` × `agent`, columns = benchmark slugs). Harbor-side storage schema is **not** defined in this repo; any quantitative study here assumes you add an **export / ETL step** that produces those matrices (or a long table) from Harbor.

---

## Objectives

| Goal | Question |
|------|----------|
| **Coverage** | Which adapters have complete metadata vs missing links/metrics? |
| **Calibration** | Do Harbor scores match official scales after normalization? |
| **Ranking agreement** | Do Harbor rankings match public leaderboards on overlapping rows? |
| **Mix diagnostics** | Does the Harbor run mix match intended categories / difficulty? |

---

## Proposed metrics (quantitative)

### A. Metadata completeness (no Harbor join required)

- Fraction of JSON files with non-null `links.website`, `evaluation.primary_metric`, at least one `results_over_time` entry.
- Per-field missingness heatmap (benchmark × field).

### B. Harbor vs documented scores (needs aligned keys)

Preconditions: stable mapping **Harbor benchmark slug ↔ `benchmark_info_jobs/<stem>.json`** and overlapping **model/agent** identifiers.

- **Point error**: for each comparable cell, \(|\hat{s}_{\text{Harbor}} - s_{\text{doc}}|\) or relative error if scales match.
- **Rank correlation**: Spearman / Kendall between Harbor ordering and documented leaderboard ordering on the **same** candidate set.
- **Scale calibration**: linear or isotonic fit of Harbor vs doc scores; report slope/intercept if linear.

### C. Temporal / versioning

- Days between `results_over_time[].date` and Harbor run date (staleness).
- Sensitivity: restrict to results after a given paper/code release date.

### D. Structure of the Harbor matrix (existing pipeline affinity)

Once scores are in `benchmark_level_matrix.csv` form, reuse or extend existing studies:

- Benchmark–benchmark correlation, variance decomposition, predictability — already implemented downstream of raw matrices.
- Stratify those summaries by **`category`** from JSON (requires joining metadata onto column names).

### E. Missingness in Harbor

- Missing `(agent, model, benchmark)` rate vs benchmark; relate to metadata `notes` or known limitations.

---

## Implementation sketch

| Step | Action |
|------|--------|
| 1 | **Slug table** — one CSV: `stem` ↔ Harbor id ↔ optional display name. |
| 2 | **ETL** — Harbor export → long format (`benchmark`, `model`, `agent`, `score`, `run_id`) → optional pivot to wide CSV for `habor-analyze`. |
| 3 | **Join** — load JSON metadata by `stem`; attach `category`, `primary_metric`. |
| 4 | **Reports** — notebooks or small scripts under `quantitative_study/scripts/` (add when ready). |

---

## Relation to main pipeline

- **Imputation / studies** in `src/habor_mix_analyzer/` operate on **numeric matrices**, not on raw Harbor JSON.
- Quantitative comparisons in this document are **orthogonal**: they validate **external consistency** before or alongside matrix-based science.

---

## Folder layout (this directory)

```
quantitative_study/
├── QUANTATIVE_STUDY.md   ← this file (plan + metrics)
├── data/                   ← optional: seed slug mappings, ignored if empty
└── scripts/               ← optional: future notebooks / one-off joins
```

Add `scripts/` and `data/` only when you start implementing; Git can track empty dirs with `.gitkeep` if needed.
