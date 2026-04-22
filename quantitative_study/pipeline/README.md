# Quantitative Analysis Pipeline

Compare Harbor benchmark scores against original (documented) benchmark scores,
detect data anomalies, and measure agent capability differences.

## Quick start

```bash
cd habor-mix-analyzer

# Full pipeline (tables + figures)
uv run python -m quantitative_study.pipeline.run_pipeline

# Specify Harbor CSV explicitly
uv run python -m quantitative_study.pipeline.run_pipeline \
  --harbor-csv benchmark_level_matrix.csv

# Tables only (skip figure generation)
uv run python -m quantitative_study.pipeline.run_pipeline --no-figures
```

## Outputs

All outputs go to `output/quantitative/`.

### CSV tables

| File | Description |
|------|-------------|
| `harbor_vs_doc_long.csv` | Full long table: every Harbor cell × doc match attempt |
| `harbor_vs_doc_summary.csv` | Rows with delta (Harbor − Original), sorted by \|delta\| |
| `harbor_anomalies.csv` | Flagged cells: negative, >1, exact zero, extreme magnitude |
| `harbor_anomaly_summary.csv` | Anomalies aggregated per benchmark |
| `agent_effect_by_match_status.csv` | Mean/median delta by match type |
| `agent_effect_by_agent.csv` | Mean/median delta by Harbor agent |
| `benchmark_direction.csv` | Per-benchmark: Harbor higher / lower / similar |
| `agent_lift_table.csv` | Agent lift over terminus-2 per (benchmark, model) |
| `rank_correlation_by_benchmark.csv` | Spearman ρ between Harbor and doc rankings |

### Figures (for paper)

| File | Description |
|------|-------------|
| `harbor_vs_doc_scatter.png` | Scatter: Harbor score vs Original score (colored by match type) |
| `delta_by_benchmark.png` | Horizontal bar: mean delta per benchmark |
| `agent_lift_heatmap.png` | Heatmap: agent lift over terminus-2 (model × benchmark) |
| `delta_distribution_boxplot.png` | Box plot: delta distribution by match status |
| `anomaly_by_benchmark.png` | Bar: anomalous cells per benchmark |
| `agent_mean_delta.png` | Bar: systematic bias by Harbor agent |
| `rank_correlation_by_benchmark.png` | Bar: Spearman ρ per benchmark |

## Architecture

```
pipeline/
├── config.py       # Paths, aliases, transforms, figure style
├── loading.py      # Load Harbor CSV + benchmark_info_jobs/*.json
├── alignment.py    # Column→stem mapping, model/agent matching
├── analysis.py     # Compare, anomaly detection, agent analysis
├── figures.py      # Paper-ready matplotlib figures
└── run_pipeline.py # CLI entry point
```

## Score transforms

Three benchmark families need score transforms to map doc values to [0, 1]:

| Benchmark | Raw scale | Transform |
|-----------|-----------|-----------|
| `algotune` | Speedup factor (≥ 1) | `y = ln(max(1, x)) / (ln(max(1, x)) + 1)` |
| `sldbench`, `ineqmath` | Reward in [-1, 1] | `y = (x + 1) / 2` |
| All others | Already [0, 1] | Identity |

## Adding model/agent aliases

Edit `config.py` → `MODEL_ALIASES` / `AGENT_ALIASES` to add new mappings
when Harbor and doc use different names for the same model or agent.
