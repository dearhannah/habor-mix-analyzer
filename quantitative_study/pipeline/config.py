"""
Central configuration: paths, column/stem mappings, model/agent aliases,
score transforms, and figure style constants.
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to habor-mix-analyzer root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

HARBOR_CSV_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / "benchmark_level_matrix.csv",
    REPO_ROOT / "benchmark_level_matrix.csv",
]
BENCHMARK_INFO_DIR = REPO_ROOT / "benchmark_info_jobs"
OUTPUT_DIR = REPO_ROOT / "output" / "quantitative"
FIGURE_DIR = OUTPUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Matrix column -> JSON stem overrides
# ---------------------------------------------------------------------------
MATRIX_COLUMN_TO_STEM: dict[str, str | None] = {
    "bixbench": "bix_bench",
    "crustbench": "crust_bench",
    "dacode": "da_code",
    "featurebench-modal": "featurebench",
    "labbench": "lab_bench",
    "mlgym": "mlgym_bench",
    "omnimath": "omni_math",
    "research-code-bench": "researchcodebench",
    "spider2": "spider_2",
    "swebench-verified": "swe_bench_verified",
    "swebench-multilingual": "swe_bench_multilingual",
    "swebenchpro": "swe_bench_pro",
    "swtbench": "swt_bench",
    "terminal-bench": "terminalbench_2_0",
    "qwen-coder": None,
    "swegym": None,
    "swesmith": "swe_smith",
}

# ---------------------------------------------------------------------------
# Model aliases: harbor_model -> list of known doc name patterns
# ---------------------------------------------------------------------------
MODEL_ALIASES: dict[str, list[str]] = {
    "gpt-5.4": ["gpt-5.4", "gpt 5.4"],
    "gpt-5-mini": ["gpt-5-mini", "gpt 5 mini", "gpt-5-mini-2025-08-07"],
    "gpt-5-nano": ["gpt-5-nano", "gpt 5 nano", "gpt-5-nano-2025-08-07"],
    "claude-haiku-4-5-20251001": ["claude-haiku-4-5", "claude haiku 4.5"],
    "claude-sonnet-4-6": ["claude-sonnet-4-6", "claude sonnet 4.6"],
    "claude-opus-4-6": ["claude-opus-4-6", "claude opus 4.6"],
    "gemini-3.1-pro-preview": [
        "gemini-3.1-pro-preview", "gemini 3.1 pro preview",
        "gemini-3.1-pro", "gemini 3.1 pro",
    ],
    "gemini-3-flash-preview": [
        "gemini-3-flash-preview", "gemini 3 flash preview",
        "gemini-3-flash", "gemini 3 flash",
    ],
    "deepseek-reasoner": ["deepseek-reasoner", "deepseek-r1", "deepseek r1"],
    "deepseek-chat": [
        "deepseek-chat", "deepseek-v3", "deepseek v3",
        "deepseek-v3.1", "deepseek v3.1",
    ],
    "kimi-k2.5": ["kimi-k2.5", "kimi k2.5"],
    "minimax-m2.5": ["minimax-m2.5", "minimax m2.5"],
    "glm-5": ["glm-5", "glm 5"],
    "mimo-v2-pro": ["mimo-v2-pro", "mimo v2 pro"],
    "qwen3-max": ["qwen3-max", "qwen 3 max", "qwen3 max", "qwen-3-max"],
}

AGENT_ALIASES: dict[str, list[str]] = {
    "terminus-2": ["terminus-2", "terminus 2", "terminus"],
    "codex": ["codex"],
    "claude-code": ["claude-code", "claude code"],
    "gemini-cli": ["gemini-cli", "gemini cli"],
    "qwen-coder": ["qwen-coder", "qwen coder"],
}

# ---------------------------------------------------------------------------
# Score transforms: map raw doc scores to [0, 1] to match Harbor scale.
# ---------------------------------------------------------------------------
TRANSFORM_LOG_SPEEDUP = {"algotune"}          # y = ln(max(1,x)) / (ln(max(1,x)) + 1)
TRANSFORM_NEG1_TO_1   = {"sldbench", "ineqmath"}  # y = (x+1) / 2

SKIP_METRICS = {"cost", "latency", "latency mean", "latency_mean"}


def transform_doc_score(score: float, stem: str) -> float:
    if stem in TRANSFORM_LOG_SPEEDUP:
        lnx = math.log(max(1.0, score))
        return lnx / (lnx + 1.0) if (lnx + 1.0) != 0 else 0.0
    if stem in TRANSFORM_NEG1_TO_1:
        return (score + 1.0) / 2.0
    return score


# ---------------------------------------------------------------------------
# Figure style — aligned with habor-analyze (core/plotting.py)
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (10, 8)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (11.5, 10)
DPI = 200

PLOT_STYLE = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.axisbelow": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
    "axes.facecolor": "#fbfbfb",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

# Palette consistent with habor-analyze
COLOR_GREEN = "#a1d99b"
COLOR_BLUE = "#9ecae1"
COLOR_ORANGE = "#fdae6b"
COLOR_PURPLE = "#bcbddc"
COLOR_RED = "#e74c3c"
COLOR_GRAY = "#95a5a6"
COLOR_GRID = "#dddddd"
COLOR_AXIS = "#666666"

# ---------------------------------------------------------------------------
# Domain taxonomy for Harbor benchmarks (8 categories)
# Maps matrix column name -> (domain, superdomain)
# superdomain is "Coding" vs "Non-Coding" for the headline split.
# ---------------------------------------------------------------------------
_CODING = "Coding"
_NON_CODING = "Non-Coding"

DOMAIN_MAP: dict[str, tuple[str, str]] = {
    # --- 1. SWE / Repo-level (14) ---
    "swebench-verified":     ("SWE / Repo-level", _CODING),
    "swebench-multilingual": ("SWE / Repo-level", _CODING),
    "swebenchpro":           ("SWE / Repo-level", _CODING),
    "swesmith":              ("SWE / Repo-level", _CODING),
    "swegym":                ("SWE / Repo-level", _CODING),
    "swtbench":              ("SWE / Repo-level", _CODING),
    "multi-swe-bench":       ("SWE / Repo-level", _CODING),
    "swe-lancer":            ("SWE / Repo-level", _CODING),
    "crustbench":            ("SWE / Repo-level", _CODING),
    "gso":                   ("SWE / Repo-level", _CODING),
    "featurebench-modal":    ("SWE / Repo-level", _CODING),
    "devopsgym":             ("SWE / Repo-level", _CODING),
    "humanevalfix":          ("SWE / Repo-level", _CODING),
    "quixbugs":              ("SWE / Repo-level", _CODING),
    # --- 2. Coding & Algorithms (7) ---
    "aider-polyglot":  ("Coding & Algorithms", _CODING),
    "bigcodebench":    ("Coding & Algorithms", _CODING),
    "livecodebench":   ("Coding & Algorithms", _CODING),
    "usaco":           ("Coding & Algorithms", _CODING),
    "algotune":        ("Coding & Algorithms", _CODING),
    "compilebench":    ("Coding & Algorithms", _CODING),
    "qwen-coder":      ("Coding & Algorithms", _CODING),
    # --- 3. Math (3) ---
    "aime":     ("Math", _NON_CODING),
    "ineqmath": ("Math", _NON_CODING),
    "omnimath": ("Math", _NON_CODING),
    # --- 4. Science & AI Research (12) ---
    "bixbench":         ("Science & AI Research", _NON_CODING),
    "codepde":          ("Science & AI Research", _NON_CODING),
    "deepsynth":        ("Science & AI Research", _NON_CODING),
    "gpqa-diamond":     ("Science & AI Research", _NON_CODING),
    "hle":              ("Science & AI Research", _NON_CODING),
    "labbench":         ("Science & AI Research", _NON_CODING),
    "qcircuitbench":    ("Science & AI Research", _NON_CODING),
    "sldbench":         ("Science & AI Research", _NON_CODING),
    "scicode":          ("Science & AI Research", _NON_CODING),
    "mlgym":            ("Science & AI Research", _NON_CODING),
    "replicationbench": ("Science & AI Research", _NON_CODING),
    "research-code-bench": ("Science & AI Research", _NON_CODING),
    # --- 5. Agents & Tool Use (7) ---
    "bfcl":            ("Agents & Tool Use", _NON_CODING),
    "gaia":            ("Agents & Tool Use", _NON_CODING),
    "gaia2":           ("Agents & Tool Use", _NON_CODING),
    "financeagent":    ("Agents & Tool Use", _NON_CODING),
    "medagentbench":   ("Agents & Tool Use", _NON_CODING),
    "skillsbench":     ("Agents & Tool Use", _NON_CODING),
    "terminal-bench":  ("Agents & Tool Use", _NON_CODING),
    # --- 6. Data Science & Analysis (5) ---
    "dacode":          ("Data Science & Analysis", _NON_CODING),
    "ds-1000":         ("Data Science & Analysis", _NON_CODING),
    "kumo":            ("Data Science & Analysis", _NON_CODING),
    "spider2":         ("Data Science & Analysis", _NON_CODING),
    "spreadsheetbench": ("Data Science & Analysis", _NON_CODING),
    # --- 7. Reasoning & Knowledge (7) ---
    "arc-agi-2":      ("Reasoning & Knowledge", _NON_CODING),
    "reasoning-gym":  ("Reasoning & Knowledge", _NON_CODING),
    "simpleqa":       ("Reasoning & Knowledge", _NON_CODING),
    "mmmlu":          ("Reasoning & Knowledge", _NON_CODING),
    "lawbench":       ("Reasoning & Knowledge", _NON_CODING),
    "mmau":           ("Reasoning & Knowledge", _NON_CODING),
    "pixiu":          ("Reasoning & Knowledge", _NON_CODING),
    # --- 8. Safety & Security (2) ---
    "seal0":          ("Safety & Security", _NON_CODING),
    "strongreject":   ("Safety & Security", _NON_CODING),
}

DOMAIN_COLORS = {
    "SWE / Repo-level":        "#74c476",
    "Coding & Algorithms":     "#a1d99b",
    "Math":                    "#6baed6",
    "Science & AI Research":   "#9e9ac8",
    "Agents & Tool Use":       "#fdae6b",
    "Data Science & Analysis": "#fdd0a2",
    "Reasoning & Knowledge":   "#fc9272",
    "Safety & Security":       "#969696",
}

SUPERDOMAIN_COLORS = {
    _CODING:     "#74c476",
    _NON_CODING: "#9ecae1",
}


MATCH_STATUS_COLORS = {
    "matched_model_and_agent": COLOR_GREEN,
    "matched_model_only_doc_no_agent": COLOR_BLUE,
    "matched_model_agent_mismatch": COLOR_ORANGE,
}

MATCH_STATUS_LABELS = {
    "matched_model_and_agent": "Model + Agent matched",
    "matched_model_only_doc_no_agent": "Model matched (doc = bare LLM)",
    "matched_model_agent_mismatch": "Model matched (agent differs)",
}
