"""
Align Harbor matrix columns/rows to benchmark_info_jobs doc entries.

Two-stage alignment:
  1. Column name  -> JSON stem  (resolve_stem)
  2. (model, agent) -> doc row  (find_doc_match)
"""

from __future__ import annotations

import re

from .config import AGENT_ALIASES, MATRIX_COLUMN_TO_STEM, MODEL_ALIASES
from .loading import DocSlice, DocRow, norm


# ---------------------------------------------------------------------------
# Stem resolution
# ---------------------------------------------------------------------------

def resolve_stem(column: str, stems: set[str]) -> str | None:
    col = column.strip()
    if col in ("model", "agent"):
        return None
    if col in MATRIX_COLUMN_TO_STEM:
        mapped = MATRIX_COLUMN_TO_STEM[col]
        if mapped is None:
            return None
        return mapped if mapped in stems else None
    cand = col.replace("-", "_")
    return cand if cand in stems else None


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

def _strip_parenthetical(s: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", s).strip()


def _strip_date_suffix(s: str) -> str:
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", s).strip()


def _build_index(aliases: dict[str, list[str]]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for canonical, patterns in aliases.items():
        for p in patterns:
            idx[norm(p)] = norm(canonical)
    return idx


_MODEL_IDX: dict[str, str] | None = None
_AGENT_IDX: dict[str, str] | None = None


def _model_idx() -> dict[str, str]:
    global _MODEL_IDX
    if _MODEL_IDX is None:
        _MODEL_IDX = _build_index(MODEL_ALIASES)
    return _MODEL_IDX


def _agent_idx() -> dict[str, str]:
    global _AGENT_IDX
    if _AGENT_IDX is None:
        _AGENT_IDX = _build_index(AGENT_ALIASES)
    return _AGENT_IDX


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def _try_match_model(harbor_model_norm: str, doc_model_norm: str) -> bool:
    if harbor_model_norm == doc_model_norm:
        return True
    idx = _model_idx()
    candidates = [
        doc_model_norm,
        _strip_parenthetical(doc_model_norm),
        _strip_date_suffix(doc_model_norm),
        _strip_date_suffix(_strip_parenthetical(doc_model_norm)),
    ]
    for c in candidates:
        cn = norm(c)
        if idx.get(cn) == harbor_model_norm:
            return True
    return False


def _try_match_agent(harbor_agent_norm: str, doc_system_norm: str) -> bool:
    if not doc_system_norm:
        return False
    if harbor_agent_norm == doc_system_norm:
        return True
    return _agent_idx().get(doc_system_norm) == harbor_agent_norm


def find_doc_match(
    harbor_model: str,
    harbor_agent: str,
    doc: DocSlice,
) -> tuple[float | None, str, str, str, str]:
    """
    Returns (score_doc, doc_model_raw, doc_system_raw, metric_used, match_status).
    """
    if not doc.rows:
        return None, "", "", "", "no_doc_data"

    hm = norm(harbor_model)
    ha = norm(harbor_agent)

    model_matches: list[DocRow] = [
        r for r in doc.rows if _try_match_model(hm, r.model_norm)
    ]
    if not model_matches:
        return None, "", "", "", "no_model_match"

    for row in model_matches:
        if row.system_raw and _try_match_agent(ha, row.system_norm):
            return (row.score, row.model_raw, row.system_raw or "",
                    row.metric_name, "matched_model_and_agent")

    null_system = [r for r in model_matches if not r.system_raw]
    if null_system:
        row = null_system[0]
        return (row.score, row.model_raw, "", row.metric_name,
                "matched_model_only_doc_no_agent")

    row = model_matches[0]
    return (row.score, row.model_raw, row.system_raw or "", row.metric_name,
            "matched_model_agent_mismatch")
