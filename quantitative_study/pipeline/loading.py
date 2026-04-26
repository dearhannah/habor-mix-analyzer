"""
Data loading: Harbor CSV matrix and benchmark_info_jobs/*.json files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    BENCHMARK_INFO_DIR,
    HARBOR_CSV_CANDIDATES,
    SKIP_METRICS,
    transform_doc_score,
)


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[_]", "-", s)
    return s


# ---------------------------------------------------------------------------
# Harbor matrix
# ---------------------------------------------------------------------------

def load_harbor_matrix(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        for c in HARBOR_CSV_CANDIDATES:
            if c.is_file():
                path = c
                break
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Harbor CSV not found; tried {HARBOR_CSV_CANDIDATES}")
    return pd.read_csv(path)


def harbor_to_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
        id_vars=["model", "agent"],
        var_name="matrix_column",
        value_name="score_harbor",
    )


# ---------------------------------------------------------------------------
# Doc JSON structures
# ---------------------------------------------------------------------------

@dataclass
class DocRow:
    model_raw: str
    effort_raw: str | None
    system_raw: str | None
    metric_name: str
    score: float
    model_norm: str = ""
    system_norm: str = ""

    def __post_init__(self):
        self.model_norm = norm(self.model_raw)
        self.system_norm = norm(self.system_raw) if self.system_raw else ""


@dataclass
class DocSlice:
    stem: str
    benchmark_name: str
    primary_metric: str
    aligned_metric: str | None
    slice_date: str
    source_url: str
    rows: list[DocRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_doc_date(s: str) -> tuple[int, int, int]:
    s = (s or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return tuple(map(int, s.split("-")))  # type: ignore[return-value]
    if re.fullmatch(r"\d{4}-\d{2}", s):
        y, m = map(int, s.split("-"))
        return (y, m, 1)
    return (0, 0, 0)


def _pick_newest_slice(rot: list[dict]) -> dict | None:
    if not rot:
        return None
    best_key, best_idx = (0, 0, 0), 0
    for i, block in enumerate(rot):
        k = _parse_doc_date(str(block.get("date", "")))
        if k > best_key:
            best_key, best_idx = k, i
    return rot[best_idx]


def _extract_metric_value(
    scores: list[dict], primary: str, aligned: str | None
) -> tuple[float | None, str]:
    if not scores:
        return None, ""

    want: list[str] = []
    if aligned and aligned.strip():
        want.append(aligned.strip().lower())
    if primary and primary.strip():
        want.append(primary.strip().lower())

    for w in want:
        for s in scores:
            m = (s.get("metric") or "").strip()
            if m.lower() == w and isinstance(s.get("value"), (int, float)):
                return float(s["value"]), m

    for w in want:
        for s in scores:
            m = (s.get("metric") or "").strip()
            if w in m.lower() and isinstance(s.get("value"), (int, float)):
                return float(s["value"]), m

    for s in scores:
        m = (s.get("metric") or "").strip().lower()
        if m in SKIP_METRICS:
            continue
        if isinstance(s.get("value"), (int, float)):
            return float(s["value"]), str(s.get("metric", ""))

    for s in scores:
        if isinstance(s.get("value"), (int, float)):
            return float(s["value"]), str(s.get("metric", ""))
    return None, ""


def load_doc_slice(path: Path) -> DocSlice:
    data = json.loads(path.read_text())
    stem = path.stem
    ev = data.get("evaluation") or {}
    primary = str(ev.get("primary_metric") or "").strip()
    aligned = ev.get("harbor_aligned_metric")
    if aligned:
        aligned_clean = re.split(r"[.;]", str(aligned))[0].strip()
        aligned = aligned_clean or None

    rot = data.get("results_over_time") or []
    block = _pick_newest_slice(rot)
    if block is None:
        return DocSlice(stem=stem, benchmark_name=str(data.get("name") or stem),
                        primary_metric=primary, aligned_metric=aligned,
                        slice_date="", source_url="")

    rows: list[DocRow] = []
    for r in block.get("results") or []:
        model = str(r.get("model") or "").strip()
        if not model:
            continue
        system = r.get("system_description")
        if system is not None:
            system = str(system).strip() or None
        effort = r.get("effort")
        if effort is not None:
            effort = str(effort).strip() or None
        val, mname = _extract_metric_value(r.get("scores") or [], primary, aligned)
        if val is None:
            continue
        val = transform_doc_score(val, stem)
        rows.append(DocRow(
            model_raw=model, effort_raw=effort, system_raw=system,
            metric_name=mname, score=val,
        ))

    return DocSlice(
        stem=stem, benchmark_name=str(data.get("name") or stem),
        primary_metric=primary, aligned_metric=aligned,
        slice_date=str(block.get("date", "")),
        source_url=str(block.get("source_url", "")),
        rows=rows,
    )


def load_all_docs(info_dir: Path | None = None) -> dict[str, DocSlice]:
    info_dir = info_dir or BENCHMARK_INFO_DIR
    docs: dict[str, DocSlice] = {}
    for p in sorted(info_dir.glob("*.json")):
        docs[p.stem] = load_doc_slice(p)
    return docs
