# ~/extensions/visualizations/utils.py
"""
Utility functions for processing evaluation results and handling file operations.
"""

import json
from pathlib import Path
from typing import List, Dict

from .config import DEFAULT_RESULTS_DIR


# Normalize code string for duplicate detection
def normalize_code(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").strip("\n")
    lines = [ln.rstrip() for ln in s.splitlines()]
    return "\n".join(lines)


# Read and parse results from JSONL file
def read_results_jsonl(path: Path) -> List[Dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


# Resolve provided path, falling back to DEFAULT_RESULTS_DIR when relative
def resolve_in_results_dir(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    candidate = DEFAULT_RESULTS_DIR / path
    return candidate if candidate.exists() else path