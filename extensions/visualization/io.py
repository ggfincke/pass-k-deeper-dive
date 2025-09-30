# ~/extensions/visualization/io.py
"""
Input/output helpers for visualization CLI commands.

Resolves default result paths and provides JSONL reading utilities.
"""

# imports
import json
from pathlib import Path
from typing import Dict, List

from extensions.config import REPO_ROOT


# Establish repo-relative defaults for result artifacts
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_DIR / "samples.jsonl_results.jsonl"

try:
    DEFAULT_RESULTS_HELP = DEFAULT_RESULTS_FILE.relative_to(REPO_ROOT)
except ValueError:
    DEFAULT_RESULTS_HELP = DEFAULT_RESULTS_FILE


# Read JSONL evaluation results from ``path``
def read_results_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


# Resolve ``path`` relative to the default results directory when missing
def resolve_in_results_dir(path: Path) -> Path:
    if path.exists() or path.is_absolute():
        return path
    candidate = DEFAULT_RESULTS_DIR / path
    return candidate if candidate.exists() else path


__all__ = [
    "read_results_jsonl",
    "resolve_in_results_dir",
    "DEFAULT_RESULTS_FILE",
    "DEFAULT_RESULTS_HELP",
]
