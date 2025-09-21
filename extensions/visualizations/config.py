# ~/extensions/visualizations/config.py
"""
Configuration constants and default paths for the evaluation metrics visualization system.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_RESULTS_FILE = DEFAULT_RESULTS_DIR / "samples.jsonl_results.jsonl"

try:
    DEFAULT_RESULTS_HELP = DEFAULT_RESULTS_FILE.relative_to(REPO_ROOT)
except ValueError:
    DEFAULT_RESULTS_HELP = DEFAULT_RESULTS_FILE