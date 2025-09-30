# ~/extensions/visualization/tables.py
"""
Table visualization helpers for evaluation metrics.

Generates formatted comparison tables for pass@k and coverage@k metrics
across different models and temperature settings.
"""

# imports
from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from extensions.visualization.io import read_results_jsonl, resolve_in_results_dir
from extensions.visualization.metrics import compute_macro, compute_per_task
from extensions.visualization.settings import HIGHLIGHT_KS, TABLE_PRECISION

# Default output directory for generated table images
DEFAULT_TABLE_DIR = Path("figures/article_figures")


# Convert labels into filename-friendly slugs
def _slugify(label: str) -> str:
    cleaned = label.strip().lower().replace("/", "-")
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in cleaned)


# Extract metric values for specified k values from macro dataframe
def extract_metrics_at_k(
    macro_df: pd.DataFrame, metric_col: str, ks: Tuple[int, ...] = HIGHLIGHT_KS
) -> List[float]:
    values = []
    for k in ks:
        row = macro_df.loc[macro_df["k"] == k, metric_col]
        if not row.empty:
            values.append(float(row.values[0]))
        else:
            values.append(float("nan"))
    return values


# Render a metric comparison table as a matplotlib figure
def create_table_viz(
    data: List[List[str]], title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("tight")
    ax.axis("off")

    # Render table with matplotlib
    table = ax.table(cellText=data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1])

    # Apply table styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Style header row with background color and bold text
    for i in range(len(data[0])):
        cell = table[(0, i)]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(weight="bold")

    # Add title above table
    plt.title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# Generate pass@k comparison table between two runs
def generate_pass_at_k_table(
    macro_a: pd.DataFrame,
    macro_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    title: str,
    output_path: Path,
) -> None:
    data = [["", label_a, label_b]]
    vals_a = extract_metrics_at_k(macro_a, "pass@k_macro")
    vals_b = extract_metrics_at_k(macro_b, "pass@k_macro")

    for k, val_a, val_b in zip(HIGHLIGHT_KS, vals_a, vals_b):
        data.append([f"k={k}", f"{val_a:.{TABLE_PRECISION}f}", f"{val_b:.{TABLE_PRECISION}f}"])

    create_table_viz(data, title, output_path)


# Generate coverage@k comparison table between two runs
def generate_coverage_at_k_table(
    macro_a: pd.DataFrame,
    macro_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    title: str,
    output_path: Path,
) -> None:
    data = [["", label_a, label_b]]
    vals_a = extract_metrics_at_k(macro_a, "coverage@k")
    vals_b = extract_metrics_at_k(macro_b, "coverage@k")

    for k, val_a, val_b in zip(HIGHLIGHT_KS, vals_a, vals_b):
        data.append([f"k={k}", f"{val_a:.{TABLE_PRECISION}f}", f"{val_b:.{TABLE_PRECISION}f}"])

    create_table_viz(data, title, output_path)


def _load_macro(path: Path, max_k: Optional[int]) -> pd.DataFrame:
    rows = read_results_jsonl(path)
    per_task = compute_per_task(rows)
    return compute_macro(per_task, max_k=max_k)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Generate pass@k and coverage@k comparison tables.")
    parser.add_argument(
        "--run",
        metavar=("LABEL", "PATH"),
        nargs=2,
        action="append",
        required=True,
        help="Label and JSONL results file produced by functional evaluation (may repeat).",
    )
    parser.add_argument(
        "--pair",
        metavar=("LABEL_A", "LABEL_B"),
        nargs=2,
        action="append",
        help="Specific label pairs to compare; defaults to all pairwise combinations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
        help=f"Directory for generated tables (default: {DEFAULT_TABLE_DIR}).",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Optional maximum k to evaluate; defaults to per-run maximum unique completions.",
    )
    parser.add_argument(
        "--prefix",
        default="table",
        help="Filename prefix for generated tables (default: 'table').",
    )
    return parser


# Build and export table images for all metric comparisons
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_paths: Dict[str, Path] = {}
    for label, raw_path in args.run:
        if label in run_paths:
            parser.error(f"Duplicate run label provided: {label!r}")
        resolved = resolve_in_results_dir(Path(raw_path)).resolve()
        if not resolved.exists():
            parser.error(f"Results file not found: {raw_path}")
        run_paths[label] = resolved

    if len(run_paths) < 2:
        parser.error("Provide at least two --run entries to compute comparisons.")

    if args.pair:
        pairs = []
        for label_a, label_b in args.pair:
            if label_a not in run_paths:
                parser.error(f"Unknown label referenced in --pair: {label_a!r}")
            if label_b not in run_paths:
                parser.error(f"Unknown label referenced in --pair: {label_b!r}")
            pairs.append((label_a, label_b))
    else:
        pairs = list(combinations(run_paths.keys(), 2))

    macros: Dict[str, pd.DataFrame] = {}
    for label, path in run_paths.items():
        macros[label] = _load_macro(path, args.max_k)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.rstrip("_") or "table"
    for label_a, label_b in pairs:
        macro_a = macros[label_a]
        macro_b = macros[label_b]
        slug_a = _slugify(label_a)
        slug_b = _slugify(label_b)

        pass_title = f"pass@k (unbiased) — {label_a} vs {label_b}"
        pass_path = output_dir / f"{prefix}_pass_at_k_{slug_a}_vs_{slug_b}.png"
        generate_pass_at_k_table(macro_a, macro_b, label_a, label_b, pass_title, pass_path)

        coverage_title = f"coverage@k — {label_a} vs {label_b}"
        coverage_path = output_dir / f"{prefix}_coverage_at_k_{slug_a}_vs_{slug_b}.png"
        generate_coverage_at_k_table(
            macro_a, macro_b, label_a, label_b, coverage_title, coverage_path
        )


if __name__ == "__main__":
    main()


__all__ = [
    "create_table_viz",
    "generate_pass_at_k_table",
    "generate_coverage_at_k_table",
    "extract_metrics_at_k",
    "HIGHLIGHT_KS",
    "TABLE_PRECISION",
]
