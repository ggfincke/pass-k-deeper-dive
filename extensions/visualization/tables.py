# ~/extensions/visualization/tables.py
"""
Table visualization helpers for evaluation metrics.

Generates formatted comparison tables for pass@k and coverage@k metrics
across different models and temperature settings.
"""

# imports
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from extensions.visualization.io import read_results_jsonl
from extensions.visualization.metrics import compute_macro, compute_per_task

# K values to include in comparison tables
HIGHLIGHT_KS = (1, 5, 10, 25)
# Decimal precision for metric values in tables
TABLE_PRECISION = 4
# Default output directory for generated table images
DEFAULT_TABLE_DIR = Path("figures/article_figures")


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


# Build and export table images for all metric comparisons
def main() -> None:
    # Load evaluation results from three experimental runs
    results_02 = read_results_jsonl(Path("results/0.2results.jsonl"))
    results_08 = read_results_jsonl(Path("results/0.8results.jsonl"))
    results_gemma = read_results_jsonl(Path("results/gemma3results.jsonl"))

    # Compute per-task and macro-averaged metrics
    df_02 = compute_per_task(results_02)
    df_08 = compute_per_task(results_08)
    df_gemma = compute_per_task(results_gemma)

    macro_02 = compute_macro(df_02, max_k=25)
    macro_08 = compute_macro(df_08, max_k=25)
    macro_gemma = compute_macro(df_gemma, max_k=25)

    # Ensure output directory exists
    output_dir = DEFAULT_TABLE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pass@k comparison tables
    generate_pass_at_k_table(
        macro_08,
        macro_gemma,
        "gpt-oss:20b",
        "gemma3:latest",
        "pass@k (unbiased) — gpt-oss:20b vs gemma3:latest",
        output_dir / "table_pass_at_k_model_comparison.png",
    )

    generate_pass_at_k_table(
        macro_02,
        macro_08,
        "0.2",
        "0.8",
        "gpt-oss:20b pass@k (unbiased) — temp=0.2 vs temp=0.8",
        output_dir / "table_pass_at_k_temp_comparison.png",
    )

    # Generate coverage@k comparison tables
    generate_coverage_at_k_table(
        macro_08,
        macro_gemma,
        "gpt-oss:20b",
        "gemma3:latest",
        "coverage@k — gpt-oss:20b vs gemma3:latest",
        output_dir / "table_coverage_at_k_model_comparison.png",
    )

    generate_coverage_at_k_table(
        macro_02,
        macro_08,
        "0.2",
        "0.8",
        "gpt-oss:20b coverage@k — temp=0.2 vs temp=0.8",
        output_dir / "table_coverage_at_k_temp_comparison.png",
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