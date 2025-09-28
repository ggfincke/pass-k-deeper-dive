"""Plotting helpers for evaluation metrics."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from .settings import (
    COVERAGE_LABEL_OFFSET,
    HIGHLIGHT_KS,
    PASS_K_XTICKS,
    PASS_K_XLIM,
    PASS_K_YLIM,
    STACKED_LABEL_BOTTOM_MARGIN_FRAC,
    STACKED_LABEL_EDGE_MARGIN,
    STACKED_LABEL_FORCE_RIGHT_KS,
    STACKED_LABEL_X_OFFSET,
    STACKED_LABEL_Y_GAP_FRAC,
    STACKED_LABEL_Y_PAD_FRAC,
    STACKED_LABEL_TOP_MARGIN_FRAC,
)


# Apply shared axis settings for pass@k plots
def _configure_pass_axes() -> None:
    plt.xlim(PASS_K_XLIM)
    plt.ylim(PASS_K_YLIM)
    if PASS_K_XTICKS is not None:
        plt.xticks(PASS_K_XTICKS)


# Plot the pass@k curve with coverage annotations
def plot_pass_vs_k_with_coverage(
    macro_df: pd.DataFrame, title: str, out_path: Path
) -> None:
    plt.figure()
    xs = macro_df["k"].tolist()
    ys = macro_df["pass@k_macro"].tolist()
    plt.plot(xs, ys, marker="o", label="pass@k (unbiased)")

    # Add coverage annotations for highlighted k values
    covs = macro_df["coverage@k"].tolist()
    for x, y, coverage in zip(xs, ys, covs):
        if int(round(x)) not in HIGHLIGHT_KS:
            continue
        plt.annotate(
            f"coverage={coverage:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=COVERAGE_LABEL_OFFSET,
            ha="center",
        )

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("pass@k (macro)")
    _configure_pass_axes()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


# Plot naive vs unbiased pass@k curves with stacked label annotations
def _stacked_label_positions(ax: Axes, values_desc: List[float]) -> List[float]:
    """Return y-positions for stacked labels, respecting plot margins."""

    if not values_desc:
        return []

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_pad = y_range * STACKED_LABEL_Y_PAD_FRAC
    y_gap = y_range * STACKED_LABEL_Y_GAP_FRAC
    top_margin = y_range * STACKED_LABEL_TOP_MARGIN_FRAC
    bottom_margin = y_range * STACKED_LABEL_BOTTOM_MARGIN_FRAC

    highest = values_desc[0]
    lowest = values_desc[-1]

    # Try to stack above the highest point first
    top_candidate = min(highest + y_pad, y_max - top_margin)
    positions = [top_candidate - i * y_gap for i in range(len(values_desc))]
    if not positions or positions[-1] >= y_min + bottom_margin:
        return positions

    # Otherwise, place the stack below the lowest point
    top_candidate = min(lowest - y_pad, y_max - top_margin)
    min_allowed = y_min + bottom_margin + y_gap * (len(values_desc) - 1)
    if top_candidate < min_allowed:
        top_candidate = min_allowed
    return [top_candidate - i * y_gap for i in range(len(values_desc))]


def plot_pass_vs_k_naive_vs_unbiased(
    macro_df: pd.DataFrame, title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots()
    xs = macro_df["k"].tolist()
    unbiased = macro_df["pass@k_macro"].tolist()
    naive = macro_df["pass@k_macro_naive"].tolist()

    ax.plot(xs, unbiased, marker="o", label="pass@k (unbiased)")
    ax.plot(xs, naive, marker="o", label="pass@k (naive)")

    _configure_pass_axes()

    # Add stacked annotations for highlighted k values with smarter positioning
    indexed = macro_df.set_index("k")
    for k in sorted(HIGHLIGHT_KS):
        if k not in indexed.index:
            continue
        row = indexed.loc[k]
        u_val = float(row["pass@k_macro"].item())
        n_val = float(row["pass@k_macro_naive"].item())
        labels = [
            (f"pass@{k}={u_val:.2f}", u_val, "tab:blue"),
            (f"naive@{k}={n_val:.2f}", n_val, "tab:orange"),
        ]

        labels_desc = sorted(labels, key=lambda item: item[1], reverse=True)
        y_positions = _stacked_label_positions(
            ax, [item[1] for item in labels_desc]
        )

        x_min, x_max = ax.get_xlim()
        align_right = True
        if (x_max - k) < STACKED_LABEL_EDGE_MARGIN and k not in STACKED_LABEL_FORCE_RIGHT_KS:
            align_right = False
        x_offset = STACKED_LABEL_X_OFFSET if align_right else -STACKED_LABEL_X_OFFSET
        ha = "left" if align_right else "right"

        for (text, _, color), y_pos in zip(labels_desc, y_positions):
            ax.annotate(
                text,
                xy=(k, y_pos),
                xycoords="data",
                textcoords="offset points",
                xytext=(x_offset, 0),
                ha=ha,
                va="center",
                color=color,
                annotation_clip=False,
            )

    ax.set_title(title)
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


# Plot a histogram of duplicates collapsed per task
def plot_duplicates_hist(
    per_task_df: pd.DataFrame, title: str, out_path: Path
) -> None:
    plt.figure()
    data = per_task_df["duplicates_collapsed"].tolist()
    if not data:
        data = [0]
    # Create bins from 0 to max duplicates + 1
    bins = range(int(max(data)) + 2)
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("Duplicates collapsed per task (n_raw - n_unique)")
    plt.ylabel("Count of tasks")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


# Compare two runs by plotting their pass@k curves
def compare_two_runs(
    file_a: Path, file_b: Path, label_a: str, label_b: str, out_path: Path
) -> None:
    from extensions.visualization.io import read_results_jsonl
    from extensions.visualization.metrics import compute_macro, compute_per_task

    # Load and process both result files
    rows_a = read_results_jsonl(file_a)
    rows_b = read_results_jsonl(file_b)
    df_a = compute_per_task(rows_a)
    df_b = compute_per_task(rows_b)
    max_k = int(
        max(
            df_a["n_unique"].max() if not df_a.empty else 0,
            df_b["n_unique"].max() if not df_b.empty else 0,
        )
    )
    macro_a = compute_macro(df_a, max_k=max_k)
    macro_b = compute_macro(df_b, max_k=max_k)

    plt.figure()
    plt.plot(macro_a["k"], macro_a["pass@k_macro"], marker="o", label=label_a)
    plt.plot(macro_b["k"], macro_b["pass@k_macro"], marker="o", label=label_b)
    plt.title("pass@k comparison")
    plt.xlabel("k")
    plt.ylabel("pass@k (macro)")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


__all__ = [
    "plot_pass_vs_k_with_coverage",
    "plot_pass_vs_k_naive_vs_unbiased",
    "plot_duplicates_hist",
    "compare_two_runs",
]
