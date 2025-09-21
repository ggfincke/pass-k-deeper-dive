# ~/extensions/visualizations/plotting.py
"""
Visualization functions for creating charts and plots of evaluation metrics.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .utils import read_results_jsonl
from .metrics import compute_per_task, compute_macro


# Plot pass@k curve with coverage annotations
def plot_pass_vs_k_with_coverage(macro_df: pd.DataFrame, title: str, out_path: Path):
    plt.figure()
    xs = macro_df["k"].tolist()
    ys = macro_df["pass@k_macro"].tolist()
    plt.plot(xs, ys, marker="o", label="pass@k (macro)")
    
    # annotate coverage as text near each point
    covs = macro_df["coverage@k"].tolist()
    for x, y, c in zip(xs, ys, covs):
        plt.annotate(f"coverage={c:.2f}", (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("pass@k (macro)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


# Plot histogram of duplicates collapsed per task
def plot_duplicates_hist(per_task_df: pd.DataFrame, title: str, out_path: Path):
    plt.figure()
    data = per_task_df["duplicates_collapsed"].tolist()
    if len(data) == 0:
        data = [0]
    bins = range(int(max(data)) + 2)
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("Duplicates collapsed per task (n_raw - n_unique)")
    plt.ylabel("Count of tasks")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


# Compare pass@k curves between two evaluation runs
def compare_two_runs(file_a: Path, file_b: Path, label_a: str, label_b: str, out_path: Path):
    rows_a = read_results_jsonl(file_a)
    rows_b = read_results_jsonl(file_b)
    df_a = compute_per_task(rows_a)
    df_b = compute_per_task(rows_b)
    max_k = int(max(df_a["n_unique"].max() if not df_a.empty else 0,
                    df_b["n_unique"].max() if not df_b.empty else 0))
    macro_a = compute_macro(df_a, max_k=max_k)
    macro_b = compute_macro(df_b, max_k=max_k)

    plt.figure()
    plt.plot(macro_a["k"], macro_a["pass@k_macro"], marker="o", label=label_a)
    plt.plot(macro_b["k"], macro_b["pass@k_macro"], marker="o", label=label_b)
    plt.title("pass@k comparison")
    plt.xlabel("k")
    plt.ylabel("pass@k (macro)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
