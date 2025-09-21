# ~/extensions/visualizations/main.py
"""
Main entry point for the evaluation metrics visualization tool.

Visualize pass@k, coverage@k, and uniqueness from a results JSONL using matplotlib.
- Uses matplotlib (no seaborn), one chart per figure, default colors.
- Expects JSONL lines with: {"task_id": str, "completion": str, "result": "passed" or "failed: ..."}

Usage:
  python main.py --outdir ./figures --max_k 3
  python main.py results/custom_run_results.jsonl --compare baseline_results.jsonl --labels "temp=0.2" "temp=0.8"
"""

import argparse
from pathlib import Path

from .config import DEFAULT_RESULTS_FILE, DEFAULT_RESULTS_HELP
from .utils import read_results_jsonl, resolve_in_results_dir
from .metrics import compute_per_task, compute_macro
from .plotting import plot_pass_vs_k_with_coverage, plot_duplicates_hist, compare_two_runs


# Main function with command-line interface
def main():
    ap = argparse.ArgumentParser(description="Visualize pass@k and coverage@k metrics from results JSONL")
    ap.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to results JSONL (default: {DEFAULT_RESULTS_HELP})",
    )
    ap.add_argument("--outdir", type=Path, default=Path("./figures"), 
                   help="Output directory for figures")
    ap.add_argument("--max_k", type=int, default=None, 
                   help="Override max k (default: inferred from n_unique)")
    ap.add_argument("--compare", type=Path, default=None, 
                   help="Optional: second results JSONL to compare")
    ap.add_argument("--labels", nargs=2, default=None, 
                   help="Labels for the comparison plot (two strings)")
    args = ap.parse_args()

    results_path = resolve_in_results_dir(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = read_results_jsonl(results_path)
    per_task_df = compute_per_task(rows)
    max_k = args.max_k if args.max_k is not None else (
        int(per_task_df["n_unique"].max()) if not per_task_df.empty else 0)
    macro_df = compute_macro(per_task_df, max_k=max_k)

    # save tables as CSV alongside plots for convenience
    per_task_csv = args.outdir / "per_task_metrics.csv"
    macro_csv = args.outdir / "macro_metrics.csv"
    per_task_df.to_csv(per_task_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    print(f"Saved: {per_task_csv}")
    print(f"Saved: {macro_csv}")

    # print summary statistics
    print(f"\nSummary:")
    print(f"- Total tasks: {len(per_task_df)}")
    print(f"- Total raw samples: {per_task_df['n_raw'].sum()}")
    print(f"- Total unique samples: {per_task_df['n_unique'].sum()}")
    print(f"- Average deduplication rate: {per_task_df['dedup_rate'].mean():.3f}")
    if max_k >= 1:
        print(f"- pass@1 (macro): {macro_df.iloc[0]['pass@k_macro']:.3f}")

    # plots
    pass_fig = args.outdir / "pass_vs_k_with_coverage.png"
    dups_fig = args.outdir / "duplicates_hist.png"
    if max_k >= 1:
        plot_pass_vs_k_with_coverage(macro_df, "pass@k with coverage", pass_fig)
    plot_duplicates_hist(per_task_df, "Duplicates collapsed per task", dups_fig)

    # optional comparison
    if args.compare is not None:
        labels = args.labels if args.labels else ["run A", "run B"]
        cmp_fig = args.outdir / "pass_vs_k_comparison.png"
        compare_path = resolve_in_results_dir(args.compare)
        if not compare_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {compare_path}")
        compare_two_runs(results_path, compare_path, labels[0], labels[1], cmp_fig)


if __name__ == "__main__":
    main()
