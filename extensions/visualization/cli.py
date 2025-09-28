# ~/extensions/visualization/cli.py
'''
CLI entry point for plotting evaluation metrics.

Usage:
    python -m extensions.visualization.cli [results.jsonl]

Outputs:
    - per_task_metrics.csv        : Per-task aggregate metrics
    - macro_metrics.csv           : Macro-averaged pass@k and coverage values
    - pass_vs_k_with_coverage.png : Curve plotting pass@k with coverage labels
    - duplicates_hist.png         : Histogram of collapsed duplicates per task
    - pass_vs_k_comparison.png    : Optional comparison when --compare is provided
'''

# imports
import argparse
from pathlib import Path

from extensions.config import MODEL, TEMP
from extensions.visualization.io import (
    DEFAULT_RESULTS_FILE,
    DEFAULT_RESULTS_HELP,
    read_results_jsonl,
    resolve_in_results_dir,
)
from extensions.visualization.metrics import compute_macro, compute_per_task
from extensions.visualization.plots import (
    compare_two_runs,
    plot_duplicates_hist,
    plot_pass_vs_k_naive_vs_unbiased,
    plot_pass_vs_k_with_coverage,
)


# Parse CLI args and generate plots
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pass@k and coverage metrics from evaluation results",
    )
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULTS_FILE,
        help=f"Path to results JSONL (default: {DEFAULT_RESULTS_HELP})",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./figures"),
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Override the maximum k to plot (default: inferred from data)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second results JSONL to compare",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        default=None,
        help="Labels when plotting comparisons (two strings)",
    )
    args = parser.parse_args()

    # Resolve input file path and validate existence
    results_path = resolve_in_results_dir(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and process evaluation results
    rows = read_results_jsonl(results_path)
    per_task_df = compute_per_task(rows)
    inferred_max_k = int(per_task_df["n_unique"].max()) if not per_task_df.empty else 0
    max_k = args.max_k if args.max_k is not None else inferred_max_k
    macro_df = compute_macro(per_task_df, max_k=max_k if max_k >= 1 else None)

    # Export computed metrics to CSV files
    per_task_csv = args.outdir / "per_task_metrics.csv"
    macro_csv = args.outdir / "macro_metrics.csv"
    per_task_df.to_csv(per_task_csv, index=False)
    macro_df.to_csv(macro_csv, index=False)
    print(f"Saved: {per_task_csv}")
    print(f"Saved: {macro_csv}")

    print("\nSummary:")
    print(f"- Total tasks: {len(per_task_df)}")
    print(f"- Total raw samples: {per_task_df['n_raw'].sum() if not per_task_df.empty else 0}")
    print(f"- Total unique samples: {per_task_df['n_unique'].sum() if not per_task_df.empty else 0}")
    if not per_task_df.empty:
        print(f"- Average deduplication rate: {per_task_df['dedup_rate'].mean():.3f}")
    if max_k >= 1 and not macro_df.empty:
        print(f"- pass@1 (macro): {macro_df.iloc[0]['pass@k_macro']:.3f}")

    # Generate visualization plots
    descriptor = f"{MODEL}, temp={TEMP:g}"

    pass_plot = plot_pass_vs_k_with_coverage
    dual_pass_plot = plot_pass_vs_k_naive_vs_unbiased
    dups_plot = plot_duplicates_hist
    compare_plot = compare_two_runs
    extension = ".png"

    pass_fig = args.outdir / f"pass_vs_k_with_coverage{extension}"
    dual_pass_fig = args.outdir / f"pass_vs_k_naive_vs_unbiased{extension}"
    dups_fig = args.outdir / f"duplicates_hist{extension}"
    if max_k >= 1 and not macro_df.empty:
        pass_plot(
            macro_df, f"pass@k with coverage — {descriptor}", pass_fig
        )
        dual_pass_plot(
            macro_df, f"pass@k naive vs unbiased — {descriptor}", dual_pass_fig
        )
    dups_plot(per_task_df, "Duplicates collapsed per task", dups_fig)

    # Generate comparison plot if second results file provided
    if args.compare is not None:
        labels = args.labels if args.labels else ["run A", "run B"]
        compare_path = resolve_in_results_dir(args.compare)
        if not compare_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {compare_path}")
        cmp_fig = args.outdir / f"pass_vs_k_comparison{extension}"
        compare_plot(results_path, compare_path, labels[0], labels[1], cmp_fig)


if __name__ == "__main__":
    main()
