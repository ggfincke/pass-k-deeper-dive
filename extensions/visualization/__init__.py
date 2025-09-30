# ~/extensions/visualization/__init__.py
"""
Visualization utilities for evaluation metrics.
"""

from extensions.visualization.plots import (
    plot_pass_vs_k_with_coverage,
    plot_pass_vs_k_naive_vs_unbiased,
    plot_pass_vs_k_unbiased_comparison,
    plot_coverage_vs_k_comparison,
    plot_duplicates_hist,
    compare_two_runs,
)

from extensions.visualization.tables import (
    create_table_viz,
    generate_pass_at_k_table,
    generate_coverage_at_k_table,
    extract_metrics_at_k,
)

__all__ = [
    "plot_pass_vs_k_with_coverage",
    "plot_pass_vs_k_naive_vs_unbiased",
    "plot_pass_vs_k_unbiased_comparison",
    "plot_coverage_vs_k_comparison",
    "plot_duplicates_hist",
    "compare_two_runs",
    "create_table_viz",
    "generate_pass_at_k_table",
    "generate_coverage_at_k_table",
    "extract_metrics_at_k",
]
