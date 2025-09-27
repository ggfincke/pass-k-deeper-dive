# ~/extensions/visualization/settings.py
"""Shared configuration for visualization plots."""

# Pass@k highlighting and axis configuration
HIGHLIGHT_KS = (1, 5, 10, 25)  # K values to annotate with coverage info
PASS_K_XLIM = (-2, 27)  # X-axis limits for pass@k plots
PASS_K_YLIM = (0.8, 1.05)  # Y-axis limits for pass@k plots
PASS_K_XTICKS = (0, 1, 5, 10, 25)  # Custom tick marks; None for autoset

# Coverage annotation positioning (offset in points)
COVERAGE_LABEL_OFFSET = (0, 10)  # Standard coverage label offset
COVERAGE_ABOVE_OFFSET = (0, 28)  # Extended offset for stacked labels

# Stacked label positioning configuration for naive vs unbiased plots
STACKED_LABEL_X_OFFSET = 16  # Horizontal offset in points
STACKED_LABEL_EDGE_MARGIN = 1.2  # Distance from plot edge to trigger left alignment
STACKED_LABEL_FORCE_RIGHT_KS = (1,)  # K values forced to right alignment
STACKED_LABEL_Y_PAD_FRAC = 0.035  # Vertical padding above curve as fraction of y-range
STACKED_LABEL_Y_GAP_FRAC = 0.055  # Gap between stacked labels as fraction of y-range
STACKED_LABEL_TOP_MARGIN_FRAC = 0.02  # Top margin as fraction of y-range
STACKED_LABEL_BOTTOM_MARGIN_FRAC = 0.012  # Bottom margin as fraction of y-range

__all__ = [
    "HIGHLIGHT_KS",
    "PASS_K_XLIM",
    "PASS_K_XTICKS",
    "PASS_K_YLIM",
    "COVERAGE_LABEL_OFFSET",
    "COVERAGE_ABOVE_OFFSET",
    "STACKED_LABEL_X_OFFSET",
    "STACKED_LABEL_EDGE_MARGIN",
    "STACKED_LABEL_FORCE_RIGHT_KS",
    "STACKED_LABEL_Y_PAD_FRAC",
    "STACKED_LABEL_Y_GAP_FRAC",
    "STACKED_LABEL_TOP_MARGIN_FRAC",
    "STACKED_LABEL_BOTTOM_MARGIN_FRAC",
]

