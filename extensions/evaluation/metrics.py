# ~/extensions/evaluation/metrics.py
"""
Shared metric helpers for evaluating HumanEval completions.

Implements normalization, unbiased pass@k estimation, and bootstrap intervals.
"""

# imports
from typing import Tuple

import numpy as np


# Normalize code for duplicate detection
def normalize_code(source: str) -> str:
    if source is None:
        return ""
    text = source.replace("\r\n", "\n").strip("\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines)


# Compute unbiased pass@k estimates per task
def estimate_pass_at_k_vector(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(n, np.nan, dtype=float)
    mask = n >= k
    if not np.any(mask):
        return out

    n_v = n[mask]
    c_v = c[mask]

    zero_mask = c_v == 0
    vals = np.zeros_like(n_v, dtype=float)

    pos_mask = ~zero_mask
    if np.any(pos_mask):
        n_p = n_v[pos_mask].astype(float)
        c_p = c_v[pos_mask].astype(float)
        j = np.arange(k, dtype=float)
        num = np.prod((n_p[:, None] - c_p[:, None] - j) / (n_p[:, None] - j), axis=1)
        vals[pos_mask] = 1.0 - num

    out[mask] = vals
    return out


# Compute a bootstrap confidence interval over task-level pass@k values
def bootstrap_ci(values: np.ndarray, iters: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    xs = values[~np.isnan(values)]
    if xs.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1234)
    idx = rng.integers(0, xs.size, size=(iters, xs.size))
    means = np.nanmean(xs[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


__all__ = ["normalize_code", "estimate_pass_at_k_vector", "bootstrap_ci"]

