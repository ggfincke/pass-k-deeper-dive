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


# Compute the binomial coefficient (n choose r)
def nCr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    numer = 1
    denom = 1
    for i in range(1, r + 1):
        numer *= n - r + i
        denom *= i
    return numer // denom if denom else 0


# Probability that at least one of k samples is correct
def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0 or n <= 0 or c < 0 or c > n or k > n:
        return float("nan")
    denom = nCr(n, k)
    if denom == 0:
        return float("nan")
    return 1.0 - (nCr(n - c, k) / denom)


# Compute unbiased pass@k estimates per task
def estimate_pass_at_k_vector(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(n, np.nan, dtype=float)
    if k <= 0:
        return out

    mask = (n >= k) & (n > 0)
    if not np.any(mask):
        return out

    n_v = n[mask].astype(int)
    c_v = c[mask].astype(int)
    vals = np.array([pass_at_k(int(n_i), int(c_i), k) for n_i, c_i in zip(n_v, c_v)], dtype=float)
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


__all__ = ["normalize_code", "nCr", "pass_at_k", "estimate_pass_at_k_vector", "bootstrap_ci"]
