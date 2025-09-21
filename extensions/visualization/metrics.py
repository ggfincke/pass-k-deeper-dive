# ~/extensions/visualization/metrics.py
"""
Metric aggregation utilities for visualization.

Summarizes per-task statistics and computes macro pass@k curves for plotting.
"""

# imports
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import pandas as pd

from extensions.evaluation.metrics import normalize_code


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
    return 1.0 - (nCr(n - c, k) / nCr(n, k))


# Summarize raw rows into per-task metrics with deduplication
def compute_per_task(rows: List[Dict]) -> pd.DataFrame:
    by_task = defaultdict(list)
    for row in rows:
        task_id = row.get("task_id")
        code = row.get("completion", "")
        res = str(row.get("result", "")).strip().lower()
        passed = res == "passed"
        by_task[task_id].append((code, passed))

    records = []
    for tid, items in sorted(by_task.items()):
        n_raw = len(items)
        unique = OrderedDict()
        for code, passed in items:
            key = normalize_code(code)
            unique[key] = unique.get(key, False) or passed
        n_unique = len(unique)
        c_unique = sum(1 for passed in unique.values() if passed)
        dedup_rate = (n_raw - n_unique) / n_raw if n_raw else 0.0

        record = {
            "task_id": tid,
            "n_raw": n_raw,
            "n_unique": n_unique,
            "c_unique": c_unique,
            "duplicates_collapsed": n_raw - n_unique,
            "dedup_rate": dedup_rate,
        }

        for k in range(1, n_unique + 1):
            record[f"pass@{k}_task"] = pass_at_k(n_unique, c_unique, k)
        records.append(record)

    return pd.DataFrame.from_records(records)


# Macro-average pass@k and coverage@k metrics across tasks
def compute_macro(df: pd.DataFrame, max_k: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["k", "coverage@k", "pass@k_macro"])
    if max_k is None:
        max_k = int(df["n_unique"].max())

    rows = []
    for k in range(1, max_k + 1):
        eligible = df[df["n_unique"] >= k]
        coverage = len(eligible) / len(df)
        mean_pass = eligible[f"pass@{k}_task"].mean() if not eligible.empty else float("nan")
        rows.append({"k": k, "coverage@k": coverage, "pass@k_macro": mean_pass})

    return pd.DataFrame(rows)


__all__ = ["compute_per_task", "compute_macro"]

