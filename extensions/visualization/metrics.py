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
    # Group results by task and extract pass/fail status
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
        c_raw = sum(1 for _, passed in items if passed)
        # Deduplicate code samples while preserving any passing result
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
            "c_raw": c_raw,
            "c_unique": c_unique,
            "duplicates_collapsed": n_raw - n_unique,
            "dedup_rate": dedup_rate,
        }

        # Compute pass@k for all valid k values using unbiased estimator on unique samples
        for k in range(1, n_unique + 1):
            pass_unique_k = pass_at_k(n_unique, c_unique, k)
            record[f"pass@{k}_task"] = pass_unique_k
            record[f"pass_unique@{k}_task"] = pass_unique_k

        # Compute unbiased pass@k on raw samples (without deduplication)
        for k in range(1, n_raw + 1):
            pass_raw_k = pass_at_k(n_raw, c_raw, k)
            record[f"pass_raw@{k}_task"] = pass_raw_k

        # Compute order-dependent naive pass@k using the first k raw samples
        any_so_far = False
        cumulative_any: List[float] = []
        for _, passed in items:
            any_so_far = any_so_far or bool(passed)
            cumulative_any.append(1.0 if any_so_far else 0.0)

        for k in range(1, n_raw + 1):
            record[f"naive_pass@{k}_task"] = cumulative_any[k - 1]
        records.append(record)

    return pd.DataFrame.from_records(records)


# Macro-average pass@k and coverage@k metrics across tasks
def compute_macro(df: pd.DataFrame, max_k: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["k", "coverage@k", "pass@k_macro", "pass@k_macro_naive"]
        )
    if max_k is None:
        max_k = int(df["n_unique"].max())

    rows = []
    total_tasks = len(df)
    for k in range(1, max_k + 1):
        # Clamp per-task pass@k to available completions to ensure monotonic curves
        # while maintaining contribution from all tasks
        per_task_unbiased: List[float] = []
        per_task_naive: List[float] = []
        for _, row in df.iterrows():
            n_unique = int(row["n_unique"])
            n_raw = int(row["n_raw"])

            if n_raw <= 0:
                per_task_unbiased.append(0.0)
            else:
                capped_raw_k = min(k, n_raw)
                col_unbiased = f"pass_raw@{capped_raw_k}_task"
                if col_unbiased in row.index:
                    value_raw = row[col_unbiased]
                    value_raw = 0.0 if pd.isna(value_raw) else float(value_raw)
                else:
                    value_raw = 0.0
                per_task_unbiased.append(float(value_raw))

            if n_raw <= 0:
                per_task_naive.append(0.0)
            else:
                capped_raw_k = min(k, n_raw)
                col_naive = f"naive_pass@{capped_raw_k}_task"
                if col_naive in row.index:
                    value_naive = row[col_naive]
                    value_naive = 0.0 if pd.isna(value_naive) else float(value_naive)
                else:
                    value_naive = 0.0
                per_task_naive.append(float(value_naive))

        # Calculate coverage as fraction of tasks with sufficient unique samples
        coverage = float((df["n_unique"] >= k).mean())
        mean_unbiased = sum(per_task_unbiased) / total_tasks if total_tasks else float("nan")
        mean_naive = sum(per_task_naive) / total_tasks if total_tasks else float("nan")
        rows.append(
            {
                "k": k,
                "coverage@k": coverage,
                "pass@k_macro": mean_unbiased,
                "pass@k_macro_naive": mean_naive,
            }
        )

    return pd.DataFrame(rows)


__all__ = ["compute_per_task", "compute_macro"]
