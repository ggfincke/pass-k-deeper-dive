# ~/extensions/visualization/metrics.py
"""
Metric aggregation utilities for visualization.

Summarizes per-task statistics and computes macro pass@k curves for plotting.
"""

# imports
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import pandas as pd

from extensions.evaluation.metrics import normalize_code, pass_at_k


# Summarize raw rows into per-task metrics with deduplication
def compute_per_task(rows: List[Dict]) -> pd.DataFrame:
    by_task = defaultdict(list)

    # Group results by task and extract pass/fail status
    for row in rows:
        task_id = row.get("task_id")
        code = row.get("completion", "")
        res = str(row.get("result", "")).strip().lower()
        passed = (row.get("passed") is True) or (res == "passed")
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

        # Compute unbiased pass@k on raw samples without deduplication
        for k in range(1, n_raw + 1):
            pass_raw_k = pass_at_k(n_raw, c_raw, k)
            record[f"pass_raw@{k}_task"] = pass_raw_k
        records.append(record)

    return pd.DataFrame.from_records(records)


# Macro-average pass@k and coverage@k metrics across tasks
def compute_macro(df: pd.DataFrame, max_k: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["k", "coverage@k", "pass@k_macro"])
    if max_k is None:
        max_k = int(df["n_unique"].max())

    rows = []
    total_tasks = len(df)
    all_task_ids = df["task_id"].unique()

    # Compute c_unique per task once before k-loop
    c_unique_by_task = (
        df[["task_id", "c_unique"]].set_index("task_id")["c_unique"]
        .reindex(all_task_ids, fill_value=0)
    )

    for k in range(1, max_k + 1):
        # Filter to tasks with sufficient samples (n_raw >= k)
        mask = df["n_raw"] >= k
        tasks_with_k_samples = df[mask]

        if len(tasks_with_k_samples) == 0:
            # No tasks have k samples; report NaN
            mean_unbiased = float("nan")
        else:
            per_task_unbiased: List[float] = []
            for _, row in tasks_with_k_samples.iterrows():
                n_raw = int(row["n_raw"])
                col_unbiased = f"pass_raw@{k}_task"

                value_raw = row.get(col_unbiased, 0.0)
                value_raw = 0.0 if pd.isna(value_raw) else float(value_raw)
                per_task_unbiased.append(value_raw)

            mean_unbiased = sum(per_task_unbiased) / len(per_task_unbiased)

        # Calculate coverage@k using c_unique >= k pattern
        coverage = float((c_unique_by_task >= k).mean())
        rows.append(
            {
                "k": k,
                "coverage@k": coverage,
                "pass@k_macro": mean_unbiased,
            }
        )

    return pd.DataFrame(rows)


__all__ = ["compute_per_task", "compute_macro"]
