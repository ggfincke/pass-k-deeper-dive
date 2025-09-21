# ~/extensions/visualizations/metrics.py
"""
Core metric calculations for evaluation including pass@k, coverage@k, and uniqueness metrics.
"""

from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional
import pandas as pd

from .utils import normalize_code


# Calculate binomial coefficient n choose r
def nCr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    numer = 1
    denom = 1
    for i in range(1, r + 1):
        numer *= (n - r + i)
        denom *= i
    return numer // denom if denom else 0


# Calculate pass@k metric: probability that at least one of k samples is correct
def pass_at_k(n: int, c: int, k: int) -> float:
    if k <= 0 or n <= 0 or c < 0 or c > n or k > n:
        return float("nan")
    return 1.0 - (nCr(n - c, k) / nCr(n, k))


# Compute per-task metrics including pass@k for various k values
def compute_per_task(rows: List[Dict]) -> pd.DataFrame:
    by_task = defaultdict(list)
    for r in rows:
        task_id = r.get("task_id")
        code = r.get("completion", "")
        res = str(r.get("result", "")).strip().lower()
        passed = res == "passed"  # tolerate "failed: ..." variants
        by_task[task_id].append((code, passed))
    
    records = []
    for tid, items in sorted(by_task.items()):
        n_raw = len(items)
        unique = OrderedDict()
        for code, p in items:
            k = normalize_code(code)
            unique[k] = unique.get(k, False) or p
        n_unique = len(unique)
        c_unique = sum(1 for p in unique.values() if p)
        dedup_rate = (n_raw - n_unique) / n_raw if n_raw else 0.0
        
        rec = {
            "task_id": tid,
            "n_raw": n_raw,
            "n_unique": n_unique,
            "c_unique": c_unique,
            "duplicates_collapsed": n_raw - n_unique,
            "dedup_rate": dedup_rate,
        }
        
        # add pass@k per task for k up to n_unique
        for k in range(1, n_unique + 1):
            rec[f"pass@{k}_task"] = pass_at_k(n_unique, c_unique, k)
        records.append(rec)
    
    return pd.DataFrame.from_records(records)


# Compute macro-averaged pass@k and coverage@k metrics
def compute_macro(df: pd.DataFrame, max_k: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["k", "coverage@k", "pass@k_macro"])
    if max_k is None:
        max_k = int(df["n_unique"].max())
    
    rows = []
    for k in range(1, max_k + 1):
        eligible = df[df["n_unique"] >= k]
        coverage = len(eligible) / len(df)
        if not eligible.empty:
            mean_pass = eligible[f"pass@{k}_task"].mean()
        else:
            mean_pass = float("nan")
        rows.append({"k": k, "coverage@k": coverage, "pass@k_macro": mean_pass})
    
    return pd.DataFrame(rows)
