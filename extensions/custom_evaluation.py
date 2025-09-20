# ~/extensions/custom_evaluation.py
'''
Subset-friendly HumanEval evaluator with unbiased pass@k, de-duplication,
stable result mapping, and macro-averaging that ignores only tasks with n<k.
'''
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Union
import numpy as np

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness
from schemas import SampleRow
from constants import K

# Normalize for de-duplication
def _normalize_code(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").strip("\n")
    lines = [ln.rstrip() for ln in s.splitlines()]
    return "\n".join(lines)

# Compute unbiased estimator: 1 - C(n-c, k)/C(n, k) for each task (returns NaN for tasks with n < k)
def _estimate_pass_at_k_vector(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(n, np.nan, dtype=float)
    mask = n >= k
    if not np.any(mask):
        return out
    n_v = n[mask]
    c_v = c[mask]
    # Where c == 0, estimator is exactly 0.0
    zero_mask = c_v == 0
    vals = np.zeros_like(n_v, dtype=float)

    # For c > 0, compute 1 - ∏_{j=0..k-1} (n - c - j)/(n - j)
    pos_mask = ~zero_mask
    n_p = n_v[pos_mask].astype(float)
    c_p = c_v[pos_mask].astype(float)
    if n_p.size:
        j = np.arange(k, dtype=float)
        num = np.prod((n_p[:, None] - c_p[:, None] - j) / (n_p[:, None] - j), axis=1)
        vals[pos_mask] = 1.0 - num

    out[mask] = vals
    return out


# Compute basic bootstrap confidence interval over tasks (ignores NaNs)
def _bootstrap_ci(x: np.ndarray, iters: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(1234)
    idx = rng.integers(0, x.size, size=(iters, x.size))
    means = np.nanmean(x[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# evaluate HumanEval pass@k for a subset JSONL of samples
# JSONL format: {"task_id": "...", "completion": "..."}
# returns dict with pass@k, coverage@k metrics
# writes ${sample_file}_results.jsonl with result+passed attached per sample
def evaluate_functional_correctness_subset(
    sample_file: str,
    k: Union[int, List[int]] = [1, K],
    timeout: float = 10.0,
    n_workers: int = 8,
    compute_ci: bool = False,
) -> Dict[str, float]:
    problems = read_problems()

    # Load samples with stable indices
    rows: List[SampleRow] = []
    for i, rec in enumerate(stream_jsonl(sample_file)):
        rows.append(SampleRow(idx=i, task_id=rec["task_id"], completion=rec["completion"]))

    # Execute tests concurrently and maintain index to result mapping
    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for row in rows:
            fut = ex.submit(
                check_correctness,
                problem=problems[row.task_id],
                completion=row.completion,
                timeout=timeout,
            )
            futures[fut] = row

        results_by_idx: Dict[int, Dict] = {}
        for fut in as_completed(futures):
            row = futures[fut]
            res = fut.result()
            # HumanEval returns a dict with "passed" and "result" fields
            results_by_idx[row.idx] = res

    # Group by task and de-duplicate
    per_task_norm_to_firstpass: Dict[str, Dict[str, bool]] = {}
    per_task_all_passes: Dict[str, List[bool]] = {}
    for row in rows:
        res = results_by_idx[row.idx]
        passed = bool(res.get("passed", False))
        key = _normalize_code(row.completion)
        per_task_norm_to_firstpass.setdefault(row.task_id, {})
        per_task_all_passes.setdefault(row.task_id, [])

        # Keep only first occurrence of a normalized completion
        if key not in per_task_norm_to_firstpass[row.task_id]:
            per_task_norm_to_firstpass[row.task_id][key] = passed
            per_task_all_passes[row.task_id].append(passed)

    # Compute n and c per task
    task_ids = sorted(per_task_all_passes.keys())
    n = np.array([len(per_task_all_passes[t]) for t in task_ids], dtype=int)
    c = np.array([sum(per_task_all_passes[t]) for t in task_ids], dtype=int)

    # Build metrics
    ks = [k] if isinstance(k, int) else list(k)
    metrics: Dict[str, float] = {}
    for kk in ks:
        vals = _estimate_pass_at_k_vector(n, c, kk)
        # Macro mean over tasks with n>=kk
        metrics[f"pass@{kk}"] = float(np.nanmean(vals))
        metrics[f"coverage@{kk}"] = float(np.mean(n >= kk))
        if compute_ci:
            lo, hi = _bootstrap_ci(vals)
            metrics[f"ci@{kk}.lo"] = lo
            metrics[f"ci@{kk}.hi"] = hi

    # Write augmented results in input order
    def _combine():
        for row in rows:
            res = results_by_idx[row.idx]
            yield {
                "task_id": row.task_id,
                "completion": row.completion,
                "result": res.get("result", None),
                "passed": bool(res.get("passed", False)),
                "idx": row.idx,
            }

    out_file = sample_file + "_results.jsonl"
    write_jsonl(out_file, _combine())
    return metrics
