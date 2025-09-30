# ~/extensions/evaluation/functional.py
"""
Functional correctness evaluation for HumanEval samples.

Loads generated completions, executes reference tests, and reports pass@k metrics.
"""

# imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.execution import SANDBOX_PROCESS_LIMIT, check_correctness

from extensions.config import EVAL_KS, MAX_EVAL_WORKERS, MAX_EXECUTION_TIMEOUT
from extensions.evaluation.metrics import (
    bootstrap_ci,
    estimate_pass_at_k_vector,
    normalize_code,
)
from extensions.generation.records import SampleRow


# Evaluate pass@k metrics for a subset of HumanEval samples
def evaluate_functional_correctness_subset(
    sample_file: str,
    k: Optional[Union[int, Sequence[int]]] = None,
    timeout: float = 10.0,
    n_workers: Optional[int] = None,
    compute_ci: bool = False,
) -> Dict[str, object]:
    problems = read_problems()
    
    # Use configured max workers if n_workers not specified
    if n_workers is None:
        n_workers = MAX_EVAL_WORKERS

    n_workers = max(1, min(n_workers, MAX_EVAL_WORKERS, SANDBOX_PROCESS_LIMIT))
    
    # Clamp timeout to maximum allowed
    effective_timeout = min(timeout, MAX_EXECUTION_TIMEOUT)

    rows: List[SampleRow] = []
    for idx, rec in enumerate(stream_jsonl(sample_file)):
        rows.append(SampleRow(idx=idx, task_id=rec["task_id"], completion=rec["completion"]))

    futures = {}
    results_by_idx: Dict[int, Dict] = {}
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        try:
            for row in rows:
                future = executor.submit(
                    check_correctness,
                    problem=problems[row.task_id],
                    completion=row.completion,
                    timeout=effective_timeout,
                )
                futures[future] = row

            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    results_by_idx[row.idx] = fut.result()
                except Exception as e:
                    # Handle individual task failures gracefully
                    results_by_idx[row.idx] = {
                        "passed": False,
                        "result": f"execution_error: {str(e)}",
                        "task_id": row.task_id,
                    }
                finally:
                    # Clean up future reference to prevent memory leaks
                    del futures[fut]
        except Exception:
            # Cancel remaining futures if something goes wrong
            for future in futures:
                future.cancel()
            raise

    per_task_norm_to_firstpass: Dict[str, Dict[str, bool]] = defaultdict(dict)
    per_task_unique_passes: Dict[str, List[bool]] = defaultdict(list)
    per_task_total_counts: Counter[str] = Counter()

    for row in rows:
        res = results_by_idx[row.idx]
        passed = bool(res.get("passed", False))
        key = normalize_code(row.completion)
        per_task_total_counts[row.task_id] += 1

        if key not in per_task_norm_to_firstpass[row.task_id]:
            per_task_norm_to_firstpass[row.task_id][key] = passed
            per_task_unique_passes[row.task_id].append(passed)

    task_ids = sorted(per_task_total_counts.keys())
    n_unique = np.array([len(per_task_unique_passes[tid]) for tid in task_ids], dtype=int)
    c_unique = np.array([sum(per_task_unique_passes[tid]) for tid in task_ids], dtype=int)

    resolved_k: Sequence[int]
    if k is None:
        resolved_k = EVAL_KS
    elif isinstance(k, int):
        resolved_k = [k]
    else:
        resolved_k = list(k)

    ks = sorted({kk for kk in resolved_k if kk > 0})

    # Compute naive pass@k using empirical success rate without correction
    def _naive_pass_at_k_vector(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(n, np.nan, dtype=float)
        mask = n > 0
        if not np.any(mask):
            return out

        n_f = n[mask].astype(float)
        c_f = c[mask].astype(float)
        vals = 1.0 - np.power(1.0 - (c_f / n_f), k)
        out[mask] = np.clip(vals, 0.0, 1.0)
        return out

    metrics: Dict[str, object] = {}
    for kk in ks:
        vals = estimate_pass_at_k_vector(n_unique, c_unique, kk)
        metrics[f"pass@{kk}"] = float(np.nanmean(vals))
        metrics[f"coverage@{kk}"] = float(np.mean(n_unique >= kk))
        if compute_ci:
            lo, hi = bootstrap_ci(vals)
            metrics[f"ci@{kk}.lo"] = lo
            metrics[f"ci@{kk}.hi"] = hi

        naive_vals = _naive_pass_at_k_vector(n_unique, c_unique, kk)
        metrics[f"naive_pass@{kk}"] = float(np.nanmean(naive_vals))

    per_task_counts = {
        tid: {
            "n_total": per_task_total_counts[tid],
            "n_unique": len(per_task_unique_passes[tid]),
            "n_correct_unique": sum(per_task_unique_passes[tid]),
        }
        for tid in task_ids
    }

    unique_hist = Counter(counts["n_unique"] for counts in per_task_counts.values())
    metrics["per_task_counts"] = per_task_counts
    metrics["n_unique_histogram"] = dict(sorted(unique_hist.items()))

    if task_ids:
        n_unique_stats = {
            "min": int(n_unique.min()),
            "median": float(np.median(n_unique)),
            "mean": float(np.mean(n_unique)),
            "max": int(n_unique.max()),
        }
    else:
        n_unique_stats = {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }

    metrics["n_unique_stats"] = n_unique_stats

    # Merge sample rows with evaluation results for output
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


__all__ = ["evaluate_functional_correctness_subset"]
