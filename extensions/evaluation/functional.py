# ~/extensions/evaluation/functional.py
"""
Functional correctness evaluation for HumanEval samples.

Loads generated completions, executes reference tests, and reports pass@k metrics.
"""

# imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union

import numpy as np

from human_eval.data import read_problems, stream_jsonl, write_jsonl
from human_eval.execution import check_correctness

from extensions.config import K
from extensions.evaluation.metrics import (
    bootstrap_ci,
    estimate_pass_at_k_vector,
    normalize_code,
)
from extensions.generation.records import SampleRow


# Evaluate pass@k metrics for a subset of HumanEval samples
def evaluate_functional_correctness_subset(
    sample_file: str,
    k: Union[int, List[int]] = (1, K),
    timeout: float = 10.0,
    n_workers: int = 8,
    compute_ci: bool = False,
) -> Dict[str, float]:
    problems = read_problems()

    rows: List[SampleRow] = []
    for idx, rec in enumerate(stream_jsonl(sample_file)):
        rows.append(SampleRow(idx=idx, task_id=rec["task_id"], completion=rec["completion"]))

    futures = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for row in rows:
            futures[executor.submit(
                check_correctness,
                problem=problems[row.task_id],
                completion=row.completion,
                timeout=timeout,
            )] = row

        results_by_idx: Dict[int, Dict] = {}
        for fut in as_completed(futures):
            row = futures[fut]
            results_by_idx[row.idx] = fut.result()

    per_task_norm_to_firstpass: Dict[str, Dict[str, bool]] = {}
    per_task_all_passes: Dict[str, List[bool]] = {}

    for row in rows:
        res = results_by_idx[row.idx]
        passed = bool(res.get("passed", False))
        key = normalize_code(row.completion)
        per_task_norm_to_firstpass.setdefault(row.task_id, {})
        per_task_all_passes.setdefault(row.task_id, [])

        if key not in per_task_norm_to_firstpass[row.task_id]:
            per_task_norm_to_firstpass[row.task_id][key] = passed
            per_task_all_passes[row.task_id].append(passed)

    task_ids = sorted(per_task_all_passes.keys())
    n = np.array([len(per_task_all_passes[tid]) for tid in task_ids], dtype=int)
    c = np.array([sum(per_task_all_passes[tid]) for tid in task_ids], dtype=int)

    ks = [k] if isinstance(k, int) else list(k)
    metrics: Dict[str, float] = {}
    for kk in ks:
        vals = estimate_pass_at_k_vector(n, c, kk)
        metrics[f"pass@{kk}"] = float(np.nanmean(vals))
        metrics[f"coverage@{kk}"] = float(np.mean(n >= kk))
        if compute_ci:
            lo, hi = bootstrap_ci(vals)
            metrics[f"ci@{kk}.lo"] = lo
            metrics[f"ci@{kk}.hi"] = hi

    # Yield evaluation rows with verdicts
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

