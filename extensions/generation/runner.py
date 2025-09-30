# ~/extensions/generation/runner.py
"""
Orchestration helpers for generating HumanEval completions.

Manages retry policy, writes artifacts, and optionally triggers evaluation.
"""

# imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import time

from human_eval.data import read_problems, write_jsonl

from extensions.clients.ollama import generate as ollama_generate
from extensions.config import (
    CONCURRENCY,
    EVAL_KS,
    EMPTY_COMPLETION_BACKOFF_BASE,
    EMPTY_COMPLETION_BACKOFF_MAX,
    EMPTY_COMPLETION_MAX_RETRIES,
    LIMIT,
    MAX_RETRIES,
    N_SAMPLES,
    TEMP,
)
from extensions.evaluation.functional import evaluate_functional_correctness_subset
from extensions.generation.records import (
    AttemptRecord,
    EmptySampleRecord,
    GenerationResult,
    SampleRecord,
)


# Format path as relative to base directory for display
def _format_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# Generate completions for HumanEval and optionally evaluate pass@k
def generate_humaneval_completions(
    results_dir: Optional[Path] = None,
    run_evaluation: bool = True,
    verbose: bool = False,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = Path(results_dir) if results_dir is not None else repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    samples_path = results_dir / "samples.jsonl"
    empty_samples_path = results_dir / "empty_samples.jsonl"

    problems: Dict[str, Dict] = read_problems()
    task_ids = sorted(problems.keys())
    if LIMIT is not None:
        task_ids = task_ids[:LIMIT]

    total_tasks = len(task_ids)

    samples: List[SampleRecord] = []
    empty_samples: List[EmptySampleRecord] = []

    # Print debug message when verbose mode is enabled
    def _debug(message: str) -> None:
        if verbose:
            print(f"[debug] {message}", flush=True)

    _debug(
        "Resolved paths -> "
        f"results_dir={_format_relative(results_dir, repo_root)}, "
        f"samples={_format_relative(samples_path, repo_root)}, "
        f"empty_samples={_format_relative(empty_samples_path, repo_root)}"
    )
    max_generation_workers = max(1, min(CONCURRENCY, N_SAMPLES))

    _debug(
        "Config -> "
        f"N_SAMPLES={N_SAMPLES}, CONCURRENCY={CONCURRENCY}, LIMIT={LIMIT}, TEMP={TEMP}, "
        f"MAX_RETRIES={MAX_RETRIES}, EMPTY_COMPLETION_MAX_RETRIES={EMPTY_COMPLETION_MAX_RETRIES}"
    )
    _debug(
        "Generation parameters -> "
        f"TOTAL_TASKS={len(problems)}, ACTIVE_TASKS={total_tasks}, RUN_EVAL={run_evaluation}"
    )
    if verbose and task_ids:
        preview_ids = ", ".join(task_ids[:5])
        if len(task_ids) > 5:
            preview_ids += ", ..."
        _debug(f"Task preview -> {preview_ids}")

    with ThreadPoolExecutor(max_workers=max_generation_workers) as executor:
        for ti, task_id in enumerate(task_ids):
            prompt = problems[task_id]["prompt"]

            if N_SAMPLES <= 0:
                continue

            _debug(f"{task_id}: scheduling {N_SAMPLES} sample(s) with base index {ti}")

            # Generate a single completion with retry logic for empty results
            def _sample_once(sample_idx: int) -> Tuple[SampleRecord, Optional[EmptySampleRecord]]:
                base_seed = 1337 + 1000 * ti + sample_idx
                completion = ""
                attempts: List[AttemptRecord] = []
                _debug(f"{task_id}[sample={sample_idx}] base_seed={base_seed}")

                resolved_attempt: Optional[int] = None
                retry_budget = EMPTY_COMPLETION_MAX_RETRIES
                for attempt in range(retry_budget + 1):
                    seed = base_seed + attempt
                    temperature = TEMP

                    _debug(
                        f"{task_id}[sample={sample_idx}] attempt={attempt} seed={seed} temp={temperature}"
                    )

                    result = cast(
                        GenerationResult,
                        ollama_generate(prompt, seed=seed, temperature=temperature),
                    )
                    raw_text_value = result.get("text")
                    raw_text = raw_text_value if isinstance(raw_text_value, str) else ""
                    attempt_completion = raw_text
                    raw_response = result.get("raw_response")

                    attempts.append(
                        {
                            "prompt": prompt,
                            "seed": seed,
                            "temperature": temperature,
                            "raw_text": raw_text,
                            "raw_response": raw_response,
                            "completion": attempt_completion,
                        }
                    )

                    status = "non-empty" if attempt_completion.strip() else "empty"
                    snippet: str = ""
                    if attempt_completion.strip():
                        first_line = attempt_completion.strip().splitlines()[0]
                        snippet = first_line[:80]
                    if snippet:
                        _debug(
                            f"{task_id}[sample={sample_idx}] attempt={attempt} -> {status}; first line: {snippet!r}"
                        )
                    else:
                        _debug(f"{task_id}[sample={sample_idx}] attempt={attempt} -> {status}")

                    if attempt_completion.strip():
                        completion = attempt_completion
                        resolved_attempt = attempt
                        if attempt > 0:
                            print(
                                f"[info] Non-empty completion recovered for {task_id} on retry {attempt}"
                            )
                        break

                    if attempt < retry_budget:
                        backoff = min(
                            EMPTY_COMPLETION_BACKOFF_MAX,
                            EMPTY_COMPLETION_BACKOFF_BASE * (2 ** attempt),
                        )
                        if backoff > 0:
                            time.sleep(backoff)

                empty_record: Optional[EmptySampleRecord] = None
                if not completion.strip():
                    print(
                        f"[warn] Empty completion for {task_id} after {retry_budget + 1} attempts; falling back to 'pass'"
                    )
                    completion = "    pass\n"
                    empty_record = {
                        "task_id": task_id,
                        "resolved": False,
                        "attempts": attempts,
                    }
                    _debug(f"{task_id}[sample={sample_idx}] flagged as empty after retries")
                elif not attempts[0]["completion"].strip():
                    empty_record = {
                        "task_id": task_id,
                        "resolved": True,
                        "attempts": attempts,
                        "final_completion": completion,
                    }
                    _debug(
                        f"{task_id}[sample={sample_idx}] recovered from empty initial attempt at attempt={resolved_attempt}"
                    )

                if resolved_attempt is not None:
                    _debug(
                        f"{task_id}[sample={sample_idx}] accepted completion from attempt={resolved_attempt}"
                    )

                return (
                    {"task_id": task_id, "completion": completion},
                    empty_record,
                )

            futures = {}
            results: Dict[int, Tuple[SampleRecord, Optional[EmptySampleRecord]]] = {}

            try:
                for j in range(N_SAMPLES):
                    future = executor.submit(_sample_once, j)
                    futures[future] = j

                for fut in as_completed(futures):
                    idx = futures.pop(fut)
                    try:
                        results[idx] = fut.result()
                    except Exception as e:
                        # Handle individual generation failures gracefully
                        _debug(f"{task_id}[sample={idx}] failed with error: {e}")
                        results[idx] = (
                            {"task_id": task_id, "completion": "    pass\n"},
                            {
                                "task_id": task_id,
                                "resolved": False,
                                "attempts": [{"error": str(e)}],
                            }
                        )
            except Exception:
                for future in list(futures):
                    future.cancel()
                raise

            for j in range(N_SAMPLES):
                sample_record, empty_record = results[j]
                samples.append(sample_record)
                if empty_record:
                    empty_samples.append(empty_record)

            _debug(
                f"{task_id}: collected {N_SAMPLES} sample(s); empty recovery count={sum(1 for _, rec in results.values() if rec)}"
            )

            completed = ti + 1
            if total_tasks > 0:
                percent = (completed / total_tasks) * 100
                print(
                    f"[progress] Completed {completed}/{total_tasks} tasks ({percent:5.1f}%)",
                    flush=True,
                )

    write_jsonl(str(samples_path), cast(Iterable[Dict[str, object]], samples))
    print(
        "Wrote "
        f"{len(samples)} samples across {len(task_ids)} tasks (n_total={N_SAMPLES}) -> "
        f"{_format_relative(samples_path, repo_root)}"
    )

    if empty_samples:
        write_jsonl(
            str(empty_samples_path),
            cast(Iterable[Dict[str, object]], empty_samples),
        )
        print(
            "Captured "
            f"{len(empty_samples)} problem completions -> "
            f"{_format_relative(empty_samples_path, repo_root)}"
        )
        _debug(f"Empty sample records persisted for {len(empty_samples)} task(s)")

    metrics: Optional[Dict[str, object]] = None
    if run_evaluation:
        eval_ks = sorted({k for k in EVAL_KS if k > 0})
        print(f"Evaluating functional correctness for k={eval_ks} ...")
        _debug(f"Evaluation request -> samples_path={samples_path}, eval_ks={eval_ks}")
        try:
            metrics = evaluate_functional_correctness_subset(str(samples_path), k=eval_ks)
        except Exception as exc:  # Eval failures don't crash generation
            print(f"[error] Evaluation failed: {exc}")
            _debug("Evaluation raised exception; continuing without metrics")
        else:
            if metrics:
                per_task_counts = cast(Dict[str, Dict[str, int]], metrics.get("per_task_counts", {}))
                unique_hist = cast(Dict[int, int], metrics.get("n_unique_histogram", {}))
                n_unique_stats = cast(Dict[str, Optional[float]], metrics.get("n_unique_stats", {}))

                scalar_metrics = {
                    metric: value
                    for metric, value in metrics.items()
                    if metric not in {"per_task_counts", "n_unique_histogram", "n_unique_stats"}
                }

                pass_entries = []
                naive_entries = []
                coverage_entries = []
                for kk in eval_ks:
                    pass_key = f"pass@{kk}"
                    naive_key = f"naive_pass@{kk}"
                    coverage_key = f"coverage@{kk}"
                    if pass_key in scalar_metrics:
                        pass_entries.append(f"{pass_key}={scalar_metrics[pass_key]:.4f}")
                    if naive_key in scalar_metrics:
                        naive_entries.append(
                            f"{naive_key}={scalar_metrics[naive_key]:.4f}"
                        )
                    if coverage_key in scalar_metrics:
                        coverage_entries.append(
                            f"{coverage_key}={scalar_metrics[coverage_key]:.4f}"
                        )

                if pass_entries:
                    print(f"pass@k (unbiased) -> {', '.join(pass_entries)}")
                if naive_entries:
                    print(f"pass@k (naive) -> {', '.join(naive_entries)}")
                if coverage_entries:
                    print(f"coverage@k -> {', '.join(coverage_entries)}")

                if per_task_counts:
                    observed_budget = sorted({info["n_total"] for info in per_task_counts.values()})
                    budget_label = observed_budget[0] if len(observed_budget) == 1 else observed_budget
                    print(f"Observed n_total per task -> {budget_label}")

                    hist_str = ", ".join(
                        f"{unique}:{count}"
                        for unique, count in sorted(unique_hist.items())
                    ) if unique_hist else ""
                    print(
                        f"n_unique histogram -> {{{hist_str}}}"
                        if hist_str
                        else "n_unique histogram -> {}"
                    )

                    total_unique = sum(info["n_unique"] for info in per_task_counts.values())
                    total_samples = sum(info["n_total"] for info in per_task_counts.values())
                    unique_rate = (total_unique / total_samples) if total_samples else 0.0
                    mean_unique = (
                        total_unique / len(per_task_counts)
                        if per_task_counts
                        else 0.0
                    )
                    print(
                        f"Unique stats -> mean_n_unique={mean_unique:.2f}, unique_rate={unique_rate:.4f}"
                    )

                    if n_unique_stats:
                        detail = ", ".join(
                            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                            for key, value in (
                                ("min", n_unique_stats.get("min")),
                                ("median", n_unique_stats.get("median")),
                                ("mean", n_unique_stats.get("mean")),
                                ("max", n_unique_stats.get("max")),
                            )
                        )
                        print(f"n_unique stats -> {detail}")

                        max_unique = n_unique_stats.get("max")
                        if max_unique is not None and eval_ks:
                            max_eval = max(eval_ks)
                            if max_unique < max_eval:
                                print(
                                    f"[warn] Coverage collapse: max(EVAL_KS)={max_eval} exceeds observed max n_unique={int(max_unique)}"
                                )

                _debug(f"Evaluation metrics -> {metrics}")
            else:
                print(
                    "[warn] Evaluation produced no pass@k metrics; ensure each task has ≥ k samples."
                )
                _debug("Evaluation returned empty metrics")
    else:
        _debug("Evaluation disabled via run_evaluation flag")

    return {
        "samples_path": samples_path,
        "empty_samples_path": empty_samples_path,
        "metrics": metrics,
    }


__all__ = ["generate_humaneval_completions"]
