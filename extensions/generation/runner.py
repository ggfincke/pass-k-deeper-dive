# ~/extensions/generation/runner.py
"""
Orchestration helpers for generating HumanEval completions.

Manages retry policy, writes artifacts, and optionally triggers evaluation.
"""

# imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

from human_eval.data import read_problems, write_jsonl

from extensions.clients.ollama import generate as ollama_generate
from extensions.config import CONCURRENCY, K, LIMIT, MAX_RETRIES, TEMP
from extensions.evaluation.functional import evaluate_functional_correctness_subset
from extensions.generation.records import (
    AttemptRecord,
    EmptySampleRecord,
    GenerationResult,
    SampleRecord,
)


# Render ``path`` relative to ``base`` when possible
def _format_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


# Generate completions for HumanEval and optionally evaluate pass@k
def generate_humaneval_completions(
    results_dir: Optional[Path] = None,
    run_evaluation: bool = True,
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

    samples: List[SampleRecord] = []
    empty_samples: List[EmptySampleRecord] = []

    for ti, task_id in enumerate(task_ids):
        prompt = problems[task_id]["prompt"]

        if K <= 0:
            continue

        def _sample_once(sample_idx: int) -> Tuple[SampleRecord, Optional[EmptySampleRecord]]:
            base_seed = 1337 + 1000 * ti + sample_idx
            completion = ""
            attempts: List[AttemptRecord] = []

            for attempt in range(MAX_RETRIES + 1):
                seed = base_seed + attempt
                temperature = TEMP

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

                if attempt_completion.strip():
                    completion = attempt_completion
                    if attempt > 0:
                        print(
                            f"[info] Non-empty completion recovered for {task_id} on retry {attempt}"
                        )
                    break

            empty_record: Optional[EmptySampleRecord] = None
            if not completion.strip():
                print(
                    f"[warn] Empty completion for {task_id} after {MAX_RETRIES + 1} attempts; falling back to 'pass'"
                )
                completion = "    pass\n"
                empty_record = {
                    "task_id": task_id,
                    "resolved": False,
                    "attempts": attempts,
                }
            elif not attempts[0]["completion"].strip():
                empty_record = {
                    "task_id": task_id,
                    "resolved": True,
                    "attempts": attempts,
                    "final_completion": completion,
                }

            return (
                {"task_id": task_id, "completion": completion},
                empty_record,
            )

        futures = {}
        results: Dict[int, Tuple[SampleRecord, Optional[EmptySampleRecord]]] = {}
        max_workers = min(CONCURRENCY, K)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for j in range(K):
                futures[executor.submit(_sample_once, j)] = j

            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

        for j in range(K):
            sample_record, empty_record = results[j]
            samples.append(sample_record)
            if empty_record:
                empty_samples.append(empty_record)

    write_jsonl(str(samples_path), cast(Iterable[Dict[str, object]], samples))
    print(
        "Wrote "
        f"{len(samples)} samples across {len(task_ids)} tasks (k={K}) -> "
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

    metrics: Optional[Dict[str, float]] = None
    if run_evaluation:
        eval_ks = sorted({k for k in (1, K) if k > 0})
        print(f"Evaluating functional correctness for k={eval_ks} ...")
        try:
            metrics = evaluate_functional_correctness_subset(str(samples_path), k=eval_ks)
        except Exception as exc:  # Eval failures don't crash generation
            print(f"[error] Evaluation failed: {exc}")
        else:
            if metrics:
                formatted = ", ".join(
                    f"{metric}={value:.4f}" for metric, value in metrics.items()
                )
                print(f"pass@k -> {formatted}")
            else:
                print(
                    "[warn] Evaluation produced no pass@k metrics; ensure each task has ≥ k samples."
                )

    return {
        "samples_path": samples_path,
        "empty_samples_path": empty_samples_path,
        "metrics": metrics,
    }


__all__ = ["generate_humaneval_completions"]
