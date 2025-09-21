# ~/extensions/main.py
'''
Main script for generating code completions using Ollama on HumanEval dataset

Usage:
    python main.py

Outputs:
    - samples.jsonl        : Generated code completions for HumanEval problems
    - empty_samples.jsonl  : Debug info for problems that produced empty completions

'''
from pathlib import Path
from typing import Dict, Iterable, List, Optional, cast
from human_eval.data import read_problems, write_jsonl
from custom_evaluation import evaluate_functional_correctness_subset
from constants import K, LIMIT, TEMP, MAX_RETRIES
from ollama_client import ollama_generate
from schemas import (
    AttemptRecord,
    EmptySampleRecord,
    GenerationResult,
    JsonDict,
    SampleRecord,
)

# Main execution
def main():
    # create results dir
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_path = results_dir / "samples.jsonl"
    empty_samples_path = results_dir / "empty_samples.jsonl"

    # format paths relative to repo root
    def fmt(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    # Read HumanEval problems
    problems: Dict[str, Dict] = read_problems()
    task_ids = sorted(problems.keys())
    if LIMIT is not None:
        task_ids = task_ids[:LIMIT]

    # Generate completions
    samples: List[SampleRecord] = []
    empty_samples: List[EmptySampleRecord] = []
    # For each task, generate K completions with retries for empty outputs
    for ti, tid in enumerate(task_ids):
        prompt = problems[tid]["prompt"]
        # K completions per task
        for j in range(K):
            base_seed = 1337 + 1000 * ti + j
            completion = ""
            attempts: List[AttemptRecord] = []
            # try up to MAX_RETRIES times
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
                raw_response: Optional[object] = result.get("raw_response")
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
                        print(f"[info] Non-empty completion recovered for {tid} on retry {attempt}")
                    break

            if not completion.strip():
                print(
                    f"[warn] Empty completion for {tid} after {MAX_RETRIES + 1} attempts; falling back to 'pass'"
                )
                completion = "    pass\n"
                empty_samples.append({
                    "task_id": tid,
                    "resolved": False,
                    "attempts": attempts,
                })
            elif not attempts[0]["completion"].strip():
                empty_samples.append(
                    {
                        "task_id": tid,
                        "resolved": True,
                        "attempts": attempts,
                        "final_completion": completion,
                    }
                )

            samples.append({"task_id": tid, "completion": completion})

    # Write outputs
    write_jsonl(str(samples_path), cast(Iterable[JsonDict], samples))
    print(
        "Wrote "
        f"{len(samples)} samples across {len(task_ids)} tasks (k={K}) -> {fmt(samples_path)}"
    )

    # Debug info for empty completions (generally caused by reasoning consuming more tokens than given)
    if empty_samples:
        write_jsonl(str(empty_samples_path), cast(Iterable[JsonDict], empty_samples))
        print(
            f"Captured {len(empty_samples)} problem completions -> {fmt(empty_samples_path)}"
        )

    # Run functional evaluation on generated samples and report pass@k
    eval_ks = sorted({k for k in (1, K) if k > 0})
    print(f"Evaluating functional correctness for k={eval_ks} ...")
    try:
        pass_at_k = evaluate_functional_correctness_subset(str(samples_path), k=eval_ks)
    # Eval failures don't crash generation
    except Exception as exc:
        print(f"[error] Evaluation failed: {exc}")
    else:
        if pass_at_k:
            metrics = ", ".join(f"{metric}={value:.4f}" for metric, value in pass_at_k.items())
            print(f"pass@k -> {metrics}")
        else:
            print("[warn] Evaluation produced no pass@k metrics; ensure each task has ≥ k samples.")


if __name__ == "__main__":
    main()
