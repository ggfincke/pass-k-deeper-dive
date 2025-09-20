# ~/extensions/main.py
'''
Main script for generating code completions using Ollama on HumanEval dataset

Usage:
    python main.py

Outputs:
    - samples.jsonl        : Generated code completions for HumanEval problems
    - empty_samples.jsonl  : Debug info for problems that produced empty completions

'''
from typing import Dict, Iterable, List, Optional, cast
from human_eval.data import read_problems, write_jsonl
from custom_evaluation import evaluate_functional_correctness_subset
from constants import K, LIMIT, TEMP, RETRY_TEMP, MAX_RETRIES
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
    # Read HumanEval problems
    problems: Dict[str, Dict] = read_problems()
    task_ids = sorted(problems.keys())
    if LIMIT > 0:
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
            augmented_prompt = prompt
            # try up to MAX_RETRIES times
            for attempt in range(MAX_RETRIES + 1):
                seed = base_seed + attempt
                temperature = TEMP if attempt == 0 else RETRY_TEMP
                if attempt > 0:
                    augmented_prompt = (
                        f"{prompt}\n# Reminder: Start each line with 4 spaces for proper indentation. "
                        "Example: '    return None' (4 spaces before 'return')."
                    )

                result = cast(
                    GenerationResult,
                    ollama_generate(augmented_prompt, seed=seed, temperature=temperature),
                )
                raw_text_value = result.get("text")
                raw_text = raw_text_value if isinstance(raw_text_value, str) else ""
                attempt_completion = raw_text
                raw_response: Optional[object] = result.get("raw_response")
                attempts.append(
                    {
                        "prompt": augmented_prompt,
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
    write_jsonl("samples.jsonl", cast(Iterable[JsonDict], samples))
    print(f"Wrote {len(samples)} samples across {len(task_ids)} tasks (k={K}) → samples.jsonl")
    
    # Debug info for empty completions (generally caused by reasoning consuming more tokens than given)
    if empty_samples:
        write_jsonl("empty_samples.jsonl", cast(Iterable[JsonDict], empty_samples))
        print(f"Captured {len(empty_samples)} problem completions → empty_samples.jsonl")

    # Run functional evaluation on generated samples and report pass@k
    eval_ks = sorted({k for k in (1, K) if k > 0})
    print(f"Evaluating functional correctness for k={eval_ks} ...")
    try:
        pass_at_k = evaluate_functional_correctness_subset("samples.jsonl", k=eval_ks)
    # Eval failures don't crash generation
    except Exception as exc:
        print(f"[error] Evaluation failed: {exc}")
    else:
        if pass_at_k:
            metrics = ", ".join(f"{metric}={value:.4f}" for metric, value in pass_at_k.items())
            print(f"pass@k → {metrics}")
        else:
            print("[warn] Evaluation produced no pass@k metrics; ensure each task has ≥ k samples.")


if __name__ == "__main__":
    main()
