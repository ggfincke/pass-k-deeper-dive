# HumanEval Extensions

Experiments for the accompanying article, built on OpenAI's HumanEval harness. This fork adds an Ollama-powered generation pipeline, subset-safe evaluation that reports unbiased pass@k/coverage@k, plotting utilities, and dataset download helpers. See [`originalREADME.md`](originalREADME.md) for upstream documentation.

**Article:** [How pass@k is used to evaluate LLM coding performance](https://medium.com/@ggfincke/how-pass-k-is-used-to-evaluate-llm-coding-performance-296e5c4565bc)

## Getting Started

### 1) Prerequisites

- Python 3.10–3.12
- [Ollama](https://ollama.com/) running locally with at least one model installed (default: [`gpt-oss:20b`](https://ollama.com/library/gpt-oss:20b))
- Git and a POSIX shell

### 2) Environment setup

Creating a dedicated environment helps isolate dependencies:

```bash
conda create -n humaneval python=3.12
conda activate humaneval
```

### 3) Install the repo

```bash
git clone https://github.com/ggfincke/pass-k-deeper-dive
cd pass-k-deeper-dive
pip install -e .  # installs upstream HumanEval + these extensions
```

### 4) Start Ollama and pull a model

```bash
ollama serve &
ollama pull gpt-oss:20b  # needs ~13 GB disk
```

## Running experiments

### Generate + evaluate in one step

[`extensions/cli/generate.py`](extensions/cli/generate.py) reads HumanEval prompts, samples completions, retries empty responses, writes artifacts, and then calls the evaluator.

```bash
python -m extensions.cli.generate          # Standard run
python -m extensions.cli.generate -v       # Verbose mode with debug output
```

Key defaults (from [`extensions/config.py`](extensions/config.py)):

- `MODEL="gpt-oss:20b"` — Ollama model to use
- `N_SAMPLES=100` — Completions per task
- `PROBLEM_LIMIT=None` — Max tasks to process (None = all 164 HumanEval tasks; legacy `LIMIT` still works)
- `EVAL_KS=[1,5,10,25]` — k-values for pass@k/coverage@k computation
- `CONCURRENCY=5` — Parallel sampling workers
- `MAX_RETRIES=2` — Retry budget for empty completions before falling back to `pass`
- `TEMP=0.2` — Sampling temperature (override via `PASSK_TEMP` or `PASSK_TEMPERATURE`)
- `MAX_NEW_TOKENS=131072` — Token generation limit per completion
- `OLLAMA_HTTP_TIMEOUT=120.0` — Timeout per Ollama API call (seconds)
- `OLLAMA_HTTP_MAX_RETRIES=3` — Retry attempts for throttled (429/503) Ollama responses
- `EMPTY_COMPLETION_MAX_RETRIES=2` — Specific retry limit for empty completions
- `EMPTY_COMPLETION_BACKOFF_BASE=0.1` — Initial backoff between empty retries (seconds)

Environment variables override any value; the full list lives in [`extensions/config.py`](extensions/config.py), and [`.env-example`](.env-example) shows sane defaults you can copy to `.env` for local overrides.

**Windows tip:** The OS sets `TEMP` to a directory path, so use `PASSK_TEMP` or `PASSK_TEMPERATURE` when you actually mean sampling temperature. Numeric `TEMP` values still work when parseable as floats.

### Evaluate an existing JSONL

If you already have completions, the evaluator backfills pass@k/coverage metrics and emits a new JSONL with verdicts.

```python
from extensions.evaluation.functional import evaluate_functional_correctness_subset

metrics = evaluate_functional_correctness_subset("results/samples.jsonl")
print(metrics["pass@1"], metrics["coverage@1"])
```

Outputs land at `results/samples.jsonl_results.jsonl` by default and include unbiased pass@k, coverage@k, per-task breakdowns, and optional bootstrap confidence intervals.

### Visualize pass@k trends

The visualization CLI aggregates evaluation output and produces CSV metrics and comparison plots.

```bash
# Generate figures from default results file
python -m extensions.visualization.cli --outdir ./figures

# Compare multiple runs with custom labels
python -m extensions.visualization.cli results/custom_run_results.jsonl \
  --compare baseline_results.jsonl \
  --labels "temp=0.2" "temp=0.8" \
  --outdir ./figures
```

Artifacts:

- `per_task_metrics.csv` — Detailed metrics for each HumanEval task
- `macro_metrics.csv` — Aggregated pass@k and coverage@k across all tasks
- `plot_pass_vs_k_unbiased_comparison.png` — Pass@k curves (with comparison when `--compare` is used)
- `plot_coverage_vs_k_comparison.png` — Coverage@k curves (with comparison when `--compare` is used)

### Download and export HumanEval

```bash
python -m extensions.cli.humaneval --outdir ./data/humaneval
```

Produces `HumanEval.jsonl.gz`, `HumanEval.jsonl`, `humaneval.json`, and `humaneval.csv` in the selected directory.

## Generated artifacts

### From generation (`python -m extensions.cli.generate`):
- `results/samples.jsonl` — Deduplicated completions (empty ones are replaced with `pass` and logged)
- `results/empty_samples.jsonl` — Debug info on retries that never produced output
- `results/samples.jsonl_results.jsonl` — Evaluation verdicts with pass@k/coverage@k metrics and execution traces
- Terminal summary — Unbiased pass@k, coverage@k, unique sample counts, and retry diagnostics

### From visualization (`python -m extensions.visualization.cli`):
- `per_task_metrics.csv` — Per-task pass@k and coverage@k breakdown
- `macro_metrics.csv` — Aggregated metrics across all tasks
- `plot_pass_vs_k_unbiased_comparison.png` — Pass@k curve plot
- `plot_coverage_vs_k_comparison.png` — Coverage@k curve plot

### Pre-generated results:
- `article_results/` — Evaluation results from experiments used in the accompanying article

## Project layout

```
extensions/
  cli/            # Entry points for generation, dataset export, visualization
  clients/        # Ollama HTTP wrapper with retry/concurrency guards
  config.py       # Environment-driven configuration shared across modules
  datasets/       # HumanEval acquisition and export helpers
  evaluation/     # Subset-safe evaluator + metrics and normalization utilities
  generation/     # Sampling orchestration, record schemas, retry logic
  visualization/  # Aggregation + plotting helpers for pass@k curves
```

## Reproducibility tips

- Keep temperature/top-p/top-k fixed across runs to isolate model variance.
- Prefer `LIMIT=None` (or `0`) when you need reliable coverage at higher k.
- Deduplication is automatic; monitor `n_unique` vs `n_raw` in the macro CSV to size future runs.
- Choose `N_SAMPLES` so `n_unique` comfortably exceeds your largest `k` (for example, if `k=25` and unique rate is ~0.16, aim for `N_SAMPLES` around 150–200).

## Troubleshooting

- Empty or truncated completions -> increase `MAX_NEW_TOKENS`, inspect `results/empty_samples.jsonl`.
- `malloc: can't allocate region` / OOM -> free memory/VRAM, lower concurrency, or run a smaller Ollama model.
- Ollama connection refused -> ensure `ollama serve` is active and `OLLAMA_URL` matches (`http://localhost:11434` by default).

## Citation

If you use this repo, please cite the original HumanEval paper (see exact citation in [`originalREADME.md`](originalREADME.md)).

## License

I maintain the same MIT license from the source repo.
