# HumanEval Extensions

Experiments for the accompanying article, built on OpenAI's HumanEval harness. This fork adds an Ollama-powered generation pipeline, subset-safe evaluation that reports unbiased pass@k/coverage@k, plotting utilities, and dataset download helpers. See [`originalREADME.md`](originalREADME.md) for upstream documentation.


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
python -m extensions.cli.generate
```

Key defaults (from [`extensions/config.py`](extensions/config.py)):

- `MODEL="gpt-oss:20b"`
- `N_SAMPLES=100` completions per task (set `LIMIT` to cap tasks)
- `EVAL_KS=[1,5,10,25]`
- `CONCURRENCY=5` parallel sampling workers
- `MAX_RETRIES=2` retries for empty completions before falling back to `pass`
- `TEMP=0.2` (override via `PASSK_TEMP` or `PASSK_TEMPERATURE`)

Environment variables override any value; the full list lives in `extensions/config.py`, and `.env-example` shows sane defaults you can copy to `.env` for local overrides. Tip for Windows: the OS fills `TEMP` with a directory path, so set `PASSK_TEMP`/`PASSK_TEMPERATURE` when you actually mean sampling temperature. Numeric `TEMP` values still work when parseable as floats.

### Evaluate an existing JSONL

If you already have completions, the evaluator backfills pass@k/coverage metrics and emits a new JSONL with verdicts.

```python
from extensions.evaluation.functional import evaluate_functional_correctness_subset

metrics = evaluate_functional_correctness_subset("results/samples.jsonl")
print(metrics["pass@1"], metrics["coverage@1"])
```

Outputs land at `results/samples.jsonl_results.jsonl` by default and include naive vs unbiased pass@k, coverage, per-task breakdowns, and optional bootstrap confidence intervals.

### Visualize pass@k trends

The visualization CLI aggregates evaluation output and produces ready-to-drop-in figures.

```bash
python -m extensions.visualization.cli --outdir ./figures
python -m extensions.visualization.cli results/custom_run_results.jsonl --compare baseline.jsonl --labels "temp=0.2" "temp=0.8"
```

Artifacts:

- `figures/per_task_metrics.csv` and `figures/macro_metrics.csv`
- `figures/pass_vs_k_with_coverage.png`, `figures/pass_vs_k_naive_vs_unbiased.png`
- `figures/duplicates_hist.png`
- Optional comparison plots when `--compare` is supplied

### Download and export HumanEval

```bash
python -m extensions.cli.humaneval --outdir ./data/humaneval
```

Produces `HumanEval.jsonl.gz`, `HumanEval.jsonl`, `humaneval.json`, and `humaneval.csv` in the selected directory.

## Generated artifacts

- `results/samples.jsonl` — deduplicated completions (empty ones are replaced with `pass` and logged)
- `results/empty_samples.jsonl` — detail on retries that never produced output
- `results/samples.jsonl_results.jsonl` — evaluation verdicts, including raw/unique pass@k fields and execution traces
- Terminal summary — unbiased pass@k, coverage@k, n_unique histogram, and retry diagnostics

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
