# HumanEval Extensions

Reproducible experiments for the accompanying article. This repo builds on OpenAI's HumanEval harness by adding a local generation pipeline (via Ollama), a subset-safe evaluator with unbiased pass@k and coverage@k, visualization tools, and dataset helpers. See [`originalREADME.md`](https://github.com/ggfincke/pass-k-deeper-dive/blob/master/originalREADME.md) for upstream details.

# Getting Started

## 1) Prereqs

- Python 3.10–3.12
- An [Ollama](https://ollama.com/) server with at least one model available (default: [gpt-oss:20b](https://ollama.com/library/gpt-oss:20b))
- Git + a POSIX shell

## 2) Setup Environment & Install

**Recommended: Create a conda environment**
```bash
conda create -n humaneval python=3.12
conda activate humaneval
```

**Install the repository**
```bash
git clone https://github.com/ggfincke/pass-k-deeper-dive
cd pass-k-deeper-dive
pip install -e .  # installs HumanEval + this repo as editable
```

## 3) Start Ollama & pull a model
```bash
ollama serve &
ollama pull gpt-oss:20b  # requires about 13GB of storage
```

## 4) Run a small experiment
```python
# Customize by editing 'extensions/config.py'
# Default values from config.py:
MODEL = "gpt-oss:20b"
N_SAMPLES = 5             # generation budget (completions sampled per task)
EVAL_KS = [1, 5, 10, 25]  # pass@k / coverage@k evaluation targets
LIMIT = 10                # None means all 164 tasks
MAX_NEW_TOKENS = 131072   # max context window in gpt-oss
DEFAULT_TEMP = 0.2        # override with PASSK_TEMP or PASSK_TEMPERATURE
```

> Windows tip: the OS populates `TEMP` with a temp-directory path. When you want a different sampling temperature, set `PASSK_TEMP` or `PASSK_TEMPERATURE`. Numeric `TEMP` values are still honored when they parse as floats.

Kick off a full generation + evaluation pass:
```bash
python -m extensions.cli.generate
```

## Artifacts

- **results/samples.jsonl** — all non-empty completions
- **results/empty_samples.jsonl** — tasks that needed a fallback stub (for debugging)
- **results/samples.jsonl_results.jsonl** — evaluation verdicts for each sample
- **Console** — unbiased pass@k / coverage@k (for each k in `EVAL_KS`) plus n_unique diagnostics

# Usage

### A) Generate with Ollama and evaluate automatically

[`extensions/cli/generate.py`](extensions/cli/generate.py) reads tasks, samples `N_SAMPLES` completions (with retries on empty), writes artifacts, then runs the evaluator for you.
```bash
python -m extensions.cli.generate
```
Tune via environment variables documented in [`extensions/config.py`](extensions/config.py):
- `OLLAMA_URL` (default http://localhost:11434)
- `MODEL` (default gpt-oss:20b)
- `N_SAMPLES`, `EVAL_KS`, `LIMIT` (0 -> all, else <= 164), `TEMP` (override via `PASSK_TEMP`/`PASSK_TEMPERATURE`), `TOP_P`, `TOP_K`, `REPEAT_PENALTY`
- `MAX_NEW_TOKENS`, `MAX_RETRIES`, `STOP`

### B) Evaluate an existing JSONL (subset-safe)

If you already have completions (from this repo or elsewhere), call:
```python
from extensions.evaluation.functional import evaluate_functional_correctness_subset

# Uses EVAL_KS from config.py by default; override via k=[...] if desired
evaluate_functional_correctness_subset("results/samples.jsonl")
```
This writes `results/samples.jsonl_results.jsonl` with per-sample pass/fail verdicts and prints pass@k, coverage@k, and n_unique diagnostics.

### C) Download and export HumanEval
```bash
python -m extensions.cli.humaneval --outdir ./data/humaneval
# Produces: HumanEval.jsonl.gz, HumanEval.jsonl, humaneval.json, humaneval.csv
```

### D) Visualize pass@k trends
```bash
python -m extensions.visualization.cli --outdir ./figures
python -m extensions.visualization.cli results/custom_run_results.jsonl --compare baseline.jsonl --labels "temp=0.2" "temp=0.8"
```
Exports CSV summaries plus PNG charts in the chosen outdir.

# Project Layout

```
extensions/
  cli/               # Command-line entrypoints (generate, humaneval, visualization)
  clients/           # External service clients (Ollama HTTP wrapper)
  config.py          # Environment-driven configuration shared across modules
  datasets/          # Dataset acquisition utilities
  evaluation/        # pass@k evaluation and metric helpers
  generation/        # Sampling orchestration and record schemas
  visualization/     # Metrics aggregation and plotting helpers
```

# Reproducibility tips

- **Sampling variance**: pass@k moves with temperature/top-p/top-k; keep these fixed when comparing models.
- **Subsets**: prefer larger `LIMIT` (or 0) so more tasks meet n ≥ k.
- **Duplicates**: completions are normalized and deduped before counting to avoid inflated pass@k.
- **Budgeting samples**: pick `N_SAMPLES` so expected `n_unique ≥ max(EVAL_KS)`; e.g., with a 0.16 unique rate and `k=25`, target `N_SAMPLES ≈ 150–200`.

# Troubleshooting

### Empty or truncated completions
Model "thinking" can produce empty output while still consuming tokens. Try increasing `MAX_NEW_TOKENS`, inspect `empty_samples.jsonl`.

### "malloc: can't allocate region" / OOM
Free RAM/VRAM, reduce parallel load, or try a smaller model.

### Ollama connection refused
Ensure `ollama serve` is running and `OLLAMA_URL` matches (default http://localhost:11434). See [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/README.md) for setup details.

# Citation

If you use this repo, please cite the original HumanEval paper (see exact citation in [`originalREADME.md`](originalREADME.md)).

# License

I maintain the same MIT license from the source repo.
