# HumanEval Extensions

Reproducible experiments for the accompanying article. This repo builds on OpenAI's HumanEval harness by adding a local generation pipeline (via Ollama), a subset-safe evaluator with unbiased pass@k and coverage@k, and small dataset utilities. See [`originalREADME.md`](https://github.com/ggfincke/pass-k-deeper-dive/blob/master/originalREADME.md) for upstream details.

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
ollama pull gpt-oss:20b # requires about 13GB of storage
```

## 4) Run a small experiment
```python
# Customize by editing 'extensions/constants.py'
# Default values from constants.py:
MODEL = "gpt-oss:20b"
K = 5                     # 'k' in pass@k
LIMIT = 10                # None means all 164 tasks
MAX_NEW_TOKENS = 131072   # max context window in gpt-oss
TEMP = 0.2
```

```bash
python extensions/main.py
```

## Artifacts:

- **samples.jsonl** — all non-empty completions
- **empty_samples.jsonl** — tasks that needed a fallback stub (for debugging)
- **samples.jsonl_results.jsonl** - evaluation of all samples
- **Console** — unbiased pass@k (for k in [1, K]) and coverage@k for the subset

# Usage

### A) Generate with Ollama and evaluate automatically

[`extensions/main.py`](extensions/main.py) reads tasks, samples K completions each (with retries on empty), writes samples.jsonl, then calls the evaluator for you.

```bash
python extensions/main.py
```

Tune via env (all optional):
- `OLLAMA_URL` (default http://localhost:11434)
- `MODEL` (default gpt-oss:20b)
- `K`, `LIMIT` (0 -> all, else ≤ 164), `TEMP`, `TOP_P`, `TOP_K`, `REPEAT_PENALTY`
- `MAX_NEW_TOKENS`, `MAX_RETRIES`, `STOP`

### B) Evaluate an existing JSONL (subset-safe)

If you already have completions (from this repo or elsewhere), call:

```python
from extensions.custom_evaluation import evaluate_functional_correctness_subset
evaluate_functional_correctness_subset("samples.jsonl", k=[1, 5])
```

This writes `samples.jsonl_results.jsonl` with per-sample pass/fail and prints pass@k + coverage@k.

### C) Download and export HumanEval
```bash
python extensions/process_human_eval.py --outdir ./data/humaneval
# Produces: humaneval.jsonl, humaneval.json, humaneval.csv
```

## Reproducibility tips

- **Sampling variance**: pass@k moves with temperature/top-p/top-k; keep these fixed when comparing models.
- **Subsets**: prefer larger `LIMIT` (or 0) so more tasks meet n ≥ k.
- **Duplicates**: we normalize and dedupe completions before counting to avoid inflated pass@k.

# Troubleshooting

### Empty or truncated completions
Model "thinking" can produce empty output while still consuming tokens. Try increasing MAX_NEW_TOKENS, inspect `empty_samples.jsonl`.

### "malloc: can't allocate region" / OOM
Free RAM/VRAM, reduce parallel load, or try a smaller model.

### Ollama connection refused
Ensure `ollama serve` is running and `OLLAMA_URL` matches (default http://localhost:11434). See [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/README.md) for setup details.


# Citation

If you use this repo, please cite the original HumanEval paper (see exact citation in [`originalREADME.md`](originalREADME.md))

# License

I maintain the same MIT license from the source repo