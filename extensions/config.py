# ~/extensions/config.py
"""
Shared configuration resolved from environment variables.

Provides constants used by generation, evaluation, and visualization modules.
"""

# imports
import os
from typing import Dict, Optional


# Interpret LIMIT environment value; zero or "all" means no limit
def _resolve_limit(raw: Optional[str], default: Optional[int]) -> Optional[int]:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "all", "none", ""}:
        return None
    value = int(normalized)
    return None if value <= 0 else value


# Check if environment value looks like a directory path
def _looks_like_os_tempdir(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    if any(sep in normalized for sep in ("/", "\\")):
        return True
    if os.name == "nt" and ":" in normalized:
        return True
    return False


# Resolve temperature from environment; fallback on TEMP directory values
def _resolve_temperature(default: float) -> float:
    candidates = (
        ("PASSK_TEMPERATURE", os.getenv("PASSK_TEMPERATURE")),
        ("PASSK_TEMP", os.getenv("PASSK_TEMP")),
        ("TEMP", os.getenv("TEMP")),
    )
    for name, raw in candidates:
        if raw is None:
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        try:
            return float(cleaned)
        except ValueError as exc:
            if name == "TEMP" and _looks_like_os_tempdir(cleaned):
                continue
            raise ValueError(
                f"Environment variable {name} must be a float; received {raw!r}"
            ) from exc
    return default


# URL of the local Ollama server; override via env when pointing elsewhere
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Model name served by Ollama; must appear in 'ollama list'
MODEL = os.getenv("MODEL", "gpt-oss:20b")
# MODEL = os.getenv("MODEL", "deepseek-r1:14b")
# Generation budget per HumanEval task.
N_SAMPLES = int(os.getenv("N_SAMPLES", "100"))
# Evaluation ks (comma-separated ints) applied during pass@k computation
_raw_eval = os.getenv("EVAL_KS", "1,5,10,25")
EVAL_KS = [int(x.strip()) for x in _raw_eval.split(",") if x.strip()]
# Concurrent generation workers when sampling pass@k completions
CONCURRENCY = max(1, int(os.getenv("CONCURRENCY", "5")))
# Maximum workers for evaluation tasks (defaults to min of CPU count or 8)
MAX_EVAL_WORKERS = max(1, min(int(os.getenv("MAX_EVAL_WORKERS", str(min(os.cpu_count() or 4, 8)))), 32))
# Maximum execution timeout in seconds
MAX_EXECUTION_TIMEOUT = float(os.getenv("MAX_EXECUTION_TIMEOUT", "30.0"))
# Maximum number of HumanEval tasks to process; None means all tasks
LIMIT: Optional[int] = _resolve_limit(os.getenv("LIMIT"), default=None)
# Temperature used for generation requests
DEFAULT_TEMP = 0.2
TEMP = _resolve_temperature(DEFAULT_TEMP)
# Model options forwarded to Ollama for GPU-friendly defaults
OLLAMA_OPTIONS: Dict[str, object] = {
    "num_ctx": 16384,
    "num_batch": 256,
    "temperature": TEMP,
}
# Upper bound on new tokens produced per completion
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "131072"))
# Retry budget when the model returns empty completions
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
# Max retries for empty completions can be overridden separately
EMPTY_COMPLETION_MAX_RETRIES = max(
    0, int(os.getenv("EMPTY_COMPLETION_MAX_RETRIES", str(MAX_RETRIES)))
)
EMPTY_COMPLETION_BACKOFF_BASE = max(
    0.0, float(os.getenv("EMPTY_COMPLETION_BACKOFF_BASE", "0.1"))
)
EMPTY_COMPLETION_BACKOFF_MAX = max(
    0.0, float(os.getenv("EMPTY_COMPLETION_BACKOFF_MAX", "1.0"))
)
# Nucleus sampling parameter (set to 1.0 for pure sampling)
TOP_P = float(os.getenv("TOP_P", "1.0"))
# Top-k sampling cutoff; zero disables the constraint
TOP_K = int(os.getenv("TOP_K", "0"))
# Penalty applied to repeated tokens to reduce loops
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.0"))
# Comma-separated stop sequences applied during generation
_STOP = os.getenv("STOP", "").strip()
# Parsed list of stop sequences excluding empty entries
STOP_SEQS = [s for s in _STOP.split(",") if s.strip()]
# Instruction prompt sent as the Ollama system message
SYSTEM = (
    "You will be given a Python function signature and docstring.\n"
    "FORMAT CONTRACT (must follow exactly):\n"
    "• Output ONLY the function body (statements that would live inside the function).\n"
    "• The FIRST line must begin with exactly four spaces (example: '    return None').\n"
    "• Use four-space indentation for every subsequent line.\n"
    "• Do NOT output the 'def ...' line.\n"
    "• Do NOT output imports, markdown fences, explanations, or text outside the body.\n"
    "• Do NOT include top-level comments or print statements.\n"
    "• No leading blank lines. End with a single trailing newline.\n"
)
# Concurrency controls for Ollama interactions
OLLAMA_MAX_PARALLEL_REQUESTS = max(
    1, int(os.getenv("OLLAMA_MAX_PARALLEL_REQUESTS", str(CONCURRENCY)))
)
OLLAMA_HTTP_TIMEOUT = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "120.0"))
OLLAMA_HTTP_MAX_RETRIES = max(0, int(os.getenv("OLLAMA_HTTP_MAX_RETRIES", "3")))
OLLAMA_RETRY_BASE_DELAY = max(
    0.0, float(os.getenv("OLLAMA_RETRY_BASE_DELAY", "0.1"))
)
OLLAMA_RETRY_MAX_DELAY = max(
    0.0, float(os.getenv("OLLAMA_RETRY_MAX_DELAY", "2.0"))
)

__all__ = [
    "OLLAMA_URL",
    "MODEL",
    "N_SAMPLES",
    "EVAL_KS",
    "CONCURRENCY",
    "MAX_EVAL_WORKERS",
    "MAX_EXECUTION_TIMEOUT",
    "LIMIT",
    "TEMP",
    "EMPTY_COMPLETION_MAX_RETRIES",
    "EMPTY_COMPLETION_BACKOFF_BASE",
    "EMPTY_COMPLETION_BACKOFF_MAX",
    "OLLAMA_OPTIONS",
    "MAX_NEW_TOKENS",
    "MAX_RETRIES",
    "OLLAMA_MAX_PARALLEL_REQUESTS",
    "OLLAMA_HTTP_TIMEOUT",
    "OLLAMA_HTTP_MAX_RETRIES",
    "OLLAMA_RETRY_BASE_DELAY",
    "OLLAMA_RETRY_MAX_DELAY",
    "TOP_P",
    "TOP_K",
    "REPEAT_PENALTY",
    "STOP_SEQS",
    "SYSTEM",
]
