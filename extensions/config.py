# ~/extensions/config.py
"""
Shared configuration resolved from environment variables.

Provides constants used by generation, evaluation, and visualization modules.
"""

# imports
import os
from typing import Dict, Optional


# Interpret LIMIT semantics from environment values
def _resolve_limit(raw: Optional[str], default: Optional[int]) -> Optional[int]:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"0", "all", "none", ""}:
        return None
    value = int(normalized)
    return None if value <= 0 else value


# URL of the local Ollama server; override via env when pointing elsewhere
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Model name served by Ollama; must appear in 'ollama list'
MODEL = os.getenv("MODEL", "gpt-oss:20b")
# Number of completions to sample per HumanEval task
K = int(os.getenv("K", "10"))
# Concurrent generation workers when sampling pass@k completions
CONCURRENCY = max(1, int(os.getenv("CONCURRENCY", "5")))
# Maximum number of HumanEval tasks to process; None means all tasks
LIMIT: Optional[int] = _resolve_limit(os.getenv("LIMIT"), default=10)
# Temperature used for generation requests
TEMP = float(os.getenv("TEMP", "0.2"))
# Model options forwarded to Ollama for GPU-friendly defaults
OLLAMA_OPTIONS: Dict[str, object] = {
    # "num_ctx": 16384,
    # "num_batch": 256,
    "temperature": TEMP,
}
# Upper bound on new tokens produced per completion
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "131072"))
# Retry budget when the model returns empty completions
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
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

__all__ = [
    "OLLAMA_URL",
    "MODEL",
    "K",
    "CONCURRENCY",
    "LIMIT",
    "TEMP",
    "OLLAMA_OPTIONS",
    "MAX_NEW_TOKENS",
    "MAX_RETRIES",
    "TOP_P",
    "TOP_K",
    "REPEAT_PENALTY",
    "STOP_SEQS",
    "SYSTEM",
]
