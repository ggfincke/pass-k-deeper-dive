# ~/extensions/constants.py
'''
Constant values for our scripts
'''
import os
from typing import Optional

# default Ollama server URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# must be a model in `ollama list`
MODEL = os.getenv("MODEL", "gpt-oss:20b") 
# pass@k
K = int(os.getenv("K", "3"))
# helper func to parse max number of problems
def parse_limit(env: str = "LIMIT", default: Optional[int] = None) -> Optional[int]:
    raw = os.getenv(env)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in {"0", "all", "none", ""}:
        return None
    v = int(s)
    return None if v <= 0 else v
# max number of problems to evaluate (None = all)
LIMIT: Optional[int] = parse_limit("LIMIT", default=2)
# temperature for first attempt
TEMP = float(os.getenv("TEMP", "0.2"))
# temperature for retries
RETRY_TEMP = float(os.getenv("RETRY_TEMP", "0.6"))
# max tokens for generation (128k context window)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "131072"))
# max attempts (1 initial + retries)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# system prompt for Ollama
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
