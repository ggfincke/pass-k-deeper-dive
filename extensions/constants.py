# ~/extensions/constants.py
'''
Constant values for our scripts
'''
import os

# default Ollama server URL
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# must be a model in `ollama list`
MODEL = os.getenv("MODEL", "gpt-oss:20b") 
# pass@k
K = int(os.getenv("K", "5"))
# amount of problems from HumanEval to process (max 164)
LIMIT = int(os.getenv("LIMIT", "10"))
# temperature for first attempt
TEMP = float(os.getenv("TEMP", "0.2"))
# temperature for retries
RETRY_TEMP = float(os.getenv("RETRY_TEMP", "0.6"))
# max tokens for generation
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "8192")) # 16384 32768?
# max attempts (1 initial + retries)
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# system prompt for Ollama
SYSTEM = (
    "You will be given a Python function signature and docstring.\n"
    "FORMAT CONTRACT (must follow exactly):\n"
    "• Output ONLY the function body (statements that would live inside the function).\n"
    "• The FIRST line must begin with exactly four spaces.\n"
    "• Do NOT output the 'def ...' line.\n"
    "• Do NOT output imports, markdown fences, explanations, or text outside the body.\n"
    "• Do NOT include top-level comments or print statements.\n"
    "• No leading blank lines. End with a single trailing newline.\n"
) 