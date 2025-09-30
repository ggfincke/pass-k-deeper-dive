# ~/extensions/clients/ollama.py
"""
Client interface for talking to a local Ollama server.

Handles authentication-independent JSON payloads for generation requests.
"""

# imports
from typing import Dict

import requests
import threading
import time

from extensions.config import (
    MAX_NEW_TOKENS,
    MODEL,
    OLLAMA_HTTP_MAX_RETRIES,
    OLLAMA_HTTP_TIMEOUT,
    OLLAMA_MAX_PARALLEL_REQUESTS,
    OLLAMA_OPTIONS,
    OLLAMA_RETRY_BASE_DELAY,
    OLLAMA_RETRY_MAX_DELAY,
    OLLAMA_URL,
    REPEAT_PENALTY,
    STOP_SEQS,
    SYSTEM,
    TEMP,
    TOP_K,
    TOP_P,
)


# Remove ``None`` values so the payload stays compact
def _clean_options(options: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in options.items() if v is not None}


# Issue a generation request and return text plus raw response
_THREAD_LOCAL = threading.local()
_REQUEST_GUARD = threading.BoundedSemaphore(OLLAMA_MAX_PARALLEL_REQUESTS)


# Retrieve thread-local requests session for connection pooling
def _session() -> requests.Session:
    sess = getattr(_THREAD_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        _THREAD_LOCAL.session = sess
    return sess


# Send a generation request to Ollama with retry logic
def generate(prompt: str, seed: int, temperature: float = TEMP) -> Dict[str, object]:
    options = dict(OLLAMA_OPTIONS)
    options.update(
        {
            "temperature": temperature,
            "num_predict": MAX_NEW_TOKENS,
            "seed": seed,
            "top_p": TOP_P,
            "top_k": (TOP_K if TOP_K > 0 else None),
            "repeat_penalty": REPEAT_PENALTY,
            "stop": (STOP_SEQS if STOP_SEQS else None),
        }
    )
    options = _clean_options(options)
    payload = {
        "model": MODEL,
        "system": SYSTEM,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }
    attempt = 0
    delay = OLLAMA_RETRY_BASE_DELAY
    session = _session()

    while True:
        attempt += 1
        _REQUEST_GUARD.acquire()
        try:
            response = session.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_HTTP_TIMEOUT,
            )
        finally:
            _REQUEST_GUARD.release()

        if response.status_code in {429, 503} and attempt <= OLLAMA_HTTP_MAX_RETRIES:
            sleep_for = min(delay, OLLAMA_RETRY_MAX_DELAY)
            if sleep_for > 0:
                time.sleep(sleep_for)
            delay = min(delay * 2, OLLAMA_RETRY_MAX_DELAY)
            continue

        response.raise_for_status()
        data = response.json()
        return {
            "text": data.get("response", ""),
            "raw_response": data,
        }


__all__ = ["generate"]
