# ~/extensions/clients/ollama.py
"""
Client interface for talking to a local Ollama server.

Handles authentication-independent JSON payloads for generation requests.
"""

# imports
from typing import Dict

import requests

from extensions.config import (
    MAX_NEW_TOKENS,
    MODEL,
    OLLAMA_OPTIONS,
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
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    response.raise_for_status()
    data = response.json()
    return {
        "text": data.get("response", ""),
        "raw_response": data,
    }


__all__ = ["generate"]
