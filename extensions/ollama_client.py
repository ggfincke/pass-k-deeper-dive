# ~/extensions/ollama_client.py
"""
Client interface for Ollama API to generate code completions.

Handles communication with the local Ollama server for LLM inference and relies
on shared configuration from ``constants``.
"""

from typing import Dict

import requests

from constants import (
    MAX_NEW_TOKENS,
    MODEL,
    OLLAMA_URL,
    REPEAT_PENALTY,
    STOP_SEQS,
    SYSTEM,
    TEMP,
    TOP_K,
    TOP_P,
)

def _clean_options(d: dict) -> dict:
    # Drop None so we don't send nulls to Ollama
    return {k: v for k, v in d.items() if v is not None}

# use /api/generate endpoint (simple prompt style; embed instruction + prompt)
def ollama_generate(prompt: str, seed: int, temperature: float = TEMP) -> Dict[str, object]:
    options = _clean_options({
        "temperature": temperature,
        "num_predict": MAX_NEW_TOKENS,
        "seed": seed,
        "top_p": TOP_P,
        "top_k": (TOP_K if TOP_K > 0 else None),
        "repeat_penalty": REPEAT_PENALTY,
        "stop": (STOP_SEQS if STOP_SEQS else None),
    })
    payload = {
        "model": MODEL,
        "system": SYSTEM,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data.get("response", ""),
        "raw_response": data,
    }
