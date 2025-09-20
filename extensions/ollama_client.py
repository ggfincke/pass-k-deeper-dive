# ~/extensions/ollama_client.py
'''
Client interface for Ollama API to generate code completions

Handles communication with local Ollama server for LLM inference.
Formats prompts with system instructions and manages generation parameters.

'''
import requests
from typing import Dict
from constants import OLLAMA_URL, MODEL, SYSTEM, MAX_NEW_TOKENS, TEMP

# use /api/generate endpoint (simple prompt style; embed instruction + prompt)
def ollama_generate(prompt: str, seed: int, temperature: float = TEMP) -> Dict[str, object]:
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM}\n\n{prompt}\n",
        "options": {
            "temperature": temperature,
            "num_predict": MAX_NEW_TOKENS,
            # vary seed to get k distinct samples
            "seed": seed,
        },
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
    r.raise_for_status()
    data = r.json()
    return {
        "text": data.get("response", ""),
        "raw_response": data,
    }