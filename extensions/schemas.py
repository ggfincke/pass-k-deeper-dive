# ~/extensions/schemas.py
'''
TypedDict schemas shared across extensions modules

Defines data structures for:
- LLM generation results and attempts
- Sample records for HumanEval completions  
- Debug records for empty generation tracking
- JSON-compatible type aliases

'''
from typing import Dict, List, Optional, TypedDict


# Result from LLM generation (text + raw API response)
class GenerationResult(TypedDict, total=False):
    text: str
    raw_response: object


# Individual generation attempt with all metadata
class AttemptRecord(TypedDict):
    prompt: str
    seed: int
    temperature: float
    raw_text: str
    raw_response: Optional[object]
    completion: str


# Debug record for tracking empty completion problems
class EmptySampleRecord(TypedDict, total=False):
    task_id: str
    resolved: bool
    attempts: List[AttemptRecord]
    final_completion: str


# Final sample record for HumanEval output
class SampleRecord(TypedDict):
    task_id: str
    completion: str


# Type alias for JSON-compatible dictionaries
JsonDict = Dict[str, object]

