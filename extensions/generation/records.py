# ~/extensions/generation/records.py
"""
Typed record structures shared by the generation pipeline.

Defines Attempt, Sample, and evaluation row schemas consumed across modules.
"""

# imports
from dataclasses import dataclass
from typing import List, Optional, TypedDict


# Result payload returned by the language model
class GenerationResult(TypedDict, total=False):
    text: str
    raw_response: object


# Metadata captured for each generation attempt
class AttemptRecord(TypedDict):
    prompt: str
    seed: int
    temperature: float
    raw_text: str
    raw_response: Optional[object]
    completion: str


# Debug info retained when a task returns empty generations
class EmptySampleRecord(TypedDict, total=False):
    task_id: str
    resolved: bool
    attempts: List[AttemptRecord]
    final_completion: str


# Final record stored in the HumanEval samples JSONL
class SampleRecord(TypedDict):
    task_id: str
    completion: str


# Evaluation row with a stable index for deterministic joins
@dataclass
class SampleRow:
    idx: int
    task_id: str
    completion: str


__all__ = [
    "GenerationResult",
    "AttemptRecord",
    "EmptySampleRecord",
    "SampleRecord",
    "SampleRow",
]

