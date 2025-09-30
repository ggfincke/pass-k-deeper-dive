# ~/extensions/cli/generate.py
'''
Main script for generating HumanEval completions via Ollama.

Usage:
    python -m extensions.cli.generate

Outputs:
    - samples.jsonl             : Generated code completions for HumanEval problems
    - empty_samples.jsonl       : Debug info for tasks that fell back to the stub
    - samples.jsonl_results.jsonl : Evaluation results with pass/fail metadata
'''

from __future__ import annotations

# imports
import argparse
from typing import Optional, Sequence

from extensions.generation.runner import generate_humaneval_completions


# Build argument parser for generation CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HumanEval completions using the configured Ollama backend.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show debugging output during generation.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    generate_humaneval_completions(verbose=args.verbose)


if __name__ == "__main__":
    main()
