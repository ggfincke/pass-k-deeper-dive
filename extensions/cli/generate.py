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

# imports
from extensions.generation.runner import generate_humaneval_completions


# Run generation with default configuration
def main() -> None:
    generate_humaneval_completions()


if __name__ == "__main__":
    main()

