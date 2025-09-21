# ~/extensions/cli/humaneval.py
'''
Download and export the HumanEval dataset in multiple formats.

Usage:
    python -m extensions.cli.humaneval --outdir ./data/humaneval

Outputs:
    - HumanEval.jsonl.gz : Raw downloaded gzip archive
    - HumanEval.jsonl    : Decompressed JSONL file
    - humaneval.json     : JSON array with all tasks
    - humaneval.csv      : CSV export with selected columns
'''

# imports
import argparse
from pathlib import Path

from extensions.datasets.humaneval import fetch_humaneval


# Parse arguments and trigger dataset download
def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize the HumanEval dataset")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./data/humaneval"),
        help="Directory to place downloaded files",
    )
    args = parser.parse_args()
    fetch_humaneval(args.outdir)


if __name__ == "__main__":
    main()

