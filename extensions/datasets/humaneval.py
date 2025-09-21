# ~/extensions/datasets/humaneval.py
"""
Helpers for acquiring and exporting the HumanEval dataset.

Provides a single entrypoint that materializes gzip, JSONL, JSON, and CSV artifacts.
"""

# imports
import csv
import gzip
import json
from pathlib import Path
from typing import Dict
from urllib.request import urlopen


# Source (official loader points here)
DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


# Download ``url`` to ``destination`` without authentication
def _download(url: str, destination: Path) -> None:
    with urlopen(url) as response, destination.open("wb") as fh:  # type: ignore[arg-type]
        fh.write(response.read())


# Return decoded UTF-8 text from a gzip archive
def _gunzip_text(source: Path) -> str:
    with gzip.open(source, "rb") as fh:
        return fh.read().decode("utf-8")


# Persist plain text to disk using UTF-8 encoding
def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# Write JSON with pretty formatting to improve readability
def _write_json(path: Path, items) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)


# Export selected columns to CSV for spreadsheet analysis
def _write_csv(path: Path, items) -> None:
    columns = ["task_id", "entry_point", "prompt", "canonical_solution", "test"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for item in items:
            row = {col: item.get(col, "") for col in columns}
            writer.writerow(row)


# Download and materialize the HumanEval dataset into ``outdir``
def fetch_humaneval(outdir: Path) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    gz_path = outdir / "HumanEval.jsonl.gz"
    jsonl_path = outdir / "HumanEval.jsonl"
    json_path = outdir / "humaneval.json"
    csv_path = outdir / "humaneval.csv"

    print(f"Downloading HumanEval from {DATA_URL} ...")
    _download(DATA_URL, gz_path)
    print(f"Saved: {gz_path}")

    print("Decompressing to JSONL ...")
    text = _gunzip_text(gz_path)
    _write_text(jsonl_path, text)
    print(f"Saved: {jsonl_path}")

    print("Parsing and exporting JSON array ...")
    items = [json.loads(line) for line in text.splitlines() if line.strip()]
    _write_json(json_path, items)
    print(f"Saved: {json_path}  (records: {len(items)})")

    print("Exporting CSV ...")
    _write_csv(csv_path, items)
    print(f"Saved: {csv_path}")

    return {
        "gz_path": gz_path,
        "jsonl_path": jsonl_path,
        "json_path": json_path,
        "csv_path": csv_path,
    }


__all__ = ["fetch_humaneval", "DATA_URL"]

