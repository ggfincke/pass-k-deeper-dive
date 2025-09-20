# ~/extensions/process_human_eval.py
'''
Downloads and processes the HumanEval dataset from OpenAI

Usage:
    python process_human_eval.py --outdir ./data/humaneval

Outputs:
    - HumanEval.jsonl.gz   : Raw downloaded gzip file
    - HumanEval.jsonl      : Decompressed JSONL file
    - humaneval.json       : JSON array file
    - humaneval.csv        : CSV file with selected fields

'''
# imports
import argparse
import gzip
import io
import json
import os
import sys
import urllib.request
import csv

# Source (official loader points here)
DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"

# Helper functions

# downloads from URL
def download(url: str, path: str):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        f.write(r.read())

# decompresses gzip to text
def gunzip_to_text(gz_path: str) -> str:
    with gzip.open(gz_path, "rb") as f:
        return f.read().decode("utf-8")

# writes text to file
def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# parses JSONL text to list of dicts
def parse_jsonl(text: str):
    return [json.loads(line) for line in text.splitlines() if line.strip()]

# writes list of dicts to JSON array file
def write_json_array(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# writes list of dicts to CSV file with selected columns
def write_csv(path: str, items):
    cols = ["task_id", "entry_point", "prompt", "canonical_solution", "test"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for it in items:
            row = {k: it.get(k, "") for k in cols}
            w.writerow(row)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="Directory to place outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gz_path   = os.path.join(args.outdir, "HumanEval.jsonl.gz")
    jsonl_path= os.path.join(args.outdir, "HumanEval.jsonl")
    json_path = os.path.join(args.outdir, "humaneval.json")
    csv_path  = os.path.join(args.outdir, "humaneval.csv")

    print(f"Downloading HumanEval from {DATA_URL} ...")
    download(DATA_URL, gz_path)
    print(f"Saved: {gz_path}")

    print("Decompressing to JSONL ...")
    text = gunzip_to_text(gz_path)
    write_text(jsonl_path, text)
    print(f"Saved: {jsonl_path}")

    print("Parsing and exporting JSON array ...")
    items = parse_jsonl(text)
    write_json_array(json_path, items)
    print(f"Saved: {json_path}  (records: {len(items)})")

    print("Exporting CSV ...")
    write_csv(csv_path, items)
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    sys.exit(main())
