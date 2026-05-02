"""
Dataset builder — converts raw scenario JSONs to HuggingFace Dataset format
and pushes to Hub.

Usage:
    python build_dataset.py --input ./scenarios --output ./hf_dataset
    python build_dataset.py --input ./scenarios --push-to-hub your_username/nonclassical-logics-benchmark
"""

import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Sequence


# HuggingFace dataset schema
FEATURES = Features({
    "id": Value("string"),
    "category": Value("string"),
    "logic": Value("string"),
    "domain": Value("string"),
    "difficulty": Value("string"),
    "context": Value("string"),
    "premises": Sequence(Value("string")),
    "query": Value("string"),
    "classical_answer": Value("string"),
    "classical_fails": Value("bool"),
    "correct_answer": Value("string"),
    "formal_proof": Value("string"),
    "classical_failure_mode": Value("string"),
    "llm_expected_failure": Value("string"),
    "notes": Value("string"),
})


def load_scenarios(input_path: str) -> list[dict]:
    """Load scenarios from a single JSON file or directory of JSON files."""
    p = Path(input_path)
    scenarios = []

    if p.is_file():
        with open(p) as f:
            data = json.load(f)
            if isinstance(data, list):
                scenarios = data
            elif isinstance(data, dict):
                # Could be {"scenarios": [...]} or {"para_001": {...}, ...}
                scenarios = data.get("scenarios", list(data.values()))
    elif p.is_dir():
        for f in sorted(p.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    scenarios.extend(data)
                else:
                    scenarios.append(data)

    return scenarios


def normalize(scenario: dict) -> dict:
    """Ensure all required fields exist with correct types."""
    return {
        "id": scenario.get("id", ""),
        "category": scenario.get("category", ""),
        "logic": scenario.get("logic", ""),
        "domain": scenario.get("domain", ""),
        "difficulty": scenario.get("difficulty", "medium"),
        "context": scenario.get("context", ""),
        "premises": scenario.get("premises", []),
        "query": scenario.get("query", ""),
        "classical_answer": scenario.get("classical_answer", ""),
        "classical_fails": bool(scenario.get("classical_fails", True)),
        "correct_answer": scenario.get("correct_answer", ""),
        "formal_proof": scenario.get("formal_proof", ""),
        "classical_failure_mode": scenario.get("classical_failure_mode", ""),
        "llm_expected_failure": scenario.get("llm_expected_failure", ""),
        "notes": scenario.get("notes", ""),
    }


def build_splits(scenarios: list[dict]) -> DatasetDict:
    """
    Split into train/test/hard.
    - hard: difficulty == "hard"
    - test: remaining (easy + medium)
    - train: empty for now — benchmark is evaluation only
    """
    hard = [s for s in scenarios if s["difficulty"] == "hard"]
    test = [s for s in scenarios if s["difficulty"] != "hard"]

    return DatasetDict({
        "test": Dataset.from_list(test, features=FEATURES),
        "hard": Dataset.from_list(hard, features=FEATURES),
        "full": Dataset.from_list(scenarios, features=FEATURES),
    })


def print_stats(scenarios: list[dict]):
    from collections import Counter
    print(f"\nTotal scenarios: {len(scenarios)}")
    print("\nBy logic:")
    for k, v in Counter(s["logic"] for s in scenarios).most_common():
        print(f"  {k:<35} {v}")
    print("\nBy difficulty:")
    for k, v in Counter(s["difficulty"] for s in scenarios).most_common():
        print(f"  {k:<10} {v}")
    print("\nBy domain:")
    for k, v in Counter(s["domain"] for s in scenarios).most_common():
        print(f"  {k:<20} {v}")
    print("\nClassical fails:")
    fails = sum(1 for s in scenarios if s["classical_fails"])
    print(f"  {fails}/{len(scenarios)} ({fails/len(scenarios):.0%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="C:\Users\user\Repos\non-classical-logics\nonclassical_logics\benchmark_data")
    parser.add_argument("--output", default="./hf_dataset", help="Local output path")
    parser.add_argument("--push-to-hub", default=None, help="HF Hub repo ID to push to")
    args = parser.parse_args()

    scenarios = load_scenarios(args.input)
    scenarios = [normalize(s) for s in scenarios]
    print_stats(scenarios)

    dataset = build_splits(scenarios)
    print(f"\nDataset splits: {list(dataset.keys())}")

    # Save locally
    Path(args.output).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(args.output)
    print(f"Saved to {args.output}")

    # Push to Hub
    if args.push_to_hub:
        dataset.push_to_hub(
            args.push_to_hub,
            private=False,
            commit_message="Initial benchmark dataset — 70 nonclassical logic scenarios",
        )
        print(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
