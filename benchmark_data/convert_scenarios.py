"""
Converts scenarios from our generated format to llm_logic_benchmark.json format.

Our format:
{
    "id": "para_001",
    "category": "contradiction",
    "logic": "paraconsistent_LP",
    "domain": "medical",
    "difficulty": "medium",
    "context": "...",
    "premises": ["...", "..."],
    "query": "...",
    "classical_answer": "...",
    "classical_fails": true,
    "correct_answer": "...",
    "formal_proof": "...",
    ...
}

Target format (llm_logic_benchmark.json):
{
    "id": "LEM-01",
    "category": "LEM",
    "premises": ["..."],
    "conclusion": "...",
    "formal": "...",
    "expected": {
        "classical": "entails" | "does_not_entail",
        "intuitionistic": "entails" | "does_not_entail",
        "paraconsistent": "entails" | "does_not_entail",
        "relevance": "entails" | "does_not_entail"
    },
    "explanation": "..."
}

Usage:
    python convert_scenarios.py --input our_scenarios.json --output llm_logic_benchmark_extended.json
    python convert_scenarios.py --input our_scenarios.json --merge llm_logic_benchmark.json --output llm_logic_benchmark_full.json
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Logic mapping
# How each of our logics maps to the 4 benchmark logics
# Key insight: for each scenario we derive what each logic would say
# about whether the premises entail the conclusion
# ---------------------------------------------------------------------------

LOGIC_ENTAILMENT_MAP = {
    # Paraconsistent LP scenarios
    # Classical: fails (explosion), Paraconsistent: entails (contains contradiction)
    "paraconsistent_LP": {
        "classical": "does_not_entail",   # classical explodes on contradiction
        "intuitionistic": "does_not_entail",
        "paraconsistent": "entails",       # LP contains contradiction, gives meaningful answer
        "relevance": "does_not_entail",
    },

    # Epistemic/Default scenarios
    # Classical closed-world assumption entails, others more cautious
    "epistemic_default": {
        "classical": "entails",            # CWA: no info = false = classical entails
        "intuitionistic": "does_not_entail",  # requires constructive witness
        "paraconsistent": "entails",
        "relevance": "entails",
    },

    # Deontic SDL
    "deontic_SDL": {
        "classical": "entails",
        "intuitionistic": "entails",
        "paraconsistent": "entails",
        "relevance": "entails",
    },

    # Fuzzy/Lukasiewicz — classical binary fails on gradual truth
    "fuzzy_lukasiewicz": {
        "classical": "does_not_entail",   # binary threshold gives wrong answer
        "intuitionistic": "does_not_entail",
        "paraconsistent": "does_not_entail",
        "relevance": "does_not_entail",
    },

    # LTL — classical misses temporal ordering
    "linear_temporal_logic": {
        "classical": "does_not_entail",   # classical has no temporal operators
        "intuitionistic": "does_not_entail",
        "paraconsistent": "does_not_entail",
        "relevance": "does_not_entail",
    },

    # Intuitionistic — requires constructive proof
    "intuitionistic": {
        "classical": "entails",            # classical accepts without witness
        "intuitionistic": "does_not_entail",  # requires explicit witness
        "paraconsistent": "entails",
        "relevance": "entails",
    },

    # Dempster-Shafer — classical averaging fails
    "dempster_shafer": {
        "classical": "does_not_entail",   # naive averaging gives wrong answer
        "intuitionistic": "does_not_entail",
        "paraconsistent": "does_not_entail",
        "relevance": "does_not_entail",
    },
}

# For scenarios where classical_fails=False (true negatives/positives),
# all logics agree
ALL_ENTAIL = {
    "classical": "entails",
    "intuitionistic": "entails",
    "paraconsistent": "entails",
    "relevance": "entails",
}

ALL_NOT_ENTAIL = {
    "classical": "does_not_entail",
    "intuitionistic": "does_not_entail",
    "paraconsistent": "does_not_entail",
    "relevance": "does_not_entail",
}


def derive_expected(scenario: dict) -> dict:
    """
    Derive per-logic expected verdicts from scenario metadata.
    
    For classical_fails=True: use LOGIC_ENTAILMENT_MAP
    For classical_fails=False: all logics agree (correct_answer is correct classically too)
    """
    logic = scenario.get("logic", "")
    classical_fails = scenario.get("classical_fails", True)

    if not classical_fails:
        # True negative or true positive — all logics agree with classical
        # Check correct_answer to determine direction
        correct = scenario.get("correct_answer", "").upper()
        if any(word in correct for word in ["VIOLATION", "FALSE", "NO", "DENIED", "BLOCKED"]):
            return ALL_NOT_ENTAIL.copy()
        else:
            return ALL_ENTAIL.copy()

    # classical_fails=True — use logic-specific mapping
    base = LOGIC_ENTAILMENT_MAP.get(logic, {
        "classical": "entails",
        "intuitionistic": "does_not_entail",
        "paraconsistent": "entails",
        "relevance": "entails",
    })
    return base.copy()


def build_conclusion(scenario: dict) -> str:
    """Build a conclusion statement from query + correct_answer."""
    query = scenario.get("query", "")
    correct = scenario.get("correct_answer", "")
    # Take first sentence of correct answer as conclusion
    first_sentence = correct.split("—")[0].split(".")[0].strip()
    if len(first_sentence) > 10:
        return first_sentence
    return query


def build_formal(scenario: dict) -> str:
    """Extract or build formal notation."""
    proof = scenario.get("formal_proof", "")
    if proof:
        # Take first line of formal proof
        first_line = proof.split(".")[0].strip()
        if len(first_line) < 200:
            return first_line
    logic = scenario.get("logic", "")
    return f"[{logic}]"


def convert_scenario(scenario: dict) -> dict:
    """Convert single scenario from our format to benchmark format."""
    premises = scenario.get("premises", [])
    # Add context as first premise if present
    context = scenario.get("context", "")
    if context and context not in premises[0] if premises else True:
        all_premises = [context] + premises
    else:
        all_premises = premises

    return {
        "id": scenario["id"],
        "category": scenario.get("category", scenario.get("logic", "unknown")),
        "logic": scenario.get("logic", ""),
        "domain": scenario.get("domain", ""),
        "difficulty": scenario.get("difficulty", "medium"),
        "premises": all_premises,
        "conclusion": build_conclusion(scenario),
        "query": scenario.get("query", ""),
        "formal": build_formal(scenario),
        "expected": derive_expected(scenario),
        "explanation": scenario.get("notes", scenario.get("classical_failure_mode", "")),
        # Keep original fields for our evaluator
        "correct_answer": scenario.get("correct_answer", ""),
        "classical_answer": scenario.get("classical_answer", ""),
        "classical_fails": scenario.get("classical_fails", True),
        "llm_expected_failure": scenario.get("llm_expected_failure", ""),
    }


def load_our_scenarios(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("scenarios", list(data.values()))
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="Path to our generated scenarios JSON"
    )
    parser.add_argument(
        "--merge", default=None,
        help="Path to existing llm_logic_benchmark.json to merge with"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for converted/merged benchmark JSON"
    )
    args = parser.parse_args()

    # Load our scenarios
    our_scenarios = load_our_scenarios(args.input)
    print(f"Loaded {len(our_scenarios)} scenarios from {args.input}")

    # Convert
    converted = [convert_scenario(s) for s in our_scenarios]
    print(f"Converted {len(converted)} scenarios")

    # Merge with existing if provided
    if args.merge:
        with open(args.merge, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing scenarios from {args.merge}")
        # Avoid duplicates by ID
        existing_ids = {s["id"] for s in existing}
        new_only = [s for s in converted if s["id"] not in existing_ids]
        print(f"Adding {len(new_only)} new scenarios (skipping {len(converted)-len(new_only)} duplicates)")
        merged = existing + new_only
    else:
        merged = converted

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(merged)} scenarios to {args.output}")

    # Stats
    from collections import Counter
    logics = Counter(s.get("logic", s.get("category", "?")) for s in merged)
    print("\nBy logic/category:")
    for k, v in logics.most_common():
        print(f"  {k:<35} {v}")


if __name__ == "__main__":
    main()
