"""
NonClassical Logics LLM Benchmark — Evaluator
Runs any HuggingFace-compatible model through the benchmark dataset
and computes agreement metrics against formal correct answers.
"""

import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    logic: str
    domain: str
    difficulty: str
    model_output: str
    correct_answer: str
    classical_answer: str
    classical_fails: bool
    # Scored fields — filled by scorer
    agrees_with_correct: Optional[bool] = None
    agrees_with_classical: Optional[bool] = None
    flags_uncertainty: Optional[bool] = None
    flags_escalation: Optional[bool] = None
    score: Optional[float] = None


@dataclass
class BenchmarkResults:
    model_id: str
    total_scenarios: int = 0
    results: list[ScenarioResult] = field(default_factory=list)

    # Aggregate metrics — filled after scoring
    overall_agreement: float = 0.0
    classical_failure_rate: float = 0.0        # how often model gives classical (wrong) answer
    uncertainty_calibration: float = 0.0       # how often model flags uncertainty when it should
    escalation_rate: float = 0.0

    # Per-category breakdown
    category_scores: dict = field(default_factory=dict)
    logic_scores: dict = field(default_factory=dict)
    domain_scores: dict = field(default_factory=dict)
    difficulty_scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a reasoning AI assistant. Answer the following question based ONLY on the premises provided.
Be explicit about uncertainty. If you cannot determine a definitive answer, say so clearly.
Do not use outside knowledge beyond what is given in the premises."""

def build_prompt(scenario: dict) -> str:
    premises_text = "\n".join(f"- {p}" for p in scenario["premises"])
    return f"""CONTEXT: {scenario["context"]}

PREMISES:
{premises_text}

QUESTION: {scenario["query"]}

Provide your answer and reasoning:"""


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

class ModelRunner:
    def __init__(self, model_id: str, device: str = "auto", max_new_tokens: int = 512):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        print(f"Loading {model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic for reproducibility
            temperature=1.0,
        )
        print("Model loaded.")

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Chat template if available, else raw prompt
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

        output = self.pipe(formatted)
        generated = output[0]["generated_text"]
        # Strip the prompt from output
        if formatted in generated:
            generated = generated[len(formatted):]
        return generated.strip()


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

# Keywords that indicate model is flagging uncertainty / escalation
UNCERTAINTY_KEYWORDS = [
    "cannot determine", "uncertain", "unknown", "not provable",
    "insufficient evidence", "unclear", "ambiguous", "escalate",
    "human review", "flag", "cannot confirm", "not enough information",
    "unprovable", "indeterminate",
]

ESCALATION_KEYWORDS = [
    "escalate", "human review", "manual review", "consult", "refer",
    "physician", "legal review", "compliance", "investigate",
]

CLASSICAL_FAILURE_INDICATORS = [
    # These phrases suggest model fell into classical reasoning trap
    "yes", "no", "approved", "compliant", "proceed", "confirmed",
]


def score_result(result: ScenarioResult) -> ScenarioResult:
    """
    Heuristic scoring — compares model output to correct_answer keywords.
    For production use, replace with LLM-as-judge or human annotation.
    """
    output_lower = result.model_output.lower()
    correct_lower = result.correct_answer.lower()

    # Extract key decision tokens from correct answer
    correct_tokens = set(correct_lower.split())

    # Flag uncertainty
    result.flags_uncertainty = any(kw in output_lower for kw in UNCERTAINTY_KEYWORDS)
    result.flags_escalation = any(kw in output_lower for kw in ESCALATION_KEYWORDS)

    # Classical failure: model gives simple yes/no when correct answer requires nuance
    if result.classical_fails:
        classical_lower = result.classical_answer.lower()
        # Check if model output resembles classical answer more than correct answer
        classical_tokens = set(classical_lower.split())
        output_tokens = set(output_lower.split())
        overlap_classical = len(output_tokens & classical_tokens)
        overlap_correct = len(output_tokens & correct_tokens)
        result.agrees_with_classical = overlap_classical > overlap_correct
    else:
        result.agrees_with_classical = False

    # Agreement with correct answer — keyword overlap heuristic
    key_correct_phrases = [
        phrase.strip().lower()
        for phrase in result.correct_answer.replace("—", " ").split(".")
        if len(phrase.strip()) > 10
    ]
    matches = sum(
        1 for phrase in key_correct_phrases[:3]
        if any(word in output_lower for word in phrase.split() if len(word) > 5)
    )
    result.agrees_with_correct = matches >= 1

    # Composite score
    score = 0.0
    if result.agrees_with_correct:
        score += 0.6
    if result.classical_fails and not result.agrees_with_classical:
        score += 0.2  # didn't fall into classical trap
    if result.flags_uncertainty and "UNKNOWN" in result.correct_answer.upper():
        score += 0.1
    if result.flags_escalation and "ESCALAT" in result.correct_answer.upper():
        score += 0.1

    result.score = round(score, 3)
    return result


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate(results: BenchmarkResults) -> BenchmarkResults:
    scored = results.results
    n = len(scored)
    if n == 0:
        return results

    results.total_scenarios = n
    results.overall_agreement = round(
        sum(1 for r in scored if r.agrees_with_correct) / n, 3
    )
    classical_fail_cases = [r for r in scored if r.classical_fails]
    if classical_fail_cases:
        results.classical_failure_rate = round(
            sum(1 for r in classical_fail_cases if r.agrees_with_classical)
            / len(classical_fail_cases), 3
        )
    uncertainty_needed = [
        r for r in scored if "UNKNOWN" in r.correct_answer.upper()
        or "NOT PROVABLE" in r.correct_answer.upper()
    ]
    if uncertainty_needed:
        results.uncertainty_calibration = round(
            sum(1 for r in uncertainty_needed if r.flags_uncertainty)
            / len(uncertainty_needed), 3
        )

    # Breakdowns
    for attr in ("category", "logic", "domain", "difficulty"):
        breakdown = {}
        for r in scored:
            key = getattr(r, attr)
            if key not in breakdown:
                breakdown[key] = {"total": 0, "correct": 0, "scores": []}
            breakdown[key]["total"] += 1
            breakdown[key]["correct"] += int(bool(r.agrees_with_correct))
            breakdown[key]["scores"].append(r.score or 0.0)
        summary = {
            k: {
                "accuracy": round(v["correct"] / v["total"], 3),
                "avg_score": round(sum(v["scores"]) / v["total"], 3),
                "n": v["total"],
            }
            for k, v in breakdown.items()
        }
        setattr(results, f"{attr}_scores", summary)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_benchmark(
    model_id: str,
    dataset_path: str,
    output_path: str,
    max_scenarios: Optional[int] = None,
    device: str = "auto",
) -> BenchmarkResults:

    # Load dataset
    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        with open(dataset_path) as f:
            scenarios = json.load(f)
            if isinstance(scenarios, dict):
                scenarios = scenarios.get("scenarios", list(scenarios.values()))
    else:
        # HuggingFace Hub dataset
        ds = load_dataset(dataset_path, split="test")
        scenarios = list(ds)

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"Loaded {len(scenarios)} scenarios.")

    runner = ModelRunner(model_id, device=device)
    benchmark = BenchmarkResults(model_id=model_id)

    for scenario in tqdm(scenarios, desc="Evaluating"):
        prompt = build_prompt(scenario)
        try:
            output = runner.generate(prompt)
        except Exception as e:
            output = f"ERROR: {e}"

        result = ScenarioResult(
            scenario_id=scenario["id"],
            category=scenario["category"],
            logic=scenario["logic"],
            domain=scenario["domain"],
            difficulty=scenario["difficulty"],
            model_output=output,
            correct_answer=scenario["correct_answer"],
            classical_answer=scenario["classical_answer"],
            classical_fails=scenario["classical_fails"],
        )
        result = score_result(result)
        benchmark.results.append(result)

        # Small delay to avoid OOM on sequential runs
        time.sleep(0.05)

    benchmark = aggregate(benchmark)

    # Save
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / f"{model_id.replace('/', '_')}_results.json"
    with open(results_file, "w") as f:
        json.dump(asdict(benchmark), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    return benchmark


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NonClassical Logics LLM Benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON or HF Hub ID")
    parser.add_argument("--output", default="./results", help="Output directory")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    results = run_benchmark(
        model_id=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        max_scenarios=args.max_scenarios,
        device=args.device,
    )

    # Print summary
    print("\n" + "="*60)
    print(f"Model: {results.model_id}")
    print(f"Scenarios: {results.total_scenarios}")
    print(f"Overall agreement: {results.overall_agreement:.1%}")
    print(f"Classical failure rate: {results.classical_failure_rate:.1%}")
    print(f"Uncertainty calibration: {results.uncertainty_calibration:.1%}")
    print("\nBy logic:")
    for logic, stats in sorted(results.logic_scores.items()):
        print(f"  {logic:<30} acc={stats['accuracy']:.1%}  avg_score={stats['avg_score']:.2f}  n={stats['n']}")
    print("\nBy difficulty:")
    for diff, stats in sorted(results.difficulty_scores.items()):
        print(f"  {diff:<10} acc={stats['accuracy']:.1%}  avg_score={stats['avg_score']:.2f}")


if __name__ == "__main__":
    main()
