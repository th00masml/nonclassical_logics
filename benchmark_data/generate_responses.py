"""
Generates LLM responses for benchmark scenarios via API.
Outputs in llm_responses.json format compatible with evaluate_responses.py.

Usage:
    # Claude (default)
    python generate_responses.py --benchmark llm_logic_benchmark_full.json --model claude-sonnet-4-20250514

    # OpenAI
    python generate_responses.py --benchmark llm_logic_benchmark_full.json --provider openai --model gpt-4o

    # Dry run (first 5 scenarios)
    python generate_responses.py --benchmark llm_logic_benchmark_full.json --max 5

Requirements:
    pip install anthropic openai tqdm
    set ANTHROPIC_API_KEY=your_key   (Windows CMD)
    $env:ANTHROPIC_API_KEY="your_key"  (PowerShell)
"""

import json
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm


SYSTEM_PROMPT = """You are a formal logic reasoning assistant.

Your task: given a set of premises and a conclusion, determine whether the premises ENTAIL the conclusion under standard logical reasoning.

Respond with exactly one of:
- "entails" - the conclusion follows from the premises
- "does_not_entail" - the conclusion does not follow from the premises

Then provide your confidence on a scale of 1-5 (5 = very confident).
Then provide brief reasoning (2-4 sentences).

Format your response as JSON:
{
  "verdict": "entails" or "does_not_entail",
  "confidence": 1-5,
  "reasoning": "your reasoning here"
}

Respond ONLY with the JSON object. No preamble, no markdown code blocks."""


def build_prompt(scenario: dict) -> str:
    premises_text = "\n".join(f"- {p}" for p in scenario.get("premises", []))
    conclusion = scenario.get("conclusion", scenario.get("query", ""))
    return f"""PREMISES:
{premises_text}

CONCLUSION: {conclusion}

Does the above set of premises entail the conclusion?"""


def call_anthropic(prompt: str, model: str, client) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def call_openai(prompt: str, model: str, client) -> dict:
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def run(benchmark_path, output_path, provider, model, trials, max_scenarios, delay):
    with open(benchmark_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"Loaded {len(scenarios)} scenarios. Trials per scenario: {trials}")
    print(f"Provider: {provider}, Model: {model}")
    print(f"Total API calls: {len(scenarios) * trials}\n")

    existing_responses = []
    if Path(output_path).exists():
        with open(output_path, encoding="utf-8") as f:
            existing_responses = json.load(f)
        existing_keys = {(r["model"], r["item_id"], r["trial"]) for r in existing_responses}
        print(f"Resuming - {len(existing_responses)} responses already exist.")
    else:
        existing_keys = set()

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        call_fn = lambda p: call_anthropic(p, model, client)
    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        call_fn = lambda p: call_openai(p, model, client)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    responses = list(existing_responses)
    errors = 0

    for scenario in tqdm(scenarios, desc="Scenarios"):
        for trial in range(trials):
            key = (model, scenario["id"], trial)
            if key in existing_keys:
                continue

            prompt = build_prompt(scenario)

            try:
                result = call_fn(prompt)
                verdict = result.get("verdict", "").lower().strip()
                if verdict not in ("entails", "does_not_entail"):
                    verdict = "does_not_entail" if "not" in verdict else "entails"

                response = {
                    "model": model,
                    "item_id": scenario["id"],
                    "category": scenario.get("category", ""),
                    "trial": trial,
                    "verdict": verdict,
                    "confidence": int(result.get("confidence", 3)),
                    "reasoning": result.get("reasoning", ""),
                    "expected": scenario.get("expected", {}),
                }
                responses.append(response)

            except Exception as e:
                errors += 1
                print(f"\nError on {scenario['id']} trial {trial}: {e}")
                responses.append({
                    "model": model,
                    "item_id": scenario["id"],
                    "category": scenario.get("category", ""),
                    "trial": trial,
                    "verdict": "error",
                    "confidence": 0,
                    "reasoning": f"ERROR: {str(e)}",
                    "expected": scenario.get("expected", {}),
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=2, ensure_ascii=False)

            time.sleep(delay)

    print(f"\nDone. {len(responses)} responses saved to {output_path}")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="llm_logic_benchmark_full.json")
    parser.add_argument("--output", default="llm_responses_full.json")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    run(args.benchmark, args.output, args.provider, args.model, args.trials, args.max, args.delay)


if __name__ == "__main__":
    main()
