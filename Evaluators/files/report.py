"""
Report generator — reads results JSON(s) and produces:
  - Markdown tables for the paper
  - Per-logic breakdown
  - Per-model comparison table
  - Failure mode analysis

Usage:
    python report.py --results ./results/
    python report.py --results ./results/ --compare llama_8b mistral_7b llama_70b
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def model_short_name(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", "_")


def table_overall(results_list: list[dict]) -> str:
    header = "| Model | Overall Acc | Classical Fail Rate | Uncertainty Cal | Scenarios |"
    sep =    "|-------|-------------|---------------------|-----------------|-----------|"
    rows = []
    for r in results_list:
        rows.append(
            f"| {model_short_name(r['model_id'])} "
            f"| {r['overall_agreement']:.1%} "
            f"| {r['classical_failure_rate']:.1%} "
            f"| {r['uncertainty_calibration']:.1%} "
            f"| {r['total_scenarios']} |"
        )
    return "\n".join([header, sep] + rows)


def table_by_logic(results_list: list[dict]) -> str:
    # Collect all logic names
    all_logics = set()
    for r in results_list:
        all_logics.update(r.get("logic_scores", {}).keys())
    all_logics = sorted(all_logics)

    model_names = [model_short_name(r["model_id"]) for r in results_list]
    header = "| Logic | " + " | ".join(model_names) + " |"
    sep = "|" + "---|" * (len(model_names) + 1)

    rows = []
    for logic in all_logics:
        cells = []
        for r in results_list:
            stats = r.get("logic_scores", {}).get(logic, {})
            acc = stats.get("accuracy", 0.0)
            cells.append(f"{acc:.1%}")
        rows.append(f"| {logic} | " + " | ".join(cells) + " |")

    return "\n".join([header, sep] + rows)


def table_by_difficulty(results_list: list[dict]) -> str:
    model_names = [model_short_name(r["model_id"]) for r in results_list]
    header = "| Difficulty | " + " | ".join(model_names) + " |"
    sep = "|" + "---|" * (len(model_names) + 1)

    rows = []
    for diff in ["easy", "medium", "hard"]:
        cells = []
        for r in results_list:
            stats = r.get("difficulty_scores", {}).get(diff, {})
            acc = stats.get("accuracy", 0.0)
            cells.append(f"{acc:.1%}")
        rows.append(f"| {diff} | " + " | ".join(cells) + " |")

    return "\n".join([header, sep] + rows)


def failure_analysis(results: dict) -> str:
    """
    Identify scenarios where model consistently fails —
    these are the most interesting for paper analysis section.
    """
    scenario_results = defaultdict(list)
    for r in results.get("results", []):
        scenario_results[r["scenario_id"]].append(r)

    failures = []
    for sid, rs in scenario_results.items():
        if all(not r.get("agrees_with_correct") for r in rs):
            failures.append({
                "id": sid,
                "logic": rs[0]["logic"],
                "domain": rs[0]["domain"],
                "difficulty": rs[0]["difficulty"],
                "classical_fails": rs[0]["classical_fails"],
            })

    if not failures:
        return "No consistent failures found."

    lines = ["### Consistent Failure Cases\n"]
    lines.append("| ID | Logic | Domain | Difficulty | Classical Trap |")
    lines.append("|-------|-------|--------|------------|----------------|")
    for f in sorted(failures, key=lambda x: x["difficulty"]):
        lines.append(
            f"| {f['id']} | {f['logic']} | {f['domain']} "
            f"| {f['difficulty']} | {'✓' if f['classical_fails'] else '✗'} |"
        )
    return "\n".join(lines)


def generate_report(results_list: list[dict], output_path: str):
    report = []
    report.append("# NonClassical Logics Benchmark — Results\n")

    report.append("## Overall Performance\n")
    report.append(table_overall(results_list))
    report.append("")

    if len(results_list) > 0:
        report.append("## Accuracy by Logic\n")
        report.append(table_by_logic(results_list))
        report.append("")

        report.append("## Accuracy by Difficulty\n")
        report.append(table_by_difficulty(results_list))
        report.append("")

    for r in results_list:
        report.append(f"## Failure Analysis — {model_short_name(r['model_id'])}\n")
        report.append(failure_analysis(r))
        report.append("")

    report_text = "\n".join(report)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(report_text)
    print(f"Report saved to {out}")
    print("\n" + report_text[:2000])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Directory containing results JSON files")
    parser.add_argument("--output", default="./results/report.md")
    args = parser.parse_args()

    results_dir = Path(args.results)
    result_files = list(results_dir.glob("*_results.json"))

    if not result_files:
        print(f"No *_results.json files found in {results_dir}")
        return

    results_list = [load_results(f) for f in sorted(result_files)]
    print(f"Loaded {len(results_list)} model results.")

    generate_report(results_list, args.output)


if __name__ == "__main__":
    main()
