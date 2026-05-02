# NonClassical Logics LLM Benchmark — Evaluator

Evaluates LLMs on 70 scenarios across 7 non-classical logics.

## Project Overview

This repository explores how Large Language Models (LLMs) handle non-classical logics, including intuitionistic, paraconsistent, relevance, and modal logics. The project includes:

- **Notebook series (01-11)**: Step-by-step analysis, from basics of agent communication to advanced benchmarks, logic routing, and fingerprinting.
- **Benchmark Data**: Scenarios, responses, and evaluation results in JSON/MD formats.
- **Evaluators**: Automated tools for running benchmarks and generating reports.

Key insights from notebooks 01-11: LLMs often default to classical logic but show biases towards paraconsistent reasoning (e.g., resisting explosion) and relevance awareness.

## General Conclusions

Based on the analysis across all notebooks (01-11), the project reveals several expanded insights into how LLMs handle non-classical logics:

- **Dominance of Classical Defaults**: Most LLMs align with classical logic as a baseline, providing truth-functional reasoning, but deviations occur in specific scenarios like contradictions and vacuous conditionals.
- **Paraconsistent Tendencies**: Consistent rejection of ex falso quodlibet (EFQ) across models suggests implicit paraconsistent logic, where contradictions do not lead to arbitrary conclusions.
- **Relevance Awareness**: LLMs frequently reject irrelevant or vacuous implications, indicating an underlying sensitivity to relevance logic principles.
- **Scale and Consistency**: Larger models exhibit more stable logic fingerprints, while smaller ones show higher variability, supporting the hypothesis that model scale contributes to logical coherence.
- **Priming Potential**: Logic fingerprints can be influenced by prompts, opening possibilities for controllable logical behavior in LLMs across the notebook series 01-11.
- **Agent Communication Insights**: Notebooks 01-08 on agent communication highlight how non-classical logics can enhance multi-agent interactions, reducing paradoxes and improving robustness in scenarios involving uncertainty or incomplete information. Notebooks 09-11 then extend these insights to benchmarking, routing, and fingerprint analysis of model reasoning.
- **Benchmark Limitations and Recommendations**: Current benchmarks from the full notebook series 01-11 are small-scale (~12 items); expanding to 50+ items per logic with multiple paraphrases, prompt-stability tests, and cross-language evaluations (e.g., Polish/Mandarin) is essential for generalizable results.
- **Chain-of-Thought Analysis**: Encouraging step-by-step reasoning in LLMs reveals traces of logical moves (e.g., avoiding LEM in intuitionistic contexts), providing deeper insights into implicit logical frameworks across the entire notebook series.
- **Cross-Logic Composition**: Combining results from notebook 03 in the 01-11 series with LLM benchmarks shows discrepancies, suggesting LLMs do not fully emulate formal logics but approximate them heuristically.
- **Implications for AI Development**: These findings suggest designing AI systems with explicit logical priming to tailor behavior for specific domains, such as paraconsistent reasoning for error-tolerant applications or relevance logic for focused inference.
- **Future Directions**: Investigate logic priming across languages, integrate with tools like LangGraph (notebook 07w in the 01-11 series), and explore modal logics for temporal or epistemic reasoning in LLMs.

## Structure

```
benchmark/
  evaluator.py       # Main evaluation pipeline
  build_dataset.py   # Converts scenarios JSON → HuggingFace Dataset
  report.py          # Generates paper-ready tables from results
  requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Build dataset from scenarios JSON
python build_dataset.py \
  --input ../scenarios/all_scenarios.json \
  --output ./hf_dataset \
  --push-to-hub your_username/nonclassical-logics-benchmark  # optional

# 2. Run evaluation — Llama 8B on 4090
python evaluator.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset ./hf_dataset \
  --output ./results

# 3. Run evaluation — Mistral 7B
python evaluator.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset ./hf_dataset \
  --output ./results

# 4. Generate report
python report.py \
  --results ./results \
  --output ./results/report.md
```

## Scoring

Current scoring is heuristic (keyword overlap). For paper-quality results,
replace `score_result()` in `evaluator.py` with one of:

- **LLM-as-judge**: pass (model_output, correct_answer) to GPT-4 / Claude
- **Human annotation**: export results CSV, annotate manually

## Related Notebooks

- **01_nonclassical_agent_comm_basics.ipynb**: Basics of agent communication in non-classical logics.
- **02_nonclassical_agent_comm_advanced.ipynb**: Advanced concepts in agent communication.
- **03_nonclassical_agent_comm_synthesis.ipynb**: Synthesis of communication strategies.
- **04_nonclassical_agent_comm_applications.ipynb**: Practical applications of agent communication.
- **05_nonclassical_agent_comm_language.ipynb**: Language aspects in non-classical agent communication.
- **06_nonclassical_agent_comm_workflow.ipynb**: Workflow for agent communication.
- **07w_nonclassical_agent_comm_langgraph.ipynb**: LangGraph integration for agent communication.
- **08_nonclassical_agent_comm_experimental_composition.ipynb**: Experimental composition techniques.
- **09_llm_logic_benchmark.ipynb**: Full benchmark execution and results visualization.
- **10_llm_logic_router_eval.ipynb**: Evaluation of logic routing in LLMs.
- **11_llm_logic_fingerprint.ipynb**: Logic fingerprint analysis — hypotheses, conclusions, and observations on LLM logical behavior.
- **Formal verifier**: for LTL/DS scenarios, add domain-specific checkers

## Metrics

| Metric | Description |
|--------|-------------|
| `overall_agreement` | % scenarios where model answer agrees with correct answer |
| `classical_failure_rate` | % of classical_fails=True scenarios where model gives classical (wrong) answer |
| `uncertainty_calibration` | % of UNKNOWN/NOT_PROVABLE scenarios where model flags uncertainty |
| `escalation_rate` | % of ESCALATE scenarios where model recommends escalation |

## Adding models

Pass any HuggingFace model ID to `--model`. Works with:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-70B-Instruct` (requires 4×A100 or quantized)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2-7B-Instruct`

For GPT-4 baseline, implement `OpenAIRunner` following same interface as `ModelRunner`.

## LLM-as-judge scorer (recommended for paper)

Replace `score_result()` with:

```python
def score_result_llm(result: ScenarioResult, judge_client) -> ScenarioResult:
    prompt = f\"\"\"
    Correct answer: {result.correct_answer}
    Model output: {result.model_output}
    
    Does the model output capture the key insight of the correct answer?
    Answer: YES or NO, then one sentence explanation.
    \"\"\"
    judgment = judge_client.generate(prompt)
    result.agrees_with_correct = "YES" in judgment.upper()
    result.score = 1.0 if result.agrees_with_correct else 0.0
    return result
```


Problem badawczy
Wielkoskalowe modele językowe (LLM) są coraz powszechniej wdrażane jako komponenty autonomicznych systemów agentowych w obszarach wymagających formalnych gwarancji poprawności rozumowania: compliance regulacyjny, triage kliniczny, wykrywanie nadużyć finansowych, weryfikacja protokołów bezpieczeństwa w systemach robotycznych. Klasyczna logika dwuwartościowa, stanowiąca domyślny aparat wnioskowania LLM, jest strukturalnie niewystarczająca dla tych zastosowań: nie obsługuje sprzecznych dowodów bez eksplozji logicznej, wymaga kompletności wiedzy, traktuje prawdziwość binarnie i nie posiada natywnej reprezentacji czasu, zobowiązań normatywnych ani gradualnych stopni pewności.
Nieklasyczne systemy logiczne logika parakonsystentna LP, logika epistemiczna i domyślna, deontyczna logika SDL, logika rozmyta Łukasiewicza, liniowa logika temporalna LTL, logika intuicjonistyczna oraz teoria evidencji Dempstera-Shafera zostały formalnie uzasadnione jako właściwe frameworki dla tych domenowych wymagań. Kluczowe pytanie badawcze brzmi: czy LLM-y w ogóle wnioskują zgodnie z tymi logikami, czy jedynie aproksymują klasyczne wnioskowanie przez dopasowanie statystyczne?
Wnioskodawca przeprowadził wstępne badania empiryczne (proof-of-concept, dostępny publicznie na GitHub: github.com/th00masml/nonclassical_logics) obejmujące opracowanie benchmarku 82 scenariuszy formalnych z poprawnymi odpowiedziami wywiedzionymi z semantyki każdej z 7 logik oraz ewaluację modelu Claude Sonnet 4.5. Wyniki wykazują systematyczny bias klasyczny: dokładność modelu wynosi poniżej poziomu losowego dla wszystkich czterech ewaluowanych systemów logicznych (klasyczny: 41,1%, parakonsystentny: 36,9%, intuicjonistyczny: 31,0%, logika relewancji: 29,8%). Cztery kategorie rozumowania — temporalne, deontyczne, kombinacja evidencji oraz wymaganie konstruktywnego świadka — wykazują zerową dokładność przy jednoczesnej wysokiej pewności modelu (średnia 4,6/5,0), wskazując na systematyczne błędy pewności siebie, nie losowe.
Cel projektu
Celem projektu jest opracowanie, implementacja i walidacja systemu Formalis — który:

Implementuje 9-etapowy pipeline weryfikacji wykorzystujący 7 nieklasycznych logik jako odrębne moduły ewaluacji
Integruje się z dowolnym frameworkiem agentowym (LangGraph, AutoGen, CrewAI) jako transparentna warstwa pośrednicząca
Wykrywa i zawiera kategorie błędów rozumowania nieobsługiwane przez natywny aparat wnioskowania LLM
Spełnia wymagania art. 13 EU AI Act dotyczące transparentności i wyjaśnialności dla systemów AI wysokiego ryzyka
