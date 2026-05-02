# NonClassical Logics LLM Benchmark — Evaluator

Evaluates LLMs on 70 scenarios across 7 non-classical logics.

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
