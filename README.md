# NonClassical Logics LLM Benchmark — Evaluator

Evaluates LLMs on 70 scenarios across 7 non-classical logics.

## Project Overview Options

This repository explores how Large Language Models (LLMs) handle non-classical logics, including intuitionistic, paraconsistent, relevance, deontic, temporal, and modal logics. The project includes:

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

## Notebook Series (01-11)

### Agent Communication Basics (Notebooks 01-08)

- **01_nonclassical_agent_comm_basics.ipynb**: Foundations of agent communication in non-classical logic frameworks. Introduces basic agent interactions and reasoning constraints.
- **02_nonclassical_agent_comm_advanced.ipynb**: Advanced agent communication patterns including multi-turn dialogue, conflict resolution, and knowledge sharing under non-classical semantics.
- **03_nonclassical_agent_comm_synthesis.ipynb**: Synthesis and composition of communication strategies. Combining multiple logic systems within single agent conversations.
- **04_nonclassical_agent_comm_applications.ipynb**: Real-world application domains for non-classical agent communication: compliance, triage, fraud detection, robotics.
- **05_nonclassical_agent_comm_language.ipynb**: Language pragmatics and linguistic aspects of non-classical reasoning. How different semantic frameworks affect expression and interpretation.
- **06_nonclassical_agent_comm_workflow.ipynb**: End-to-end workflow design for agent communication pipelines. Integration patterns and deployment considerations.
- **07w_nonclassical_agent_comm_langgraph.ipynb**: LangGraph integration for structured multi-agent systems. State machines, transitions, and event handling in non-classical settings.
- **08_nonclassical_agent_comm_experimental_composition.ipynb**: Experimental techniques for composing heterogeneous logical systems. Mixing classical, paraconsistent, and intuitionistic reasoning in single workflows.

### Benchmark & Evaluation (Notebooks 09-11)

- **09_llm_logic_benchmark.ipynb**: Full benchmark execution framework. Scenario generation, model evaluation, response collection, and result aggregation for all 7 non-classical logics.
- **10_llm_logic_router_eval.ipynb**: Logic routing evaluation—can LLMs recognize which logic applies to a scenario? Classification accuracy and routing confidence metrics.
- **11_llm_logic_fingerprint.ipynb**: Logic fingerprint analysis. Clustering model reasoning patterns, extracting implicit logical preferences, hypothesis testing on LLM semantic biases.

## Structure

```
nonclassical_logics/
  ├── 01_nonclassical_agent_comm_basics.ipynb
  ├── 02_nonclassical_agent_comm_advanced.ipynb
  ├── 03_nonclassical_agent_comm_synthesis.ipynb
  ├── 04_nonclassical_agent_comm_applications.ipynb
  ├── 05_nonclassical_agent_comm_language.ipynb
  ├── 06_nonclassical_agent_comm_workflow.ipynb
  ├── 07w_nonclassical_agent_comm_langgraph.ipynb
  ├── 08_nonclassical_agent_comm_experimental_composition.ipynb
  ├── 09_llm_logic_benchmark.ipynb
  ├── 10_llm_logic_router_eval.ipynb
  ├── 11_llm_logic_fingerprint.ipynb
  ├── benchmark_data/
  │   ├── scenarios/
  │   ├── responses/
  │   ├── evaluation_results.json
  │   ├── evaluation_results.md
  │   ├── logic_fingerprints.json
  │   └── convert_scenarios.py
  ├── Evaluators/files/
  │   ├── evaluator.py       # Main evaluation pipeline
  │   ├── build_dataset.py   # Converts scenarios JSON → HuggingFace Dataset
  │   ├── report.py          # Generates paper-ready tables from results
  │   └── requirements.txt
  └── CLAUDE.md
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

## Remote / Distributed Testing

For remote evaluation on dedicated GPU hosts or cluster nodes:

- Copy the dataset and evaluator files to the remote machine or mount via network storage.
- Use the same `python evaluator.py` command on the remote host.
- For large models, prefer remote GPUs or quantized runtimes:
  - `meta-llama/Meta-Llama-3-70B-Instruct` requires 4×A100 or an 8-bit/4-bit quantized runtime.
  - `mistralai/Mistral-7B-Instruct-v0.3` works well on a single 24GB GPU.
- If you need to run multiple models in parallel, use a remote job scheduler or SSH script to launch separate runs.
- Copy `./results` back to your local machine for reporting and analysis.

## Scoring

Current scoring is heuristic (keyword overlap). For paper-quality results, replace `score_result()` in `evaluator.py` with one of:

- **LLM-as-judge**: pass (model_output, correct_answer) to GPT-4 / Claude
- **Human annotation**: export results CSV, annotate manually

## Metrics

| Metric | Description |
|--------|-------------|
| `overall_agreement` | % scenarios where model answer agrees with correct answer |
| `classical_failure_rate` | % of classical_fails=True scenarios where model gives classical (wrong) answer |
| `uncertainty_calibration` | % of UNKNOWN/NOT_PROVABLE scenarios where model flags uncertainty |
| `escalation_rate` | % of ESCALATE scenarios where model recommends escalation |

## Adding Models

Pass any HuggingFace model ID to `--model`. Works with:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-70B-Instruct` (requires 4×A100 or quantized)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `Qwen/Qwen2-7B-Instruct`
- `claude-opus-4` (requires Anthropic API key)
- `gpt-4` (requires OpenAI API key)

For custom model runners, implement `ModelRunner` interface.

## LLM-as-judge Scorer (Recommended)

Replace `score_result()` with:

```python
def score_result_llm(result: ScenarioResult, judge_client) -> ScenarioResult:
    prompt = f"""
    Correct answer: {result.correct_answer}
    Model output: {result.model_output}
    
    Does the model output capture the key insight of the correct answer?
    Answer: YES or NO, then one sentence explanation.
    """
    judgment = judge_client.generate(prompt)
    result.agrees_with_correct = "YES" in judgment.upper()
    result.score = 1.0 if result.agrees_with_correct else 0.0
    return result
```

---

## Research Problem Statement (Problem badawczy)

### Motivation

Large language models (LLMs) are increasingly deployed as components of autonomous agent systems in domains requiring formal reasoning guarantees:
- Regulatory compliance
- Clinical triage
- Fraud detection
- Security protocol verification in robotic systems

However, classical two-valued logic—the default reasoning apparatus for LLMs—is structurally insufficient for these applications:

- Cannot handle contradictory evidence without logical explosion
- Requires complete knowledge (closed-world assumption)
- Treats truth as binary (no graded uncertainty)
- Lacks native support for time, normative obligations, or graded certainty degrees

### Research Question

Do LLMs reason in accordance with non-classical logic frameworks, or do they merely approximate classical inference through statistical pattern matching?

### Non-Classical Logic Systems Under Investigation

- **Paraconsistent Logic (LP)**: Handles contradictions without explosion
- **Epistemic & Default Logic**: Models incomplete knowledge and reasoning under uncertainty
- **Deontic Logic (SDL)**: Represents normative obligations and permissions
- **Fuzzy Logic (Łukasiewicz)**: Handles graded truth values
- **Linear Temporal Logic (LTL)**: Represents temporal reasoning and sequences
- **Intuitionistic Logic**: Constructive reasoning, rejection of excluded middle
- **Dempster-Shafer Theory**: Formal evidence combination framework

### Empirical Evidence (Proof-of-Concept)

Preliminary empirical evaluation of Claude Sonnet 4.5 against a benchmark of 82 formal scenarios with correct answers derived from each logic's semantics revealed:

**Accuracy Results (Below Random):**
- Classical logic: 41.1%
- Paraconsistent logic: 36.9%
- Intuitionistic logic: 31.0%
- Relevance logic: 29.8%

**Critical Failure Categories (Zero Accuracy):**
- Temporal reasoning
- Deontic reasoning
- Evidence combination
- Constructive witness requirements

**Overconfidence Pattern:**
- Average model confidence: 4.6/5.0 despite incorrect answers
- Suggests systematic confidence calibration failures, not random error

### Conclusion from Preliminary Work

LLMs exhibit a strong **classical bias** and often approximate classical reasoning rather than following formal non-classical semantics. This poses significant risk in high-stakes domains requiring multi-logic reasoning.

---

## Project Goal: Formalis System

The objective is to develop, implement, and validate **Formalis**—a verification pipeline system that:

### Core Requirements

1. **9-Stage Verification Pipeline**: Implements 7 non-classical logics as distinct evaluation modules
   - Stage 1: Input parsing and scenario classification
   - Stage 2-8: Logic-specific reasoning verification (one per logic)
   - Stage 9: Reasoning conflict detection and escalation

2. **Agent Framework Integration**: Transparent middleware layer for:
   - LangGraph
   - AutoGen
   - CrewAI
   - Custom agent systems

3. **Reasoning Error Detection & Containment**:
   - Identifies reasoning failures outside LLM's native inference apparatus
   - Flags high-uncertainty outputs for human review
   - Proposes alternative reasoning paths

4. **EU AI Act Compliance** (Article 13):
   - Transparency: Explicit reasoning trace visibility
   - Explainability: Per-decision audit logs
   - High-risk certification for agent systems

### System Architecture

```
Agent Input
    ↓
[Formalis Verification Pipeline]
    ├─ Stage 1: Classification (which logics apply?)
    ├─ Stage 2: Classical Logic Check
    ├─ Stage 3: Paraconsistent Logic Check
    ├─ Stage 4: Intuitionistic Logic Check
    ├─ Stage 5: Relevance Logic Check
    ├─ Stage 6: Deontic Logic Check
    ├─ Stage 7: Temporal Logic Check
    ├─ Stage 8: Evidential Logic Check
    └─ Stage 9: Conflict Resolution & Escalation
    ↓
[Audit Trail + Decision Recommendations]
    ↓
Agent Output / Human Review
```

### Deliverables

- Formalis pipeline implementation (Python)
- Integration adapters for LangGraph, AutoGen, CrewAI
- Benchmark datasets (70-500 scenarios per logic)
- Evaluation metrics and reporting
- EU AI Act compliance documentation
- Academic publication (peer review)

### Success Metrics

- **Reasoning Accuracy**: > 80% on non-classical benchmarks
- **Confidence Calibration**: Overconfidence gap < 10%
- **Latency**: < 500ms per decision (with agent framework integration)
- **False Positive Rate**: < 5% escalation on valid classical reasoning
- **Auditability**: 100% decision traceability

---

## Roadmap & Milestones

### Phase 1: Foundation (Notebooks 01-05)
- Establish agent communication patterns in non-classical logics
- Define benchmark scenarios for each logic system
- Implement baseline evaluators

### Phase 2: Benchmarking (Notebooks 06-09)
- LangGraph integration
- Full benchmark execution on multiple models
- Comparative analysis (model scales, architectures)

### Phase 3: Analysis & Fingerprinting (Notebooks 10-11)
- Logic routing evaluation
- Model fingerprint extraction
- Hypothesis testing on LLM biases

### Phase 4: Formalis System Development (Future)
- 9-stage verification pipeline
- Agent framework middleware
- EU AI Act compliance harness
- Production deployment

### Phase 5: Expansion
- Cross-language evaluation (Polish, Mandarin, others)
- Modal logics integration
- Temporal/epistemic reasoning extensions
- Continuous monitoring in production systems

---

## Contributing

Contributions welcome for:
- Additional benchmark scenarios
- New model evaluations
- Agent framework integrations
- Non-classical logic extensions
- EU AI Act compliance features

Please open an issue or PR with your proposal.

## License

MIT License

## Contact

For questions or collaboration inquiries, please reach out via GitHub issues or email.
