# NonClassical Logics LLM Benchmark

This repository explores how Large Language Models (LLMs) handle non-classical logics, including intuitionistic, paraconsistent, relevance, deontic, temporal, and modal logics.

## Project overview

Large language models are increasingly deployed inside autonomous agent systems in domains that require formal reasoning guarantees: regulatory compliance, clinical triage, fraud detection, and secure robotics protocols.

Classical two-valued logic is the default reasoning apparatus for LLMs, but it is often insufficient for these applications:

- it cannot handle inconsistent evidence without explosive conclusions,
- it assumes complete knowledge,
- it treats truth as binary,
- it lacks native support for time, obligations, and graded certainty.

This project investigates whether LLMs actually reason in line with non-classical logics, or whether they only approximate classical inference through statistical patterns.

## Research motivation

The work is grounded in empirical evaluation of large language models against structured formal reasoning benchmarks. It asks:

- do LLMs respect paraconsistent reasoning or do they still explode on contradictions?
- do they avoid irrelevant implications and reflect relevance logic principles?
- can they represent temporal, deontic, or evidential uncertainty reliably?
- are their reasoning patterns closer to intuitionistic, modal, or classical semantics?

## What’s included

- **Notebook series (01-11)**: analysis from agent communication basics to benchmark execution, logic routing, and fingerprinting.
- **Benchmark data**: scenarios, responses, and evaluation results in JSON/MD formats.
- **Evaluators**: tools for running benchmarks and generating reports.
- **Formal project goal**: design a pipeline for non-classical reasoning evaluation in agent systems.

## Repository structure

- `01_nonclassical_agent_comm_basics.ipynb`
- `02_nonclassical_agent_comm_advanced.ipynb`
- `03_nonclassical_agent_comm_synthesis.ipynb`
- `04_nonclassical_agent_comm_applications.ipynb`
- `05_nonclassical_agent_comm_language.ipynb`
- `06_nonclassical_agent_comm_workflow.ipynb`
- `07w_nonclassical_agent_comm_langgraph.ipynb`
- `08_nonclassical_agent_comm_experimental_composition.ipynb`
- `09_llm_logic_benchmark.ipynb`
- `10_llm_logic_router_eval.ipynb`
- `11_llm_logic_fingerprint.ipynb`
- `benchmark_data/`
- `Evaluators/`
- `CLAUDE.md`

## Key findings

Early experimental results show a strong classical bias in LLM outputs:

- accuracy below random on non-classical benchmark items,
- consistent failure on temporal, deontic, evidential, and constructive witness reasoning,
- high model confidence despite incorrect answers.

These results suggest LLMs often approximate classical reasoning rather than following formal non-classical semantics.

## Project goal

The proposed system, Formalis, aims to:

1. implement a 9-stage verification pipeline using 7 non-classical logics,
2. integrate transparently with agent frameworks like LangGraph, AutoGen, or CrewAI,
3. detect and contain reasoning failures outside the LLM’s native inference,
4. satisfy EU AI Act transparency and explainability requirements for high-risk systems.

## Running the evaluation

See `Evaluators/files/README.md` for detailed instructions on dataset building, model evaluation, and report generation.

## Notes

A more detailed evaluator README is available at `Evaluators/files/README.md`.
