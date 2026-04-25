# Non-Classical Logics in Multi-Agent Communication

A series of eight Jupyter notebooks exploring how non-classical logics shape the way AI agents exchange, update, and reconcile information. Each notebook builds on the last — from individual logic definitions through real-world applications to a full LangGraph pipeline and experimental logic composition.

All evaluators are hand-rolled with no external logic libraries, so the mechanics stay visible.

---

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | `01_nonclassical_agent_comm_basics.ipynb` | Eight foundational logics, one short multi-agent scenario each |
| 02 | `02_nonclassical_agent_comm_advanced.ipynb` | Fourteen rarer logics with richer expressive power |
| 03 | `03_nonclassical_agent_comm_synthesis.ipynb` | Cross-logic benchmarks: the same scenarios replayed under multiple logics |
| 04 | `04_nonclassical_agent_comm_applications.ipynb` | Ten real-world domains where each logic earns its keep |
| 05 | `05_nonclassical_agent_comm_language.ipynb` | Non-classical logics applied to natural-language tasks |
| 06 | `06_nonclassical_agent_comm_workflow.ipynb` | End-to-end pipeline composing the best logics into a single agent |
| 07 | `07w_nonclassical_agent_comm_langgraph.ipynb` | The same pipeline rebuilt as a LangGraph state machine |
| 08 | `08_nonclassical_agent_comm_experimental_composition.ipynb` | What happens when two logics overlap on the same linguistic phenomenon |

---

## Logics covered

### Basics (notebook 01)

| Logic | One-line intuition |
|-------|--------------------|
| Łukasiewicz Ł3 | Three truth values; *unknown* survives. |
| Fuzzy logic | Continuous `[0,1]` truth, t-norms/t-conorms. |
| Modal K / S4 / S5 | `□`/`◇` over Kripke frames. |
| Epistemic logic | Per-agent `K_i` with public announcements. |
| Paraconsistent LP | Contradictions contained, no explosion. |
| Intuitionistic logic | Truth requires a constructive witness. |
| Relevance logic / FDE | Four values, blocks irrelevant inference. |
| Linear Temporal Logic (LTL) | `X`, `F`, `G`, `U` over linear traces. |
| Default logic | Tentative conclusions, retractable. |

### Advanced (notebook 02)

CTL, Alternating-time Temporal Logic (ATL), Dynamic Epistemic Logic (DEL), Public Announcement Logic, AGM belief revision, Possibilistic logic, Subjective logic, Dempster–Shafer evidence theory, Bilattice logic (Belnap), Free logic, Standard Deontic Logic (SDL), Quantum logic, Linear logic, Independence-Friendly (IF) logic.

---

## Applications (notebook 04)

| Domain | Logic |
|--------|-------|
| Self-driving sensor fusion | Bilattice (Belnap) |
| Medical triage | Possibilistic |
| Distributed commit protocol | Epistemic + PAL |
| Smart contracts | Linear logic + SDL |
| Sealed-bid auctions | IF logic |
| Chatbot retraction | Default logic |
| Rover mission planning | CTL |
| Recommender fusion | Dempster–Shafer |
| Legal hypothetical referents | Free logic |
| Federated-learning privacy | DEL action models |

---

## Language tasks (notebook 05)

| Task | Logic |
|------|-------|
| Vague predicates (*tall*, *warm*) | Fuzzy / Łukasiewicz |
| Contradictory dialogue | Paraconsistent LP |
| Presupposition failure | Free logic |
| Conversational implicature & defaults | Default logic |
| Nested belief reports | Epistemic logic |
| Tense and aspect | LTL |
| Hedged translation (*allegedly*, *reportedly*) | Possibilistic |
| Annotator disagreement / moderation | Subjective logic |
| Word-sense fusion | Dempster–Shafer |
| Grounded QA / RAG | Intuitionistic logic |

---

## End-to-end pipeline (notebooks 06 & 07)

A medical-information assistant that uses nine logics as distinct pipeline stages:

```
START
 └─ parse_intent          (Default logic — retractable conversational defaults)
 └─ check_norms           (SDL — forbidden vs. permitted answers)
 └─ retrieve_with_witness (Intuitionistic — claim only if a passage supports it)
 └─ disambiguate_terms    (Dempster–Shafer — belief mass over sense families)
 └─ fuse_evidence         (Possibilistic + LP — graded reliability, tagged contradictions)
 └─ aggregate_opinions    (Subjective logic — residual uncertainty routed to escalation)
 └─ check_privacy         (DEL — epistemic formula for information leakage)
 └─ check_protocol        (LTL — conversation trace must satisfy protocol spec)
 └─ decide                (Bilattice — final {T, F, N, B} collapse → answer / refuse / escalate)
```

Notebook 07 promotes this pipeline to a **LangGraph state machine** with conditional routing, typed state, and a drawable graph topology.

---

## Experimental composition (notebook 08)

Six experiments probing what happens when two logics speak about the same atom simultaneously:

| # | Overlap | Logics |
|---|---------|--------|
| 1 | Vagueness under contradiction | Fuzzy × LP → `FuzzyLP(μ_T, μ_F)` |
| 2 | Presupposition failure inside a hedged report | Free × Possibilistic |
| 3 | Defaults firing on contradicted premises | Default × LP |
| 4 | Tense operators over intuitionistic claims | LTL × Intuitionistic |
| 5 | Annotator disagreement on a vague label | Subjective × Fuzzy |
| 6 | Nested beliefs about a word's sense | Epistemic × Dempster–Shafer |

---

## Setup

```bash
git clone https://github.com/th00masml/nonclassical_logics.git
cd nonclassical_logics
pip install jupyter langgraph
jupyter notebook
```

Python 3.10+. Core logic evaluators have no external dependencies. Notebook 07 requires `langgraph`.

---

## Key references

- Łukasiewicz, J. (1920). *O logice trójwartościowej*. Ruch Filozoficzny 5, 170–171.
- Zadeh, L. A. (1965). *Fuzzy sets*. Information and Control 8(3), 338–353.
- Kripke, S. A. (1963). *Semantical analysis of modal logic I*. Zeitschrift für mathematische Logik 9, 67–96.
- Hintikka, J. (1962). *Knowledge and Belief*. Cornell University Press.
- Priest, G. (1979). *The logic of paradox*. Journal of Philosophical Logic 8(1), 219–241.
- Heyting, A. (1930). *Die formalen Regeln der intuitionistischen Logik*. Sitzungsberichte der Preußischen Akademie.
- Anderson, A. R. & Belnap, N. D. (1975). *Entailment*, Vol. 1. Princeton University Press.
- Pnueli, A. (1977). *The temporal logic of programs*. FOCS '77, 46–57.
- Reiter, R. (1980). *A logic for default reasoning*. Artificial Intelligence 13(1–2), 81–132.
- Girard, J.-Y. (1987). *Linear logic*. Theoretical Computer Science 50(1), 1–101.
- von Wright, G. H. (1951). *Deontic logic*. Mind 60(237), 1–15.
- Belnap, N. D. (1977). *A useful four-valued logic*. In *Modern Uses of Multiple-Valued Logic*, Reidel, 5–37.
- Alchourrón, C. E., Gärdenfors, P. & Makinson, D. (1985). *On the logic of theory change*. Journal of Symbolic Logic 50(2), 510–530.
- Baltag, A., Moss, L. S. & Solecki, S. (1998). *The logic of public announcements, common knowledge, and private suspicions*. TARK '98.
- Jøsang, A. (2001). *A logic for uncertain probabilities*. Int. J. Uncertainty 9(3), 279–311.
- Shafer, G. (1976). *A Mathematical Theory of Evidence*. Princeton University Press.
- Clarke, E. M. & Emerson, E. A. (1981). *Design and synthesis of synchronization skeletons using branching-time temporal logic*. LNCS 131, 52–71.
- Alur, R., Henzinger, T. A. & Kupferman, O. (2002). *Alternating-time temporal logic*. Journal of the ACM 49(5), 672–713.
- Blackburn, P., de Rijke, M. & Venema, Y. (2001). *Modal Logic*. Cambridge University Press.
- Priest, G. (2008). *An Introduction to Non-Classical Logic* (2nd ed.). Cambridge University Press.

---

Ideas for future work
New logics to add

Probabilistic logic / Nilsson (1986) — assign probabilities to classical sentences and propagate via linear programming; natural bridge between LP/possibilistic and full Bayesian reasoning. Interesting comparison point for notebook 03 benchmarks.
Conditional logic (Stalnaker / Lewis) — φ > ψ ("if φ were the case, ψ would hold") interpreted over similarity-ordered worlds. Relevant for counterfactual reasoning in chatbot retraction (notebook 04 §6) and implicature (notebook 05 §4).
Description logics (ALC, EL, OWL) — the logical backbone of knowledge graphs. Would fit naturally into the retrieval-grounding stage of the pipeline (notebook 06 §2) as a typed ontology layer on top of intuitionistic witnesses.
Hybrid logic — adds nominals (names for specific worlds) to modal logic; @_i φ means "at world i, φ holds". Useful when agents need to reason about specific named situations, not just possible ones.
Separation logic — extends Hoare logic with * (separating conjunction) for reasoning about disjoint memory regions. Could replace or complement linear logic in the smart-contract scenario (notebook 04 §4).
Rough set logic — approximation spaces with lower/upper bounds on set membership; complementary to fuzzy for discrete, categorical uncertainty rather than graded.

Extensions to existing notebooks
Notebook 03 (Synthesis)

Add a third benchmark scenario: conflicting temporal reports (two agents disagree about when an event happened, not just whether). Would bring LTL and CTL into the cross-logic comparison.
Quantitative noise sweep: currently checks agreement at fixed conflict level; vary the conflict rate from 0% to 100% and plot how each logic's "survival rate" degrades. Reveals which logics are robust vs. brittle.

Notebook 04 (Applications)

Rover scenario (§7): replace CTL with ATL to model two competing rovers each trying to satisfy their own mission spec — coalition quantifier <<C>> makes the adversarial aspect explicit.
Legal referents (§9): extend free logic scenario to partial denotation (a term denotes in some worlds but not others), bridging free logic and modal logic.

Notebook 05 (Language)

Test the grounded QA / intuitionistic RAG scenario (§10) against a real retrieval pipeline (FAISS + small LM) and measure hallucination rate with vs. without witness requirement.
Add a section on scalar implicature (default logic + Łukasiewicz): "some students passed" implicates "not all"; default rule fires unless explicitly blocked by "in fact, all did".

Notebook 06 / 07 (Pipeline)

Add a confidence-threshold router between fuse_evidence and decide: if possibilistic weight drops below a tunable threshold, route directly to escalation without going through subjective aggregation. Makes the pipeline adaptive.
Swap out the toy retriever in retrieve_with_witness for a real one (ChromaDB / FAISS) and measure end-to-end latency and refusal rate on a small QA dataset.
Benchmark the LangGraph version (notebook 07) against a flat-function baseline (notebook 06) on throughput and correctness — does the graph overhead pay off at scale?

Experimental composition (notebook 08 extensions)

Experiment 7: AGM revision under fuzzy evidence — what is the right revision policy when the incoming evidence is itself a fuzzy degree rather than a crisp formula? Does the Levi identity still hold?
Experiment 8: Default logic inside a DEL action model — can a private announcement trigger or block a default? Tests whether the pipeline's stage-isolation assumption holds when events are not public.
Experiment 9: Linear logic resources consumed by epistemic updates — model the cognitive cost of learning as a consumable resource; !K_i φ (reusable knowledge) vs. K_i φ (one-shot information). Connects subjective logic uncertainty to resource accounting.
Composition laws summary table — each experiment currently ends with an ad hoc observation; consolidate into a matrix (Logic A) × (Logic B) → {safe | requires new type | degenerates to classical | explodes}.

Testing and evaluation

Property-based tests (Hypothesis library) for every hand-rolled evaluator: e.g. for Łukasiewicz, verify ¬¬φ = φ for all φ ∈ {0, 0.5, 1}, double-negation fails for intermediate values in FDE, etc.
Cross-notebook regression suite: fix a set of canonical scenarios (contradictory sensor reports, medical triage, sealed-bid auction) and assert that the logic-specific evaluators give the same outputs as notebook 03 after any refactor.
LLM-in-the-loop variant: replace the hand-rolled evaluators with LLM calls that are prompted to reason in the target logic, then compare outputs to the ground-truth hand-rolled results. Measures how well current LLMs internalise non-classical semantics.
Ablation on pipeline stage order (notebook 06/07): permute the order of stages and measure how many test traces change outcome. Identifies which stages are truly order-independent and which have hidden dependencies.

## License

MIT
