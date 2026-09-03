# PROJECT_STATE

Development log for **catalyst-kg-agent**. The README is the public-facing document; this file is the working record of what was built, what broke, and what was decided.

**Last updated:** 2026-09-03

---

## Current status

The pipeline runs end to end and the surrogate evaluation has been completed with results reported in the README. A substantial correctness pass was carried out on 2026-09-01/02 (see [Correctness pass](#correctness-pass-2026-09-0102)); several previously-claimed results turned out to be invalid and were either fixed or withdrawn.

**Working:**

- MP data pull → KG build → 686 nodes, 921 edges, 130 materials, 390 properties, 0 CIF failures
- MACE surrogate with formation energies via self-consistent elemental references (validated to 12 meV/atom against MP on mp-2790)
- Residual-force escalation gate, calibrated against the corpus distribution and **firing on real candidates** (mp-943, Co₃S₄, 0.795 eV/Å)
- CGCNN baseline trained and cross-validated, with composition-disjoint splits and out-of-fold predictions
- MACE vs CGCNN comparison, reported split by oxide/non-oxide
- Budget-bounded campaign loop with journal + MLflow logging; three-campaign trace verified end to end (see [Campaign trace](#campaign-trace-2026-09-03))
- Two genuine LLM agents: Retriever (query → structured intent) and Planner (candidate prioritisation)

**Not done / simulated:**

- The "expensive experiment" escalation deducts cost and logs; nothing is actually computed
- No catalytic-activity modelling of any kind
- UMA relaxation showcase requires FAIRChem (not a listed dependency); reports "skipped" without it

---

## Correctness pass (2026-09-01/02)

A file-by-file review found that several components either could not run or were producing physically wrong numbers. Recorded here because the earlier version of this document reported some of these as verified.

### Critical: results that were wrong

| Issue | Impact | Resolution |
|---|---|---|
| **Missing periodic boundary conditions** | `predictor.py` rebuilt pymatgen's `MSONAtoms` into a fresh `ase.Atoms` without `pbc=`, which defaults to `False`. MACE evaluated every crystal as an isolated cluster in vacuum. Formation energies were wrong by ~1.6 eV/atom and came out **positive for hull-stable compounds**. `elemental_references.py` never re-wrapped, so references were correct — the asymmetry was the entire error. | Removed the re-wrap. `MSONAtoms` is an `ase.Atoms` subclass and is accepted directly. mp-2790 went from +1.1035 to −0.4837 eV/atom vs MP's −0.4714 (a 128× improvement). |
| **`e_above_hull` was never computed** | `predictor.py` referenced an undefined variable `structure` inside the formation-energy block. The `NameError` was swallowed by a bare `except`, falling through to return the raw MACE energy per atom **labelled as e_above_hull**. This produced the −4.5591 "e_above_hull" recorded throughout the previous PROJECT_STATE — a physically impossible negative hull distance. | Removed the fake calculation. The predictor now reports raw energy per atom and a properly-referenced formation energy, and states plainly that it does not compute e_above_hull. Stability is read from the KG's MP value. |
| **MC-Dropout uncertainty was identically zero** | MACE inference is deterministic with no active dropout, so N "dropout passes" returned bit-identical energies and `np.std` was exactly 0.0 for every material. The Critic's uncertainty gate could never fire on any input — the escalation path was dead in practice. Tests passed only because they injected uncertainty values by hand. | Replaced with **max residual force** (eV/Å) as the trust signal. See [Escalation signal](#escalation-signal-design-note). |
| **`kg/schema.py` would not parse** | Decorator arguments had been merged into adjacent docstrings (`propesource", mode="before")`, `@field_validator("rty. Use ...`). `SyntaxError` on import; nothing depending on it could run. | Reassembled the `PropertyNode` block from the intact fragments. |
| **Scribe would have corrupted MP data** | `_add_prediction_to_kg` hardcoded `prop_name = "energy_above_hull"` and averaged MACE predictions into the same PropertyNode holding MP's ground-truth value. | MACE predictions now write to `mace_energy_per_atom`, never merged with MP values. |

### Components that could not run at all

- `MACE_CGCNN_surrogate_comparison.py` — **SyntaxError** (stray un-commented text). Also: missing `import sys`, `G` used as an undefined global in three functions, `config.CGCNN_CONFIG` (nonexistent), `evaluate(...)[:0]` returning an empty tuple, non-existent pymatgen methods, `np.polyfit(x, x, 1)` correlating a variable with itself. Rewritten from scratch as `models/surrogate_comparison.py`.
- `UMA_final_fidelity_check.py` — missing `import sys`, undefined `G`, nonexistent `fairchem.relaxation.relax` API, keys read that were never written, `mp_eah:.4f` on a possible `None`. **Deleted**; the notebook is now the single UMA artifact.
- `baseline_cgcnn.py` — training, prediction, and comparison functions were all `[TODO]` print stubs. The model itself would not have run: `edge_index` built transposed, `dx_cart[:, 0]` indexing a 1-D array, a 94-dim one-hot layer fed 1-D atomic numbers, `DataLoader` imported from the wrong module, and a duplicated `Th` entry in the atom table holding thallium's values. Rewritten as a working implementation.

### Structural bugs

- `agent/planner.py` was fully implemented and tested but **never imported by `campaign.py`** — the orchestrator ran its own inline loop while the README diagram described the Planner as live. Now wired in: campaign.py owns the `BudgetTracker`, Planner advises on continue/escalate/stop.
- Batch mode only ever persisted the **last step's** predictions (`predictions` was reassigned each iteration), and a `step > 1` guard meant single-iteration campaigns persisted nothing at all. Now accumulates across all steps.
- `tracker.deduct()`'s return value was ignored in the prediction loop, allowing overspend past zero.
- `kg/queries.py` — three methods (`edge_types`, `node_types`, `custom_filter`) were permanently shadowed by instance attributes of the same name; calling them raised `TypeError: 'list' object is not callable`. `find_materials_by_property_range` hardcoded a dummy start node `material:mp-123` and raised `ValueError` on any graph lacking it.
- `kg/build_graph.py` — `crystal_system` was `None` on all 130 structures (the key does not exist in `get_symmetry_dataset()`'s return; it needs `SpacegroupAnalyzer`). CIF paths were stored with Windows backslashes, unusable on Linux/macOS. The CIF cache was re-loaded and re-dumped once per material — O(N²) file I/O.
- `kg/graph_store.py` — dead branch referencing an undefined `edge_data`.

### Documentation corruption

The previous PROJECT_STATE.md contained systematic text corruption: `\u0007gent/campaign.py`, `\formula_pretty`, `\band_gap`, `\num_sites`, `` `ash `` code fences. Root cause: the file was processed with **escape-sequence interpretation**, so Windows-style paths like `\agent` became `\a` (bell) + `gent`, `\band_gap` became backspace + `and_gap`, and so on. Avoid running this file through any tool that interprets backslash escapes.

---

## Escalation signal (design note)

The Critic escalates on **max residual force** (eV/Å) rather than a model-uncertainty estimate.

**Rationale.** Every structure in the corpus is a DFT-relaxed Materials Project geometry, so DFT's own forces on those atoms are ≈0 by construction. Whatever residual force MACE reports is therefore direct MACE-vs-DFT geometric disagreement — a per-material, physically interpretable measure of whether the surrogate can be trusted here. It costs no extra inference (forces come from the same evaluation as the energy) and it is the standard practitioner check for MLIP trustworthiness.

**Alternatives rejected:**

- *MC Dropout* — structurally always 0.0 (see above).
- *Two-checkpoint ensemble* (`mace-mpa-0` + `mace-omat-0`) — the checkpoints have different training sets and energy references, so their spread would be dominated by a near-constant offset rather than genuine per-material disagreement.
- *Positional-perturbation sensitivity* — viable and cheap, but residual force measures the same underlying property more directly.

**Calibration.** The gate was set from the measured distribution across all 130 materials, not chosen a priori:

| Statistic | eV/Å |
|---|---|
| median | 0.2229 |
| mean ± std | 0.2578 ± 0.1879 |
| p90 / p95 | 0.5197 / 0.6343 |
| max | 0.9181 |

`FORCE_GATE_EV_PER_ANG = 0.5` → 14/130 escalate (10.8%). Lower gates escalate the majority of the corpus (0.1 → 78.5%) and collapse the cost-tiering premise. The distribution is smooth with no natural boundary, so this is a judgement about escalation rate rather than a threshold the data selected.

**Honest caveat.** A median of 0.223 eV/Å is higher than would be expected for exactly-reproduced DFT geometries. Contributors: the CIF round-trip idealises fractional coordinates (pymatgen warns on ~9 structures), and MP's GGA+U systems are not reproducible by the surrogate. The gate therefore separates *relative* disagreement, not absolute trustworthiness.

---

## Results summary

Full tables in the README. Headline figures:

| Measurement | Value |
|---|---|
| MACE formation energy, mp-2790 vs MP | 0.0123 eV/atom error |
| MACE MAE, non-oxides (fair figure) | 0.121 eV/atom |
| MACE MAE, oxides | 1.100 eV/atom |
| CGCNN 5-fold CV, random split | 0.0838 ± 0.0191 eV/atom |
| CGCNN 5-fold CV, composition-disjoint | 0.1065 ± 0.0335 eV/atom |
| Mean-predictor baseline | 0.412 eV/atom |
| MACE inference | 0.191 s/material (CPU, mean) |

### Finding: transition-metal oxide discrepancy

MACE's error against MP formation energies scales with **metal** content, not oxygen content:

| Element | Non-oxides (eV/metal atom) | Oxides |
|---|---|---|
| Co | +0.03 | +2.58 |
| Fe | +0.23 | +2.96 |
| Mn | +0.35 | +2.44 |
| Ni | +0.46 | +3.19 |

Consistent with MP computing transition-metal oxides under GGA+U while computing elemental references under plain GGA, then reconciling via a fitted scheme (Jain 2011, Wang 2021). MP formation energies for these systems are not on a single level of theory.

Two hypotheses were tested and **rejected** before arriving at this: (1) MP's oxygen anion correction, and (2) a poor molecular-O₂ MACE reference. Both predict error scaling with oxygen fraction; a fit of error vs oxygen fraction gave slope −3.61 with intercept +3.24, i.e. scaling with metal fraction instead. Stated as a consistent explanation, not a proven mechanism — magnitudes are ~half the corresponding Hubbard U values and element ordering is not an exact match.

The same subset is independently flagged by the residual-force distribution (CoO₂, MnO₂, Co₃O₄, IrO₃ dominate the high-force tail).

### Finding: composition leakage in cross-validation

130 materials span only ~75 distinct compositions (7 MoS₂ polymorphs, 7 MnO₂, 6 NiS₂, 6 WS₂, 5 CoO₂); 62% share a formula with another entry. Random CV splits leak polymorphs between folds, letting the model score via composition lookup.

Composition-disjoint CV degrades MAE by ~27% (0.0838 → 0.1065) — modest, indicating the model learned genuine structure–property signal. Both numbers are reported; the gap is the measurement. `--group-by-composition` implements this, and k-fold runs now emit out-of-fold predictions so downstream comparison uses unbiased CGCNN numbers.

---

## Campaign trace (2026-09-03)

Three campaigns from a freshly built KG, verified end to end. Full table and
commentary in the README's *Worked example*.

| | demo-001 | demo-002 | demo-003 |
|---|---|---|---|
| Query | broad | broad | "Find stable Ni-P HER catalysts" |
| Skipped (already in KG) | 0 | 19 | 36 |
| Evaluated | 19 | 17 | 5 |
| Escalations | 0 | 1 (paid) | 0 |
| Spent / remaining | 96.5 / 3.5 | 96.5 / 3.5 | 27.5 / 72.5 |
| Termination | `budget_exhausted` | `budget_exhausted` | `completed` |

Cross-checks that passed: 19+17+5 = 41 distinct `mace_energy_per_atom`
properties in the KG (no material scored twice); demo-002's arithmetic
reconciles exactly (1.0 + 17×5.0 + 10.0 + 0.5 overhead = 96.5); escalation
counts agree between the summary log and the budget tracker.

**Nondeterminism.** The Planner is an LLM, so candidate ordering varies
between runs. mp-943 was in demo-001's candidate pool but ordered outside the
affordable window, and surfaced in demo-002 instead. Aggregate behaviour is
stable; the specific escalation is not. Any published trace is *a* run.

### Campaign-loop bugs found and fixed during this trace

The loop needed six rounds of correction before producing a trustworthy run.
Recorded because several of these reported plausible-looking numbers while
being wrong.

| Bug | Symptom | Fix |
|---|---|---|
| Retriever returned the whole corpus every step | `n_materials_evaluated: 520` on a 130-material corpus (130 × 4 retrievals) | Track scored mpids; count materials actually scored, not retrieved |
| `get_eah()` read a nonexistent `properties` attribute | Every material scored `inf`, sort was a no-op, "best candidate" was simply first in retrieval order; `best_candidate_e_above_hull` always null | Traverse `HAS_PROPERTY` edges |
| `status` overwritten after the loop | Every campaign reported `completed`, including budget exhaustion | Preserve terminal status; added distinct `budget_exhausted` |
| Prediction ran before its cost was charged | 23 predictions computed, 19 charged — 20% unbilled compute | Deduct before computing |
| Escalation `deduct()` return ignored | Summary reported "Escalations triggered: 1" while the tracker recorded 0 | Log only affordable escalations; record `escalation_unaffordable` otherwise |
| Critic called once per batch, after all predictions | All 19 predictions spent 95 of 99.5 units, then the Critic flagged an escalation that could no longer be afforded | Screen incrementally: predict → validate → escalate per candidate |

Also fixed: `prediction_count` off by one (initialised on update rather than
creation); `budget_remaining` hardcoded to 0.0 in `campaign_state`, contradicting
the tracker; a KG lookup paid for when the remaining budget could not fund a
single subsequent prediction.

### Retriever bugs (three stacked failures)

Chemsys filtering had never worked. Three separate faults, each masking the next:

1. `_execute_intent` dispatched on `intent.tool` with `if/elif`, so exactly one
   constraint was applied and the rest discarded. "Find stable Ni-P HER
   catalysts" parsed correctly as `stability` + `chemsys=[Ni-P]`, took the
   stability branch, and returned all 130 — every material in the corpus is
   below the threshold by construction, so the only discriminating constraint
   was the one thrown away.
2. `find_materials_in_chemsys` required a string; the Retriever passed a list.
   Before the `queries.py` rewrite this silently returned `[]`; after, it
   raised, and the Retriever's `except` swallowed it into an all-materials
   fallback.
3. `_normalize_chemsys` uppercased the input and then tested `part[1].islower()`
   — a condition the uppercase had just made impossible. Every two-letter
   element symbol was dropped: `Ni-P` → `['P']`, `Pt` → `[]`.

All three fixed; the query now correctly resolves to the 8 nickel phosphides.

### Planner: from hollow agent to real one

`planner.py` previously constructed a pydantic-ai `Agent` with a system prompt
and **never called it** — `plan_next_step()` was pure `if/elif`, and
`campaign.py` passed `use_llm=False` so the agent was not even built. The
project described itself as multi-agent while only the Retriever made LLM calls.

The Planner now uses an LLM for **candidate prioritisation** — choosing which
~19 of 130 candidates to spend the budget on, given what has been learned. Loop
control (continue/escalate/stop) remains fully deterministic. Ordering is
validated against the candidate list so the output is always a permutation of
the input: hallucinated ids dropped, duplicates ignored, omissions appended.
Verified against seven adversarial response shapes.

Also fixed in the same pass: `dataclasses.field` used inside a pydantic
`BaseModel`, and `MAX_ACTIONS_PER_CAMPAIGN` referenced without being imported
(`is_campaign_complete()` raised `NameError` whenever called).

---

## Technical debt

- `ScribeAgent.get_materials_with_properties()` is dead code with three
  independent defects, any one of which makes it return an empty list:
  `str(property_name)` on an enum member gives `"PropertyName.ENERGY_ABOVE_HULL"`
  not `"energy_above_hull"`; `G.edges(prop_nid, edges[0])` passes a node id into
  the `data` parameter, so the traversal returns nothing; and `edges[0]` raises
  `IndexError` for a property node with no predecessors. `kg/queries.py`'s
  `find_materials_by_property_range` already does this correctly — either
  delegate to it or delete the method.
- `PlannerAgent.state.remaining_budget` is not synchronised with the campaign's `BudgetTracker`. It exists for standalone testing; `campaign.py`'s tracker is authoritative. Reading it mid-campaign gives stale numbers.
- `requirements.txt` pins Windows-only packages (`pywin32`, `triton-windows`) and omits `torch_geometric`. Needs regenerating cross-platform.
- `scripts/run_campaign.py` previously read `result["campaign_state"]` and `result["final_materials"]`, which `CampaignOrchestrator.run()` returns flattened at the top level, so the CLI reported zeros on successful runs; it also double-logged to MLflow. **Fixed 2026-09-03.**
- A campaign that exhausts its candidate pool spends one final KG lookup discovering there is nothing left (demo-003: 2 lookups, 5 predictions). Defensible — you cannot know the pool is empty without querying — but it is 1 unit of avoidable cost.
- MLflow creates one *experiment* per campaign id rather than one experiment with many runs, which is unusual and makes cross-campaign comparison awkward in the UI.
- `PropertyUnit.FORMATION_ENERGY_PER_ATOM` is an enum alias of `ENERGY_ABOVE_HULL` (both are `"eV/atom"`; Python enums collapse duplicate values). Serialisation is correct; do not identify properties by their unit member — use `PropertyName`.
- GraphML export JSON-encodes list attributes as a workaround; `kg.json` remains canonical.

---

## Locked decisions

### Data: Materials Project via mp-api, clean-energy/catalyst subset

Filter (from `data/download.py`):

- **chemsys groups:**
  - HER: Ni-P, Co-P, Fe-P, Mo-P, W-P, Mn-P, Ni-S, Co-S, Fe-S, Mo-S, W-S, Mn-S, Ni-C, Co-C, Fe-C, Mo-C, W-C, Mn-C
  - OER: Ni-O, Co-O, Fe-O, Mn-O, Ni-Fe-O, Co-Fe-O, Ni-Co-O
  - Benchmarks: Pt, Ir-O
- **constraints:** `energy_above_hull` ∈ [0, 0.05], `num_sites` ∈ [0, 20]
- **fields pulled:** `material_id`, `formula_pretty`, `elements`, `energy_above_hull`, `formation_energy_per_atom`, `band_gap`
- **post-processing:** dedupe by material_id, rank by energy_above_hull, cap at 300
- **yield:** 130 unique candidates

`formation_energy_per_atom` was added during the correctness pass; it is the CGCNN training target and the surrogate-comparison target. `band_gap` is ingested to exercise the multi-property schema but is **not read by the agent loop**.

### KG layer: NetworkX + Pydantic schemas

Canonical store `data/processed/kg.json` (node-link JSON). GraphML is best-effort interop. Every node and edge is a pydantic model in `kg/schema.py`; `PropertyName`/`PropertyUnit`/`PropertySource` are open-vocabulary enums that coerce unknown strings rather than requiring a schema edit per new property.

### Property tiers — never mixed numerically

| Tier | Source | Role |
|---|---|---|
| MP | Materials Project DFT | stability gate, training target, comparison ground truth |
| MACE | `mace-mpa-0-medium` | surrogate ranking, formation energy, residual-force signal |
| UMA | FAIRChem OMat24 | optional showcase only |

`PropertyNode.source` records the tier. MACE writes to `mace_energy_per_atom`, never to `energy_above_hull`.

### Surrogates — two, compared not merged

MACE (`mace-mpa-0-medium`) runs zero-shot in the live loop. CGCNN is an **offline baseline only** — it is not a second real-time predictor path. Both predict `formation_energy_per_atom` so the comparison is like-for-like.

Formation energy uses MACE-evaluated elemental references (`models/elemental_references.py`), cached to `data/processed/mace_elemental_refs.json`. Solid elements use the lowest-hull MP elemental crystal; oxygen uses molecular O₂ in a 15 Å vacuum box. All references come from the same checkpoint as the material energies, so the subtraction is self-consistent.

### Agent layer: pydantic-ai, schema-first, Ollama for LLM

Typed contracts between agents. Ollama drives the Retriever's optional natural-language query parsing; the Planner is instantiated with `use_llm=False` from the campaign loop, since its decision logic is rule-based.

### Loop: budget-bounded, Critic-gated, Scribe persists

`agent/campaign.py` owns the `BudgetTracker` and performs all deduction. The Planner is consulted per step for continue/escalate/stop. The Critic gates escalation on stability (MP-sourced) and residual force (MACE-sourced). The Scribe writes property predictions to the KG; campaign metadata stays in the journal and MLflow.

### Tracking: MLflow + JSON journals

Per-campaign journal at `agent/journal/<campaign_id>.json`; MLflow to `sqlite:///mlflow.db`.

---

## Considered and rejected

Full table in the README's *Design Decisions* section. Additions from the correctness pass:

| Considered | Rejected because |
|---|---|
| Computing true `e_above_hull` in the Predictor | Requires the convex hull of all competing phases per chemical system — an MP phase-diagram call plus MACE evaluation of every competing phase, turning an O(1) surrogate into O(phases) and defeating the cost-tiering premise. Stability is read from MP instead. |
| MC-Dropout uncertainty on MACE | Deterministic inference, no active dropout: always exactly 0.0. |
| Two-checkpoint MACE ensemble for uncertainty | Different energy references between checkpoints; spread would be a constant offset, not model disagreement. |
| Excluding oxides from the comparison | Would remove 49/130 materials and the entire OER half of the corpus. Kept with molecular-O₂ reference and oxide/non-oxide split reporting instead. |
| Fine-tuning MACE on this corpus | Would make the MACE-vs-CGCNN comparison harder to interpret (both trained on the same 130 materials). Zero-shot keeps the contrast clean. |

---

## Next steps (prioritised)

1. **Relax `E_ABOVE_HULL_RANGE`** in `data/download.py` beyond the current
   `(0.0, 0.05)`. Effects: the Critic's stability threshold (0.1) would
   finally reject something — at present nothing in the corpus can fail it;
   more high-residual-force materials, so escalations become common rather
   than occasional; and a larger, more chemically varied CGCNN training set,
   which is currently the weakest part of the evaluation. Note this
   regenerates every downstream artifact — KG, elemental references, CGCNN
   training, force calibration, and all README numbers.
2. **Re-run the surrogate comparison and CGCNN training** after (1), and
   update the README Results tables.
3. Consider whether `TOP_N_BY_STABILITY = 300` should rise; it is currently
   unreached at 130 but becomes live once the stability filter is relaxed.

Completed since the last revision: `run_campaign.py` fixed; full campaign
trace recorded; `requirements.txt` regenerated cross-platform with
`torch_geometric`; `test_predictor.py` and `test_critic.py` updated for the
removed `uncertainty` field.

Deliberately **not** doing: fitting elemental references by least squares
against training-fold targets. It would shrink the oxide error and make the
comparison look tidier, but it would absorb the GGA+U discrepancy into fitted
constants and hide the most interesting finding in the project. It would also
break the clean "zero-shot foundation model vs corpus-trained baseline"
contrast. Kept as a stretch goal with that rationale attached.

---

## Stretch goals

- Target-property model (adsorption energy / overpotential proxy) so the pipeline selects for catalytic activity, not just stability.
- Neo4j Community + Cypher once the KG schema stabilises.
- `rdflib` RDF/OWL layer aligned with CMSO/ASMO ontologies.
- Novelty checking in the Scribe before treating candidates as new discoveries.
- Literature-mined synthesis-route edges via `lematerial-llm-synthesis`.
- Uncertainty-weighted averaging when the Scribe merges repeated predictions (currently a simple mean).

---

## Source rationale (job-sourcing)

Themes mined from job descriptions and community posts: evaluation as its own discipline (Dunia, NVIDIA), Bayesian optimisation breaking down in green-field spaces (Dunia), negative results as a neglected data source (RadicalAI, Dunia), closed-loop orchestration (Siemens Energy, alqem.ai, CuspAI), cost-aware agents (Dunia), physical plausibility and uncertainty (Dunia, alqem.ai, CuspAI), FAIR/traceable data (Siemens, alqem.ai), production DFT pipelines (alqem.ai), cross-campaign learning (Dunia).

Academic lineage: Bai et al. *Nat. Commun.* 2024 and *JACS Au* 2022 (KG-SDL, Cambridge/World Avatar group); Tejs Vegge's MaterialsCommons talk (FAIR + KG, AI4X 2026); Ian Foster, CMSC 35370 *AI Agents for Science* (University of Chicago, 2026) — reference architecture for Reasoning Core / Memory / Trust Layer.

Related projects: [ai-mandel](https://github.com/artificial-scientist-lab/ai-mandel) (agent loop pattern), [atomic-agents](https://github.com/Eigenwise/atomic-agents) (schema discipline), [AdsMind](https://arxiv.org/abs/2606.19152) (physics-grounded multi-agent, similar architecture but single-candidate focus).