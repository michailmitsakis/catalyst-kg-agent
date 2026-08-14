# Knowledge-Graph-Grounded, Cost-Aware Decision Agent for Materials Discovery

**mat-kg-agent** is a multi-agent system that helps a materials-discovery campaign choose the *cheapest sufficient* next action — a knowledge-graph lookup, a fine-tuned MLIP query, or an expensive simulated experiment — the same decision real self-driving-lab (SDL) orchestration has to make under a budget.

**Status:** work in progress. Core agent loop and knowledge-graph build are functional; evaluation results and the final Materials Project data filter are still being finalized (see [Evaluation & Testing Status](#evaluation--testing-status) and [Data](#data) below — stated plainly rather than implied).

---

## Table of Contents

- [Knowledge-Graph-Grounded, Cost-Aware Decision Agent for Materials Discovery](#knowledge-graph-grounded-cost-aware-decision-agent-for-materials-discovery)
  - [Table of Contents](#table-of-contents)
  - [The Problem](#the-problem)
  - [How It Works](#how-it-works)
  - [Evaluation \& Testing Status](#evaluation--testing-status)
  - [Setup](#setup)
  - [Architecture](#architecture)
    - [Agent roles](#agent-roles)
    - [Cost model](#cost-model)
  - [Repository Structure](#repository-structure)
  - [Data](#data)
  - [Limitations](#limitations)
  - [Design Decisions \& Alternatives Considered](#design-decisions--alternatives-considered)
  - [Background \& Motivation](#background--motivation)
  - [Stretch Goals](#stretch-goals)
  - [Citations \& Further Reading](#citations--further-reading)
  - [Acknowledgments](#acknowledgments)

---

## The Problem

Materials-discovery teams building self-driving labs face a recurring decision: given a target property and a limited budget, what should be tried next — a cheap database lookup, a fast ML surrogate estimate, or an expensive real (or simulated) experiment? Get this wrong and either the budget is wasted on redundant expensive steps, or an unreliable surrogate result gets trusted without a check.

This project builds a small, locally-runnable version of that decision system: a knowledge graph as the memory/grounding layer, a fine-tuned MLIP as the property surrogate, and a role-specialized multi-agent system that picks actions under an explicit cost budget — with a dedicated safety check before anything expensive is allowed to run.

---

## How It Works

```
Materials Project  →  kg/build_graph.py  →  NetworkX knowledge graph
                                                  │
                                                  ▼
                                   agent/planner.py (holds budget)
                                                  │
                       ┌──────────────┬───────────┼───────────┬──────────────┐
                       ▼              ▼           ▼           ▼              ▼
                  Retriever      Predictor     Critic       Scribe    (optional, final only)
                (KG lookup)   (MACE/CGCNN)  (plausibility  (writes    UMA relaxation check
                                            + uncertainty   result
                                             gate; e_above   back
                                             _hull check)   to KG)
```

At each step, the **Planner** decides whether to call the **Retriever** (cheap KG lookup), the **Predictor** (moderate-cost MACE/GNN surrogate), or escalate to a simulated expensive experiment. The **Critic** must approve before any escalation — checking physical plausibility (including an `e_above_hull` stability threshold) and flagging when the Predictor's uncertainty is too high to trust. The **Scribe** writes each campaign's outcome back into the knowledge graph, so later campaigns start from accumulated understanding rather than from scratch.

All inter-agent messages use strict Pydantic schemas (`agent/schemas.py`) — agents chain by matching typed contracts, not free-form text.

This decomposition mirrors the "AI-native Scientific Discovery Platform" reference architecture (Reasoning Core / Memory / Trust Layer / Knowledge Substrate / Domain Foundation Models / 1st-principles models) presented in Ian Foster's *AI Agents for Science* course (University of Chicago, CMSC 35370): the Planner plays the Reasoning Core role, the knowledge graph is the Knowledge Substrate, the Critic is the Trust Layer, MACE/CGCNN are the Domain Foundation Model, and the optional UMA relaxation check stands in for a 1st-principles verification step.

**Scope relative to similar work:** [`AdsMind`](https://github.com/NagatoBigSeven/AdsMind) — a physics-grounded multi-agent system that self-corrects a *single* adsorption configuration using MLIP-relaxation feedback — solves an adjacent but distinct problem: per-candidate structural correction, rather than this project's focus on cross-candidate, budget-constrained decisions about which action to spend resources on next.

A worked example (input query → agent trace → final candidate + cost breakdown) will be added here once the first end-to-end campaign run is complete.

---

## Evaluation & Testing Status

Stated plainly, since this is still in progress:

- **Surrogate comparison (in progress):** `notebooks/03_gnn_surrogate_eval.ipynb` will compare a from-scratch CGCNN-style baseline against the fine-tuned `mace-mp-0` model on accuracy, training cost, and data-efficiency. Not yet run to completion — no numbers to report here yet.
- **Unit tests (present):** `tests/test_kg_build.py`, `tests/test_cost_model.py`, and `tests/test_agent_tools.py` cover graph construction correctness, cost-unit accounting, and agent schema validation. These are structural/unit-level tests, not end-to-end campaign evaluation.
- **Campaign-level evaluation (not yet done):** no cost-vs-outcome results have been generated yet. `notebooks/04_campaign_analysis.ipynb` is scaffolded to read from `agent/journal/*.json` run logs and MLflow, but no full campaign has been run against real data yet.
- **Monitoring:** each campaign run logs cost-per-action, cost-per-agent-role, and final outcome to both a JSON journal file and MLflow, for later inspection.

This section will be updated with concrete numbers once the first full evaluation pass is complete — flagged here rather than left implicit.

---

## Setup

Requires:
- Python 3.10+
- A Materials Project API key (the only external credential needed — no VASP/HPC scheduler credentials, since no remote DFT job submission is used)
- Unsloth Studio Desktop running locally, with a small instruction-tuned model pulled (e.g. Llama 3.1 8B or Qwen2.5 7B class) --> TBD
- A local MLflow tracking instance (or its default file-based store)
- Optional: a CUDA-capable GPU (12GB VRAM is sufficient for fine-tuning the small MACE checkpoint and training the CGCNN baseline)

```bash
git clone <this-repo>
cd mat-kg-agent
pip install -r requirements.txt
cp .env.example .env   # fill in MP_API_KEY
```

Running the test suite:
```bash
pytest tests/
```

Detailed dataset pull and filter criteria are documented in `data/download.py` and `notebooks/01_explore_dataset.ipynb` (in progress — see [Data](#data)).

---

## Architecture

### Agent roles

- **Retriever** — owns knowledge-graph traversal only. Given a target property and constraints, returns candidate materials with provenance (which KG edges/sources support the answer).
- **Predictor** — owns surrogate model calls (MACE or the CGCNN baseline). Returns a property estimate *with* uncertainty, not a bare point value.
- **Critic** — validates before anything gets "spent." Checks physical plausibility (including an `e_above_hull` stability threshold) and flags when the Predictor's uncertainty is too high to trust without escalation.
- **Planner** — the only agent that sees the remaining budget. Decides, at each step, whether to call the Retriever, the Predictor, or escalate.
- **Scribe** — writes each campaign's outcome back into the knowledge graph as new edges or updated confidence values.

### Cost model

Each action type (`kg_lookup`, `surrogate_query`, `experiment`) has an explicit cost unit in `agent/cost_model.py`. Decisions and the resulting cost-vs-outcome tradeoff are logged per campaign.

---

## Repository Structure

```
mat-kg-agent/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example                    # MP_API_KEY, Unsloth endpoint, MLflow URI
│
├── data/
│   ├── raw/                        # gitignored
│   ├── processed/
│   └── download.py                 # Materials Project API pull
│
├── kg/
│   ├── build_graph.py
│   ├── schema.py                   # node/edge types; ontology stub for RDF stretch goal
│   ├── graph_store.py
│   └── queries.py
│
├── models/
│   ├── gnn_surrogate/
│   │   ├── dataset.py
│   │   ├── baseline_cgcnn.py       # from-scratch CGCNN-style GNN
│   │   └── mace_finetune.py        # fine-tuned mace-mp-0
│   ├── bo/
│   │   ├── search_space.py
│   │   └── optimize.py
│   └── verification/
│       └── uma_relax.py            # optional, final-candidate-only fidelity check
│
├── agent/
│   ├── schemas.py
│   ├── retriever.py
│   ├── predictor.py
│   ├── critic.py
│   ├── scribe.py
│   ├── planner.py
│   ├── cost_model.py
│   ├── campaign.py
│   └── journal/                    # per-run JSON logs, gitignored except .gitkeep
│
├── tracking/
│   └── mlflow_setup.py
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_build_kg_explore.ipynb
│   ├── 03_gnn_surrogate_eval.ipynb
│   └── 04_campaign_analysis.ipynb
│
├── stretch/
│   └── rdf_ontology.py             # rdflib layer, CMSO/ASMO-aligned, optional
│
└── tests/
    ├── test_kg_build.py
    ├── test_cost_model.py
    └── test_agent_tools.py
```

---

## Data

**Source:** Materials Project, via the `mp-api` client.

**Scope:** a clean-energy/catalyst-relevant subset of bulk inorganic structures. Final property-tag filtering criteria are still being finalized — this section will be updated with the exact query once locked, rather than described vaguely in the meantime.

**Why not adsorption-energy datasets (OC20/OC25/AQCat25):** more directly relevant to catalytic activity, but a different scale and physics (millions of DFT calculations of slab + adsorbate + solvent systems, vs. Materials Project's bulk-only structures). Noted as a natural extension for a version of this project targeting adsorption energy directly.

---

## Limitations

- Toy-scale demonstration, not a production SDL controller. The "expensive experiment" step is a held-out dataset lookup, not a real synthesis or characterization run.
- The knowledge graph is built from a single structured database (Materials Project) rather than a federated, multi-source graph — a meaningfully harder problem real infrastructure (e.g. MaterialsCommons) is built to solve.
- UMA/OMat24-derived energies are not numerically compatible with Materials Project-derived energies without additional correction terms; these are kept in strictly separate, labeled tiers rather than corrected against each other.
- Multi-agent design adds real coordination overhead and additional failure surface versus a single-agent pipeline. Chosen because independent role separation — particularly the Critic's gate before escalation — mattered more than raw simplicity for this problem, not because more agents are inherently better.
- No end-to-end evaluation numbers yet (see [Evaluation & Testing Status](#evaluation--testing-status)).

---

## Design Decisions & Alternatives Considered

*This section is for readers who want the full rationale behind each choice.*

| Decision | Chosen | Alternatives considered | Why chosen |
|---|---|---|---|
| **Dataset source** | Materials Project (bulk properties) | OQMD (weaker natural graph structure); literature-mined synthesis data via `lematerial-llm-synthesis` (extraction-accuracy risk, frontier-API dependency); OC20/OC25/AQCat25 (adsorption-energy datasets, millions of DFT calculations, slab/solvent complexity) | MP gives structured, versioned, citable provenance out of the box, a natural node/edge shape for the KG, and direct compatibility with MACE's own MP-trained foundation checkpoint. |
| **Surrogate / MLIP** | MACE (`mace-mp-0`, fine-tuned) | CHGNet (better fit if dataset skewed ionic/oxide-heavy); GNN-from-scratch as sole surrogate | More mature fine-tuning workflow; more commonly cited across current MLIP literature and industry tooling. |
| **GNN baseline** | Custom CGCNN-style GNN (PyTorch Geometric), used as an ablation against MACE | Pretrained MLIP only; from-scratch GNN as sole surrogate | Produces an actual accuracy/cost/data-efficiency comparison rather than just "a model was used." |
| **Final-fidelity check** | Optional single-shot ASE relaxation via FAIRChem's UMA, run only on the top candidate | Treating surrogate output as final; UMA as a third competing surrogate | One heavier-fidelity sanity check on the winning candidate without complicating the core cost loop; kept in a separate, non-comparable tier due to DFT-setting incompatibility with MP. |
| **KG storage** | NetworkX | Neo4j Community + Cypher; RDF/OWL via `rdflib` | Zero infrastructure, fastest to iterate on schema while it's still evolving. Neo4j/RDF kept as stretch goals. |
| **Agent framework** | pydantic-ai | atomic-agents (Instructor-based) | Reuses an existing personal toolchain; borrowed atomic-agents' strict I/O schema discipline as a pattern, not a framework switch. |
| **Multi-agent architecture** | Role-specialized agents with a hard budget and a Critic gate before escalation | Single monolithic agent; open-ended exploration loop (`ai-mandel`-style) | A real SDL mistake costs materials and time, not tokens — auditable, budget-constrained decisions prioritized over open-ended novelty-seeking. |
| **Orchestration granularity** | Single `campaign.py` runner with separated internal stages | Fully independent scripts per stage (`ai-mandel`-style) | Loop is tighter and budget-bounded rather than open-ended — a deliberate simplification given narrower scope. |
| **MD/batching infrastructure** | Not used | NVIDIA ALCHEMI Toolkit (`nvalchemi`) | Solves large-scale MD-throughput efficiency; this project does single-point inference, not large-scale MD sampling. |
| **Stability screening rule** | `e_above_hull` threshold as the Critic's first concrete gate | Vague, unspecified "plausibility check" | Mirrors the stability-screening step used in production materials-discovery agent workflows; a well-defined, MP-derivable threshold. |
| **Distributed/federated agent execution** | Not used — all agents run locally, in-process | [Academy](https://academy.proxystore.dev) (Globus Compute + Parsl agentic middleware for federated, actor-model agent deployment across HPC/experimental facilities) | This project targets single-machine, local execution; Academy-style federated middleware is the natural path if the same agent roles were later deployed across real HPC and instrument resources rather than simulated ones. |

---

## Background & Motivation

*Full reasoning trail for readers curious how this project's scope was chosen.*

This project started from three separate observations that turned out to point at the same gap.

**1. Conference signal.** At the AI4X 2026 conference, the field's center of gravity was visibly the same triangle everywhere: ML surrogates (MLIPs, GNNs), automated/robotic experimentation, and orchestration logic tying the two together into a closed loop. Talks from Ulrich Schubert (self-driving labs), Curtis Berlinguette (Ada-Carbon), Tejs Vegge (MaterialsCommons, FAIR workflows and knowledge-graph-backed federated infrastructure), and Mohamad Moosavi (literature-informed autonomous discovery) all described variations of the same architecture, at very different scales.

**2. Job-market signal.** Mining current job descriptions from materials-informatics startups and labs (CuspAI, Atomscale, Siemens Energy, Mistral AI, Dunia Innovations, alqem.ai, NVIDIA, Meta FAIR Chemistry, BAM) surfaced recurring problems, almost verbatim, across otherwise unrelated companies: evaluation as its own discipline, Bayesian optimization breaking down in "green-field" search spaces, negative results as a neglected data source, closed-loop orchestration, agentic workflows and where they're dangerous (physical actions costing far more than a token call), preventing physically invalid model outputs, FAIR/traceable data as a bottleneck, and — repeatedly — knowledge graphs and ontologies for materials research, including in a BAM permanent research position posted during this project's development.

**3. Literature signal.** Once knowledge graphs were flagged as recurring, a direct academic lineage confirmed it wasn't just hiring-post language: Bai et al., *"A dynamic knowledge graph approach to distributed self-driving laboratories"* (*Nature Communications*, 2024) and Bai et al., *"From Platform to Knowledge Graph: Evolution of Laboratory Automation"* (*JACS Au*, 2022) frame knowledge graphs explicitly as the next stage of laboratory automation.

This project combines those three signals into one small, locally-runnable artifact.

---

## Stretch Goals

- Replace NetworkX with Neo4j Community + Cypher once the KG schema stabilizes.
- Add an `rdflib`-based RDF/OWL layer aligned with the CMSO/ASMO ontologies, following the ontology-mapping skill from [`materials-simulation-skills`](https://github.com/HeshamFS/materials-simulation-skills).
- Extend the Scribe agent's novelty-checking logic against known structures before treating candidates as new discoveries, following the [AtomisticSkills materials-discovery workflow](https://github.com/learningmatter-mit/AtomisticSkills/blob/main/.agents/workflows/materials-discovery.md).
- Enrich KG synthesis-route edges with literature-mined synthesis parameters, following [`lematerial-llm-synthesis`](https://github.com/LeMaterial/lematerial-llm-synthesis), adapted to local inference.

---

## Citations & Further Reading

- Bai, J. et al. *A dynamic knowledge graph approach to distributed self-driving laboratories.* Nature Communications (2024).
- Bai, J. et al. *From Platform to Knowledge Graph: Evolution of Laboratory Automation.* JACS Au (2022).
- [Acceleration Consortium — Awesome Self-Driving Labs](https://github.com/AccelerationConsortium/awesome-self-driving-labs)
- [`ai-mandel`](https://github.com/artificial-scientist-lab/ai-mandel) — iterative Researcher/Novelty-Supervisor/Judge agent loop pattern.
- [`atomic-agents`](https://github.com/Eigenwise/atomic-agents) — strict input/output schema chaining discipline for agents.
- [`GNN-materials`](https://github.com/polbeni/GNN-materials) — CGCNN-style GNN implementation reference.
- [`AtomisticSkills`](https://github.com/learningmatter-mit/AtomisticSkills) — materials-discovery agent workflow, including the stability-screening pattern adopted for the Critic agent.
- [`materials-simulation-skills`](https://github.com/HeshamFS/materials-simulation-skills) — CMSO/ASMO ontology skill referenced for the RDF stretch goal.
- [`lematerial-llm-synthesis`](https://github.com/LeMaterial/lematerial-llm-synthesis) — literature-mined synthesis-parameter extraction, noted as future work.
- [FAIRChem / UMA](https://github.com/facebookresearch/fairchem) — used for the optional final-fidelity relaxation check.
- Zhang, Z. et al. *AdsMind: A Physics-Grounded Multi-Agent System for Self-Correcting Discovery of Adsorption Configurations on Heterogeneous Catalyst Surfaces.* arXiv:2606.19152 (2026) — closely related architecture (LLM planner + MACE-MP relaxation feedback, multi-backend LLM including Ollama); see the Scope note above for how this project differs.
- Prince, C. et al. *Opportunities for retrieval and tool augmented large language models in scientific facilities.* npj Computational Materials (2024) — the CALMS retrieve-then-escalate-if-uncertain pattern this project's Retriever/Predictor/Critic loop follows.
- Foster, I. & Kamatar, A. *AI Agents for Science*, Lecture 6: HPC Systems and Self-Driving Labs. CMSC 35370, University of Chicago (2026). [agents4science.github.io](https://agents4science.github.io) — source of the AI-native Scientific Discovery Platform reference architecture cited above, and of the Academy federated-agent middleware noted in Design Decisions.

---

## Acknowledgments

Parts of this codebase were developed with assistance from an AI coding assistant using Agent Skills from [`materials-simulation-skills`](https://github.com/HeshamFS/materials-simulation-skills) and [`AtomisticSkills`](https://github.com/learningmatter-mit/AtomisticSkills).
