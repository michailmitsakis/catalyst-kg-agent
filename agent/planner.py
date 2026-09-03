"""Planner agent: decides what to spend the next surrogate call on.

Two responsibilities, deliberately split by whether an LLM helps:

1. LOOP CONTROL (deterministic).
   continue / escalate / stop, from remaining budget and experiment count.
   This is arithmetic and a safety property. An LLM would make it slower,
   non-deterministic, and no better, so `plan_next_step` contains no model
   call at all.

2. CANDIDATE PRIORITISATION (LLM).
   Given what the campaign has already learned -- which materials were
   scored, their surrogate energies, and which ones the surrogate was
   least confident about -- decide the ORDER in which the remaining
   candidates should be evaluated. This is the acquisition-function role
   in an active-learning loop: with budget for ~19 of 130 candidates, the
   choice of which 19 is the whole game.

   The LLM proposes an ordering only. It cannot authorise spending, cannot
   change costs, and cannot introduce candidates: every mpid it returns is
   validated against the supplied candidate list, unknown ids are dropped,
   and anything it omits is appended in the original order. The worst a bad
   LLM response can do is give a poor ordering -- never an unsafe action.

Falling back: if `use_llm=False`, Ollama is unreachable, or the response
fails validation, prioritisation degrades to the corpus's existing order
(stability-ranked from Materials Project) and the campaign proceeds. The
provenance field on PrioritizationResult records which path was taken.

Related work: using an LLM in place of, or alongside, a classical
acquisition function is an active area -- see e.g. LLAMBO (Liu et al.,
"Large Language Models to Enhance Bayesian Optimization", ICLR 2024) for
LLM-driven candidate proposal in Bayesian optimisation. Cite whichever
reference you have verified; this implementation is a much simpler,
budget-bounded variant of the same idea.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from kg.schema import MaterialNode, PropertyNode, NodeType
from kg.graph_store import load_graph
from agent.cost_model import (
    INITIAL_BUDGET,
    MAX_ACTIONS_PER_CAMPAIGN,
    KG_LOOKUP_COST,
    SURROGATE_COST,
    EXPERIMENT_COST,
)
from agent.critic import CriticDecision


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_max_experiments() -> int:
    """Get max experiments limit from environment or default."""
    try:
        return int(os.environ.get("MAX_EXPERIMENTS", "10"))
    except ValueError:
        return 10


# How many already-scored materials to summarise for the LLM. Keeps the
# prompt small enough for a local model to handle reliably.
MAX_HISTORY_IN_PROMPT = 15

# How many candidates to offer the LLM per prioritisation call. A local
# model asked to rank 130 ids will usually truncate or hallucinate.
MAX_CANDIDATES_IN_PROMPT = 40


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass
class PlannerState:
    """Planner state tracking loop progress."""
    remaining_budget: float = INITIAL_BUDGET
    actions_taken: int = 0
    materials_evaluated: List[MaterialNode] = field(default_factory=list)
    critic_rejections: int = 0
    experiments_count: int = 0
    escalation_needed: bool = False
    campaign_id: str = "default"


class PlannerDecision(BaseModel):
    """Planner output decision for loop control."""
    next_action: str  # "continue", "escalate", "stop"
    reason: str
    cost_update: float  # Cost the decision implies (campaign.py does the deducting)
    # NOTE: pydantic's Field, not dataclasses.field. The previous version
    # used dataclasses.field inside a BaseModel, which is the wrong API.
    materials_approved: List[MaterialNode] = Field(default_factory=list)


class PrioritizationIntent(BaseModel):
    """What the LLM is asked to return: an ordering plus its reasoning.

    Deliberately minimal. The model returns identifiers and a sentence, not
    structured material data -- asking a small local model to emit nested
    objects is what causes retry loops and malformed output.
    """
    ordered_mpids: List[str] = Field(
        default_factory=list,
        description="Candidate mpids in the order they should be evaluated, best first",
    )
    reason: str = Field(
        default="",
        description="One or two sentences explaining the ordering",
    )


class PrioritizationResult(BaseModel):
    """Validated prioritisation, safe for the campaign loop to act on."""
    ordered_materials: List[MaterialNode]
    reason: str
    provenance: str  # "llm", "llm_partial", "fallback_order", "llm_failed"
    n_llm_ranked: int = 0
    n_dropped_unknown: int = 0


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Budget-bounded loop control plus LLM-guided candidate prioritisation."""

    def __init__(
        self,
        graph_path: Path = None,
        campaign_id: str = "default",
        use_llm: bool = True,
    ):
        """Initialize planner.

        Args:
            graph_path: Path to the knowledge graph JSON
            campaign_id: Identifier for this campaign run
            use_llm: Enable LLM-guided prioritisation. When False the planner
                is fully deterministic and makes no network calls, which is
                what the unit tests use.
        """
        self.graph_path = graph_path or Path("data/processed/kg.json")
        self.campaign_id = campaign_id
        self.G = load_graph(self.graph_path)

        self.state = PlannerState(campaign_id=campaign_id)
        self.max_experiments = get_max_experiments()
        self.use_llm = use_llm
        self.agent = None

        if use_llm:
            try:
                from pydantic_ai import Agent
                from pydantic_ai.models.ollama import OllamaModel
                from pydantic_ai.providers.ollama import OllamaProvider

                ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_model = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

                model = OllamaModel(
                    model_name=ollama_model,
                    provider=OllamaProvider(base_url=ollama_base),
                )
                # output_type pins the response to PrioritizationIntent, so
                # the model returns ids and prose rather than free text that
                # would need parsing.
                self.agent = Agent(
                    model=model,
                    output_type=PrioritizationIntent,
                    system_prompt=self._build_system_prompt(),
                )
            except Exception as exc:
                print(f"[WARN] Planner LLM unavailable ({exc}); using deterministic ordering")
                self.agent = None

    # -----------------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """System prompt for the prioritisation agent.

        Scoped narrowly: the model orders candidates and nothing else. It is
        told explicitly that it does not control spending, because a model
        that believes it can authorise budget will try to.
        """
        return f"""You are the Planner in a budget-constrained materials discovery campaign.

YOUR ONLY JOB: given a list of candidate materials and what has already been
learned this campaign, return the order in which the remaining candidates
should be evaluated by the surrogate model. Best candidate first.

You do NOT decide whether to spend, how much anything costs, or when to
stop. Those are handled outside you. Return an ordering, nothing else.

CONTEXT YOU ARE WORKING IN:
- A cheap surrogate (MACE) estimates formation energy per atom. Lower
  (more negative) means more thermodynamically favourable.
- Each surrogate call costs {SURROGATE_COST} units out of a {INITIAL_BUDGET}
  unit budget, so only a fraction of the corpus can ever be evaluated. Which
  fraction is what you are choosing.
- Each material also has a max residual force (eV/Angstrom). High values
  mean the surrogate disagrees with the reference geometry and is less
  trustworthy there; those materials get escalated to an expensive check.

HOW TO ORDER:
- Prefer candidates whose chemistry resembles the low-formation-energy
  materials already found this campaign.
- Prefer exploring chemical systems not yet sampled over re-sampling a
  system already well covered -- a campaign that evaluates six polymorphs
  of one compound learns less than one covering six systems.
- Do not prioritise a material purely because it is unusual; the budget is
  for finding good candidates, not curiosities.

RULES:
- Return ONLY mpids that appear in the candidate list given to you.
- Never invent an mpid.
- Include every candidate you were given, in your preferred order.
- Give a one or two sentence reason for the ordering."""

    # -----------------------------------------------------------------------
    # Prioritisation (LLM)
    # -----------------------------------------------------------------------

    def _summarise_history(self, predictions: Optional[List[Any]]) -> str:
        """Render what has been learned so far, compactly, for the prompt."""
        if not predictions:
            return "Nothing evaluated yet this campaign."

        scored = [p for p in predictions if not getattr(p, "prediction_failed", False)]
        if not scored:
            return "Nothing evaluated successfully yet this campaign."

        # Show the most favourable results -- that is the signal the model
        # should be extrapolating from.
        def energy(p):
            value = getattr(p, "formation_energy_per_atom", None)
            return value if value is not None else float("inf")

        best = sorted(scored, key=energy)[:MAX_HISTORY_IN_PROMPT]

        lines = []
        for p in best:
            e = getattr(p, "formation_energy_per_atom", None)
            formula = self._formula_for(getattr(p, "material_id", ""))
            e_str = f"{e:+.3f}" if e is not None else "n/a"
            lines.append(
                f"  {p.material_id} ({formula}): formation energy {e_str} eV/atom, "
                f"residual force {getattr(p, 'max_residual_force', float('nan')):.2f} eV/A"
            )
        return f"Evaluated so far ({len(scored)} materials), most favourable first:\n" + "\n".join(lines)

    def _formula_for(self, mpid: str) -> str:
        """Look up a material's pretty formula from the graph, if present."""
        node = self.G.nodes.get(f"material:{mpid}")
        if node:
            return node.get("formula_pretty", "?")
        return "?"

    def prioritize_candidates(
        self,
        candidates: List[MaterialNode],
        predictions_so_far: Optional[List[Any]] = None,
        remaining_budget: Optional[float] = None,
    ) -> PrioritizationResult:
        """Order candidates for evaluation, LLM-guided where available.

        Args:
            candidates: Materials not yet scored this campaign
            predictions_so_far: PredictorResults already obtained, used as
                the "what we have learned" context
            remaining_budget: Passed to the prompt for context only -- the
                planner never spends it

        Returns:
            PrioritizationResult whose ordered_materials is always a
            permutation of `candidates` (never a subset, never with
            additions), so the caller can use it unconditionally.
        """
        if not candidates:
            return PrioritizationResult(
                ordered_materials=[],
                reason="No candidates to order.",
                provenance="fallback_order",
            )

        if self.agent is None:
            return PrioritizationResult(
                ordered_materials=list(candidates),
                reason="LLM disabled; using the corpus's stability-ranked order.",
                provenance="fallback_order",
            )

        # Offer a bounded slice; a local model asked to rank 130 ids will
        # truncate or hallucinate. Anything beyond the slice keeps its
        # original relative position behind the ranked ones.
        offered = candidates[:MAX_CANDIDATES_IN_PROMPT]
        remainder = candidates[MAX_CANDIDATES_IN_PROMPT:]

        listing = "\n".join(
            f"  {m.mpid} ({m.formula_pretty}, elements: {', '.join(m.elements)})"
            for m in offered
        )
        budget_line = (
            f"Remaining budget: {remaining_budget:.1f} units "
            f"(~{int(remaining_budget // SURROGATE_COST)} surrogate calls)."
            if remaining_budget is not None else ""
        )

        prompt = f"""{self._summarise_history(predictions_so_far)}

{budget_line}

Candidates to order ({len(offered)}):
{listing}

Return these {len(offered)} mpids in your preferred evaluation order."""

        try:
            result = self.agent.run_sync(prompt)
            intent = result.output
        except Exception as exc:
            print(f"[WARN] Planner prioritisation failed ({exc}); using existing order")
            return PrioritizationResult(
                ordered_materials=list(candidates),
                reason=f"LLM call failed: {exc}",
                provenance="llm_failed",
            )

        return self._apply_ordering(intent, offered, remainder)

    def _apply_ordering(
        self,
        intent: PrioritizationIntent,
        offered: List[MaterialNode],
        remainder: List[MaterialNode],
    ) -> PrioritizationResult:
        """Validate the LLM's ordering and rebuild a safe candidate list.

        Guarantees the output is a permutation of the input:
          - mpids the model invented are dropped
          - duplicates are ignored
          - candidates the model omitted are appended in their original order

        A malformed response degrades to the original ordering rather than
        losing candidates or acting on fabricated ids.
        """
        by_mpid = {m.mpid: m for m in offered}
        seen: set[str] = set()
        ordered: List[MaterialNode] = []
        dropped = 0

        for mpid in intent.ordered_mpids or []:
            key = str(mpid).strip()
            if key in seen:
                continue
            material = by_mpid.get(key)
            if material is None:
                dropped += 1  # hallucinated or out-of-scope id
                continue
            ordered.append(material)
            seen.add(key)

        n_ranked = len(ordered)

        # Anything the model left out keeps its original relative order.
        for material in offered:
            if material.mpid not in seen:
                ordered.append(material)

        ordered.extend(remainder)

        if n_ranked == 0:
            provenance = "llm_failed"
            reason = "LLM returned no usable mpids; kept the original order."
        elif n_ranked < len(offered) or dropped:
            provenance = "llm_partial"
            reason = intent.reason or "Partial ordering from the LLM."
        else:
            provenance = "llm"
            reason = intent.reason or "LLM ordering."

        return PrioritizationResult(
            ordered_materials=ordered,
            reason=reason,
            provenance=provenance,
            n_llm_ranked=n_ranked,
            n_dropped_unknown=dropped,
        )

    # -----------------------------------------------------------------------
    # Loop control (deterministic)
    # -----------------------------------------------------------------------

    def plan_next_step(
        self,
        retrieved_materials: List[MaterialNode],
        predictions: Optional[List[Any]] = None,
        critic_decisions: Optional[List[CriticDecision]] = None,
    ) -> PlannerDecision:
        """Decide continue / escalate / stop.

        Deliberately contains no LLM call: this is budget arithmetic and a
        termination guarantee. Non-determinism here would make campaigns
        unreproducible and could let a model talk the loop into overspending.

        Args:
            retrieved_materials: Materials scored this step
            predictions: Predictor outputs for them
            critic_decisions: Critic results, carrying requires_escalation

        Returns:
            PlannerDecision with next_action and the cost it implies
            (campaign.py owns the BudgetTracker and does the deducting)
        """
        self.state.materials_evaluated.extend(retrieved_materials)
        self.state.actions_taken += 1

        escalation_needed = any(
            getattr(d, "requires_escalation", False) for d in (critic_decisions or [])
        )

        can_afford_surrogate = self.state.remaining_budget >= SURROGATE_COST
        can_afford_experiment = self.state.remaining_budget >= EXPERIMENT_COST
        experiments_limit_reached = self.state.experiments_count >= self.max_experiments

        if escalation_needed and not experiments_limit_reached:
            decision = PlannerDecision(
                next_action="escalate",
                reason=(
                    f"Critic flagged a low-confidence prediction "
                    f"(experiments so far: {self.state.experiments_count}/{self.max_experiments})"
                ),
                cost_update=EXPERIMENT_COST,
                materials_approved=list(retrieved_materials),
            )
            self.state.experiments_count += 1

        elif experiments_limit_reached:
            decision = PlannerDecision(
                next_action="stop",
                reason=f"Maximum experiments ({self.max_experiments}) reached",
                cost_update=0.0,
                materials_approved=list(retrieved_materials),
            )

        elif not can_afford_experiment and not can_afford_surrogate:
            decision = PlannerDecision(
                next_action="stop",
                reason="Budget exhausted (insufficient funds for any further action)",
                cost_update=0.0,
                materials_approved=list(retrieved_materials),
            )

        elif retrieved_materials and can_afford_surrogate:
            decision = PlannerDecision(
                next_action="continue",
                reason=(
                    f"{len(retrieved_materials)} candidates evaluated "
                    f"(experiments so far: {self.state.experiments_count}/{self.max_experiments})"
                ),
                cost_update=SURROGATE_COST,
                materials_approved=list(retrieved_materials),
            )

        else:
            decision = PlannerDecision(
                next_action="stop",
                reason="No candidates remaining or insufficient budget for a surrogate call",
                cost_update=0.0,
                materials_approved=[],
            )

        self.state.remaining_budget -= decision.cost_update
        return decision

    def get_remaining_budget(self) -> float:
        """Planner's own budget view.

        Not authoritative during a campaign: agent/campaign.py owns the
        BudgetTracker and this value does not receive its deductions. Kept
        for standalone use and tests.
        """
        return self.state.remaining_budget

    def is_campaign_complete(self) -> bool:
        """Check whether the planner's own state says to terminate.

        MAX_ACTIONS_PER_CAMPAIGN is imported at module level; an earlier
        version referenced it without importing it, so this method raised
        NameError whenever it was called.
        """
        return (
            self.state.remaining_budget <= 0
            or self.state.actions_taken >= MAX_ACTIONS_PER_CAMPAIGN
            or self.state.experiments_count >= self.max_experiments
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_planner(
    graph_path: Path = None,
    campaign_id: str = "default",
    use_llm: bool = True,
) -> PlannerAgent:
    """Factory function to create a planner agent.

    Args:
        graph_path: Path to knowledge graph JSON
        campaign_id: Unique identifier for this campaign run
        use_llm: Enable LLM-guided prioritisation

    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(graph_path=graph_path, campaign_id=campaign_id, use_llm=use_llm)


__all__ = [
    "PlannerAgent",
    "PlannerDecision",
    "PlannerState",
    "PrioritizationIntent",
    "PrioritizationResult",
    "create_planner",
]