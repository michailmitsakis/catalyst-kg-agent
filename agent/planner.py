"""Planner agent: Orchestrates budget-bounded discovery loop.

Decides next action based on:
1. Remaining budget
2. Critic approval status
3. Agent feedback (Retriever, Predictor, Critic)

Loop flow:
- Retriever (cheap KG lookup) → get candidate materials
- (Optional) Predictor (medium cost surrogate) → property estimates
- Critic (validation) → gate before escalation
- Planner decides: continue loop OR escalate to expensive DFT OR stop

Tracks cost per action, logs to journal for campaign analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from pydantic_ai import Agent

from kg.schema import MaterialNode, PropertyNode, NodeType
from kg.graph_store import load_graph


# ---------------------------------------------------------------------------
# Cost constants (must sync with agent/cost_model.py)
# ---------------------------------------------------------------------------

INITIAL_BUDGET = 100.0  # Total campaign budget
KG_LOOKUP_COST = 1.0    # Retriever cost
SURROGATE_COST = 5.0    # Predictor cost
EXPERIMENT_COST = 10.0  # Expensive DFT/UMA check


# ---------------------------------------------------------------------------
# Planner output schemas
# ---------------------------------------------------------------------------

@dataclass
class PlannerState:
    """Planner state tracking loop progress."""
    remaining_budget: float = INITIAL_BUDGET
    actions_taken: int = 0
    materials_evaluated: List[MaterialNode] = field(default_factory=list)
    critic_rejections: int = 0
    escalation_needed: bool = False
    campaign_id: str = "default"


class PlannerDecision(BaseModel):
    """Planner output decision."""
    next_action: str  # "continue", "escalate", "stop"
    reason: str
    cost_update: float  # Cost deducted this step
    materials_approved: List[MaterialNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

class PlannerAgent:
    """Budget-bounded discovery loop orchestrator."""

    def __init__(self, graph_path: Path = None, campaign_id: str = "default"):
        """Initialize planner with KG path and campaign ID."""
        self.graph_path = graph_path or Path("data/processed/kg.json")
        self.campaign_id = campaign_id
        self.G = load_graph(self.graph_path)

        # State tracking
        self.state = PlannerState(campaign_id=campaign_id)

        # Agent setup (Pydantic-ai for typed I/O)
        self.agent = Agent(
            model="ollama/llama3.1:8b",
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt describing budget loop logic."""
        return f"""You are the Planner agent for catalyst discovery campaigns.

BUDGET MANAGEMENT:
- Total budget: {INITIAL_BUDGET} units
- Action costs: KG_LOOKUP={KG_LOOKUP_COST}, SURROGATE={SURROGATE_COST}, EXPERIMENT={EXPERIMENT_COST}
- Stop when budget depleted OR target found

LOOP FLOW:
1. Call Retriever → get candidate materials from KG
2. (Optional) Call Predictor → get property estimates with uncertainty
3. Call Critic → validate stability and uncertainty thresholds
4. Decide next action:
   • "continue": Keep looping if budget remains
   • "escalate": Call expensive DFT/UMA on best candidate
   • "stop": Campaign complete

DECISION RULES:
- Escalate if: Critic requires escalation OR best candidate meets target property
- Stop if: Budget exhausted OR no viable candidates remain
- Continue if: More materials to evaluate within budget

TRACKING:
- Log each action to journal (JSON file)
- Track cost per step, total actions, rejection count

OUTPUT: PlannerDecision with next_action + reason + cost_update."""

    def plan_next_step(
        self,
        retrieved_materials: List[MaterialNode],
        predictions: Optional[List[Any]] = None,
        critic_decisions: Optional[List[Any]] = None
    ) -> PlannerDecision:
        """Plan the next action in the discovery loop.

        Args:
            retrieved_materials: Materials from Retriever
            predictions: Optional Predictor outputs
            critic_decisions: Optional Critic validation results

        Returns:
            PlannerDecision with next_action and cost update
        """
        # Update state
        self.state.materials_evaluated.extend(retrieved_materials)
        self.state.actions_taken += 1

        # Calculate step cost (assume KG lookup already done)
        step_cost = KG_LOOKUP_COST

        # Check if escalation needed
        escalation_needed = False
        if critic_decisions:
            for decision in critic_decisions:
                if hasattr(decision, 'requires_escalation') and decision.requires_escalation:
                    escalation_needed = True
                    break

        # Check if budget allows continuation
        can_continue = self.state.remaining_budget >= step_cost + SURROGATE_COST

        # Decision logic
        if escalation_needed or not can_continue:
            decision = PlannerDecision(
                next_action="escalate" if escalation_needed else "stop",
                reason="Escalation required OR budget exhausted",
                cost_update=EXPERIMENT_COST if escalation_needed else step_cost,
                materials_approved=[m for m in retrieved_materials if hasattr(m, 'formula_pretty')],
            )
        elif len(retrieved_materials) > 0:
            decision = PlannerDecision(
                next_action="continue",
                reason=f"{len(retrieved_materials)} viable candidates found",
                cost_update=step_cost,
                materials_approved=retrieved_materials,
            )
        else:
            decision = PlannerDecision(
                next_action="stop",
                reason="No candidates retrieved from KG",
                cost_update=0.0,
                materials_approved=[],
            )

        # Update remaining budget
        self.state.remaining_budget -= decision.cost_update

        return decision

    def get_remaining_budget(self) -> float:
        """Get current campaign budget."""
        return self.state.remaining_budget

    def is_campaign_complete(self) -> bool:
        """Check if campaign should terminate."""
        return self.state.remaining_budget <= 0 or self.state.actions_taken >= 50  # Max iterations


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_planner(graph_path: Path = None, campaign_id: str = "default") -> PlannerAgent:
    """Factory function to create a planner agent.

    Args:
        graph_path: Path to knowledge graph JSON
        campaign_id: Unique identifier for this campaign run

    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(graph_path=graph_path, campaign_id=campaign_id)


__all__ = [
    "PlannerAgent",
    "PlannerDecision",
    "PlannerState",
    "INITIAL_BUDGET",
    "KG_LOOKUP_COST",
    "SURROGATE_COST",
    "EXPERIMENT_COST",
    "create_planner",
]
