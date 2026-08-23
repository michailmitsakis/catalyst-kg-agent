"""Planner agent: Orchestrates budget-bounded discovery loop.

Decides next action based on:
1. Remaining budget
2. Critic approval status (including uncertainty gate)
3. Agent feedback (Retriever, Predictor, Critic)

Loop flow:
- Retriever (cheap KG lookup) → get candidate materials
- (Optional) Predictor (medium cost surrogate) → property estimates with uncertainty
- Critic (validation) → gate before escalation (stability + uncertainty check)
- Planner decides: continue loop OR escalate to expensive DFT OR stop

Tracks cost per action, logs to journal for campaign analysis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from pydantic_ai import Agent
from pydantic import BaseModel

from kg.schema import MaterialNode, PropertyNode, NodeType
from kg.graph_store import load_graph
from agent.cost_model import (
    INITIAL_BUDGET,
    KG_LOOKUP_COST,
    SURROGATE_COST,
    EXPERIMENT_COST,
)


# ---------------------------------------------------------------------------
# Configuration (load from .env)
# ---------------------------------------------------------------------------

def get_max_experiments() -> int:
    """Get max experiments limit from environment or default."""
    try:
        return int(os.environ.get("MAX_EXPERIMENTS", "10"))
    except ValueError:
        return 10


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
    experiments_count: int = 0
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

    def __init__(self, graph_path: Path = None, campaign_id: str = 'default', use_llm: bool = True):
        """Initialize planner with KG path and campaign ID."""
        self.graph_path = graph_path or Path("data/processed/kg.json")
        self.campaign_id = campaign_id
        self.G = load_graph(self.graph_path)

        # State tracking
        self.state = PlannerState(campaign_id=campaign_id)

        # Load max experiments from environment
        self.max_experiments = get_max_experiments()

        # Agent setup (Pydantic-ai for typed I/O)
        if use_llm:
            unsloth_url = os.environ.get("UNSLOTH_BASE_URL", "http://localhost:11434/v1")
            self.agent = Agent(
                model=f"{unsloth_url}/llama3.1:8b",
                system_prompt=self._build_system_prompt(),
            )
        else:
            self.agent = None
    def _build_system_prompt(self) -> str:
        """Build system prompt describing budget loop logic."""
        return f"""You are the Planner agent for catalyst discovery campaigns.

BUDGET MANAGEMENT:
- Total budget: {INITIAL_BUDGET} units
- Action costs: KG_LOOKUP={KG_LOOKUP_COST}, SURROGATE={SURROGATE_COST}, EXPERIMENT={EXPERIMENT_COST}
- Max experiments allowed: {self.max_experiments}
- Stop when budget depleted OR max experiments reached OR target found

LOOP FLOW:
1. Call Retriever → get candidate materials from KG
2. (Optional) Call Predictor → get property estimates with uncertainty
3. Call Critic → validate stability and uncertainty thresholds
4. Decide next action:
   • "continue": Keep looping if budget remains AND experiments_count < max_experiments
   • "escalate": Call expensive DFT/UMA on best candidate (if uncertainty gate triggered)
   • "stop": Campaign complete

DECISION RULES:
- Escalate if: Critic requires escalation OR best candidate meets target property
- Stop if: Budget exhausted OR max experiments reached OR no viable candidates remain
- Continue if: More materials to evaluate within budget AND experiments_count < max_experiments

TRACKING:
- Log each action to journal (JSON file)
- Track cost per step, total actions, rejection count, experiments count

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

        # Check if escalation needed from Critic (uncertainty gate triggered)
        escalation_needed = False
        if critic_decisions:
            for decision in critic_decisions:
                if hasattr(decision, 'requires_escalation') and decision.requires_escalation:
                    escalation_needed = True
                    break

        # Check budget constraints
        can_afford_surrogate = self.state.remaining_budget >= SURROGATE_COST
        can_afford_experiment = self.state.remaining_budget >= EXPERIMENT_COST
        
        # Check experiment count limit
        experiments_limit_reached = self.state.experiments_count >= self.max_experiments

        # Decision logic
        if escalation_needed:
            # Critic requires escalation (high uncertainty)
            decision = PlannerDecision(
                next_action="escalate",
                reason=f"Critic requires escalation: uncertainty gate triggered (experiments so far: {self.state.experiments_count}/{self.max_experiments})",
                cost_update=EXPERIMENT_COST,
                materials_approved=[m for m in retrieved_materials if hasattr(m, 'formula_pretty')],
            )
            self.state.experiments_count += 1

        elif experiments_limit_reached:
            # Max experiments reached
            decision = PlannerDecision(
                next_action="stop",
                reason=f"Maximum experiments ({self.max_experiments}) reached",
                cost_update=0.0,
                materials_approved=[m for m in retrieved_materials if hasattr(m, 'formula_pretty')],
            )

        elif not can_afford_experiment and not can_afford_surrogate:
            # Budget exhausted
            decision = PlannerDecision(
                next_action="stop",
                reason="Budget exhausted (insufficient funds for any further actions)",
                cost_update=0.0,
                materials_approved=[m for m in retrieved_materials if hasattr(m, 'formula_pretty')],
            )

        elif len(retrieved_materials) > 0 and can_afford_surrogate:
            # Continue with surrogate prediction
            decision = PlannerDecision(
                next_action="continue",
                reason=f"{len(retrieved_materials)} viable candidates found (experiments so far: {self.state.experiments_count}/{self.max_experiments})",
                cost_update=SURROGATE_COST,
                materials_approved=retrieved_materials,
            )

        else:
            # No candidates or cannot afford surrogate
            decision = PlannerDecision(
                next_action="stop",
                reason="No candidates retrieved from KG or insufficient budget for surrogate",
                cost_update=0.0,
                materials_approved=[],
            )

        # Update remaining budget and experiment count
        self.state.remaining_budget -= decision.cost_update

        return decision

    def get_remaining_budget(self) -> float:
        """Get current campaign budget."""
        return self.state.remaining_budget

    def is_campaign_complete(self) -> bool:
        """Check if campaign should terminate."""
        return (
            self.state.remaining_budget <= 0 or 
            self.state.actions_taken >= MAX_ACTIONS_PER_CAMPAIGN or
            self.state.experiments_count >= self.max_experiments
        )


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
    "create_planner",
]
