"""Campaign orchestrator: Runs the full budget-bounded discovery loop.

Coordinates all agents (Retriever, Predictor, Critic, Scribe) and manages
budget tracking via cost_model.BudgetTracker. Logs each step to journal + MLflow.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from pydantic_ai import Agent

from kg.graph_store import load_graph, DEFAULT_KG_JSON
from kg.schema import MaterialNode, NodeType
from agent.cost_model import (
    BudgetTracker,
    ActionCategory,
    INITIAL_BUDGET,
    MAX_ACTIONS_PER_CAMPAIGN,
)


# ---------------------------------------------------------------------------
# Campaign state
# ---------------------------------------------------------------------------

class CampaignState:
    """Track campaign progress and results."""

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status: str = "running"  # running, completed, failed
        self.final_materials: list[MaterialNode] = []
        self.best_candidate: Optional[MaterialNode] = None
        self.logs: list[dict[str, Any]] = []

    def log(self, event: str, details: dict[str, Any]) -> None:
        """Append log entry.

        Args:
            event: Event type (e.g., "retriever_call", "critic_gate")
            details: Event-specific data
        """
        entry = {
            "campaign_id": self.campaign_id,
            "timestamp": datetime.now().isoformat(),
            "event": event,
            **details,
        }
        self.logs.append(entry)

    def to_dict(self) -> dict[str, Any]:
        """Export state for journal + MLflow."""
        return {
            "campaign_id": self.campaign_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "n_materials_evaluated": len(self.final_materials),
            "best_candidate_mpid": (
                self.best_candidate.mpid if self.best_candidate else None
            ),
            "budget_remaining": 0.0,  # Filled by orchestrator
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class CampaignOrchestrator:
    """Run a single campaign end-to-end."""

    def __init__(
        self,
        graph_path: Path = DEFAULT_KG_JSON,
        campaign_id: Optional[str] = None,
        initial_budget: float = INITIAL_BUDGET,
    ):
        """Initialize orchestrator.

        Args:
            graph_path: Path to knowledge graph JSON
            campaign_id: Unique ID for this run (auto-generated if not provided)
            initial_budget: Starting budget amount
        """
        self.graph_path = graph_path
        self.G = load_graph(graph_path)
        self.campaign_id = campaign_id or str(uuid.uuid4())[:8]

        # State
        self.state = CampaignState(campaign_id=self.campaign_id)
        self.tracker = BudgetTracker(initial_budget=initial_budget)

        # Agent instances (lazy-loaded to avoid model load until needed)
        self._retriever: Optional[Any] = None
        self._predictor: Optional[Any] = None
        self._critic: Optional[Any] = None

    # -----------------------------------------------------------------------
    # Agent factories
    # -----------------------------------------------------------------------

    def get_retriever(self):
        """Get or create Retriever agent."""
        if self._retriever is None:
            from agent.retriever import KGRetrieverAgent, create_retriever
            self._retriever = create_retriever(self.graph_path)
        return self._retriever

    def get_predictor(self):
        """Get or create Predictor agent."""
        if self._predictor is None:
            raise NotImplementedError("Predictor not yet implemented")
        return self._predictor

    def get_critic(self):
        """Get or create Critic agent."""
        if self._critic is None:
            raise NotImplementedError("Critic not yet implemented")
        return self._critic

    # -----------------------------------------------------------------------
    # Campaign loop
    # -----------------------------------------------------------------------

    async def run(
        self,
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the full campaign loop.

        Args:
            query: Natural language query (optional, defaults to "find all stable materials")

        Returns:
            Campaign result dict with state + summary stats
        """
        # Start timing
        self.state.start_time = datetime.now()
        self.state.status = "running"

        self.state.log("campaign_start", {"query": query or "default"})

        # Main loop
        step = 0
        while step < MAX_ACTIONS_PER_CAMPAIGN and self.tracker.remaining_budget > 0:
            step += 1
            self.state.log(f"step_{step}", {"remaining_budget": self.tracker.remaining_budget})

            # 1. Retrieve candidates
            retriever = self.get_retriever()
            if query:
                result = retriever.run_query(query)
            else:
                result = retriever.run_query("find all stable materials")

            cost_deducted = self.tracker.deduct(ActionCategory.KG_LOOKUP, result.query_cost)
            if not cost_deducted:
                self.state.status = "failed"
                self.state.log("budget_exhausted", {})
                break

            materials = result.materials
            self.state.final_materials.extend(materials)

            # 2. (Optional) Predict properties
            # Skip for now until Predictor impl is done
            predictions = None

            # 3. Critic validation
            critic_decisions = None
            if materials:
                # First material triggers escalation check
                first_mat = materials[0]
                requires_escalation = self._check_stability(first_mat)
                critic_decisions = [{"material_id": first_mat.mpid, "requires_escalation": requires_escalation}]

                # Escalate if needed
                if requires_escalation:
                    self.state.log("critic_escalation", {"material_id": first_mat.mpid})
                    # Would call expensive DFT/UMA here
                    break

        # Sort by stability for final output
        self.state.final_materials.sort(
            key=lambda m: (
                next((p.value for p in self.G.nodes.get(m.id, {}).get("properties", [])), float("inf"))
            )
        )

        if self.state.final_materials:
            self.state.best_candidate = self.state.final_materials[0]

        # End timing
        self.state.end_time = datetime.now()
        self.state.status = "completed"

        # Export journal
        self._write_journal()

        # Get summary stats
        result = self.state.to_dict()
        result["budget_summary"] = self.tracker.to_dict()
        result["logs"] = self.state.logs[-10:]  # Last 10 log entries

        return result

    def _check_stability(self, material: MaterialNode) -> bool:
        """Check if material meets stability threshold.

        Args:
            material: Material to check

        Returns:
            True if material should be escalated for expensive verification
        """
        # Default: escalate first candidate as demo
        return True

    def _write_journal(self):
        """Write campaign journal to JSON file."""
        journal_dir = Path("agent/journal")
        journal_dir.mkdir(parents=True, exist_ok=True)

        journal_file = journal_dir / f"{self.campaign_id}.json"
        journal_data = {
            "campaign_state": self.state.to_dict(),
            "budget_tracker": self.tracker.to_dict(),
            "end_time": datetime.now().isoformat(),
        }

        journal_file.write_text(json.dumps(journal_data, indent=2))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_campaign(
    graph_path: Path = DEFAULT_KG_JSON,
    campaign_id: Optional[str] = None,
    initial_budget: float = INITIAL_BUDGET,
) -> CampaignOrchestrator:
    """Factory function to create a campaign orchestrator.

    Args:
        graph_path: Path to knowledge graph JSON
        campaign_id: Unique ID for this run
        initial_budget: Starting budget amount

    Returns:
        Configured CampaignOrchestrator instance
    """
    return CampaignOrchestrator(
        graph_path=graph_path,
        campaign_id=campaign_id,
        initial_budget=initial_budget,
    )


__all__ = [
    "CampaignOrchestrator",
    "CampaignState",
    "create_campaign",
]
