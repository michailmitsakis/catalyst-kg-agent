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
    KG_LOOKUP_COST,
    SURROGATE_COST,
    EXPERIMENT_COST,
)
from agent.scribe import ScribeAgent


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
        mode: str = "batch",  # "sequential" or "batch" - default to batch for performance
    ):
        """Initialize orchestrator.

        Args:
            graph_path: Path to knowledge graph JSON
            campaign_id: Unique ID for this run (auto-generated if not provided)
            initial_budget: Starting budget amount
            mode: Campaign execution mode - "sequential" (write KG after each iteration) 
                  or "batch" (write KG at end of campaign, default)
        """
        self.graph_path = graph_path
        self.G = load_graph(graph_path)
        self.campaign_id = campaign_id or str(uuid.uuid4())[:8]
        self.mode = mode  # Track execution mode for logging

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
            from agent.predictor import PredictorAgent, create_predictor
            self._predictor = create_predictor()
        return self._predictor

    def get_critic(self):
        """Get or create Critic agent."""
        if self._critic is None:
            from agent.critic import CriticAgent, create_critic
            self._critic = create_critic()
        return self._critic

    # -----------------------------------------------------------------------
    # Campaign loop
    # -----------------------------------------------------------------------

    def run(
        self,
        query: Optional[str] = None,
        mode: Optional[str] = None,  # Override mode for this run
    ) -> dict[str, Any]:
        """Run the full campaign loop.

        Args:
            query: Natural language query (optional, defaults to "find all stable materials")
            mode: Execution mode override ("sequential" or "batch"). If None, uses instance default.

        Returns:
            Campaign result dict with state + summary stats
        """
        # Use provided mode or instance default
        effective_mode = mode if mode else self.mode
        
        # Start timing
        self.state.start_time = datetime.now()
        self.state.status = "running"
        self.state.log("campaign_start", {"query": query or "default", "mode": effective_mode})

        # Main loop - use MAX_ACTIONS_PER_CAMPAIGN and budget check
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

            # Deduct KG lookup cost
            if not self.tracker.deduct(ActionCategory.KG_LOOKUP, result.query_cost):
                self.state.status = "failed"
                self.state.log("budget_exhausted", {})
                break

            materials = result.materials
            self.state.final_materials.extend(materials)

            # 2. Predict properties (optional - use predictor if available)
            try:
                predictor = self.get_predictor()
                predictions = []
                
                for mat in materials:
                    pred_result = predictor.predict(mat)
                    predictions.append(pred_result)
                    
                    # Deduct surrogate cost
                    self.tracker.deduct(ActionCategory.SURROGATE_QUERY, SURROGATE_COST)

            except NotImplementedError:
                # Skip prediction if not implemented yet
                predictions = None
            
            # 3. Critic validation
            critic_decisions = []
            if materials and predictions:
                critic = self.get_critic()
                critic_decisions = critic.validate_materials(materials, predictions)

                # Track escalations (expensive DFT/UMA checks)
                for dec in critic_decisions:
                    if dec.requires_escalation:
                        self.state.log("critic_escalation", {"material_id": dec.reason})
                        # Deduct experiment cost
                        self.tracker.deduct(ActionCategory.EXPERIMENT_ESCALATION, EXPERIMENT_COST)

            # 4. Scribe - persist predictions to KG (mode-dependent)
            if predictions:
                scribe = ScribeAgent(graph_path=self.graph_path)
                n_escalations = sum(1 for d in critic_decisions if d.requires_escalation)
                
                if effective_mode == "sequential":
                    # Sequential mode: write each prediction immediately
                    self.state.log(f"step_{step}_scribe", {"mode": "sequential"})
                    for pred in predictions:
                        scribe._add_prediction_to_kg(pred)
                        self.tracker.deduct(ActionCategory.SURROGATE_QUERY, SURROGATE_COST)
                else:
                    # Batch mode: write all predictions at end of campaign (deferred)
                    self.state.log(f"step_{step}_scribe", {"mode": "batch", "n_predictions": len(predictions)})

        # Write deferred batch predictions if in batch mode and not already written
        if effective_mode == "batch" and predictions and step > 1:
            self.state.log("scribe_batch_deferred", {"mode": "batch", "n_predictions": len(predictions)})
            scribe = ScribeAgent(graph_path=self.graph_path)
            n_escalations = sum(1 for d in critic_decisions if d.requires_escalation)
            # Calculate total spent from budget tracker
            total_spent = round(self.tracker.initial_budget - self.tracker.remaining_budget, 2)
            scribe.log_campaign_results(
                campaign_id=self.campaign_id,
                materials_evaluated=materials,
                predictions_made=predictions,
                escalations_triggered=n_escalations,
                total_cost=total_spent
            )

        # Sort by stability for final output (lowest e_above_hull first)
        if self.state.final_materials:
            def get_eah(mat):
                props = self.G.nodes.get(mat.id, {}).get("properties", [])
                for p in props:
                    if isinstance(p, dict) and p.get("name") == "energy_above_hull":
                        return float(p.get("value", float("inf")))
                return float("inf")
            
            self.state.final_materials.sort(key=get_eah)

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
