"""Campaign orchestrator: Runs the full budget-bounded discovery loop.

Coordinates all agents (Retriever, Predictor, Critic, Planner, Scribe) and
manages budget tracking via cost_model.BudgetTracker. Logs each step to
journal + MLflow.

Planner's role here: CampaignOrchestrator owns the BudgetTracker and does
the actual cost deduction (KG_LOOKUP is deducted here, since Planner never
handled that cost -- see PlannerAgent.plan_next_step docstring). Planner is
consulted per step for the continue/escalate/stop decision; campaign.py
still performs the deduction its decision implies.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

import mlflow

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
from tracking.mlflow_setup import (
    setup_mlflow,
    log_campaign_metrics,
    end_campaign_tracking,
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
        # running -> completed (finished normally)
        #          -> budget_exhausted (spent its budget; expected outcome)
        #          -> failed (genuine fault)
        self.status: str = "running"
        self.final_materials: list[MaterialNode] = []
        self.budget_remaining: float = 0.0
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
            # Set by the orchestrator before the journal is written; the
            # hardcoded 0.0 it used to carry contradicted
            # budget_tracker.remaining_budget in every journal.
            "budget_remaining": self.budget_remaining,
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
        self._planner: Optional[Any] = None

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

    def get_planner(self, use_llm: bool = True):
        """Get or create Planner agent.

        use_llm defaults to True: the Planner uses an LLM to ORDER the
        remaining candidates (the acquisition-function role), which is where
        a model adds something rules cannot. Its continue/escalate/stop
        logic stays deterministic regardless of this flag -- see
        agent/planner.py. If Ollama is unreachable the Planner falls back to
        the corpus order and the campaign still runs.

        Note: PlannerAgent keeps its own internal PlannerState.remaining_budget
        for standalone use (see tests/test_planner.py), but campaign.py never
        feeds it real deductions -- self.tracker is the only source of truth
        for budget here. Treat planner.state.remaining_budget as decorative
        when Planner is driven from this campaign loop.
        """
        if self._planner is None:
            from agent.planner import PlannerAgent, create_planner
            self._planner = PlannerAgent(
                graph_path=self.graph_path,
                campaign_id=self.campaign_id,
                use_llm=use_llm,
            )
        return self._planner

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

        planner = self.get_planner()

        # Accumulates predictions across every step, regardless of mode.
        # Batch mode writes this whole list once at the end; sequential
        # mode writes incrementally but this still tracks the full-run
        # total for the final journal/MLflow summary.
        all_predictions: list[Any] = []

        # mpids already scored by the Predictor. The Retriever is stateless
        # and returns the same candidate set on every call, so without this
        # the loop re-scores the same first material every step until the
        # budget dies.
        #
        # Seeded from the KG, not empty: any material that already carries a
        # MACE prediction from an EARLIER campaign is skipped, so successive
        # campaigns advance through the corpus instead of re-deriving the
        # same results. This is what makes the graph function as compounding
        # memory rather than a write-only log.
        scored_mpids: set[str] = self._already_predicted_mpids()
        if scored_mpids:
            self.state.log("resuming_from_kg", {"n_already_predicted": len(scored_mpids)})
            print(f"Campaign: {len(scored_mpids)} materials already predicted in "
                  f"earlier campaigns; skipping them.")

        # Main loop - use MAX_ACTIONS_PER_CAMPAIGN and budget check
        step = 0
        while step < MAX_ACTIONS_PER_CAMPAIGN and self.tracker.remaining_budget > 0:
            step += 1
            self.state.log(f"step_{step}", {"remaining_budget": self.tracker.remaining_budget})

            # Stop before paying for a retrieval that cannot fund a single
            # surrogate call. demo-001 spent a KG lookup on step 2 with 3.5
            # units left -- enough for the lookup, never enough for the
            # 5.0-unit prediction it was retrieving candidates for.
            if self.tracker.remaining_budget < (KG_LOOKUP_COST + SURROGATE_COST):
                self.state.status = "budget_exhausted"
                self.state.log("budget_exhausted", {
                    "remaining": round(self.tracker.remaining_budget, 2),
                    "reason": "insufficient for a retrieval plus one surrogate call",
                })
                break

            # 1. Retrieve candidates
            retriever = self.get_retriever()
            if query:
                result = retriever.run_query(query)
            else:
                result = retriever.run_query("find all stable materials")

            # Deduct KG lookup cost
            if not self.tracker.deduct(ActionCategory.KG_LOOKUP, result.query_cost):
                # Running out of budget is the EXPECTED termination of a
                # budget-bounded campaign, not an error. "failed" is reserved
                # for genuine faults so the two can be told apart in MLflow.
                self.state.status = "budget_exhausted"
                self.state.log("budget_exhausted", {})
                break

            # Retrieved set is deduplicated against everything already
            # scored, so `final_materials` counts unique materials rather
            # than retrievals. (Previously this extended by all 130 every
            # step, reporting 520 "materials evaluated" on a 130-material
            # corpus.)
            materials = [
                m for m in result.materials
                if getattr(m, "mpid", None) not in scored_mpids
            ]

            if not materials:
                self.state.log("candidates_exhausted", {
                    "n_scored": len(scored_mpids),
                })
                break

            # 2a. Ask the Planner which candidates to spend the budget on.
            # The budget only covers a fraction of the corpus, so the choice
            # of WHICH candidates is the substantive decision. The Planner
            # returns a permutation of `materials` -- never a subset, never
            # with additions -- so this is safe to use unconditionally.
            prioritized = planner.prioritize_candidates(
                candidates=materials,
                predictions_so_far=all_predictions,
                remaining_budget=self.tracker.remaining_budget,
            )
            materials = prioritized.ordered_materials
            self.state.log("prioritization", {
                "provenance": prioritized.provenance,
                "n_llm_ranked": prioritized.n_llm_ranked,
                "n_dropped_unknown": prioritized.n_dropped_unknown,
                "reason": prioritized.reason,
            })

            # 2b. Screen candidates one at a time: predict, then validate,
            # then escalate if warranted -- before moving to the next one.
            #
            # An earlier version ran the ENTIRE prediction batch first and
            # called the Critic once afterwards. The Critic's judgement was
            # already per-material, but its TIMING was per-batch: in a
            # 100-unit campaign all 19 surrogate calls spent 95 units, and
            # only then did the Critic flag mp-943 (Co3S4, residual force
            # 0.795 eV/A) for escalation -- with 3.5 units left against a
            # 10-unit escalation cost. The gate fired correctly and arrived
            # too late to act on.
            #
            # Screening incrementally means an escalation can be funded
            # while budget remains, which is how a real screening loop
            # behaves: you act on a bad reading before running eighteen more
            # measurements. Escalations now compete with surrogate calls for
            # the same budget, which is exactly what the cost model is for.
            critic_decisions = []
            try:
                predictor = self.get_predictor()
                critic = self.get_critic()
                predictions = []
                evaluated: list[MaterialNode] = []

                for mat in materials:
                    # Charge BEFORE computing. predict() is the expensive
                    # step, so running it first and deducting afterwards
                    # meant the last prediction of every step was real MACE
                    # compute that was never billed.
                    if not self.tracker.deduct(ActionCategory.SURROGATE_QUERY, SURROGATE_COST):
                        self.state.log("budget_exhausted_mid_predictions", {
                            "material_id": getattr(mat, "mpid", None),
                        })
                        break

                    pred_result = predictor.predict(mat)
                    predictions.append(pred_result)
                    evaluated.append(mat)
                    scored_mpids.add(getattr(mat, "mpid", None))

                    # Validate this material immediately, on its own.
                    decisions = critic.validate_materials([mat], [pred_result])
                    critic_decisions.extend(decisions)

                    for dec in decisions:
                        if not dec.requires_escalation:
                            continue
                        # Only log an escalation that was actually AFFORDED.
                        # Ignoring deduct()'s return here previously made the
                        # summary report an escalation the budget tracker had
                        # no record of.
                        if self.tracker.deduct(
                            ActionCategory.EXPERIMENT_ESCALATION, EXPERIMENT_COST
                        ):
                            self.state.log("critic_escalation", {
                                "material_id": getattr(mat, "mpid", None),
                                "reason": dec.reason,
                            })
                        else:
                            self.state.log("escalation_unaffordable", {
                                "material_id": getattr(mat, "mpid", None),
                                "reason": dec.reason,
                                "required": EXPERIMENT_COST,
                                "remaining": round(self.tracker.remaining_budget, 2),
                            })

            except NotImplementedError:
                # Skip prediction if not implemented yet
                predictions = None
                evaluated = list(materials)

            # `final_materials` counts materials actually SCORED, not merely
            # retrieved. The Retriever returns the whole candidate set each
            # call; counting retrievals reported 520 "materials evaluated" on
            # a 130-material corpus, and counting de-duplicated retrievals
            # would still have reported 457.
            self.state.final_materials.extend(evaluated)

            # 3b. Consult Planner for the continue/escalate/stop decision.
            # This part is deterministic -- no LLM call. See planner.py.
            # campaign.py still owns BudgetTracker and performs all actual
            # cost deduction above; Planner's decision informs loop control
            # (e.g. stopping early once max experiments is reached) without
            # re-deducting cost itself.
            planner_decision = planner.plan_next_step(
                retrieved_materials=evaluated,
                predictions=predictions,
                critic_decisions=critic_decisions,
            )
            self.state.log("planner_decision", {
                "next_action": planner_decision.next_action,
                "reason": planner_decision.reason,
            })

            # 4. Scribe - persist predictions to KG (mode-dependent)
            if predictions:
                all_predictions.extend(predictions)
                scribe = ScribeAgent(graph_path=self.graph_path)

                if effective_mode == "sequential":
                    # Sequential mode: write each prediction immediately
                    self.state.log(f"step_{step}_scribe", {"mode": "sequential"})
                    for pred in predictions:
                        scribe._add_prediction_to_kg(pred)
                        # Note: surrogate cost already deducted in the
                        # prediction loop above.
                else:
                    # Batch mode: defer writing until the full run completes
                    self.state.log(f"step_{step}_scribe", {"mode": "batch", "n_predictions": len(predictions)})

            if planner_decision.next_action == "stop":
                self.state.log("planner_stop", {"reason": planner_decision.reason})
                break

        # Write deferred batch predictions if in batch mode. Uses the
        # full-run accumulated list, not just the last step's predictions.
        if effective_mode == "batch" and all_predictions:
            self.state.log("scribe_batch_deferred", {"mode": "batch", "n_predictions": len(all_predictions)})
            scribe = ScribeAgent(graph_path=self.graph_path)
            n_escalations = sum(1 for d in critic_decisions if d.requires_escalation)
            # Calculate total spent from budget tracker
            total_spent = round(self.tracker.initial_budget - self.tracker.remaining_budget, 2)
            scribe.log_campaign_results(
                campaign_id=self.campaign_id,
                materials_evaluated=self.state.final_materials,
                predictions_made=all_predictions,
                escalations_triggered=n_escalations,
                total_cost=total_spent
            )

            # Reload graph after Scribe write to get latest data
            self.G = load_graph(self.graph_path)

        # Sort by stability for final output (lowest e_above_hull first).
        #
        # Properties are separate NODES joined by HAS_PROPERTY edges, not a
        # "properties" list attribute on the material node. The previous
        # version read self.G.nodes[mat.id]["properties"], which never
        # exists, so every material scored inf, the sort was a no-op, and
        # "best candidate" was simply the first material in retrieval order.
        if self.state.final_materials:
            self.state.final_materials.sort(key=self._get_e_above_hull)

        if self.state.final_materials:
            self.state.best_candidate = self.state.final_materials[0]

        # End timing. Preserve any terminal status the loop already set --
        # an earlier version overwrote it unconditionally, so a campaign that
        # ran out of budget still reported success.
        self.state.end_time = datetime.now()
        if self.state.status == "running":
            self.state.status = "completed"

        # Compute best_candidate_e_above_hull from final materials
        best_e_above_hull = None
        if self.state.best_candidate:
            value = self._get_e_above_hull(self.state.best_candidate)
            if value != float("inf"):
                best_e_above_hull = value

        # Surface the real remaining budget in the state dict.
        self.state.budget_remaining = round(self.tracker.remaining_budget, 2)

        # Export journal
        self._write_journal(best_e_above_hull)

        # Log to MLflow
        self._log_to_mlflow(
            total_cost=round(self.tracker.initial_budget - self.tracker.remaining_budget, 2),
            materials_evaluated=len(self.state.final_materials),
            predictions_made=len(all_predictions),
            escalations_triggered=sum(1 for log in self.state.logs if log.get("event") == "critic_escalation"),
            best_candidate_e_above_hull=best_e_above_hull,
            final_outcome=self.state.status,
        )

        # Get summary stats
        result = self.state.to_dict()
        result["budget_summary"] = self.tracker.to_dict()
        result["logs"] = self.state.logs[-10:]  # Last 10 log entries

        return result


    def _already_predicted_mpids(self) -> set[str]:
        """mpids that already carry a MACE prediction in the knowledge graph.

        Read at campaign start so a new campaign skips work an earlier one
        already did. Without this the graph accumulates predictions that
        nothing ever consults, and every campaign re-derives the same first
        N materials until its budget runs out.

        Returns:
            Set of mpid strings, empty if none have been predicted yet
        """
        from agent.scribe import MACE_ENERGY_PROPERTY_NAME

        found: set[str] = set()
        for _nid, data in self.G.nodes(data=True):
            if data.get("type") != NodeType.PROPERTY.value:
                continue
            if data.get("name") != MACE_ENERGY_PROPERTY_NAME:
                continue
            mpid = data.get("mpid")
            if mpid:
                found.add(mpid)
        return found

    def _get_e_above_hull(self, material) -> float:
        """Look up a material's MP-derived e_above_hull from the graph.

        Property values live on their own nodes, reached via HAS_PROPERTY
        edges from the material. Returns inf when absent so that materials
        without a stability value sort last rather than first.

        Args:
            material: MaterialNode whose e_above_hull is wanted

        Returns:
            e_above_hull in eV/atom, or inf if not present
        """
        node_id = getattr(material, "id", None)
        if node_id is None or node_id not in self.G:
            return float("inf")

        for _src, target, data in self.G.edges(node_id, data=True):
            if data.get("type") != "HAS_PROPERTY":
                continue
            prop = self.G.nodes.get(target, {})
            if prop.get("name") == "energy_above_hull":
                try:
                    return float(prop["value"])
                except (KeyError, TypeError, ValueError):
                    return float("inf")
        return float("inf")

    def _write_journal(self, best_candidate_e_above_hull: Optional[float] = None):
        """Write campaign journal to JSON file.

        Args:
            best_candidate_e_above_hull: e_above_hull of best candidate (computed in run method)
        """
        journal_dir = Path("agent/journal")
        journal_dir.mkdir(parents=True, exist_ok=True)

        journal_file = journal_dir / f"{self.campaign_id}.json"
        journal_data = {
            "campaign_state": self.state.to_dict(),
            "budget_tracker": self.tracker.to_dict(),
            "end_time": datetime.now().isoformat(),
            "best_candidate_e_above_hull": best_candidate_e_above_hull,
        }

        journal_file.write_text(json.dumps(journal_data, indent=2))

    def _log_to_mlflow(
        self,
        total_cost: float,
        materials_evaluated: int,
        predictions_made: int,
        escalations_triggered: int,
        best_candidate_e_above_hull: Optional[float],
        final_outcome: str,
    ) -> dict[str, Any]:
        """Log campaign metrics to MLflow.

        Args:
            total_cost: Total budget spent
            materials_evaluated: Count of unique materials queried
            predictions_made: Count of surrogate predictions
            escalations_triggered: Count of times escalation occurred
            best_candidate_e_above_hull: e_above_hull of best candidate (if found)
            final_outcome: Campaign result string

        Returns:
            Dict with run_id and artifact_uri
        """
        setup_mlflow(self.campaign_id)

        # Log campaign metrics
        log_campaign_metrics(
            campaign_id=self.campaign_id,
            total_cost=total_cost,
            materials_evaluated=materials_evaluated,
            predictions_made=predictions_made,
            escalations_triggered=escalations_triggered,
            best_candidate_e_above_hull=best_candidate_e_above_hull,
            final_outcome=final_outcome,
        )

        # End the run and return info (MLflow 3.x API)
        artifact_uri = mlflow.active_run().info.artifact_uri if mlflow.active_run() else None
        mlflow.end_run()

        return {
            "run_id": self.state.campaign_id,
            "artifact_uri": artifact_uri,
        }


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