#!/usr/bin/env python
"""CLI entry point for catalyst discovery campaigns.

Usage:
    python scripts/run_campaign.py [--budget 100] [--campaign-id demo-001]
                                   [--mode batch|sequential]
                                   [--ollama-model gemma4:latest]
                                   [--mace-checkpoint mace-mpa-0-medium]

This is a thin CLI wrapper. All orchestration lives in
`agent/campaign.py`: the CampaignOrchestrator runs the
Retriever -> Predictor -> Critic -> Planner -> Scribe loop, owns the
BudgetTracker, writes the journal, and logs its own MLflow run.

This script therefore does NOT log to MLflow itself. An earlier version
created a second MLflowLogger here and called log_campaign_end() alongside
the orchestrator's internal _log_to_mlflow(), producing two runs per
campaign with different numbers in each.

Journal files are written to `agent/journal/<campaign_id>.json`.

Execution modes:
- batch (default): predictions are written to the KG once at the end of the
  campaign (fewer graph writes)
- sequential: predictions are written after each iteration (slower; lets a
  later step see earlier results)
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from agent.campaign import CampaignOrchestrator
from agent.cost_model import INITIAL_BUDGET


# ---------------------------------------------------------------------------
# Campaign configuration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a catalyst discovery campaign",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Starting budget for this campaign (default: INITIAL_BUDGET from agent/cost_model.py)",
    )

    parser.add_argument(
        "--campaign-id",
        type=str,
        default=None,
        help="Unique identifier for this campaign. Used as the journal filename: "
             "agent/journal/<campaign-id>.json. Auto-generated if omitted.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["sequential", "batch"],
        help="Campaign execution mode: 'sequential' (write KG after each iteration) "
             "or 'batch' (default, write once at the end)",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Natural-language query for the Retriever (default: find all stable materials)",
    )

    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (e.g. gemma4:latest). Overrides OLLAMA_MODEL from .env.",
    )

    parser.add_argument(
        "--mace-checkpoint",
        type=str,
        default=None,
        help="MACE checkpoint name. Overrides MACE_CHECKPOINT from .env.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

def run_campaign(
    campaign_id: str,
    budget: float | None = None,
    mode: str | None = None,
    query: str | None = None,
) -> dict:
    """Run one campaign via CampaignOrchestrator.

    Args:
        campaign_id: Unique identifier; becomes the journal filename
        budget: Starting budget (falls back to INITIAL_BUDGET)
        mode: "sequential" or "batch"
        query: Optional natural-language query for the Retriever

    Returns:
        The orchestrator's result dict, unmodified.
    """
    print(f"Starting campaign {campaign_id}...")

    orchestrator = CampaignOrchestrator(
        campaign_id=campaign_id,
        initial_budget=budget if budget is not None else INITIAL_BUDGET,
        # Orchestrator defaults to "batch"; only override when asked.
        **({"mode": mode} if mode else {}),
    )

    # The orchestrator handles MLflow and journal writing internally.
    return orchestrator.run(query=query)


def print_summary(result: dict) -> None:
    """Print a campaign summary.

    Reads the keys CampaignOrchestrator.run() actually returns. Its result
    is CampaignState.to_dict() FLATTENED at the top level, plus
    "budget_summary" and "logs" -- there is no nested "campaign_state" key
    and no "final_materials" key. An earlier version of this script read
    both of those, so every summary printed zeros even on a successful run.
    """
    budget = result.get("budget_summary", {})

    print("\n" + "=" * 60)
    print("Campaign Summary")
    print("=" * 60)
    print(f"  Campaign ID       : {result.get('campaign_id')}")
    print(f"  Status            : {result.get('status')}")
    print(f"  Materials evaluated: {result.get('n_materials_evaluated', 0)}")
    print(f"  Best candidate    : {result.get('best_candidate_mpid') or 'none'}")

    if budget:
        print()
        print(f"  Budget            : {budget.get('initial_budget')}")
        print(f"  Spent             : {budget.get('total_spent')}")
        print(f"  Remaining         : {budget.get('remaining_budget')}")

        costs = budget.get("costs_by_category", {})
        actions = budget.get("actions_by_category", {})
        if costs:
            print()
            print("  Cost breakdown:")
            for category, cost in costs.items():
                count = actions.get(category, 0)
                print(f"    {category:<24s} {cost:>7.1f}  ({count} actions)")

    escalations = sum(
        1 for entry in result.get("logs", []) if entry.get("event") == "critic_escalation"
    )
    if escalations:
        print(f"\n  Escalations triggered: {escalations}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main CLI entry point."""
    load_dotenv()
    args = parse_args()

    campaign_id = args.campaign_id or str(uuid.uuid4())[:8]

    # These are read from the environment by the agents themselves, so set
    # them before the orchestrator constructs anything.
    if args.ollama_model:
        os.environ["OLLAMA_MODEL"] = args.ollama_model
    if args.mace_checkpoint:
        os.environ["MACE_CHECKPOINT"] = args.mace_checkpoint

    print("=" * 60)
    print("Catalyst Discovery Campaign")
    print("=" * 60)
    print(f"Campaign ID: {campaign_id}")
    if args.budget is not None:
        print(f"Budget     : {args.budget}")
    if args.mode:
        print(f"Mode       : {args.mode}")
    print()

    try:
        result = run_campaign(
            campaign_id=campaign_id,
            budget=args.budget,
            mode=args.mode,
            query=args.query,
        )
    except Exception as exc:
        print(f"\nCampaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print_summary(result)
    print(f"\nJournal: agent/journal/{campaign_id}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())