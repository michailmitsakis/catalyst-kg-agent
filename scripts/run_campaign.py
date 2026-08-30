#!/usr/bin/env python
"""CLI entry point for catalyst discovery campaigns.

Usage:
    python scripts/run_campaign.py --budget 100 [--ollama-model gemma4:latest] [--mode batch|sequential] [--mace-checkpoint mace-mpa-0-medium]

This script orchestrates the full agent loop:
1. Retriever → Predictor → Critic → Scribe (repeat until budget exhausted)
2. Logs to MLflow and JSON journals

Execution modes:
- batch (default): Write KG predictions at end of campaign (faster, better for large batches)
- sequential: Write KG after each iteration (slower, enables adaptive discovery within campaign)
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.campaign import (
    CampaignOrchestrator,
    INITIAL_BUDGET,
    MAX_ACTIONS_PER_CAMPAIGN,
)
from tracking.mlflow_setup import (
    create_mlflow_logger,
    log_campaign_metrics,
    log_campaign_params,
)


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
        help="Override INITIAL_BUDGET from .env"
    )
    
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Override MAX_EXPERIMENTS from .env"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["sequential", "batch"],
        help="Campaign execution mode: 'sequential' (write KG after each iteration) or 'batch' (default, write at end)"
    )
    
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (e.g., gemma4:latest)"
    )
    
    parser.add_argument(
        "--mace-checkpoint",
        type=str,
        default=None,
        help="MACE checkpoint path or name"
    )
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

def run_campaign(
    campaign_id: str,
    ollama_model: str | None = None,
    mace_checkpoint: str | None = None,
    mode: str | None = None,
) -> dict:
    """Run a complete catalyst discovery campaign using CampaignOrchestrator.

    Args:
        campaign_id: Unique identifier for this campaign (UUID recommended)
        ollama_model: Ollama model name/version used
        mace_checkpoint: MACE checkpoint version
        mode: Execution mode override ("sequential" or "batch")

    Returns:
        Dict with campaign results and MLflow run info
    """
    print(f"Starting campaign {campaign_id}...")
    
    # Initialize orchestrator (handles retriever, predictor, critic)
    orchestrator = CampaignOrchestrator()
    
    # Initialize MLflow tracking
    mlflow_logger = create_mlflow_logger(
        campaign_id=campaign_id,
        ollama_model=ollama_model,
        mace_checkpoint_version=mace_checkpoint,
    )
    
    try:
        # Run the orchestrator (now synchronous)
        result = orchestrator.run()
        
        # Update MLflow with results
        total_spent = result.get("budget_summary", {}).get("total_spent", 0)
        n_materials = result.get("campaign_state", {}).get("n_materials_evaluated", 0)
        
        mlflow_logger.log_campaign_end(
            total_cost=total_spent,
            materials_evaluated=n_materials,
            predictions_made=0,  # Not tracked yet
            escalations_triggered=0,  # Not tracked yet
            best_candidate_e_above_hull=None,  # Would compute from results
            final_outcome="completed",
        )
        
        return {
            "campaign_id": result.get("campaign_state", {}).get("campaign_id"),
            "materials_evaluated": len(result.get("final_materials", [])),
            "total_cost": result.get("budget_summary", {}).get("total_spent", 0),
            "mlflow_run_id": mlflow_logger.run_id,
            "status": "success",
            "result": result,
        }
        
    except Exception as e:
        print(f"Campaign failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Log failure to MLflow
        mlflow_logger.log_campaign_end(
            total_cost=0,
            materials_evaluated=0,
            predictions_made=0,
            escalations_triggered=0,
            best_candidate_e_above_hull=None,
            final_outcome="failed",
        )
        
        raise


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Generate campaign ID if not provided
    campaign_id = args.campaign_id if hasattr(args, 'campaign_id') else str(uuid.uuid4())
    
    print("="*60)
    print("Catalyst Discovery Campaign")
    print("="*60)
    print(f"Campaign ID: {campaign_id}")
    print()
    
    # Override config from command line if provided
    if args.budget is not None:
        import os
        os.environ["INITIAL_BUDGET"] = str(args.budget)
        print(f"Budget override: {args.budget}")
    
    if args.max_experiments is not None:
        import os
        os.environ["MAX_EXPERIMENTS"] = str(args.max_experiments)
        print(f"Max experiments override: {args.max_experiments}")
    
    try:
        # Run campaign
        result = run_campaign(
            campaign_id=campaign_id,
            ollama_model=args.ollama_model,
            mace_checkpoint=args.mace_checkpoint,
            mode=args.mode,
        )
        
        print("\n" + "="*60)
        print("Campaign completed successfully!")
        print(f"MLflow run ID: {result['mlflow_run_id']}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nCampaign failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
