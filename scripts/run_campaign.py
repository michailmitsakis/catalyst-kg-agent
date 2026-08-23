#!/usr/bin/env python
"""CLI entry point for catalyst discovery campaigns.

Usage:
    python scripts/run_campaign.py --budget 100 [--unsloth-model llama3.1:8b] [--mace-checkpoint mace-mpa-0-medium]

This script orchestrates the full agent loop:
1. Retriever → Predictor → Critic → Planner (repeat until budget exhausted)
2. Logs to MLflow and JSON journals
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.retriever import create_retriever
from agent.predictor import create_predictor
from agent.critic import create_critic
from agent.planner import create_planner
from agent.scribe import create_scribe
from tracking.mlflow_setup import (
    create_mlflow_logger,
    log_campaign_metrics,
    log_campaign_params,
)
from agent.logging import create_logging_context


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
        "--unsloth-model",
        type=str,
        default=None,
        help="Unsloth model name (e.g., llama3.1:8b)"
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
    unsloth_model: str | None = None,
    mace_checkpoint: str | None = None,
) -> dict:
    """Run a complete catalyst discovery campaign.

    Args:
        campaign_id: Unique identifier for this campaign (UUID recommended)
        unsloth_model: Model name/version used
        mace_checkpoint: MACE checkpoint version

    Returns:
        Dict with campaign results and MLflow run info
    """
    print(f"Starting campaign {campaign_id}...")
    
    # Initialize agents
    retriever = create_retriever()
    predictor = create_predictor()
    critic = create_critic()
    planner = create_planner(campaign_id=campaign_id)
    scribe = create_scribe()
    
    # Initialize MLflow tracking
    mlflow_logger = create_mlflow_logger(
        campaign_id=campaign_id,
        unsloth_model=unsloth_model,
        mace_checkpoint_version=mace_checkpoint,
    )
    
    # Create logging context
    with create_logging_context(campaign_id) as logger:
        try:
            # Campaign state
            materials_pool = []  # All available materials from KG
            evaluated_materials = []  # Materials we've already queried
            best_candidate = None
            total_cost = 0.0
            
            print(f"Available materials in KG: {len(planner.G.nodes())}")
            
            # Main campaign loop
            step = 0
            while not planner.is_campaign_complete():
                step += 1
                print(f"\n--- Step {step} ---")
                
                # Log step start
                logger.log(
                    level="INFO",
                    message=f"Campaign step {step}: Remaining budget=${planner.get_remaining_budget():.1f}",
                )
                
                # 1. Retriever: Get candidate materials
                print("Retrieving candidates from KG...")
                retrieved = retriever.search(
                    query="Find stable HER/OER catalyst materials",
                    chemsys_groups=["Ni-P", "Co-P", "Fe-P"],  # Example filters
                )
                
                if not retrieved:
                    logger.log(
                        level="WARNING",
                        message="No candidates found from KG",
                    )
                    break
                
                # Filter out already-evaluated materials
                new_materials = [m for m in retrieved if m.mpid not in [e.mpid for e in evaluated_materials]]
                
                if not new_materials:
                    print("All materials already evaluated. Stopping.")
                    break
                
                print(f"Retrieved {len(new_materials)} new candidates")
                
                # 2. Planner: Decide next action
                decision = planner.plan_next_step(
                    retrieved_materials=new_materials,
                )
                
                print(f"Planner decision: {decision.next_action}")
                
                # Update evaluated materials
                evaluated_materials.extend(new_materials)
                
                # 3. Predictor: Get property estimates (if continue action)
                if decision.next_action == "continue":
                    print("Running MACE predictions...")
                    predictions = []
                    
                    for mat in new_materials:
                        result = predictor.predict(mat)
                        predictions.append(result)
                        
                        # Log prediction
                        logger.log(
                            level="INFO",
                            message=f"Predicted {mat.mpid}: e_above_hull={result.property_value:.3f} eV, uncertainty={result.uncertainty:.1%}",
                        )
                    
                    # Scribe: Log predictions to KG
                    scribe_results = scribe.log_predictions(predictions)
                    print(f"Scribed {len(scribe_results)} predictions")
                
                # 4. Critic: Validate before escalation
                if decision.next_action in ("escalate", "continue"):
                    decisions = critic.validate_materials(
                        materials=new_materials,
                        predictions=predictions if 'predictions' in locals() else None,
                    )
                    
                    # Log critic decisions
                    for dec in decisions:
                        logger.log(
                            level="WARNING" if not dec.approved else "INFO",
                            message=f"Critic decision for {dec.reason}",
                        )
                    
                    # Update state with critic rejections
                    planner.state.critic_rejections += sum(1 for d in decisions if not d.approved)
                
                # 5. Planner: Update budget and check completion
                print(f"Cost update: ${decision.cost_update:.1f}")
                total_cost += decision.cost_update
                
                # Log cost to MLflow
                category = "kg_lookup" if decision.next_action == "continue" else "experiment_escalation"
                mlflow_logger.log_cost(category, decision.cost_update)
                
                print(f"Remaining budget: ${planner.get_remaining_budget():.1f}")
            
            # Campaign complete - gather results
            print("\n=== Campaign Complete ===")
            print(f"Total steps: {step}")
            print(f"Materials evaluated: {len(evaluated_materials)}")
            print(f"Total cost: ${total_cost:.1f}")
            
            # Find best candidate (lowest e_above_hull)
            if evaluated_materials:
                # Get predictions from KG or use KG properties
                from kg.graph_store import load_graph, rehydrate_node
                G = load_graph(planner.graph_path)
                
                for mat in sorted(evaluated_materials, key=lambda m: str(m.mpid))[:5]:
                    print(f"  - {mat.formula_pretty} ({mat.mpid})")
            
            # Log final metrics to MLflow
            mlflow_logger.log_campaign_end(
                total_cost=total_cost,
                materials_evaluated=len(evaluated_materials),
                predictions_made=step,  # Approximate
                escalations_triggered=0,  # Would track from critic decisions
                best_candidate_e_above_hull=None,  # Would compute from results
                final_outcome="completed",
            )
            
            return {
                "campaign_id": campaign_id,
                "steps_completed": step,
                "materials_evaluated": len(evaluated_materials),
                "total_cost": total_cost,
                "mlflow_run_id": mlflow_logger.run_id,
                "status": "success",
            }
            
        except Exception as e:
            logger.log(
                level="ERROR",
                message=f"Campaign failed: {str(e)}",
                exception=e,
            )
            
            # Log error to MLflow
            mlflow_logger.log_campaign_end(
                total_cost=total_cost,
                materials_evaluated=len(evaluated_materials) if 'evaluated_materials' in locals() else 0,
                predictions_made=step if 'step' in locals() else 0,
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
            unsloth_model=args.unsloth_model,
            mace_checkpoint=args.mace_checkpoint,
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
