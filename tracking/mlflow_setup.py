"""MLflow tracking setup for catalyst discovery campaigns.

Provides functions to log metrics, parameters, and artifacts during campaign execution.
Uses file-based MLflow (no server required) — logs stored in `tracking/mlruns/`.

Metrics logged per campaign:
- cost/kg_lookup, cost/surrogate_query, cost/experiment_escalation
- total_cost, materials_evaluated, predictions_made, escalations_triggered
- final_outcome, best_candidate_e_above_hull

Tags (metadata):
- unsloth_model, mace_checkpoint_version, campaign_duration_seconds
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import mlflow


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_tracking_uri() -> str:
    """Get MLflow tracking URI from environment or default to file-based."""
    return (
        os.environ.get("MLFLOW_TRACKING_URI", "file:./tracking/mlruns")
    )


def setup_mlflow(campaign_id: str) -> None:
    """Configure MLflow for campaign tracking.

    Args:
        campaign_id: Unique identifier for this campaign run
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(f"catalyst_discovery/{campaign_id}")


# ---------------------------------------------------------------------------
# Metric logging functions
# ---------------------------------------------------------------------------

def log_cost_metric(metric_name: str, value: float) -> None:
    """Log a cost-related metric.

    Args:
        metric_name: One of "kg_lookup", "surrogate_query", "experiment_escalation"
        value: Cost amount in budget units
    """
    # Convert to full metric name
    full_name = f"cost/{metric_name}"
    mlflow.log_metric(full_name, value)


def log_campaign_metrics(
    campaign_id: str,
    total_cost: float,
    materials_evaluated: int,
    predictions_made: int,
    escalations_triggered: int,
    best_candidate_e_above_hull: Optional[float] = None,
    final_outcome: Optional[str] = None,
) -> None:
    """Log all campaign-level metrics at once.

    Args:
        campaign_id: Unique identifier for this campaign
        total_cost: Total budget spent
        materials_evaluated: Count of unique materials queried
        predictions_made: Count of surrogate predictions
        escalations_triggered: Count of times escalation occurred
        best_candidate_e_above_hull: e_above_hull of best candidate (if found)
        final_outcome: Campaign result string
    """
    # Core metrics
    mlflow.log_metric("total_cost", round(total_cost, 2))
    mlflow.log_metric("materials_evaluated", materials_evaluated)
    mlflow.log_metric("predictions_made", predictions_made)
    mlflow.log_metric("escalations_triggered", escalations_triggered)

    # Best candidate metric (optional)
    if best_candidate_e_above_hull is not None:
        mlflow.log_metric("best_candidate_e_above_hull", round(best_candidate_e_above_hull, 4))

    # Final outcome (as tag since it's categorical)
    if final_outcome is not None:
        mlflow.log_param("final_outcome", final_outcome)


# ---------------------------------------------------------------------------
# Parameter logging functions
# ---------------------------------------------------------------------------

def log_campaign_params(
    campaign_id: str,
    unsloth_model: Optional[str] = None,
    mace_checkpoint_version: Optional[str] = None,
    initial_budget: float = 100.0,
    max_experiments: int = 10,
) -> None:
    """Log campaign parameters/metadata as MLflow params.

    Args:
        campaign_id: Unique identifier for this campaign
        unsloth_model: Model name/version used (e.g., "llama3.1:8b")
        mace_checkpoint_version: MACE checkpoint version (e.g., "mace-mpa-0-medium")
        initial_budget: Starting budget amount
        max_experiments: Maximum experiments allowed
    """
    mlflow.log_param("unsloth_model", unsloth_model or "not_specified")
    mlflow.log_param("mace_checkpoint_version", mace_checkpoint_version or "not_specified")
    mlflow.log_param("initial_budget", initial_budget)
    mlflow.log_param("max_experiments", max_experiments)


# ---------------------------------------------------------------------------
# Artifact logging functions
# ---------------------------------------------------------------------------

def log_campaign_artifact(
    campaign_id: str,
    artifact_name: str,
    file_path: Path,
) -> None:
    """Log a file as an MLflow artifact.

    Args:
        campaign_id: Unique identifier for this campaign
        artifact_name: Name to give the artifact in MLflow
        file_path: Path to file to log
    """
    mlflow.log_artifact(str(file_path), artifact_name)


def log_campaign_log(
    campaign_id: str,
    log_file: Path,
) -> None:
    """Log a console log file as an MLflow artifact.

    Args:
        campaign_id: Unique identifier for this campaign
        log_file: Path to log file (e.g., campaign_20260820.log)
    """
    mlflow.log_artifact(str(log_file), f"logs/{log_file.name}")


# ---------------------------------------------------------------------------
# Campaign summary logging
# ---------------------------------------------------------------------------

def log_campaign_summary(
    campaign_id: str,
    metrics: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    """Log complete campaign summary.

    Args:
        campaign_id: Unique identifier for this campaign
        metrics: Dict of metric_name -> value
        params: Dict of param_name -> value
    """
    # Log all metrics
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    # Log all params
    for name, value in params.items():
        mlflow.log_param(name, value)


# ---------------------------------------------------------------------------
# Convenience functions for campaign execution
# ---------------------------------------------------------------------------

def start_campaign_tracking(
    campaign_id: str,
    unsloth_model: Optional[str] = None,
    mace_checkpoint_version: Optional[str] = None,
) -> str:
    """Start tracking a new campaign.

    Args:
        campaign_id: Unique identifier for this campaign (UUID recommended)
        unsloth_model: Model name/version used
        mace_checkpoint_version: MACE checkpoint version

    Returns:
        Active run ID from MLflow
    """
    setup_mlflow(campaign_id)
    
    with mlflow.start_run() as run:
        # Log parameters
        log_campaign_params(
            campaign_id=campaign_id,
            unsloth_model=unsloth_model,
            mace_checkpoint_version=mace_checkpoint_version,
        )
        
        # Return run ID for later reference
        return run.info.run_id


def end_campaign_tracking(
    campaign_id: str,
    total_cost: float,
    materials_evaluated: int,
    predictions_made: int,
    escalations_triggered: int,
    best_candidate_e_above_hull: Optional[float] = None,
    final_outcome: Optional[str] = None,
) -> Dict[str, Any]:
    """End tracking for a campaign and log final metrics.

    Args:
        campaign_id: Unique identifier for this campaign
        total_cost: Total budget spent
        materials_evaluated: Count of unique materials queried
        predictions_made: Count of surrogate predictions
        escalations_triggered: Count of times escalation occurred
        best_candidate_e_above_hull: e_above_hull of best candidate (if found)
        final_outcome: Campaign result string

    Returns:
        Dict with run_id and artifact_uri
    """
    setup_mlflow(campaign_id)
    
    # Log campaign metrics
    log_campaign_metrics(
        campaign_id=campaign_id,
        total_cost=total_cost,
        materials_evaluated=materials_evaluated,
        predictions_made=predictions_made,
        escalations_triggered=escalations_triggered,
        best_candidate_e_above_hull=best_candidate_e_above_hull,
        final_outcome=final_outcome,
    )
    
    with mlflow.start_run() as run:
        # Log metrics
        mlflow.log_metric("total_cost", round(total_cost, 2))
        mlflow.log_metric("materials_evaluated", materials_evaluated)
        mlflow.log_metric("predictions_made", predictions_made)
        mlflow.log_metric("escalations_triggered", escalations_triggered)
        
        if best_candidate_e_above_hull is not None:
            mlflow.log_metric("best_candidate_e_above_hull", round(best_candidate_e_above_hull, 4))
        
        # Log outcome as param
        if final_outcome is not None:
            mlflow.log_param("final_outcome", final_outcome)
        
        return {
            "run_id": run.info.run_id,
            "artifact_uri": run.info.artifact_uri,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_mlflow_logger(
    campaign_id: str,
    unsloth_model: Optional[str] = None,
    mace_checkpoint_version: Optional[str] = None,
) -> "MLflowLogger":
    """Create an MLflow logger for a campaign.

    Args:
        campaign_id: Unique identifier for this campaign
        unsloth_model: Model name/version used
        mace_checkpoint_version: MACE checkpoint version

    Returns:
        Configured MLflowLogger instance
    """
    return MLflowLogger(
        campaign_id=campaign_id,
        unsloth_model=unsloth_model,
        mace_checkpoint_version=mace_checkpoint_version,
    )


class MLflowLogger:
    """Context manager for MLflow campaign tracking.

    Usage:
        with create_mlflow_logger("campaign-uuid") as logger:
            logger.log_cost("kg_lookup", 1.0)
            logger.log_campaign_end(total_cost=45, materials_evaluated=12)
    """

    def __init__(
        self,
        campaign_id: str,
        unsloth_model: Optional[str] = None,
        mace_checkpoint_version: Optional[str] = None,
    ):
        self.campaign_id = campaign_id
        self.unsloth_model = unsloth_model
        self.mace_checkpoint_version = mace_checkpoint_version
        self.run_id: Optional[str] = None
        self.metrics: Dict[str, float] = {}
        self.params: Dict[str, Any] = {}

        setup_mlflow(campaign_id)

    def __enter__(self) -> "MLflowLogger":
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            self.params["unsloth_model"] = self.unsloth_model or "not_specified"
            self.params["mace_checkpoint_version"] = (
                self.mace_checkpoint_version or "not_specified"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # Run already started in __enter__

    def log_cost(self, metric_name: str, value: float) -> None:
        """Log a cost metric."""
        full_name = f"cost/{metric_name}"
        self.metrics[full_name] = value
        mlflow.log_metric(full_name, value)

    def log_campaign_end(
        self,
        total_cost: float,
        materials_evaluated: int,
        predictions_made: int,
        escalations_triggered: int,
        best_candidate_e_above_hull: Optional[float] = None,
        final_outcome: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log campaign completion metrics."""
        log_campaign_metrics(
            campaign_id=self.campaign_id,
            total_cost=total_cost,
            materials_evaluated=materials_evaluated,
            predictions_made=predictions_made,
            escalations_triggered=escalations_triggered,
            best_candidate_e_above_hull=best_candidate_e_above_hull,
            final_outcome=final_outcome,
        )

        return {
            "run_id": self.run_id,
            "artifact_uri": mlflow.active_run().info.artifact_uri if mlflow.active_run() else None,
        }

    def log_artifact(self, artifact_name: str, file_path: Path) -> None:
        """Log a file as an MLflow artifact."""
        mlflow.log_artifact(str(file_path), artifact_name)


__all__ = [
    "get_tracking_uri",
    "setup_mlflow",
    "log_cost_metric",
    "log_campaign_metrics",
    "log_campaign_params",
    "log_campaign_artifact",
    "log_campaign_log",
    "log_campaign_summary",
    "start_campaign_tracking",
    "end_campaign_tracking",
    "create_mlflow_logger",
    "MLflowLogger",
]
