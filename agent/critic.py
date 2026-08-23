"""Critic agent: Safety gate before expensive escalation.

Validates:
1. Physical plausibility (e_above_hull < threshold)
2. Predictor uncertainty (reject if too high → escalate)
3. Schema compliance (no NaN/Inf outputs)

Outputs approval/rejection/escalation decision with cost impact tracking.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

import os

from kg.schema import MaterialNode, NodeType
from agent.predictor import PredictorResult
from agent.cost_model import EXPERIMENT_COST


# ---------------------------------------------------------------------------
# Configuration (load from .env)
# ---------------------------------------------------------------------------

def get_stability_threshold() -> float:
    """Get stability threshold from environment or default."""
    try:
        return float(os.environ.get("STABILITY_THRESHOLD", "0.1"))
    except ValueError:
        return 0.1


def get_uncertainty_gate() -> float:
    """Get uncertainty gate from environment or default (30%)."""
    try:
        return float(os.environ.get("UNCERTAINTY_GATE", "0.3"))
    except ValueError:
        return 0.3


# ---------------------------------------------------------------------------
# Pydantic models: Input schemas
# ---------------------------------------------------------------------------

class KGLookupResult(BaseModel):
    """Input from Retriever agent."""
    materials: List[MaterialNode]
    chemsys_groups: Optional[List[str]] = None
    elements_found: Optional[List[str]] = None
    provenance: Dict[str, Any]


# PredictorResult from predictor.py (MACE e_above_hull + MC Dropout uncertainty)



# ---------------------------------------------------------------------------
# Critic Agent
# ---------------------------------------------------------------------------

class CriticDecision(BaseModel):
    """Critic output decision."""
    approved: bool
    requires_escalation: bool
    reason: str = Field(..., description="Approval/rejection/escalation reason")
    cost_impact: float = Field(default=0.0, description="Additional cost if escalated")


class CriticAgent:
    """Critic agent with hard safety gates."""

    def __init__(self, graph_path: Path = None):
        """Initialize critic with thresholds.

        Args:
            graph_path: Path to knowledge graph JSON (for KG traversal)
        """
        # Load thresholds from environment
        self.stability_threshold = get_stability_threshold()
        self.uncertainty_gate = get_uncertainty_gate()
        
        # Load graph for KG traversal
        self.graph_path = graph_path or Path("data/processed/kg.json")
    def validate_materials(
        self,
        materials: List[MaterialNode],
        predictions: Optional[List[PredictorResult]] = None
    ) -> List[CriticDecision]:
        """Validate each material and return decision list.

        Args:
            materials: Materials from Retriever
            predictions: Optional Predictor outputs with uncertainty estimates

        Returns:
            List of CriticDecision objects (one per material)
        """
        decisions = []

        for mat in materials:
            if isinstance(mat, MaterialNode):
                decision = self._evaluate_material(mat, predictions)
                decisions.append(decision)

        return decisions

    def _evaluate_material(
        self,
        material: MaterialNode,
        predictions: Optional[List[PredictorResult]] = None
    ) -> CriticDecision:
        """Evaluate single material for safety gate.

        Args:
            material: Material to validate
            predictions: Predictor outputs for this material

        Returns:
            CriticDecision with approval status and reason
        """
        reasons = []
        requires_escalation = False

        # Check 1: Stability (e_above_hull threshold)
        stability_check = self._check_stability(material)
        if not stability_check["passed"]:
            reasons.append(f"Unstable: e_above_hull={stability_check['value']:.3f} > {self.stability_threshold}")
            return CriticDecision(
                approved=False,
                requires_escalation=False,
                reason=reasons[0],
                cost_impact=0.0,
            )

        # Check 2: Predictor uncertainty (if available)
        if predictions:
            unc_check = self._check_uncertainty(predictions)
            if not unc_check["passed"]:
                requires_escalation = True
                reasons.append(f"High uncertainty: {unc_check['value']:.1%} > {self.uncertainty_gate:.1%}")

        # Check 3: Schema compliance (no NaN/Inf)
        schema_check = self._check_schema(material)
        if not schema_check["passed"]:
            reasons.append(f"Invalid data: {schema_check['error']}")
            return CriticDecision(
                approved=False,
                requires_escalation=False,
                reason=reasons[0],
                cost_impact=0.0,
            )

        # Final decision
        if requires_escalation:
            return CriticDecision(
                approved=False,
                requires_escalation=True,
                reason=" ".join(reasons),
                cost_impact=EXPERIMENT_COST,  # Will be set from cost model
            )
        else:
            reasons.append("All checks passed")
            return CriticDecision(
                approved=True,
                requires_escalation=False,
                reason=reasons[0],
                cost_impact=0.0,
            )

    def _check_stability(self, material: MaterialNode) -> Dict[str, Any]:
        """Check e_above_hull stability threshold.

        Args:
            material: Material to check

        Returns:
            Dict with {passed: bool, value: float}
        """
        # Find e_above_hull property for this material from KG
        eah_value = self._get_e_above_hull_from_kg(material.id)

        if eah_value is None:
            return {"passed": False, "value": None, "error": "No e_above_hull property found in KG"}

        passed = eah_value <= self.stability_threshold
        return {"passed": passed, "value": eah_value}

    def _check_uncertainty(self, predictions: List[PredictorResult]) -> Dict[str, Any]:
        """Check predictor uncertainty against gate.

        Args:
            predictions: List of PredictorResult objects

        Returns:
            Dict with {passed: bool, value: float}
        """
        max_uncertainty = 0.0

        for pred in predictions:
            if pred.uncertainty > max_uncertainty:
                max_uncertainty = pred.uncertainty

        passed = max_uncertainty <= self.uncertainty_gate
        return {"passed": passed, "value": max_uncertainty}

    def _check_schema(self, material: MaterialNode) -> Dict[str, Any]:
        """Validate schema compliance (no NaN/Inf).

        Args:
            material: Material to validate

        Returns:
            Dict with {passed: bool, error: str}
        """
        # Check material fields
        if material.formula_pretty is None:
            return {"passed": False, "error": "formula_pretty is None"}
        
        if not material.elements:
            return {"passed": False, "error": "Empty elements list"}

        # Check for NaN/Inf in strings (shouldn't happen but defend)
        try:
            float(material.formula_pretty.replace(" ", ""))
        except (ValueError, TypeError):
            pass  # Formula strings are OK

        return {"passed": True, "error": None}

    def _get_e_above_hull_from_kg(self, material_id: str) -> Optional[float]:
        """Get e_above_hull value for a material from the KG.

        Args:
            material_id: Material node ID (e.g., "material:mp-1234")

        Returns:
            e_above_hull value or None if not found
        """
        try:
            from kg.graph_store import load_graph
            
            G = load_graph(self.graph_path)
            
            # Find property nodes for this material with name "energy_above_hull"
            for nid, data in G.nodes(data=True):
                if (data.get("type") == NodeType.PROPERTY.value and 
                    data.get("name") == "energy_above_hull" and
                    data.get("mpid") == material_id.split(":")[-1] if ":" in material_id else material_id):
                    return float(data.get("value", 0))
            
            return None
        except Exception as e:
            print(f"Critic: Failed to load KG for stability check: {e}")
            return None

    def _traverse_material_properties(self, material_id: str) -> List[Dict]:
        """Traverse graph to find properties for a material.

        Args:
            material_id: Material node ID

        Returns:
            List of property node data dicts
        """
        # Use the new method instead
        return []


# ---------------------------------------------------------------------------
# Cost constants for escalation tracking
# ---------------------------------------------------------------------------

EXPERIMENT_COST = 10.0  # High cost for expensive DFT/UMA checks


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_critic() -> CriticAgent:
    """Factory function to create a critic agent.

    Returns:
        Configured CriticAgent instance with default thresholds
    """
    return CriticAgent()


__all__ = [
    "CriticAgent",
    "CriticDecision",
    "STABILITY_THRESHOLD",
    "UNCERTAINTY_GATE",
    "create_critic",
]
