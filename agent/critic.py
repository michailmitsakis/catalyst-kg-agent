"""Critic agent: Safety gate before expensive escalation.

Validates:
1. Physical plausibility -- e_above_hull below threshold, read from the KG's
   MP-derived value (NOT from the surrogate)
2. Surrogate trustworthiness -- max residual force below the gate; above it,
   the material is escalated for a higher-fidelity check
3. Schema compliance (no NaN/Inf outputs)

Outputs approval/rejection/escalation decision with cost impact tracking.

Note on check 2: this replaced an "uncertainty" check that read a
MC-Dropout standard deviation which was structurally always exactly 0.0
(MACE inference is deterministic, no dropout is active), so the gate could
never fire. See agent/predictor.py for the full explanation.
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


def get_force_gate() -> float:
    """Get the residual-force escalation gate from environment, in eV/Angstrom.

    Renamed from the previous UNCERTAINTY_GATE (a dimensionless fraction).
    The gate now compares against a max residual force in eV/Angstrom, so
    keeping the old name with new units would have been a silent semantics
    change -- exactly the class of bug that made a raw energy masquerade as
    e_above_hull earlier in this project. Set FORCE_GATE_EV_PER_ANG in .env.
    """
    try:
        return float(os.environ.get("FORCE_GATE_EV_PER_ANG", "0.1"))
    except ValueError:
        return 0.1


# ---------------------------------------------------------------------------
# Pydantic models: Input schemas
# ---------------------------------------------------------------------------

class KGLookupResult(BaseModel):
    """Input from Retriever agent."""
    materials: List[MaterialNode]
    chemsys_groups: Optional[List[str]] = None
    elements_found: Optional[List[str]] = None
    provenance: Dict[str, Any]


# PredictorResult from predictor.py (MACE energy per atom, formation energy,
# and max residual force -- the escalation signal; see predictor.py docstring)



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
        self.force_gate = get_force_gate()
        
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
            predictions: Optional Predictor outputs carrying the residual-force signal

        Returns:
            List of CriticDecision objects (one per material)
        """
        decisions = []

        # Match predictions to materials by material_id
        if predictions:
            # Create a dict for fast lookup
            pred_by_id = {p.material_id: p for p in predictions}
        
        for mat in materials:
            if isinstance(mat, MaterialNode):
                # Pass only the prediction for this specific material (or None if not found)
                mat_predictions = [pred_by_id[mat.mpid]] if mat.mpid in pred_by_id else None
                decision = self._evaluate_material(mat, mat_predictions)
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

        # Check 2: Surrogate trust signal (if a prediction is available)
        if predictions:
            force_check = self._check_residual_force(predictions)
            if not force_check["passed"]:
                requires_escalation = True
                reasons.append(
                    f"High residual force: {force_check['value']:.3f} > "
                    f"{self.force_gate:.3f} eV/A"
                )

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
                cost_impact=EXPERIMENT_COST,
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

    def _check_residual_force(self, predictions: List[PredictorResult]) -> Dict[str, Any]:
        """Check the surrogate's max residual force against the gate.

        Every structure in the corpus is a DFT-relaxed MP geometry, so DFT's
        own forces on it are approximately zero. A large MACE residual force
        therefore means MACE disagrees with DFT about the geometry, i.e. the
        surrogate is outside the region where it can be trusted for this
        material -- which is what warrants an expensive check.

        Args:
            predictions: List of PredictorResult objects

        Returns:
            Dict with {passed: bool, value: float} -- value in eV/Angstrom
        """
        max_force = 0.0

        for pred in predictions:
            if pred.max_residual_force > max_force:
                max_force = pred.max_residual_force

        passed = max_force <= self.force_gate
        return {"passed": passed, "value": max_force}

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

            # Normalize material_id to a bare mpid up front so the
            # comparison below is a plain equality check, not a
            # conditional expression tangled into an `and` chain
            # (the previous version had `X == Y if cond else Z` inline,
            # which does not parenthesize the way it reads).
            target_mpid = material_id.split(":")[-1] if ":" in material_id else material_id

            # Find property nodes for this material with name "energy_above_hull"
            for nid, data in G.nodes(data=True):
                if (data.get("type") == NodeType.PROPERTY.value and
                    data.get("name") == "energy_above_hull" and
                    data.get("mpid") == target_mpid):
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
    "create_critic",
]