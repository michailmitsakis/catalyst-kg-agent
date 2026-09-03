"""Scribe agent: Writes campaign results back to KG.

Parses results from Retriever/Predictor/Critic/Planner and updates KG with
new material property predictions (materials + properties only -- campaign
metadata such as escalation decisions, final candidate choice, and cost
live in the journal JSON / MLflow, not in the KG).

Keeps KG as "compounding memory" for future campaigns to start from accumulated
knowledge.

IMPORTANT -- property naming: MACE predictions are written under the
"mace_energy_per_atom" property name, NOT "energy_above_hull". The Predictor
does not compute e_above_hull (see agent/predictor.py docstring); writing
its output under the same property name MP's own e_above_hull uses would
silently average MACE energies into MP's ground-truth stability values --
exactly the tier-mixing this project's own rule (never mix UMA/MP energies)
is meant to prevent, just for MACE-vs-MP instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import networkx as nx

from pydantic import BaseModel

from kg.schema import (
    MaterialNode, PropertyNode, NodeType, property_id, PropertyName,
      PropertySource, PropertyUnit, material_id, KGNode
)
from kg.graph_store import load_graph, save_graph, rehydrate_node
from agent.predictor import PredictorResult  # Import actual predictor result type


# Property name used for MACE surrogate output. Not a PropertyName enum
# member (that enum's only members are energy_above_hull and band_gap);
# PropertyName.coerce()/schema's open-vocabulary handling accepts this raw
# string without a schema edit. Kept as a module constant so build_graph.py
# / queries.py can reference the same literal if they need to filter on it.
MACE_ENERGY_PROPERTY_NAME = "mace_energy_per_atom"


# ---------------------------------------------------------------------------
# Scribe output schemas
# ---------------------------------------------------------------------------

class CampaignResult(BaseModel):
    """Campaign run result for Scribe to log."""
    campaign_id: str
    timestamp: str  # ISO format datetime
    materials_evaluated: List[MaterialNode]
    predictions_made: List[PredictorResult] = []  # Changed from PredictorOutput
    escalations_triggered: int = 0
    total_cost: float = 0.0


class PredictorOutput(BaseModel):
    """Prediction result from Predictor agent (legacy format, kept for compatibility).
    
    Note: Now using PredictorResult directly from predictor.py instead.
    """
    material_id: str
    property_name: str
    predicted_value: float
    max_residual_force: float
    model_source: str
    
    @classmethod
    def from_result(cls, result: PredictorResult) -> "PredictorOutput":
        """Convert PredictorResult to PredictorOutput for backward compatibility."""
        return cls(
            material_id=result.material_id,
            property_name=MACE_ENERGY_PROPERTY_NAME,
            predicted_value=result.property_value if result.property_value is not None else 0.0,
            max_residual_force=result.max_residual_force,
            model_source=result.model_used,
        )


# ---------------------------------------------------------------------------
# Scribe Agent
# ---------------------------------------------------------------------------

class ScribeAgent:
    """Writes campaign results to KG for future campaigns."""

    def __init__(self, graph_path: Path = None):
        """Initialize scribe with KG path."""
        self.graph_path = graph_path or Path("data/processed/kg.json")
        self.G = load_graph(self.graph_path)

    def log_campaign_results(
        self,
        campaign_id: str,
        materials_evaluated: List[MaterialNode],
        predictions_made: Optional[List[PredictorResult]] = None,  # Changed from PredictorOutput
        escalations_triggered: int = 0,
        total_cost: float = 0.0
    ) -> Dict[str, Any]:
        """Log campaign results to KG.

        Args:
            campaign_id: Unique identifier for this campaign run
            materials_evaluated: Materials that were evaluated in this campaign
            predictions_made: Property predictions made during campaign (PredictorResult objects)
            escalations_triggered: Number of times escalation was triggered
            total_cost: Total cost of campaign run

        Returns:
            Dict with summary of what was written to KG
        """
        timestamp = datetime.now().isoformat()

        # Update KG with new property predictions
        for pred in predictions_made or []:
            self._add_prediction_to_kg(pred)  # Now takes PredictorResult directly

        # Note: campaign metadata (escalations_triggered, total_cost, final
        # outcome) is intentionally NOT written to the KG. The KG holds
        # materials and properties only; campaign-level bookkeeping lives
        # in the journal JSON / MLflow (see agent/campaign.py).

        # Save updated graph to disk
        save_graph(self.G, self.graph_path)

        return {
            "campaign_id": campaign_id,
            "timestamp": timestamp,
            "materials_logged": len(materials_evaluated),
            "predictions_logged": len(predictions_made or []),
            "escalations_logged": escalations_triggered,
            "graph_updated": True,
        }

    def _add_prediction_to_kg(self, prediction: PredictorResult):
        """Add a MACE energy-per-atom prediction to the KG as a new PropertyNode.

        Written under MACE_ENERGY_PROPERTY_NAME ("mace_energy_per_atom"),
        never under "energy_above_hull" -- see module docstring for why.

        Args:
            prediction: PredictorResult from the MACE predictor, carrying the
                energy per atom and the max residual force
        """
        material_id = prediction.material_id

        # Extract values from PredictorResult
        prop_name = MACE_ENERGY_PROPERTY_NAME
        pred_value = prediction.property_value
        # The surrogate's trust signal, stored as node metadata alongside the
        # value. This is a force (eV/Angstrom), not an uncertainty -- see
        # agent/predictor.py for why the MC-Dropout uncertainty was removed.
        residual_force = prediction.max_residual_force
        
        if pred_value is None or prediction.prediction_failed:
            print(f"Scribe: Skipping failed prediction for {material_id}")
            return

        # Check if property already exists for this material
        mpid = material_id.split(":")[-1] if ":" in material_id else material_id
        existing_props = [nid for nid, data in self.G.nodes(data=True) 
                          if data.get("type") == NodeType.PROPERTY.value and \
                             data.get("mpid") == mpid and
                             data.get("name") == prop_name]

        if existing_props:
            # Update existing property node with new prediction (average the values)
            for prop_nid in existing_props:
                prop_data = self.G.nodes[prop_nid]
                current_value = float(prop_data.get("value", 0))
                new_value = pred_value
                
                # Simple average for now (could be force-weighted later)
                avg_value = (current_value + new_value) / 2
                self.G.nodes[prop_nid]["value"] = avg_value
                self.G.nodes[prop_nid]["source"] = PropertySource.MACE_FINETUNED.value
                
                # Store the residual force as metadata (average with existing)
                current_force = float(prop_data.get("max_residual_force", 0))
                avg_force = (current_force + residual_force) / 2
                self.G.nodes[prop_nid]["max_residual_force"] = round(avg_force, 4)
                
                # Increment. Nodes created before prediction_count was
                # initialised on creation count as 1 prior prediction, so
                # this update makes 2.
                self.G.nodes[prop_nid]["prediction_count"] = (
                    int(self.G.nodes[prop_nid].get("prediction_count", 1)) + 1
                )
                
            print(f"Scribe: Updated existing property {prop_name} for {material_id} "
                  f"(avg residual force={avg_force:.3f} eV/A)")
            return

        # Create new PropertyNode for this prediction. prop_name is a raw
        # string ("mace_energy_per_atom"), not a canonical PropertyName
        # enum member -- PropertyNode's open-vocabulary validator
        # (PropertyName.coerce) accepts it without a schema edit.
        prop_node = PropertyNode(
            id=property_id(material_id, prop_name),
            mpid=mpid,
            name=prop_name,
            value=pred_value,
            unit=PropertyUnit.ENERGY_ABOVE_HULL,  # "eV/atom" -- same unit string, different property name
            source=PropertySource.MACE_FINETUNED,
        )

        self.G.add_node(prop_node.id, **prop_node.model_dump(mode="json"))

        # Add the trust signal as metadata
        self.G.nodes[prop_node.id]["max_residual_force"] = round(residual_force, 4)

        # This IS the first prediction, so the count starts at 1. Leaving it
        # unset here meant the field only appeared on the second write, so a
        # material predicted twice reported prediction_count = 1.
        self.G.nodes[prop_node.id]["prediction_count"] = 1

        # Add edge from material to property
        self.G.add_edge(material_id, prop_node.id, key=f"HAS_PROPERTY:{prop_name}", type="HAS_PROPERTY")
        
        print(f"Scribe: Added new property {prop_name}={pred_value:.4f} eV/atom "
              f"for {material_id} (residual force={residual_force:.3f} eV/A)")

    def get_materials_with_properties(
        self,
        property_name: PropertyName | str,
        min_value: float = 0.0,
        max_value: float = float('inf')
    ) -> List[MaterialNode]:
        """Query KG for materials with properties in range (for filtering).

        Args:
            property_name: Name of property to query
            min_value: Minimum value threshold
            max_value: Maximum value threshold

        Returns:
            List of MaterialNode objects matching criteria
        """
        # Find all property nodes with matching name and value range
        prop_nodes = [nid for nid, data in self.G.nodes(data=True) 
                     if (data.get("type") == NodeType.PROPERTY.value and \
                         data.get("name") == str(property_name)) and \
                        min_value <= float(data.get("value", 0)) <= max_value]

        # Get material IDs from property nodes
        material_ids = set()
        for prop_nid in prop_nodes:
            edges = list(self.G.predecessors(prop_nid))
            for edge in self.G.edges(prop_nid, edges[0]):
                if self.G[edge[0]][edge[1]].get("type") == "HAS_PROPERTY":
                    material_ids.add(edge[0] if edge[0] != prop_nid else edge[1])

        # Rehydrate and return materials
        materials = []
        for mat_id in material_ids:
            try:
                mat_node = rehydrate_node(self.G, mat_id)
                materials.append(mat_node)
            except Exception:
                pass

        return materials


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_scribe(graph_path: Path = None) -> ScribeAgent:
    """Factory function to create a scribe agent.

    Args:
        graph_path: Path to knowledge graph JSON    

    Returns:
        Configured ScribeAgent instance
    """
    return ScribeAgent(graph_path=graph_path)


__all__ = [
    "ScribeAgent",
    "CampaignResult",
    "PredictorOutput",
    "MACE_ENERGY_PROPERTY_NAME",
    "create_scribe",
]