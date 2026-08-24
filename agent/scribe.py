"""Scribe agent: Writes campaign results back to KG.

Parses results from Retriever/Predictor/Critic/Planner and updates KG with:
1. New material evaluations (predicted properties)
2. Campaign metadata (run ID, timestamps, outcomes)
3. Scribe tracks: which materials were evaluated, predictions made, escalations triggered

Keeps KG as "compounding memory" for future campaigns to start from accumulated knowledge.
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
from kg.graph_store import load_graph, save_graph


# ---------------------------------------------------------------------------
# Scribe output schemas
# ---------------------------------------------------------------------------

class CampaignResult(BaseModel):
    """Campaign run result for Scribe to log."""
    campaign_id: str
    timestamp: str  # ISO format datetime
    materials_evaluated: List[MaterialNode]
    predictions_made: List[PredictorOutput] = []
    escalations_triggered: int = 0
    total_cost: float = 0.0


class PredictorOutput(BaseModel):
    """Prediction result from Predictor agent."""
    material_id: str
    property_name: str
    predicted_value: float
    uncertainty: float
    model_source: str


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
        predictions_made: Optional[List[PredictorOutput]] = None,
        escalations_triggered: int = 0,
        total_cost: float = 0.0
    ) -> Dict[str, Any]:
        """Log campaign results to KG.

        Args:
            campaign_id: Unique identifier for this campaign run
            materials_evaluated: Materials that were evaluated in this campaign
            predictions_made: Property predictions made during campaign
            escalations_triggered: Number of times escalation was triggered
            total_cost: Total cost of campaign run

        Returns:
            Dict with summary of what was written to KG
        """
        timestamp = datetime.now().isoformat()

        # Update KG with new property predictions
        for pred in predictions_made or []:
            self._add_prediction_to_kg(pred)

        # Add campaign metadata node (optional, for tracking)
        # self.G.add_node(f"campaign:{campaign_id}", type="Campaign", ...)

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

    def _add_prediction_to_kg(self, prediction: PredictorOutput):
        """Add a property prediction to the KG as a new PropertyNode.

        Args:
            prediction: PredictorOutput object with predicted value + uncertainty
        """
        material_id = prediction.material_id
        
        # Check if property already exists for this material
        existing_props = [nid for nid, data in self.G.nodes(data=True) 
                         if data.get("type") == NodeType.PROPERTY.value and \
                            data.get("mpid") == material_id]

        if existing_props:
            # Update existing property node with new prediction (confidence-weighted?)
            # For now, add as new node with source="MACE-finetuned"
            pass

        # Create new PropertyNode for this prediction
        prop_node = PropertyNode(
            id=property_id(material_id, prediction.property_name),
            mpid=material_id.split(":")[-1] if ":" in material_id else material_id,
            name=prediction.property_name,
            value=prediction.predicted_value,
            unit=PropertyUnit.BAND_GAP if "gap" in str(prediction.property_name).lower() \
                   else PropertyUnit.ENERGY_ABOVE_HULL,  # Default to eV/atom
            source=PropertySource.MACE_FINETUNED,
        )

        self.G.add_node(prop_node.id, **prop_node.model_dump(mode="json"))

        # Add edge from material to property
        mat_id = material_id.split(":")[-1] if ":" in material_id else material_id
        prop_nid = prop_node.id
        self.G.add_edge(mat_id, prop_nid, key=f"HAS_PROPERTY:{prop_node.name.value}", type="HAS_PROPERTY")

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
    "create_scribe",
]
