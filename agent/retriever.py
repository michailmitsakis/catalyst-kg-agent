                                                                                """Knowledge Graph Retriever agent.

Queries the KG via `kg/queries.py` to find candidate materials based on:
- Element presence (e.g., "Ni")
- Chemical system (e.g., "Ni-P")
- Property ranges (e.g., "e_above_hull < 0.1")
- Natural language descriptions

Returns typed MaterialNode results with provenance tracking for agent chain verification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent
from pydantic import BaseModel

from kg.schema import (
    KGNode, KGEdge, MaterialNode, ElementNode, ChemsysNode, PropertyNode,
    NodeType, EdgeType, PropertyName, material_id, chemsys_id, element_id,
)
from kg.graph_store import load_graph, rehydrate_node, DEFAULT_KG_JSON
from kg.queries import (
    QueryBuilder, build_query, find_materials_by_element,
    find_materials_in_chemsys, find_stable_materials, find_all_materials,
)
from agent.cost_model import KG_LOOKUP_COST


# ---------------------------------------------------------------------------
# Pydantic-ai Agent: KG Retriever
# ---------------------------------------------------------------------------

class KGLookupResult(BaseModel):
    """Structured result from a KG query."""
    materials: List[MaterialNode]
    chemsys_groups: Optional[List[str]] = None
    elements_found: Optional[List[str]] = None
    provenance: Dict[str, Any]  # Which KG edges/nodes supported this answer
    query_cost: float = KG_LOOKUP_COST


class KGRetrieverAgent:
    """KG Retriever agent with dependency-injected graph access.

    Uses Pydantic-ai for typed I/O. Query results include provenance tracking
    to support Critic agent's plausibility verification.
    """

    def __init__(self, graph_path: Path = None, use_llm: bool = True):
        """Initialize retriever with KG path.

Args:
    graph_path: Path to knowledge graph JSON
    use_llm: If False, skip LLM initialization (for testing)
"""
        self.graph_path = graph_path
        self.G = load_graph(graph_path)
        
        # Pydantic-ai agent setup
        # Pydantic-ai agent setup (optional for testing)
        if use_llm:
            self.agent = Agent(
            model="ollama/llama3.1:8b",  # Default model; override via env
            system_prompt=self._build_system_prompt(),
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt describing KG structure and query patterns."""
        return f"""You are the Knowledge Graph Retriever agent for catalyst discovery.

KG STRUCTURE:
- Nodes: Material (mpid), Element (symbol), Chemsys (element set), Structure (CIF), Property (value)
- Edges: HAS_ELEMENT, IN_CHEMSYS, HAS_STRUCTURE, HAS_PROPERTY
- Query via kg.queries module using typed filters

TASK: Find materials matching user query. Return KGLookupResult with:
- materials: List of MaterialNode objects
- chemsys_groups: Unique chemsys symbols found
- elements_found: All element symbols queried/returned
- provenance: Dict showing which KG nodes/edges matched the query
- query_cost: {KG_LOOKUP_COST} (KG lookup is cheap)

QUERY PATTERNS:
1. Element-based: "Find all Ni-containing materials" -> use find_materials_by_element("Ni")
2. Chemsys-based: "Find Ni-P materials" -> use find_materials_in_chemsys(["Ni", "P"])
3. Property range: "Stable materials (e_above_hull < 0.1)" -> use find_stable_materials(0.1)
4. Natural language: Parse intent, extract filters, combine queries

ALWAYS include provenance showing KG traversal path for Critic verification."""

    def run_query(self, query_string: str) -> KGLookupResult:
        """Run a KG query and return structured results.

        Args:
            query_string: Natural language query describing what to find

        Returns:
            KGLookupResult with materials and provenance
        """
        # Parse query intent
        filters = self._parse_filters(query_string)

        # Execute appropriate query strategy
        materials, chemsys_groups, elements_found, provenance = self._execute_query(filters)

        return KGLookupResult(
            materials=materials,
            chemsys_groups=list(set(chemsys_groups)) if chemsys_groups else None,
            elements_found=list(set(elements_found)) if elements_found else None,
            provenance=provenance,
            query_cost=KG_LOOKUP_COST,
        )

    def _parse_filters(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into filter parameters.

        Args:
            query: User query string

        Returns:
            Dict of extracted filters {element: "Ni", chemsys: ["Ni","P"], ...}
        """
        filters = {}

        # Extract element mentions (e.g., "Ni-containing", "Fe materials")
        elem_pattern = r'\b([A-Z][a-z]{0,1})\b'
        # Match 1-2 letter element symbols at word boundaries
        potential_elements = re.findall(elem_pattern, query, re.IGNORECASE)

        if potential_elements:
            # Validate against periodic table (simple check: 1-2 chars, uppercase first)
            valid_elements = []
            for el in potential_elements:
                el_upper = el.upper()
                if len(el_upper) == 1 or (len(el_upper) == 2 and el_upper[0].isupper()):
                    # Check if it's a real element (basic validation)
                    if el_upper in ["H", "He", "Li", "Be", "B", "C", "N", "O", 
                                     "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", 
                                     "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
                                     "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
                                     "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
                                     "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                                     "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
                                     "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                                     "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
                                     "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
                                     "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra"]:
                        valid_elements.append(el_upper)

            if valid_elements:
                filters["elements"] = valid_elements

        # Extract chemsys patterns (e.g., "Ni-P", "Fe-Co-O")
        chemsys_pattern = r'([A-Z][a-z]{0,1})[-_]([A-Z][a-z]{0,1}(?:[-_][A-Z][a-z]{0,1})*)'
        chemsys_matches = re.findall(chemsys_pattern, query)

        if chemsys_matches:
            filters["chemsys_candidates"] = [self._normalize_chemsys(m) for m in chemsys_matches]

        # Extract property thresholds (e.g., "e_above_hull < 0.1", "band_gap > 2")
        prop_patterns = {
            r'(energy_above_hull|e_above_hull)\s*[<>]=?[\s]*(\d+\.?\d*)': ("property_range", PropertyName.ENERGY_ABOVE_HULL, lambda m: float(m.group(2))),
            r'(band_gap)\s*[<>]=?[\s]*(\d+\.?\d*)': ("property_range", PropertyName.BAND_GAP, lambda m: float(m.group(2))),
        }

        for pattern, (prop_type, name, extractor) in prop_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                # Determine direction (<, >, <=, >=, ==)
                val = extractor(match)
                direction = match.group(0).split('[')[1].strip() if '[' in match.group(0) else None
                filters[prop_type] = {"name": name, "value": val}

        # Check for stability-focused queries ("stable materials", "thermodynamically stable")
        if re.search(r'\bstable\b', query, re.IGNORECASE):
            filters["stability_mode"] = True

        return filters

    def _normalize_chemsys(self, chemsys_str: str) -> List[str]:
        """Normalize chemsys string to sorted list of element symbols."""
        # Handle "Ni-P", "Fe-Co-O", etc.
        parts = re.split(r'[-_]', chemsys_str.upper())
        elements = []
        for part in parts:
            if len(part) == 1:
                elements.append(part)
            elif len(part) == 2 and part[0].isupper() and part[1].islower():
                elements.append(part)
        
        return sorted(elements)

    def _execute_query(self, filters: Dict[str, Any]) -> tuple:
        """Execute query based on parsed filters.

        Args:
            filters: Extracted filter parameters

        Returns:
            Tuple of (materials, chemsys_groups, elements_found, provenance)
        """
        materials = []
        chemsys_groups = []
        elements_found = []
        provenance = {
            "query_string": None,  # Will be set by caller if needed
            "filters_applied": filters,
            "nodes_visited": [],
            "edges_matched": [],
        }

        # Case 1: Element-based query
        if "elements" in filters:
            for elem in filters["elements"]:
                try:
                    mats = find_materials_by_element(self.G, elem)
                    materials.extend(mats)
                    elements_found.append(elem)
                    provenance["edges_matched"].append(f"HAS_ELEMENT:{elem}")
                    
                    # Collect chemsys groups from results
                    for mat in mats:
                        if isinstance(mat, MaterialNode):
                            # Get chemsys from edge traversal
                            chemsys_nodes = list(self.G.neighbors(mat.id))
                            for cnid in chemsys_nodes:
                                node_data = self.G.nodes[cnid]
                                if node_data.get("type") == NodeType.CHEMSYS.value:
                                    chemsys_groups.append(node_data.get("symbols", []))
                                    provenance["nodes_visited"].append(cnid)

                except Exception as e:
                    provenance["error"] = f"Element query for {elem}: {str(e)}"

        # Case 2: Chemsys-based query
        elif "chemsys_candidates" in filters:
            for chemsys in filters["chemsys_candidates"]:
                try:
                    mats = find_materials_in_chemsys(self.G, chemsys)
                    materials.extend(mats)
                    
                    # Collect chemsys groups
                    for mat in mats:
                        if isinstance(mat, MaterialNode):
                            chemsys_nodes = list(self.G.neighbors(mat.id))
                            for cnid in chemsys_nodes:
                                node_data = self.G.nodes[cnid]
                                if node_data.get("type") == NodeType.CHEMSYS.value:
                                    chemsys_groups.append(node_data.get("symbols", []))
                                    provenance["nodes_visited"].append(cnid)

                except Exception as e:
                    provenance["error"] = f"Chemsys query for {chemsys}: {str(e)}"

        # Case 3: Stability-focused query
        elif "stability_mode" in filters or "property_range" in filters:
            if "stability_mode" in filters:
                max_eah = 0.1  # Default stability threshold
                materials = find_stable_materials(self.G, max_eah)
                provenance["threshold"] = max_eah

            elif "property_range" in filters:
                pr = filters["property_range"]
                if hasattr(pr, "name"):
                    name = pr.name
                    min_val = getattr(pr, "min", None)
                    max_val = getattr(pr, "max", None)

                    # Build custom query
                    qb = build_query(self.G, "material:mp-123")  # Dummy start
                    qb.edge_type("HAS_PROPERTY")
                    
                    if name == PropertyName.ENERGY_ABOVE_HULL:
                        materials = find_stable_materials(self.G, max_val or 0.5)
                        provenance["property"] = "energy_above_hull"
                        provenance["range"] = [min_val, max_val]

        # Case 4: Broad query (return all materials)
        else:
            materials = find_all_materials(self.G)
            provenance["mode"] = "all_materials"

        # Extract unique chemsys group representations
        if chemsys_groups:
            chemsys_strs = []
            for cs in chemsys_groups:
                if isinstance(cs, list):
                    chemsys_strs.append("-".join(sorted(cs)))
                else:
                    chemsys_strs.append(str(cs))
            provenance["chemsys_strings"] = list(set(chemsys_strs))

        return materials, chemsys_groups, elements_found, provenance


# ---------------------------------------------------------------------------
# Convenience factory function
# ---------------------------------------------------------------------------

def create_retriever(graph_path: Path = DEFAULT_KG_JSON) -> KGRetrieverAgent:
    """Factory function to create a retriever agent.

    Args:
        graph_path: Path to the knowledge graph JSON file

    Returns:
        Configured KGRetrieverAgent instance
    """
    return KGRetrieverAgent(graph_path=graph_path)


__all__ = [
    "KGRetrieverAgent",
    "KGLookupResult",
    "create_retriever",
]
