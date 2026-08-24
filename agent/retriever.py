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
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
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

from dotenv import load_dotenv
load_dotenv()

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


class AgentResponse(BaseModel):
    """Response from the LLM agent after parsing and querying."""
    materials: List[MaterialNode]  # MaterialNode objects returned by tools
    chemsys_groups: Optional[List[str]] = None
    elements_found: Optional[List[str]] = None
    provenance: Dict[str, Any] = {}
    query_cost: float = KG_LOOKUP_COST


class KGRetrieverAgent:
    """KG Retriever agent with dependency-injected graph access.

    Uses Pydantic-ai for typed I/O. Query results include provenance tracking
    """

    def __init__(self, graph_path: Path = None, use_llm: bool = True):
        """Initialize retriever with KG path.

Args:
    graph_path: Path to knowledge graph JSON
    use_llm: If False, skip LLM initialization (for testing)
"""
        self.graph_path = graph_path
        self.G = load_graph(graph_path)
        
        # Determine if we should use LLM for parsing
        self.use_llm_parsing = use_llm
        
        # Pydantic-ai agent setup (optional for testing)
        if use_llm:
            ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:latest")
            
            # Normalize base URL to include /v1 suffix for OpenAI-compatible API
            base_url = ollama_base.rstrip("/") + "/v1" if not ollama_base.endswith("/v1") else ollama_base
            
            # Create Ollama model with custom provider
            ollama_model_obj = OllamaModel(
                model_name=ollama_model,
                provider=OllamaProvider(base_url=base_url)
            )
            
            # Define tools the agent can use for KG queries after LLM parsing
            def execute_element_query(elements: List[str]) -> List[MaterialNode]:
                """Execute element-based query."""
                all_mats = []
                for elem in elements:
                    mats = find_materials_by_element(self.G, elem)
                    all_mats.extend(mats)
                return all_mats

            def execute_chemsys_query(chemsys_list: List[str]) -> List[MaterialNode]:
                """Execute chemsys-based query."""
                all_mats = []
                for cs in chemsys_list:
                    mats = find_materials_in_chemsys(self.G, cs)
                    all_mats.extend(mats)
                return all_mats

            def execute_stability_query(threshold: float) -> List[MaterialNode]:
                """Execute stability-based query."""
                mats = find_stable_materials(self.G, threshold)
                return mats

            def execute_broad_query() -> List[MaterialNode]:
                """Return all materials."""
                mats = find_all_materials(self.G)
                return mats

            self.agent = Agent(
                model=ollama_model_obj,
                system_prompt=self._build_system_prompt(),
                tools=[
                    execute_element_query,
                    execute_chemsys_query,
                    execute_stability_query,
                    execute_broad_query,
                ],
                result_type=AgentResponse  # Agent should return structured response
            )

    def _build_system_prompt(self) -> str:
        """Build system prompt describing KG structure and query patterns."""
        return f"""You are the Knowledge Graph Retriever agent for catalyst discovery.

KG STRUCTURE:
- Nodes: Material (mpid), Element (symbol), Chemsys (element set), Structure (CIF), Property (value)
- Edges: HAS_ELEMENT, IN_CHEMSYS, HAS_STRUCTURE, HAS_PROPERTY
- Query via kg.queries module using typed filters

AVAILABLE TOOLS (call the appropriate one based on query intent):
1. execute_element_query(elements=["Ni", "Fe"]) - Find materials containing specific elements
   Returns: List of MaterialNode objects for each element
2. execute_chemsys_query(chemsys=["Ni-P", "Fe-Co-O"]) - Find materials in chemical systems
   Returns: List of MaterialNode objects for each chemsys
3. execute_stability_query(threshold=0.1) - Find stable materials with e_above_hull < threshold
   Returns: List of MaterialNode objects that are stable
4. execute_broad_query() - Return all materials in the KG
   Returns: List of all MaterialNode objects

TASK: 
1. Analyze the user's natural language query to determine intent
2. Identify which tool(s) to call based on query patterns
3. Extract filter parameters (elements, chemsys names, thresholds) from the query
4. Call the appropriate tool with extracted parameters
5. Aggregate results and return AgentResponse with:
   - materials: List of MaterialNode dicts (mpid, elements, chemsys fields)
   - chemsys_groups: Unique chemsys symbols found (e.g., "Ni-P", "Fe-Co-O")  
   - elements_found: All element symbols queried/returned
   - provenance: Dict showing which KG nodes/edges were traversed

QUERY PATTERNS:
1. Element-based: "Find all Ni-containing materials" -> execute_element_query(["Ni"])
2. Chemsys-based: "Find Ni-P materials" -> execute_chemsys_query(["Ni-P"])
3. Property range: "Stable materials (e_above_hull < 0.1)" -> execute_stability_query(0.1)
4. Broad queries: "Show me all materials" -> execute_broad_query()

Return results with provenance showing KG traversal path for Critic verification."""

    def run_query(self, query_string: str) -> KGLookupResult:
        """Run a KG query and return structured results.

        Args:
            query_string: Natural language query describing what to find

        Returns:
            KGLookupResult with materials and provenance
        """
        # Parse query intent - use LLM if enabled, otherwise manual parsing
        if self.use_llm_parsing:
            # Use the Pydantic-ai agent which will:
            # 1. Parse the natural language query using LLM
            # 2. Call appropriate tools based on parsed intent
            # 3. Return AgentResponse with MaterialNode objects
            
            try:
                agent_result = self.agent.run_query(
                    query_string,
                    return_type=AgentResponse
                )
                
                # agent_result.materials is already a list of MaterialNode objects
                # We just need to aggregate chemsys_groups and elements_found from the results
                all_materials = list(agent_result.materials)
                
                # Extract unique chemsys groups and elements from materials
                chemsys_sets = set()
                element_sets = set()
                for mat in all_materials:
                    if hasattr(mat, 'chemsys'):
                        chemsys_sets.add(str(mat.chemsys))
                    if hasattr(mat, 'elements'):
                        element_sets.update(str(mat.elements))
                
                return KGLookupResult(
                    materials=all_materials,
                    chemsys_groups=list(chemsys_sets) if chemsys_sets else None,
                    elements_found=list(element_sets) if element_sets else None,
                    provenance=agent_result.provenance,
                    query_cost=KG_LOOKUP_COST,
                )
            except Exception as e:
                print(f"[WARN] Agent query failed: {e}, falling back to manual parsing")
                # Fall back to manual parsing on error
                filters = self._parse_filters_manual(query_string)
                materials, chemsys_groups, elements_found, provenance = self._execute_query(filters)
        else:
            # Manual parsing fallback
            filters = self._parse_filters_manual(query_string)
            materials, chemsys_groups, elements_found, provenance = self._execute_query(filters)

        return KGLookupResult(
            materials=materials,
            chemsys_groups=["-".join(sorted(cs)) for cs in chemsys_groups] if chemsys_groups else None,
            elements_found=list(set(elements_found)) if elements_found else None,
            provenance=provenance,
            query_cost=KG_LOOKUP_COST,
        )

    def _parse_filters_with_llm(self, query: str) -> Dict[str, Any]:
        """Parse natural language query using Pydantic-ai Agent.

        Args:
            query: User query string

        Returns:
            Dict of extracted filters
        """
        # Use the Pydantic-ai agent to parse the query
        try:
            result = self.agent.run_query(
                query,
                return_type=Dict[str, Any],
                default_tool_response=self._default_llm_response
            )
            
            print(f"[LLM] Parsed filters: {result}")
            return result
            
        except Exception as e:
            print(f"[WARN] LLM parsing failed: {e}, falling back to manual parsing")
            return self._parse_filters_manual(query)

    def _default_llm_response(self, tool_call: Any, default: Dict[str, Any]) -> Any:
        """Default response for KG lookup tools when no matching query found."""
        # This is called by Pydantic-ai when a tool needs to be invoked
        # Return the default result (manual parsing fallback)
        return default

    def _parse_filters_manual(self, query: str) -> Dict[str, Any]:
        """Parse natural language query manually (fallback).

        Args:
            query: User query string

        Returns:
            Dict of extracted filters
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
            filters["chemsys_candidates"] = [self._normalize_chemsys_match(m) for m in chemsys_matches]

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

    def _normalize_chemsys_match(self, match_tuple) -> List[str]:
        """Normalize a regex match tuple to sorted list of element symbols."""
        # Handle "Ni-P", "Fe-Co-O" from regex match tuples
        chemsys_str = match_tuple[0]  # The matched string
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
