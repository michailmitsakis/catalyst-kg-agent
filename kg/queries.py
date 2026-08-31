"""A typed query language over a built knowledge graph.

This module provides a fluent API for querying the catalyst-kg knowledge graph,
supporting traversal-based queries with filters on nodes and edges. The query
system is designed to work seamlessly with the agent roles (Retriever, Predictor,
Critic) by providing typed, schema-compliant results.

Design notes:
- Uses NetworkX MultiDiGraph for flexible multi-edge support
- Filters are applied during traversal using edge keys and node attributes
- Returns typed pydantic models via rehydrate_node() for agent consumption
- Supports both explicit start nodes and type-scoped queries
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, List, Callable, Union

import networkx as nx
from pydantic import BaseModel


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg.schema import (
    ChemsysNode, CrystalSystem, ElementNode, KGNode, KGEdge, MaterialNode,
    NodeType, PropertyName, PropertyNode, StructureNode,
    chemsys_id, element_id, material_id, property_id, structure_id,
)
from kg.graph_store import load_graph, save_graph, DEFAULT_KG_JSON, rehydrate_node


# ---------------------------------------------------------------------------
# QueryBuilder: Fluent API for constructing typed queries
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Fluent builder for knowledge graph queries.

    Supports traversing from a start node with filters on:
    - Edge types (HAS_ELEMENT, IN_CHEMSYS, HAS_STRUCTURE, HAS_PROPERTY)
    - Node types (MATERIAL, ELEMENT, CHEMSYS, STRUCTURE, PROPERTY)
    - Property ranges (energy_above_hull, band_gap, and any custom property)
    - Element symbols (find all materials containing an element)
    - Chemical systems (find all materials in a chemsys)

    Example:
        >>> qb = build_query(G, "material:mp-123")
        >>> results = qb.edge_type("HAS_ELEMENT").element("Ni").execute()
        # Returns node IDs of all Ni-containing materials reachable from mp-123
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        start_node_id: str,
        direction: str = "out"  # "out" or "in" for traversal direction
    ):
        """Initialize builder with a starting node.

        Args:
            graph: The NetworkX MultiDiGraph to query
            start_node_id: ID of the node to start traversal from
            direction: Traversal direction ("out" = downstream, "in" = upstream)
        """
        self.G = graph
        self.start_node_id = start_node_id

        # Validation
        if not self._is_valid_node(start_node_id):
            raise ValueError(f"Invalid node ID: {start_node_id}")

        # Traversal state
        self.direction = direction
        self.visited: set[str] = set()
        self.traversal_path: List[str] = [start_node_id]

        # Filters (applied during traversal)
        self.filters: dict = {}
        self.edge_types: List[str] = []
        self.node_types: Optional[List[NodeType]] = None
        self.property_name: Optional[PropertyName] = None
        self.property_min: Optional[float] = None
        self.property_max: Optional[float] = None
        self.element_symbols: List[str] = []
        self.chemsys_symbols: Optional[List[str]] = None

        # Callback for custom filtering (for advanced use cases)
        self.custom_filter: Optional[Callable[[KGNode], bool]] = None

    def _is_valid_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.G.nodes()

    def _get_node_type(self, node_id: str) -> Optional[NodeType]:
        """Get the NodeType for a node ID."""
        data = self.G.nodes[node_id]
        return data.get("type")

    def edge_type(self, etype: EdgeType | str) -> "QueryBuilder":
        """Filter traversal to only follow edges of a specific type.

        Args:
            etype: Edge type (use EdgeType enum or string like "HAS_ELEMENT")

        Returns:
            Self for method chaining
        """
        self.edge_types.append(etype if isinstance(etype, str) else etype.value)
        return self

    def edge_types(self, *etypes: EdgeType | str) -> "QueryBuilder":
        """Filter traversal to only follow edges of specific types.

        Args:
            *etypes: One or more edge types to match

        Returns:
            Self for method chaining
        """
        for etype in etypes:
            self.edge_types.append(etype if isinstance(etype, str) else etype.value)
        return self

    def node_type(self, ntype: NodeType | str) -> "QueryBuilder":
        """Filter traversal to only visit nodes of a specific type.

        Args:
            ntype: Node type (use NodeType enum or string)

        Returns:
            Self for method chaining
        """
        if self.node_types is None:
            self.node_types = []
        self.node_types.append(ntype if isinstance(ntype, str) else NodeType(ntype))
        return self

    def node_types(self, *ntypes: NodeType | str) -> "QueryBuilder":
        """Filter traversal to visit nodes of specific types.

        Args:
            *ntypes: One or more node types to match

        Returns:
            Self for method chaining
        """
        if self.node_types is None:
            self.node_types = []
        for ntype in ntypes:
            self.node_types.append(ntype if isinstance(ntype, str) else NodeType(ntype))
        return self

    def element(self, symbol: str) -> "QueryBuilder":
        """Find all materials containing a specific element.

        Args:
            symbol: Element symbol (e.g., "Ni", "Fe")

        Returns:
            Self for method chaining
        """
        if symbol not in self.element_symbols:
            self.element_symbols.append(symbol)
        return self

    def elements(self, *symbols: str) -> "QueryBuilder":
        """Find all materials containing any of the specified elements.

        Args:
            *symbols: Element symbols to match (OR logic)

        Returns:
            Self for method chaining
        """
        for sym in symbols:
            if sym not in self.element_symbols:
                self.element_symbols.append(sym)
        return self

    def chemsys(self, symbols: List[str]) -> "QueryBuilder":
        """Find all materials in a chemical system.

        Args:
            symbols: List of element symbols defining the chemsys (e.g., ["Ni", "P"])

        Returns:
            Self for method chaining
        """
        self.chemsys_symbols = sorted(symbols)
        return self

    def property_range(
        self,
        name: PropertyName | str,
        min_val: float,
        max_val: float
    ) -> "QueryBuilder":
        """Filter for properties within a value range.

        Args:
            name: Property name (use PropertyName enum or string)
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Self for method chaining
        """
        self.property_name = name if isinstance(name, PropertyName) else PropertyName.coerce(name)
        self.property_min = min_val
        self.property_max = max_val
        return self

    def property_value(
        self,
        name: PropertyName | str,
        value: float,
        tolerance: float = 0.0
    ) -> "QueryBuilder":
        """Filter for properties with a specific value (± tolerance).

        Args:
            name: Property name
            value: Target value
            tolerance: Acceptable deviation from target

        Returns:
            Self for method chaining
        """
        return self.property_range(name, value - tolerance, value + tolerance)

    def material_with_formula(self, formula: str) -> "QueryBuilder":
        """Find materials with a specific pretty formula.

        Args:
            formula: Pretty formula string (e.g., "LiFePO4", "Ni2P")

        Returns:
            Self for method chaining
        """
        self.formula_filter = formula
        return self

    def crystal_system(self, cs: CrystalSystem | str) -> "QueryBuilder":
        """Filter for materials with a specific crystal system.

        Args:
            cs: Crystal system (use CrystalSystem enum or string)

        Returns:
            Self for method chaining
        """
        self.crystal_system_filter = cs if isinstance(cs, CrystalSystem) else CrystalSystem(cs)
        return self

    def space_group(self, sg_num: int | str) -> "QueryBuilder":
        """Filter for materials with a specific space group number.

        Args:
            sg_num: Space group number (1-230)

        Returns:
            Self for method chaining
        """
        self.space_group_filter = int(sg_num) if isinstance(sg_num, str) else sg_num
        return self

    def custom_filter(self, func: Callable[[KGNode], bool]) -> "QueryBuilder":
        """Apply a custom filter function to nodes during traversal.

        Args:
            func: Function that takes a KGNode and returns True to include

        Returns:
            Self for method chaining
        """
        self.custom_filter = func
        return self

    def _traverse(self, current_id: str) -> List[str]:
        """Perform graph traversal from a node with applied filters.

        Args:
            current_id: Current node to traverse from/to

        Returns:
            List of matched node IDs
        """
        # Skip if already visited
        if current_id in self.visited:
            return []

        # Get node data for filtering
        node_data = self.G.nodes[current_id]
        node_type = node_data.get("type")
        
        # Node type filter - but allow start node to pass even if it doesn't match
        # (we query FROM an element TO materials, so element won't be a Material)
        is_start_node = len(self.visited) == 0
        
        if self.node_types is not None and node_type not in self.node_types and not is_start_node:
            return []

        # Custom filter
        if self.custom_filter is not None:
            try:
                if not self.custom_filter(rehydrate_node(self.G, current_id)):
                    return []
            except Exception:
                pass  # Skip nodes that fail custom filter

        # Add to results and visited
        results = [current_id]
        self.visited.add(current_id)

        # Determine neighbors based on direction
        if self.direction == "out":
            neighbors = list(self.G.successors(current_id))
        else:  # "in"
            neighbors = list(self.G.predecessors(current_id))

        for neighbor in neighbors:
            
            # Edge type filter - handle MultiDiGraph with custom string keys
            edge_type = None
            try:
                # Try simple lookup first (for single-key edges)
                edge_data = self.G.edges[current_id, neighbor]
                if isinstance(edge_data, dict):
                    edge_type = edge_data.get("type")
                elif isinstance(edge_data, str):
                    edge_type = edge_data
            except (KeyError, TypeError, ValueError):
                # For MultiDiGraph with custom keys, iterate over all edges manually
                for u, v, key, data in self.G.edges(keys=True, data=True):
                    if ((u == current_id and v == neighbor) or (v == current_id and u == neighbor)):
                        if isinstance(data, dict):
                            edge_type = data.get("type")
                        elif isinstance(data, str):
                            edge_type = data
                        break
            
            if self.edge_types and edge_type is not None and edge_type not in self.edge_types:
                continue

            # Property filter (only applies to PROPERTY nodes)
            if neighbor in self.visited:
                continue
                
            neighbor_data = self.G.nodes[neighbor]
            if neighbor_data.get("type") == NodeType.PROPERTY.value:
                prop_name = neighbor_data.get("name")
                prop_value = neighbor_data.get("value")

                if self.property_name and prop_name == self.property_name.value:
                    if prop_value is not None:
                        if self.property_min is not None and prop_value < self.property_min:
                            continue
                        if self.property_max is not None and prop_value > self.property_max:
                            continue

            # Element filter
            if self.element_symbols:
                element_nodes = [nid for nid in neighbors 
                                if self.G.nodes[nid].get("type") == NodeType.ELEMENT.value]
                if element_nodes:
                    neighbor_elements = [self.G.nodes[eln].get("symbol") for eln in element_nodes]
                    matching = any(sym in self.element_symbols for sym in neighbor_elements)
                    if not matching:
                        continue

            # Chemsys filter
            if self.chemsys_symbols:
                chemsys_nodes = [nid for nid in neighbors 
                                if self.G.nodes[nid].get("type") == NodeType.CHEMSYS.value]
                if chemsys_nodes:
                    neighbor_chemsys = self.G.nodes[chemsys_nodes[0]].get("symbols", [])
                    expected_chemsys = sorted(self.chemsys_symbols)
                    if sorted(neighbor_chemsys) != expected_chemsys:
                        continue

            # Crystal system filter
            if hasattr(self, 'crystal_system_filter'):
                structure_nodes = [nid for nid in neighbors 
                                  if self.G.nodes[nid].get("type") == NodeType.STRUCTURE.value]
                if structure_nodes:
                    cs_val = self.G.nodes[structure_nodes[0]].get("crystal_system")
                    if cs_val and cs_val != str(self.crystal_system_filter):
                        continue

            # Space group filter
            if hasattr(self, 'space_group_filter'):
                structure_nodes = [nid for nid in neighbors 
                                  if self.G.nodes[nid].get("type") == NodeType.STRUCTURE.value]
                if structure_nodes:
                    sg_val = self.G.nodes[structure_nodes[0]].get("space_group_number")
                    if sg_val is not None and sg_val != self.space_group_filter:
                        continue

            # Recursively traverse neighbors of this neighbor (for multi-hop queries)
            sub_results = self._traverse(neighbor)
            results.extend(sub_results)

        return results

    def execute(self) -> List[str]:
        """Execute the query and return matching node IDs.

        Returns:
            List of node IDs that match all applied filters
        """
        # Start from the initial node (or all nodes of a type if no start specified)
        if self.start_node_id in self.G.nodes():
            results = self._traverse(self.start_node_id)
        else:
            raise ValueError(f"Start node not found: {self.start_node_id}")

        return results

    def execute_typed(self) -> List[KGNode]:
        """Execute query and return typed pydantic models.

        Returns:
            List of KGNode models matching the query
        """
        node_ids = self.execute()
        return [rehydrate_node(self.G, nid) for nid in node_ids]

    def execute_materials_only(self) -> List[MaterialNode]:
        """Execute query and return only Material nodes.

        Returns:
            List of MaterialNode models matching the query
        """
        node_ids = self.execute()
        materials = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.MATERIAL.value:
                materials.append(rehydrate_node(self.G, nid))
        return materials

    def execute_structures_only(self) -> List[StructureNode]:
        """Execute query and return only Structure nodes.

        Returns:
            List of StructureNode models matching the query
        """
        node_ids = self.execute()
        structures = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.STRUCTURE.value:
                structures.append(rehydrate_node(self.G, nid))
        return structures

    def execute_properties_only(self) -> List[PropertyNode]:
        """Execute query and return only Property nodes.

        Returns:
            List of PropertyNode models matching the query
        """
        node_ids = self.execute()
        properties = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.PROPERTY.value:
                properties.append(rehydrate_node(self.G, nid))
        return properties

    def execute_elements_only(self) -> List[ElementNode]:
        """Execute query and return only Element nodes.

        Returns:
            List of ElementNode models matching the query
        """
        node_ids = self.execute()
        elements = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.ELEMENT.value:
                elements.append(rehydrate_node(self.G, nid))
        return elements

    def execute_chemsys_only(self) -> List[ChemsysNode]:
        """Execute query and return only Chemsys nodes.

        Returns:
            List of ChemsysNode models matching the query
        """
        node_ids = self.execute()
        chemsys_list = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.CHEMSYS.value:
                chemsys_list.append(rehydrate_node(self.G, nid))
        return chemsys_list

    def execute_structures_only(self) -> List[StructureNode]:
        """Execute query and return only Structure nodes.

        Returns:
            List of StructureNode models matching the query
        """
        node_ids = self.execute()
        structures = []
        for nid in node_ids:
            node_data = self.G.nodes[nid]
            if node_data.get("type") == NodeType.STRUCTURE.value:
                structures.append(rehydrate_node(self.G, nid))
        return structures


# ---------------------------------------------------------------------------
# Convenience query functions (high-level APIs)
# ---------------------------------------------------------------------------

def find_materials_by_element(graph: nx.MultiDiGraph, element_symbol: str) -> List[MaterialNode]:
    """Find all materials containing a specific element.

    Args:
        graph: The knowledge graph
        element_symbol: Element symbol (e.g., "Ni", "Fe")

    Returns:
        List of MaterialNode models for all materials containing the element
    """
    # Find the element node
    element_nodes = [nid for nid, data in graph.nodes(data=True) 
                     if data.get("type") == NodeType.ELEMENT.value and \
                        data.get("symbol") == element_symbol]

    if not element_nodes:
        return []

    # Query materials connected to this element (materials point TO elements, so use direction="in")
    qb = QueryBuilder(graph, element_nodes[0], direction="in")
    qb.edge_type("HAS_ELEMENT")
    qb.node_type(NodeType.MATERIAL)

    return qb.execute_materials_only()


def find_materials_in_chemsys(
    graph: nx.MultiDiGraph, 
    chemsys_name: str  # e.g., "Ni-P" or "Co-Fe-O"
) -> List[MaterialNode]:
    """Find all materials in a chemical system.

    Args:
        graph: The knowledge graph
        chemsys_name: Chemical system name (e.g., "Ni-P", "Co-Fe-O")

    Returns:
        List of MaterialNode models for materials in the chemsys
    """
    # Find the chemsys node by ID (format: "chemsys:<name>")
    chemsys_id = f"chemsys:{chemsys_name}"
    chemsys_nodes = [nid for nid, data in graph.nodes(data=True) 
                     if nid == chemsys_id]

    if not chemsys_nodes:
        return []

    # Query materials connected to this chemsys (materials point TO chemsys, so use direction="in")
    qb = QueryBuilder(graph, chemsys_nodes[0], direction="in")
    qb.edge_type("IN_CHEMSYS")
    qb.node_type(NodeType.MATERIAL)

    return qb.execute_materials_only()


def find_stable_materials(
    graph: nx.MultiDiGraph, 
    max_e_above_hull: float,
    source_filter: Optional[PropertySource] = None
) -> List[MaterialNode]:
    """Find materials with e_above_hull below a threshold (stability filter).

    Args:
        graph: The knowledge graph
        max_e_above_hull: Maximum allowed energy above hull (eV/atom)
        source_filter: Optional filter for property source (e.g., only MaterialsProject data)

    Returns:
        List of MaterialNode models for stable materials
    """
    # Find all material nodes
    mat_nodes = [nid for nid, data in graph.nodes(data=True) 
                 if data.get("type") == NodeType.MATERIAL.value]

    stable_materials = []
    for mat_nid in mat_nodes:
        # Find e_above_hull property for this material
        mat_id = mat_nid.split(":")[-1] if ":" in mat_nid else mat_nid
        
        prop_nodes = [nid for nid, data in graph.nodes(data=True) 
                      if (data.get("type") == NodeType.PROPERTY.value and \
                          data.get("name") == PropertyName.ENERGY_ABOVE_HULL.value and \
                          data.get("mpid") == mat_id)]
        
        for prop_nid in prop_nodes:
            prop_data = graph.nodes[prop_nid]
            value = prop_data.get("value")
            
            # Check source filter if specified
            if source_filter is not None and prop_data.get("source") != str(source_filter):
                continue
            
            # Check stability threshold
            if value is not None and value <= max_e_above_hull:
                stable_materials.append(rehydrate_node(graph, mat_nid))
                break  # Only add each material once

    return stable_materials


def find_materials_by_property_range(
    graph: nx.MultiDiGraph,
    property_name: PropertyName | str,
    min_val: float,
    max_val: float
) -> List[MaterialNode]:
    """Find materials with a property in a given range.

    Args:
        graph: The knowledge graph
        property_name: Name of the property to query
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        List of MaterialNode models for materials with properties in range
    """
    qb = QueryBuilder(graph, "material:mp-123")  # Dummy start
    qb.edge_type("HAS_PROPERTY")
    qb.property_range(property_name, min_val, max_val)
    qb.node_type(NodeType.MATERIAL)

    return qb.execute_materials_only()


def find_all_materials(graph: nx.MultiDiGraph) -> List[MaterialNode]:
    """Find all materials in the graph.

    Args:
        graph: The knowledge graph

    Returns:
        List of all MaterialNode models
    """
    mat_nodes = [nid for nid, data in graph.nodes(data=True) 
                 if data.get("type") == NodeType.MATERIAL.value]
    return [rehydrate_node(graph, nid) for nid in mat_nodes]


def find_materials_by_formula(graph: nx.MultiDiGraph, formula: str) -> List[MaterialNode]:
    """Find materials with a specific pretty formula.

    Args:
        graph: The knowledge graph
        formula: Pretty formula string (e.g., "LiFePO4")

    Returns:
        List of MaterialNode models matching the formula
    """
    mat_nodes = [nid for nid, data in graph.nodes(data=True) 
                 if data.get("type") == NodeType.MATERIAL.value and \
                    data.get("formula_pretty") == formula]
    return [rehydrate_node(graph, nid) for nid in mat_nodes]


def find_all_elements(graph: nx.MultiDiGraph) -> List[ElementNode]:
    """Find all elements in the graph.

    Args:
        graph: The knowledge graph

    Returns:
        List of all ElementNode models
    """
    elem_nodes = [nid for nid, data in graph.nodes(data=True) 
                  if data.get("type") == NodeType.ELEMENT.value]
    return [rehydrate_node(graph, nid) for nid in elem_nodes]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_query(
    graph: nx.MultiDiGraph,
    node_id: str,
    direction: str = "out"
) -> QueryBuilder:
    """Build a typed query object from a specific node.

    Args:
        graph: The knowledge graph
        node_id: ID of the starting node
        direction: Traversal direction ("out" or "in")

    Returns:
        QueryBuilder instance ready for filtering and execution
    """
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a NetworkX MultiDiGraph")
    
    if node_id not in graph.nodes():
        raise ValueError(f"Node ID not found: {node_id}")

    return QueryBuilder(graph=graph, start_node_id=node_id, direction=direction)


def get_typed_query(graph: nx.MultiDiGraph) -> QueryBuilder:
    """Entry point for building typed queries.

    This is a convenience function that creates a QueryBuilder with
    default settings. For more control, use build_query() directly.

    Args:
        graph: The knowledge graph

    Returns:
        QueryBuilder ready for configuration
    """
    if not graph.nodes():
        raise ValueError("Graph must contain at least one node before querying.")

    # Default to starting from the first material node
    mat_nodes = [nid for nid, data in graph.nodes(data=True) 
                 if data.get("type") == NodeType.MATERIAL.value]

    if not mat_nodes:
        raise ValueError("No Material nodes found in graph")

    return QueryBuilder(graph=graph, start_node_id=mat_nodes[0], direction="out")


__all__ = [
    # Classes
    "QueryBuilder",
    # Functions
    "build_query", 
    "get_typed_query",
    # Convenience functions
    "find_materials_by_element",
    "find_materials_in_chemsys",
    "find_stable_materials",
    "find_materials_by_property_range",
    "find_all_materials",
    "find_materials_by_formula",
    "find_all_elements",
]