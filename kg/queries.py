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

Naming note: the fluent filter methods are all singular (`edge_type`,
`node_type`, `element`, ...) and accept one or more values. Earlier
versions also defined plural variants (`edge_types`, `node_types`) whose
names collided with the instance attributes assigned in `__init__` --
the attribute always won, so calling them raised
`TypeError: 'list' object is not callable`. The plural methods are gone;
the singular ones are variadic instead.
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
    ChemsysNode, CrystalSystem, EdgeType, ElementNode, KGNode, KGEdge, MaterialNode,
    NodeType, PropertyName, PropertyNode, PropertySource, StructureNode,
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
    - Property ranges (energy_above_hull, formation_energy_per_atom, band_gap, ...)
    - Element symbols (find all materials containing an element)
    - Chemical systems (find all materials in a chemsys)

    Example:
        >>> qb = build_query(G, "element:Ni", direction="in")
        >>> results = qb.edge_type("HAS_ELEMENT").node_type(NodeType.MATERIAL).execute()
        # Returns node IDs of all Ni-containing materials
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

        # Filters (applied during traversal). These attribute names are
        # deliberately distinct from every method name on this class --
        # an attribute and a method sharing a name silently shadows the
        # method (see module docstring).
        self.edge_type_filter: List[str] = []
        self.node_type_filter: List[str] = []
        self.property_name: Optional[PropertyName] = None
        self.property_min: Optional[float] = None
        self.property_max: Optional[float] = None
        self.element_symbols: List[str] = []
        self.chemsys_symbols: Optional[List[str]] = None
        self.formula_filter: Optional[str] = None
        self.crystal_system_filter: Optional[CrystalSystem] = None
        self.space_group_filter: Optional[int] = None

        # Callback for custom filtering (for advanced use cases)
        self.custom_filter_fn: Optional[Callable[[KGNode], bool]] = None

    def _is_valid_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.G.nodes()

    def _get_node_type(self, node_id: str) -> Optional[str]:
        """Get the serialized `type` attribute for a node ID."""
        return self.G.nodes[node_id].get("type")

    @staticmethod
    def _as_edge_type_str(etype: EdgeType | str) -> str:
        return etype if isinstance(etype, str) else etype.value

    @staticmethod
    def _as_node_type_str(ntype: NodeType | str) -> str:
        """Normalize to the serialized string form stored on nodes.

        Node `type` attributes are serialized strings ("Material"), so
        filters are compared as strings rather than enum members.
        """
        if isinstance(ntype, NodeType):
            return ntype.value
        return NodeType(ntype).value

    def edge_type(self, *etypes: EdgeType | str) -> "QueryBuilder":
        """Only follow edges of the given type(s).

        Args:
            *etypes: One or more edge types (EdgeType members or strings
                like "HAS_ELEMENT")

        Returns:
            Self for method chaining
        """
        for etype in etypes:
            value = self._as_edge_type_str(etype)
            if value not in self.edge_type_filter:
                self.edge_type_filter.append(value)
        return self

    def node_type(self, *ntypes: NodeType | str) -> "QueryBuilder":
        """Only visit nodes of the given type(s).

        Args:
            *ntypes: One or more node types (NodeType members or strings)

        Returns:
            Self for method chaining
        """
        for ntype in ntypes:
            value = self._as_node_type_str(ntype)
            if value not in self.node_type_filter:
                self.node_type_filter.append(value)
        return self

    def element(self, *symbols: str) -> "QueryBuilder":
        """Restrict to materials containing the given element(s) (OR logic).

        Args:
            *symbols: Element symbols (e.g., "Ni", "Fe")

        Returns:
            Self for method chaining
        """
        for sym in symbols:
            if sym not in self.element_symbols:
                self.element_symbols.append(sym)
        return self

    def chemsys(self, symbols: List[str]) -> "QueryBuilder":
        """Restrict to materials in a chemical system.

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
        """Restrict to materials with a specific pretty formula.

        Args:
            formula: Pretty formula string (e.g., "LiFePO4", "Ni2P")

        Returns:
            Self for method chaining
        """
        self.formula_filter = formula
        return self

    def crystal_system(self, cs: CrystalSystem | str) -> "QueryBuilder":
        """Restrict to structures with a specific crystal system.

        Args:
            cs: Crystal system (use CrystalSystem enum or string)

        Returns:
            Self for method chaining
        """
        self.crystal_system_filter = cs if isinstance(cs, CrystalSystem) else CrystalSystem(cs)
        return self

    def space_group(self, sg_num: int | str) -> "QueryBuilder":
        """Restrict to structures with a specific space group number.

        Args:
            sg_num: Space group number (1-230)

        Returns:
            Self for method chaining
        """
        self.space_group_filter = int(sg_num)
        return self

    def custom(self, func: Callable[[KGNode], bool]) -> "QueryBuilder":
        """Apply a custom predicate to nodes during traversal.

        Named `custom` rather than `custom_filter` so it cannot shadow
        the `custom_filter_fn` attribute.

        Args:
            func: Function that takes a KGNode and returns True to include

        Returns:
            Self for method chaining
        """
        self.custom_filter_fn = func
        return self

    # -----------------------------------------------------------------------
    # Traversal internals
    # -----------------------------------------------------------------------

    def _edge_types_between(self, u: str, v: str) -> List[str]:
        """All edge `type` values on edges between u and v (multi-edge safe)."""
        types: List[str] = []
        if self.G.has_edge(u, v):
            for _key, data in self.G[u][v].items():
                if isinstance(data, dict):
                    t = data.get("type")
                    if t is not None:
                        types.append(t)
        return types

    def _passes_node_filters(self, node_id: str) -> bool:
        """Check a candidate node against every node-scoped filter.

        Each filter is evaluated against THIS node, not against the
        neighbourhood of whatever node we arrived from (the previous
        implementation inspected the current node's sibling list, which
        did not match the documented semantics).
        """
        data = self.G.nodes[node_id]
        ntype = data.get("type")

        if self.node_type_filter and ntype not in self.node_type_filter:
            return False

        if ntype == NodeType.MATERIAL.value:
            if self.formula_filter is not None and data.get("formula_pretty") != self.formula_filter:
                return False
            if self.element_symbols:
                mat_elements = data.get("elements") or []
                if not any(sym in mat_elements for sym in self.element_symbols):
                    return False

        if ntype == NodeType.CHEMSYS.value and self.chemsys_symbols:
            if sorted(data.get("symbols") or []) != sorted(self.chemsys_symbols):
                return False

        if ntype == NodeType.STRUCTURE.value:
            if self.crystal_system_filter is not None:
                cs_val = data.get("crystal_system")
                expected = (
                    self.crystal_system_filter.value
                    if isinstance(self.crystal_system_filter, CrystalSystem)
                    else str(self.crystal_system_filter)
                )
                if cs_val != expected:
                    return False
            if self.space_group_filter is not None:
                if data.get("space_group_number") != self.space_group_filter:
                    return False

        if ntype == NodeType.PROPERTY.value and self.property_name is not None:
            if data.get("name") != self.property_name.value:
                return False
            value = data.get("value")
            if value is None:
                return False
            if self.property_min is not None and value < self.property_min:
                return False
            if self.property_max is not None and value > self.property_max:
                return False

        if self.custom_filter_fn is not None:
            try:
                if not self.custom_filter_fn(rehydrate_node(self.G, node_id)):
                    return False
            except Exception:
                return False

        return True

    def _traverse(self, current_id: str, is_start: bool = False) -> List[str]:
        """Depth-first traversal from a node with applied filters.

        Args:
            current_id: Node to visit
            is_start: True for the initial node, which is always traversed
                through even when it does not itself match the filters
                (queries typically start FROM an Element and look FOR
                Materials, so the start node's type rarely matches).

        Returns:
            List of matched node IDs
        """
        if current_id in self.visited:
            return []
        self.visited.add(current_id)

        results: List[str] = []
        if is_start or self._passes_node_filters(current_id):
            results.append(current_id)

        if self.direction == "out":
            neighbors = list(self.G.successors(current_id))
        else:  # "in"
            neighbors = list(self.G.predecessors(current_id))

        for neighbor in neighbors:
            if neighbor in self.visited:
                continue

            # Edge type filter: check edges in the traversal direction.
            if self.edge_type_filter:
                if self.direction == "out":
                    edge_types = self._edge_types_between(current_id, neighbor)
                else:
                    edge_types = self._edge_types_between(neighbor, current_id)
                if not any(et in self.edge_type_filter for et in edge_types):
                    continue

            results.extend(self._traverse(neighbor))

        return results

    def execute(self) -> List[str]:
        """Execute the query and return matching node IDs.

        Returns:
            List of node IDs that match all applied filters
        """
        if self.start_node_id not in self.G.nodes():
            raise ValueError(f"Start node not found: {self.start_node_id}")

        self.visited = set()
        results = self._traverse(self.start_node_id, is_start=True)

        # The start node is included unconditionally to seed traversal;
        # drop it from the result set unless it genuinely matches.
        if results and results[0] == self.start_node_id:
            if not self._passes_node_filters(self.start_node_id):
                results = results[1:]
        return results

    def execute_typed(self) -> List[KGNode]:
        """Execute query and return typed pydantic models."""
        return [rehydrate_node(self.G, nid) for nid in self.execute()]

    def _execute_of_type(self, node_type: NodeType) -> List[KGNode]:
        """Execute and keep only nodes of one serialized type."""
        out: List[KGNode] = []
        for nid in self.execute():
            if self.G.nodes[nid].get("type") == node_type.value:
                out.append(rehydrate_node(self.G, nid))
        return out

    def execute_materials_only(self) -> List[MaterialNode]:
        """Execute query and return only Material nodes."""
        return self._execute_of_type(NodeType.MATERIAL)

    def execute_structures_only(self) -> List[StructureNode]:
        """Execute query and return only Structure nodes."""
        return self._execute_of_type(NodeType.STRUCTURE)

    def execute_properties_only(self) -> List[PropertyNode]:
        """Execute query and return only Property nodes."""
        return self._execute_of_type(NodeType.PROPERTY)

    def execute_elements_only(self) -> List[ElementNode]:
        """Execute query and return only Element nodes."""
        return self._execute_of_type(NodeType.ELEMENT)

    def execute_chemsys_only(self) -> List[ChemsysNode]:
        """Execute query and return only Chemsys nodes."""
        return self._execute_of_type(NodeType.CHEMSYS)


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
    element_nodes = [nid for nid, data in graph.nodes(data=True)
                     if data.get("type") == NodeType.ELEMENT.value and
                        data.get("symbol") == element_symbol]

    if not element_nodes:
        return []

    # Materials point TO elements, so walk edges backwards.
    qb = QueryBuilder(graph, element_nodes[0], direction="in")
    qb.edge_type(EdgeType.HAS_ELEMENT)
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
    # Chemsys node IDs are built from SORTED symbols (see kg.schema.chemsys_id),
    # so "P-Ni" and "Ni-P" must resolve to the same node.
    node_id = chemsys_id(chemsys_name.split("-"))
    if node_id not in graph.nodes():
        return []

    # Materials point TO chemsys, so walk edges backwards.
    qb = QueryBuilder(graph, node_id, direction="in")
    qb.edge_type(EdgeType.IN_CHEMSYS)
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
        source_filter: Optional filter for property source (e.g., only
            MaterialsProject-derived values)

    Returns:
        List of MaterialNode models for stable materials
    """
    return find_materials_by_property_range(
        graph,
        PropertyName.ENERGY_ABOVE_HULL,
        float("-inf"),
        max_e_above_hull,
        source_filter=source_filter,
    )


def find_materials_by_property_range(
    graph: nx.MultiDiGraph,
    property_name: PropertyName | str,
    min_val: float,
    max_val: float,
    source_filter: Optional[PropertySource] = None,
) -> List[MaterialNode]:
    """Find materials with a property in a given range.

    Scans Property nodes directly rather than traversing from an
    arbitrary start node (an earlier version hardcoded a dummy
    "material:mp-123" start node, which raised ValueError on any graph
    that did not happen to contain it).

    Args:
        graph: The knowledge graph
        property_name: Name of the property to query
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        source_filter: Optional PropertySource restriction

    Returns:
        List of MaterialNode models for materials with properties in range
    """
    name_value = (
        property_name.value if isinstance(property_name, PropertyName) else str(property_name)
    )
    source_value = (
        source_filter.value if isinstance(source_filter, PropertySource) else source_filter
    )

    matched_mpids: set[str] = set()
    for _nid, data in graph.nodes(data=True):
        if data.get("type") != NodeType.PROPERTY.value:
            continue
        if data.get("name") != name_value:
            continue
        if source_value is not None and data.get("source") != source_value:
            continue
        value = data.get("value")
        if value is None:
            continue
        if not (min_val <= float(value) <= max_val):
            continue
        mpid = data.get("mpid")
        if mpid:
            matched_mpids.add(mpid)

    materials: List[MaterialNode] = []
    for nid, data in graph.nodes(data=True):
        if data.get("type") == NodeType.MATERIAL.value and data.get("mpid") in matched_mpids:
            materials.append(rehydrate_node(graph, nid))
    return materials


def find_all_materials(graph: nx.MultiDiGraph) -> List[MaterialNode]:
    """Find all materials in the graph."""
    mat_nodes = [nid for nid, data in graph.nodes(data=True)
                 if data.get("type") == NodeType.MATERIAL.value]
    return [rehydrate_node(graph, nid) for nid in mat_nodes]


def find_materials_by_formula(graph: nx.MultiDiGraph, formula: str) -> List[MaterialNode]:
    """Find materials with a specific pretty formula."""
    mat_nodes = [nid for nid, data in graph.nodes(data=True)
                 if data.get("type") == NodeType.MATERIAL.value and
                    data.get("formula_pretty") == formula]
    return [rehydrate_node(graph, nid) for nid in mat_nodes]


def find_all_elements(graph: nx.MultiDiGraph) -> List[ElementNode]:
    """Find all elements in the graph."""
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

    Convenience wrapper that starts from the first Material node. For
    control over the start node, use build_query() directly.

    Args:
        graph: The knowledge graph

    Returns:
        QueryBuilder ready for configuration
    """
    if not graph.nodes():
        raise ValueError("Graph must contain at least one node before querying.")

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
