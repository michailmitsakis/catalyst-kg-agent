"""Persist and reload the knowledge graph.

Canonical store: `data/processed/kg.json` (NetworkX node_link_data, JSON).
GraphML at `data/processed/kg.graphml` is a best-effort interop export
with list-valued attrs stringified; do not treat it as the source of
truth -- it cannot round-trip list fields.

This module owns the on-disk format choices so that callers
(`kg/queries.py`, `agent/retriever.py`, notebooks) only see
`load_graph()` / `save_graph()` and the typed rehydrate helper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TypeVar, Union

import networkx as nx
from pydantic import BaseModel


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg.schema import (
    ChemsysNode,
    ElementNode,
    KGNode,
    MaterialNode,
    NodeType,
    PropertyNode,
    StructureNode,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_KG_JSON = DEFAULT_OUT_DIR / "kg.json"
DEFAULT_KG_GRAPHML = DEFAULT_OUT_DIR / "kg.graphml"


# ---------------------------------------------------------------------------
# Type discriminator -> pydantic model class
# ---------------------------------------------------------------------------

_NODE_MODELS: dict[NodeType, type[BaseModel]] = {
    NodeType.MATERIAL: MaterialNode,
    NodeType.ELEMENT: ElementNode,
    NodeType.CHEMSYS: ChemsysNode,
    NodeType.STRUCTURE: StructureNode,
    NodeType.PROPERTY: PropertyNode,
}


T = TypeVar("T", bound=BaseModel)


def _model_for(node_type: str) -> type[BaseModel]:
    """Map a serialized NodeType string back to its pydantic class.

    Raises KeyError if a graph contains an unknown type -- the failure
    should surface loudly rather than silently drop nodes.
    """
    return _NODE_MODELS[NodeType(node_type)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_graph(
    G: nx.MultiDiGraph,
    json_path: Path = DEFAULT_KG_JSON,
    graphml_path: Path | None = DEFAULT_KG_GRAPHML,
) -> None:
    """Write the graph to disk. `kg.json` is canonical; `kg.graphml` is
    best-effort interop and may be skipped if list-valued attrs make
    serialization fail. Idempotent: overwrites existing files."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(nx.node_link_data(G, edges="links"), indent=2),
        encoding="utf-8",
    )
    if graphml_path is not None:
        try:
            from kg.build_graph import _to_graphml_safe
            nx.write_graphml(_to_graphml_safe(G), graphml_path)
        except Exception:
            # Don't fail the canonical write because the interop export broke.
            # (The CLI in build_graph.py reports this; library callers can
            # pass graphml_path=None to skip entirely.)
            pass


def load_graph(json_path: Path = DEFAULT_KG_JSON) -> nx.MultiDiGraph:
    """Read `kg.json` back into a `MultiDiGraph` with the original
    attribute dicts intact (lists stay lists, strings stay strings).

    The rehydrate call sites that want typed pydantic models should
    use `rehydrate_node()` per node -- this loader stays untyped so
    that graph algorithms (`nx.shortest_path`, traversal, etc.) are
    unblocked.
    """
    blob = json.loads(Path(json_path).read_text(encoding="utf-8"))
    G = nx.node_link_graph(blob, edges="links")
    # `node_link_graph` returns a DiGraph by default; force the
    # multi-edge view so the rest of the codebase can rely on it.
    if not isinstance(G, nx.MultiDiGraph):
        G = G.to_directed(as_view=False)  # noqa: F841 -- assignment below
        G = nx.MultiDiGraph(G)
    return G


def rehydrate_node(G: nx.MultiDiGraph, node_id: str) -> KGNode:
    """Return the typed pydantic model for one node, looked up by id.

    Convenience for callers (e.g. `kg/queries.py`, the Retriever
    agent) that want a real `MaterialNode`/`PropertyNode` instead of
    the raw attribute dict.

    Note: NetworkX uses the node ID as the lookup key and strips it
    from the attribute dict, so we must inject it back before pydantic
    validation.

    For MaterialNode specifically: traverses HAS_STRUCTURE edge to populate
    structure_id field if present in the graph.
    """
    data = dict(G.nodes[node_id])
    data["id"] = node_id  # nx.node_link_graph drops id from attrs
    model_cls = _model_for(data["type"])
    
    result = model_cls.model_validate(data)
    
    # Special handling for MaterialNode: find and link its StructureNode
    if isinstance(result, MaterialNode):
        # Look for HAS_STRUCTURE edge from this material
        for edge_tuple in G.edges(node_id, data=True):
            # Handle multi-edges (tuples of 3) vs regular edges (tuples of 2)
            if len(edge_tuple) >= 3:
                src, tgt, edge_dict = edge_tuple
            else:
                src, tgt = edge_tuple
                edge_dict = edge_data
            
            if edge_dict.get("type") == "HAS_STRUCTURE":
                result.structure_id = tgt
                break
    
    return result


def rehydrate_many(G: nx.MultiDiGraph, node_ids: list[str]) -> list[KGNode]:
    """Typed version of `G.nodes(data=True)` filtered to `node_ids`."""
    return [rehydrate_node(G, nid) for nid in node_ids]


def node_ids_by_type(G: nx.MultiDiGraph, node_type: NodeType | str) -> list[str]:
    """Return all node ids whose `type` attribute matches.

    Cheap: O(N) over nodes, no traversal. Used by `queries.py` to
    scope traversal from a known node type rather than walking the
    full graph.
    """
    target = node_type.value if isinstance(node_type, NodeType) else str(node_type)
    return [n for n, d in G.nodes(data=True) if d.get("type") == target]


__all__ = [
    "save_graph", "load_graph", "rehydrate_node", "rehydrate_many",
    "node_ids_by_type",
    "DEFAULT_KG_JSON", "DEFAULT_KG_GRAPHML",
]
