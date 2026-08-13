"""A typed query language over a built knowledge graph."""
from kg.schema import *


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_query(g: nx.MultiDiGraph) -> QueryBuilder:
    """Build a typed query object from the parsed graph."""
    return GraphQueryBuilder(graph=G, default_node_type=NodeType.MATERIAL)


def get_typed_query(g: nx.MultiDiGraph) -> TypedQueryBuilder:
    """The entry point for building a typed query against a built graph."""
    if not g.nodes():
        raise ValueError("Must build graph before querying.")

    return GraphQueryBuilder(G=g)