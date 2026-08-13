"""Knowledge-graph layer for the catalyst-kg-agent.

Public surface:
    schema  -- node/edge models, ID helpers, enums (this module's primary surface)
    build_graph -- populate a NetworkX MultiDiGraph from data/raw/ artifacts
    graph_store -- load/save a built graph to disk
    queries  -- typed query functions over a built graph
"""
