import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kg.graph_store import DEFAULT_KG_JSON
from agent.retriever import create_retriever

r = create_retriever(DEFAULT_KG_JSON)
for q in ["Find stable Ni-P HER catalysts", "find all stable materials"]:
    res = r.run_query(q)
    p = res.provenance
    print(f"{q!r:42s} -> {len(res.materials):3d} materials")
    print(f"    parsed_by   = {p.get('parsed_by')}")
    print(f"    constraints = {p.get('constraints_used')}")
    print(f"    intent      = {p.get('llm_intent')}")