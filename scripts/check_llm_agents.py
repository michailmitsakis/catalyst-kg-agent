#!/usr/bin/env python
"""Verify that Ollama is actually being called by both LLM agents.

Two of the five agent roles use an LLM: the Retriever (natural-language
query -> structured intent) and the Planner (candidate prioritisation).
Both degrade silently to a deterministic fallback if Ollama is unreachable
-- by design, so a campaign still runs -- which means a broken Ollama setup
looks exactly like a working one unless you check provenance.

This script calls each agent directly and reports which path it took.

Usage:
    python scripts/check_llm_agents.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()


def check_server() -> bool:
    """Is an Ollama server reachable, and is the configured model present?"""
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    want = os.environ.get("OLLAMA_MODEL", "gemma4:latest")

    print("=" * 62)
    print("Ollama server")
    print("=" * 62)
    print(f"  OLLAMA_BASE_URL = {base}")
    print(f"  OLLAMA_MODEL    = {want}")

    # The /v1 suffix is the OpenAI-compatible path; /api/tags lives at the root.
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    try:
        import httpx

        resp = httpx.get(f"{root}/api/tags", timeout=5.0)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        print(f"  [FAIL] cannot reach Ollama: {type(exc).__name__}: {exc}")
        print("         Start it with: ollama serve")
        return False

    print(f"  [OK]   reachable, {len(tags)} model(s) installed")
    if want in tags:
        print(f"  [OK]   '{want}' is installed")
        return True

    print(f"  [FAIL] '{want}' NOT installed. Available: {tags}")
    print(f"         Install it with: ollama pull {want}")
    return False


def check_retriever() -> bool:
    """Does the Retriever parse a query with the LLM, or fall back?"""
    from kg.graph_store import DEFAULT_KG_JSON
    from agent.retriever import create_retriever

    print()
    print("=" * 62)
    print("Retriever  (query -> structured intent)")
    print("=" * 62)

    try:
        retriever = create_retriever(DEFAULT_KG_JSON)
    except Exception as exc:
        print(f"  [FAIL] could not construct: {type(exc).__name__}: {exc}")
        return False

    if getattr(retriever, "agent", None) is None:
        print("  [FAIL] no LLM agent constructed (use_llm off, or Ollama unavailable)")
        return False

    query = "Find stable Ni-P HER catalysts"
    result = retriever.run_query(query)
    parsed_by = result.provenance.get("parsed_by")

    print(f"  query       : {query!r}")
    print(f"  parsed_by   : {parsed_by}")
    print(f"  llm_intent  : {result.provenance.get('llm_intent')}")
    print(f"  constraints : {result.provenance.get('constraints_used')}")
    print(f"  materials   : {len(result.materials)}")

    if parsed_by == "llm":
        print("  [OK]   Ollama was called and its intent was used")
        return True

    print(f"  [FAIL] fell back to '{parsed_by}' -- Ollama was NOT used")
    if result.provenance.get("error"):
        print(f"         error: {result.provenance['error']}")
    return False


def check_planner() -> bool:
    """Does the Planner order candidates with the LLM, or fall back?"""
    from kg.graph_store import DEFAULT_KG_JSON, load_graph, rehydrate_node
    from kg.schema import NodeType
    from agent.planner import create_planner

    print()
    print("=" * 62)
    print("Planner  (candidate prioritisation)")
    print("=" * 62)

    planner = create_planner(graph_path=DEFAULT_KG_JSON, campaign_id="llm-check", use_llm=True)

    if planner.agent is None:
        print("  [FAIL] no LLM agent constructed -- Ollama unavailable at init")
        return False

    G = load_graph(DEFAULT_KG_JSON)
    nids = [n for n, d in G.nodes(data=True) if d.get("type") == NodeType.MATERIAL.value][:8]
    candidates = [rehydrate_node(G, n) for n in nids]

    result = planner.prioritize_candidates(
        candidates=candidates,
        predictions_so_far=None,
        remaining_budget=95.0,
    )

    print(f"  candidates  : {len(candidates)}")
    print(f"  provenance  : {result.provenance}")
    print(f"  llm_ranked  : {result.n_llm_ranked}")
    print(f"  dropped ids : {result.n_dropped_unknown}")
    print(f"  reason      : {(result.reason or '')[:110]}")
    print(f"  order       : {[m.mpid for m in result.ordered_materials]}")

    returned = {m.mpid for m in result.ordered_materials}
    given = {m.mpid for m in candidates}
    print(f"  permutation intact: {returned == given and len(result.ordered_materials) == len(candidates)}")

    if result.provenance == "llm":
        print("  [OK]   Ollama returned a complete, valid ordering")
        return True
    if result.provenance == "llm_partial":
        print("  [WARN] Ollama was called but returned a partial/invalid ordering.")
        print("         Working, but consider lowering MAX_CANDIDATES_IN_PROMPT")
        print("         in agent/planner.py if this happens consistently.")
        return True

    print(f"  [FAIL] provenance '{result.provenance}' -- Ollama was NOT used effectively")
    return False


def main() -> int:
    ok_server = check_server()
    if not ok_server:
        print()
        print("Server unavailable -- agent checks would only confirm the fallback path.")
        return 1

    ok_retriever = check_retriever()
    ok_planner = check_planner()

    print()
    print("=" * 62)
    print("Summary")
    print("=" * 62)
    for name, ok in (("server", ok_server), ("Retriever", ok_retriever), ("Planner", ok_planner)):
        print(f"  {name:11s} {'OK' if ok else 'NOT USING OLLAMA'}")

    if ok_retriever and ok_planner:
        print()
        print("Both LLM agents are calling Ollama. Note the Critic, Predictor and")
        print("Scribe are deterministic by design and make no LLM calls.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())