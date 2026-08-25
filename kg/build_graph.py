"""Build the catalyst-kg knowledge graph from data/raw/ artifacts.

Reads:
    data/raw/metadata.json   -- one record per material_id (lightweight fields)
    data/raw/structures/<mpid>.cif  -- crystal structure per structure

Writes (via main()):
    data/processed/kg.graphml  -- NetworkX-serialized graph
    data/processed/kg_build_report.json  -- per-material parse status

Caching:
    Structures are cached in a pickle file after the first build to avoid
    re-parsing CIF files on every run. Cache location: data/processed/cif_cache.pkl
    
    Cache metadata (timestamps, size) is tracked in data/processed/cif_cache_meta.json
    for monitoring cache health and invalidation decisions.

Design notes
------------
- Pure builder: `build_graph(metadata_path, struct_dir) -> nx.MultiDiGraph`.
  No file writes inside the builder so it is testable in isolation and
  safe to call from notebooks / `kg/queries.py` ad-hoc.
- Symmetry fields (`crystal_system`, `space_group_*`) are best-effort.
  pymatgen's public surface has shifted across releases, so we only
  trust `get_symmetry_dataset()`'s dict keys and tolerate absence.
- One ChemsysNode per unique sorted element set; one ElementNode per
  unique element symbol across the corpus. Multi-edge `HAS_ELEMENT`
  carries per-element count on the edge attribute.
- Properties live on PropertyNode (not Material attribute) so predicted
  and measured values can coexist with provenance.
- All multi-edges (Material -> multiple Structure / Property nodes in
  future) use NetworkX `MultiDiGraph` + `key=` from the edge id.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from pymatgen.core import Structure

from kg.schema import (
    ChemsysNode,
    CrystalSystem,
    ElementNode,
    HasElementEdge,
    HasPropertyEdge,
    HasStructureEdge,
    InChemsysEdge,
    MaterialNode,
    NodeType,
    PropertyName,
    PropertyNode,
    PropertySource,
    PropertyUnit,
    StructureNode,
    chemsys_id,
    element_id,
    material_id,
    property_id,
    structure_id,
)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

CACHE_FILE = Path("data/processed/cif_cache.pkl")
CACHE_META_FILE = Path("data/processed/cif_cache_meta.json")


def _load_cached_structures() -> dict[str, Structure]:
    """Load cached structures from pickle file if available.
    
    Returns:
        Dict mapping mpid to Structure object, or empty dict if cache missing/corrupt
    """
    if not CACHE_FILE.exists():
        return {}
    
    try:
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
            # Verify cache integrity by checking a few structures
            for mpid, struct in list(cached.items())[:3]:
                if not isinstance(struct, Structure):
                    print(f"Cache warning: Invalid structure type for {mpid}, regenerating")
                    return {}
            return cached
    except Exception as e:
        print(f"Cache load error: {e}, regenerating structures")
        return {}


def _save_cached_structures(struct_cache: dict[str, Structure]) -> None:
    """Save structures to pickle file for future builds.
    
    Args:
        struct_cache: Dict mapping mpid to Structure object
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(struct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_cache_metadata(cache_stats: dict[str, Any]) -> None:
    """Save cache metadata (timestamps, stats) for tracking.
    
    Args:
        cache_stats: Dict with timestamps and statistics
    """
    CACHE_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_META_FILE, "w") as f:
        json.dump(cache_stats, f, indent=2)


def _load_cache_metadata() -> dict[str, Any]:
    """Load cache metadata if available.
    
    Returns:
        Dict with cache statistics and timestamps, or empty dict
    """
    if not CACHE_META_FILE.exists():
        return {}
    try:
        with open(CACHE_META_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_cached_structure(mpid: str) -> Optional[Structure]:
    """Get a cached structure by mpid if available.
    
    Args:
        mpid: Material ID (e.g., "mp-2790")
        
    Returns:
        Structure object or None if not in cache
    """
    cached = _load_cached_structures()
    return cached.get(mpid)


def _update_cache(structure: Structure, mpid: str) -> None:
    """Update the cache with a new structure.
    
    Args:
        structure: Parsed Structure object
        mpid: Material ID to use as cache key
    """
    # Load existing cache
    cached = _load_cached_structures()
    # Update/insert the new structure
    cached[mpid] = structure
    # Save updated cache
    _save_cached_structures(cached)
    
    # Update metadata with timestamp and stats
    meta = _load_cache_metadata()
    meta["last_updated"] = datetime.now().isoformat()
    meta["total_structures"] = len(cached)
    meta["cache_size_mb"] = round(os.path.getsize(CACHE_FILE) / 1024 / 1024, 2)
    _save_cache_metadata(meta)


# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = REPO_ROOT / "data" / "raw" / "metadata.json"
DEFAULT_STRUCT_DIR = REPO_ROOT / "data" / "raw" / "structures"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Symmetry helpers (defensive: tolerate API drift across pymatgen releases)
# ---------------------------------------------------------------------------

def _safe_symmetry(struct: Structure) -> dict[str, Any]:
    """Return a dict with `crystal_system`, `space_group_symbol`,
    `space_group_number` keys; values are None if unavailable.

    Uses pymatgen's `get_symmetry_dataset()` (the most stable surface
    across recent pymatgen releases) and tolerates missing keys.
    """
    out: dict[str, Any] = {
        "crystal_system": None,
        "space_group_symbol": None,
        "space_group_number": None,
    }
    try:
        ds = struct.get_symmetry_dataset()
    except Exception:
        return out
    if not isinstance(ds, dict):
        return out
    cs = ds.get("crystal_system")
    if isinstance(cs, str):
        out["crystal_system"] = cs
    # 'international' is the Hermann-Mauguin symbol, 'number' is the ITA number.
    sg_sym = ds.get("international") or ds.get("spacegroup") or ds.get("symbol")
    if isinstance(sg_sym, str):
        out["space_group_symbol"] = sg_sym
    sg_num = ds.get("number") or ds.get("space_group_number")
    if isinstance(sg_num, int):
        out["space_group_number"] = sg_num
    return out


def _coerce_crystal_system(name: Optional[str]) -> Optional[CrystalSystem]:
    """Map a pymatgen crystal_system string to the CrystalSystem enum.
    Returns None if the string is missing or not one of the 7 ITCA systems."""
    if not name:
        return None
    try:
        return CrystalSystem(name.lower())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Element counting
# ---------------------------------------------------------------------------

def _element_counts(struct: Structure) -> dict[str, int]:
    """Per-element atom counts in the formula unit.

    Uses the Composition's `as_dict()` and the structure's number of
    sites, then rounds to the nearest integer. This avoids the
    `fractional_composition * num_sites` rounding traps (e.g. Fe2.0001)
    that pymatgen occasionally emits for symmetric unit cells.
    """
    counts = struct.composition.as_dict()
    return {str(el): int(round(c)) for el, c in counts.items() if c > 0}


# ---------------------------------------------------------------------------
# Per-material builder
# ---------------------------------------------------------------------------

def _add_material(
    G: nx.MultiDiGraph,
    record: dict[str, Any],
    struct: Optional[Structure],
    cif_relpath: Optional[str],
) -> dict[str, Any]:
    """Add one Material and all of its satellite nodes/edges to G.
    Returns a small report dict for the build summary."""
    mpid = record["material_id"]
    elements = [str(e) for e in record["elements"]]

    # Material node
    mat = MaterialNode(
        id=material_id(mpid),
        mpid=mpid,
        formula_pretty=record["formula_pretty"],
        elements=elements,
    )
    G.add_node(mat.id, **mat.model_dump(mode="json"))

    # Element + Chemsys nodes (deduped by id; NetworkX set-semantics make
    # this a no-op if already present)
    for sym in elements:
        enode = ElementNode(id=element_id(sym), symbol=sym)
        G.add_node(enode.id, **enode.model_dump(mode="json"))
        G.add_edge(
            mat.id, enode.id, key=f"HAS_ELEMENT:{sym}",
            type="HAS_ELEMENT", count=None,  # filled below if struct available
        )

    cs_node = ChemsysNode(id=chemsys_id(elements), symbols=elements)
    G.add_node(cs_node.id, **cs_node.model_dump(mode="json"))
    G.add_edge(
        mat.id, cs_node.id, key="IN_CHEMSYS",
        type="IN_CHEMSYS",
    )

    # Structure node (if CIF parsed)
    sym_info = _safe_symmetry(struct) if struct is not None else {
        "crystal_system": None, "space_group_symbol": None, "space_group_number": None,
    }
    snode = StructureNode(
        id=structure_id(mpid),
        mpid=mpid,
        cif_path=cif_relpath or f"data/raw/structures/{mpid}.cif",
        space_group_symbol=sym_info["space_group_symbol"],
        space_group_number=sym_info["space_group_number"],
        crystal_system=_coerce_crystal_system(sym_info["crystal_system"]),
        num_sites=len(struct) if struct is not None else None,
    )
    G.add_node(snode.id, **snode.model_dump(mode="json"))
    G.add_edge(mat.id, snode.id, key="HAS_STRUCTURE", type="HAS_STRUCTURE")

    # Fill HAS_ELEMENT counts from the parsed structure (preferred)
    # or fall back to elements list with count=1.
    if struct is not None:
        counts = _element_counts(struct)
    else:
        counts = {sym: 1 for sym in elements}
    for sym, c in counts.items():
        key = f"HAS_ELEMENT:{sym}"
        if G.has_edge(mat.id, element_id(sym), key=key):
            G.edges[mat.id, element_id(sym), key]["count"] = c

    # Property nodes (only what metadata carries today)
    e_above = record.get("energy_above_hull")
    if e_above is not None:
        p = PropertyNode(
            id=property_id(mpid, PropertyName.ENERGY_ABOVE_HULL),
            mpid=mpid,
            name=PropertyName.ENERGY_ABOVE_HULL,
            value=float(e_above),
            unit=PropertyUnit.ENERGY_ABOVE_HULL,
            source=PropertySource.MATERIALS_PROJECT,
        )
        G.add_node(p.id, **p.model_dump(mode="json"))
        G.add_edge(mat.id, p.id, key=f"HAS_PROPERTY:{p.name.value}", type="HAS_PROPERTY")

    bg = record.get("band_gap")
    if bg is not None:
        p = PropertyNode(
            id=property_id(mpid, PropertyName.BAND_GAP),
            mpid=mpid,
            name=PropertyName.BAND_GAP,
            value=float(bg),
            unit=PropertyUnit.BAND_GAP,
            source=PropertySource.MATERIALS_PROJECT,
        )
        G.add_node(p.id, **p.model_dump(mode="json"))
        G.add_edge(mat.id, p.id, key=f"HAS_PROPERTY:{p.name.value}", type="HAS_PROPERTY")

    return {
        "mpid": mpid,
        "structure_parsed": struct is not None,
        "crystal_system": sym_info["crystal_system"],
        "space_group_symbol": sym_info["space_group_symbol"],
        "element_counts": counts,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_graph(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    struct_dir: Path = DEFAULT_STRUCT_DIR,
    clear_cache: bool = False,  # Force regeneration of all structures
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]], dict[str, str]]:
    """Read metadata + CIFs, return (graph, per-material report list, cache stats).

    Pure function: no files are written unless clear_cache=True.
    CIF parse failures are recorded in the report and skipped (the
    Material still gets added, just without structure-derived fields).
    
    Caching:
        - Structures are loaded from cache if available and up-to-date
        - Failed parses are always re-attempted (cache is stale for failed files)
        - Cache stats are returned to track efficiency
        
    Args:
        metadata_path: Path to metadata.json
        struct_dir: Directory containing CIF files
        clear_cache: If True, ignore cache and parse all CIFs fresh
        
    Returns:
        Tuple of (NetworkX graph, per-material report list, cache statistics)
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    report: list[dict[str, Any]] = []
    
    # Load cached structures if not clearing cache
    struct_cache = {}
    if not clear_cache:
        struct_cache = _load_cached_structures()
    
    # Track cache stats
    cache_hits = 0
    cache_misses = 0
    cache_failures = 0
    
    for record in metadata:
        mpid = record["material_id"]
        cif_path = Path(struct_dir) / f"{mpid}.cif"
        struct: Optional[Structure] = None
        cif_relpath: Optional[str] = None
        
        # Check cache first
        if not clear_cache and mpid in struct_cache:
            struct = struct_cache[mpid]
            cache_hits += 1
        elif cif_path.exists():
            try:
                struct = Structure.from_file(str(cif_path))
                cif_relpath = str(cif_path.relative_to(REPO_ROOT))
                
                # Update cache if we successfully parsed it
                if not clear_cache:
                    _update_cache(struct, mpid)
                    
            except Exception as exc:
                # Don't fail the build for one bad CIF; the report flags it.
                report.append({
                    "mpid": mpid, "structure_parsed": False,
                    "error": f"cif parse: {type(exc).__name__}: {exc}",
                    "cache_used": False,
                })
                struct = None
                cache_failures += 1
        
        # If not in cache and CIF exists but failed to load, count as miss
        if struct is None and cif_path.exists():
            cache_misses += 1
            
        try:
            row = _add_material(G, record, struct, cif_relpath)
        except Exception as exc:
            report.append({
                "mpid": mpid, "structure_parsed": False,
                "error": f"add_material: {type(exc).__name__}: {exc}",
                "cache_used": mpid in struct_cache and not clear_cache,
            })
            continue
        
        report.append({
            **row,
            "cache_used": (mpid in struct_cache) if not clear_cache else False,
        })
    
    # Return cache statistics
    cache_stats = {
        "total_materials": len(metadata),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_failures": cache_failures,
        "cached_structures": len(struct_cache),
    }
    
    return G, report, cache_stats


# ---------------------------------------------------------------------------
# GraphML export (best-effort, list-attr-unfriendly)
# ---------------------------------------------------------------------------

def _to_graphml_safe(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a shallow copy of G with list-valued attrs JSON-encoded
    and None-valued attrs dropped. GraphML scalar-only data model
    cannot represent Python lists; JSON-encoding is the least
    lossy compromise. The canonical store is `kg.json`."""
    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        clean = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, list):
                clean[k] = json.dumps(v)
            else:
                clean[k] = v
        H.add_node(n, **clean)
    for u, v, k, d in G.edges(keys=True, data=True):
        clean = {}
        for kk, vv in d.items():
            if vv is None:
                continue
            if isinstance(vv, list):
                clean[kk] = json.dumps(vv)
            else:
                clean[kk] = vv
        H.add_edge(u, v, key=k, **clean)
    return H


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _summary(G: nx.MultiDiGraph, report: list[dict[str, Any]], cache_stats: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Generate summary statistics for the built graph.
    
    Args:
        G: NetworkX graph
        report: Per-material build report
        cache_stats: Cache statistics (optional, from build_graph return)
        
    Returns:
        Dict with node/edge counts, material counts, and cache stats if available
    """
    by_type: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Unknown")
        by_type[t] = by_type.get(t, 0) + 1
    
    parsed = sum(1 for r in report if r.get("structure_parsed"))
    
    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "nodes_by_type": by_type,
        "n_materials": len([r for r in report if "mpid" in r]),
        "n_structures_parsed": parsed,
        "n_structures_failed": len([r for r in report if not r.get("structure_parsed")]),
    }
    
    # Add cache stats if available
    if cache_stats:
        summary["cache"] = {
            "hits": cache_stats.get("cache_hits", 0),
            "misses": cache_stats.get("cache_misses", 0),
            "failures": cache_stats.get("cache_failures", 0),
            "cached_count": cache_stats.get("cached_structures", 0),
        }
    
    return summary


def main(clear_cache: bool = False) -> None:
    """Main entry point for building the knowledge graph.
    
    Args:
        clear_cache: If True, ignore cache and parse all CIFs fresh
        
    Prints summary statistics including cache efficiency metrics.
    """
    G, report, cache_stats = build_graph(clear_cache=clear_cache)
    summary = _summary(G, report, cache_stats)
    
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Canonical persistence: JSON. Preserves list-valued attrs (Material.elements,
    # Chemsys.symbols) that GraphML cannot serialize. GraphML export is a
    # best-effort convenience for gephi/cytoscape; lists get JSON-encoded.
    (out_dir / "kg.json").write_text(
        json.dumps(nx.node_link_data(G, edges="links"), indent=2),
        encoding="utf-8",
    )
    (out_dir / "kg_build_report.json").write_text(
        json.dumps({"summary": summary, "per_material": report}, indent=2),
        encoding="utf-8",
    )
    
    # Best-effort GraphML: stringify list-valued attrs to JSON, drop None.
    try:
        G_gml = _to_graphml_safe(G)
        nx.write_graphml(G_gml, out_dir / "kg.graphml")
        gml_msg = f"Wrote: data/processed/kg.graphml (list attrs JSON-encoded)"
    except Exception as exc:
        gml_msg = f"GraphML skipped: {type(exc).__name__}: {exc}"
    
    # Print summary with cache stats
    print(f"Nodes: {summary['n_nodes']} ({summary['nodes_by_type']})")
    print(f"Edges: {summary['n_edges']}")
    print(f"Materials: {summary['n_materials']}  "
          f"structures parsed: {summary['n_structures_parsed']}  "
          f"failed: {summary['n_structures_failed']}")
    
    # Display cache statistics if available
    if "cache" in summary and summary["cache"]:
        cache = summary["cache"]
        print(f"\nCache efficiency:")
        print(f"  Loaded from cache: {cache['hits']} / {cache['cached_count']} structures ({100*cache['hits']/max(cache['cached_count'],1):.1f}%)")
        print(f"  Re-parsed: {cache['misses']} structures")
        print(f"  Parse failures: {cache['failures']} structures")
    
    print(f"Wrote: data/processed/kg.json")
    print(gml_msg)
    print(f"Wrote: data/processed/kg_build_report.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build catalyst-kg knowledge graph from CIF files")
    parser.add_argument("--clear-cache", action="store_true", 
                       help="Force regeneration of all structures (ignore cache)")
    args = parser.parse_args()
    
    # Use clear_cache flag to force regeneration
    main(clear_cache=args.clear_cache)
