"""Build the catalyst-kg knowledge graph from data/raw/ artifacts.

Reads:
    data/raw/metadata.json   -- one record per material_id (lightweight fields)
    data/raw/structures/<mpid>.cif  -- crystal structure per structure

Writes (via main()):
    data/processed/kg.json     -- canonical NetworkX node_link_data store
    data/processed/kg.graphml  -- best-effort interop export (list attrs JSON-encoded)
    data/processed/kg_build_report.json  -- per-material parse status

Caching:
    Structures are cached in a pickle file after the first build to avoid
    re-parsing CIF files on every run. Cache location: data/processed/cif_cache.pkl

    Cache metadata (timestamps, size) is tracked in data/processed/cif_cache_meta.json
    for monitoring cache health and invalidation decisions.

    The cache is written ONCE at the end of a build, not per-material -- see
    `_save_cached_structures` / the tail of `build_graph`.

Design notes
------------
- Pure-ish builder: `build_graph(metadata_path, struct_dir) -> (graph, report, stats)`.
  The graph itself is built in memory; the only side effect is the CIF cache
  write (one pickle + one metadata JSON at the end of the run). Graph
  persistence happens in `main()`, not here.
- Symmetry fields (`crystal_system`, `space_group_*`) are best-effort.
  `Structure.get_symmetry_dataset()` supplies the space group symbol
  (`international`) and number (`number`) but has NO `crystal_system` key,
  so the crystal system comes from `SpacegroupAnalyzer.get_crystal_system()`.
- CIF paths are stored POSIX-style (forward slashes) via `as_posix()` so a
  KG built on Windows still resolves on Linux/macOS. Backslash-separated
  paths are a single literal filename on POSIX, not a path.
- One ChemsysNode per unique sorted element set; one ElementNode per
  unique element symbol across the corpus. Multi-edge `HAS_ELEMENT`
  carries per-element count on the edge attribute.
- Properties live on PropertyNode (not Material attribute) so predicted
  and measured values can coexist with provenance. Three properties are
  ingested from MP metadata when present:
    energy_above_hull        -- Critic's stability gate + campaign ranking
    formation_energy_per_atom -- CGCNN training target / MACE comparison target
    band_gap                 -- ingested for schema demonstration; not read
                                by the agent loop today
- All multi-edges (Material -> multiple Structure / Property nodes in
  future) use NetworkX `MultiDiGraph` + `key=` from the edge id.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    """Save structures to pickle file and refresh cache metadata.

    Called ONCE per build (not per material): the previous per-material
    version re-loaded and re-dumped the entire pickle for every record,
    which is O(N^2) file I/O over the corpus.

    Args:
        struct_cache: Dict mapping mpid to Structure object
    """
    if not struct_cache:
        return

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(struct_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = _load_cache_metadata()
    meta["last_updated"] = datetime.now().isoformat()
    meta["total_structures"] = len(struct_cache)
    meta["cache_size_mb"] = round(os.path.getsize(CACHE_FILE) / 1024 / 1024, 2)
    _save_cache_metadata(meta)


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

    Two different pymatgen surfaces are needed here:
      - `Structure.get_symmetry_dataset()` returns a dict carrying
        `international` (Hermann-Mauguin symbol) and `number` (ITA number),
        but it has NO `crystal_system` key -- reading one always yields None.
      - `SpacegroupAnalyzer(struct).get_crystal_system()` is the actual
        source for the crystal system string.

    Each lookup is guarded separately so a failure in one does not blank
    the other.
    """
    out: dict[str, Any] = {
        "crystal_system": None,
        "space_group_symbol": None,
        "space_group_number": None,
    }

    # Space group symbol / number from the symmetry dataset.
    try:
        ds = struct.get_symmetry_dataset()
        if isinstance(ds, dict):
            sg_sym = ds.get("international") or ds.get("spacegroup") or ds.get("symbol")
            if isinstance(sg_sym, str):
                out["space_group_symbol"] = sg_sym
            sg_num = ds.get("number") or ds.get("space_group_number")
            if isinstance(sg_num, int):
                out["space_group_number"] = sg_num
    except Exception:
        pass

    # Crystal system from SpacegroupAnalyzer (not present in the dataset dict).
    try:
        sga = SpacegroupAnalyzer(struct)
        cs = sga.get_crystal_system()
        if isinstance(cs, str):
            out["crystal_system"] = cs
        # Backfill space group fields from the analyzer if the dataset lookup
        # above came up empty.
        if out["space_group_symbol"] is None:
            sym = sga.get_space_group_symbol()
            if isinstance(sym, str):
                out["space_group_symbol"] = sym
        if out["space_group_number"] is None:
            num = sga.get_space_group_number()
            if isinstance(num, int):
                out["space_group_number"] = num
    except Exception:
        pass

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
# Property ingestion
# ---------------------------------------------------------------------------

# (metadata key, PropertyName member, PropertyUnit member)
# Every property MP metadata may carry. Missing keys are skipped silently --
# a KG built before formation_energy_per_atom was added to download.py's
# METADATA_FIELDS simply won't have those nodes.
_METADATA_PROPERTIES = [
    ("energy_above_hull", PropertyName.ENERGY_ABOVE_HULL, PropertyUnit.ENERGY_ABOVE_HULL),
    ("formation_energy_per_atom", PropertyName.FORMATION_ENERGY_PER_ATOM, PropertyUnit.FORMATION_ENERGY_PER_ATOM),
    ("band_gap", PropertyName.BAND_GAP, PropertyUnit.BAND_GAP),
]


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

    # Property nodes (whatever the metadata record carries)
    properties_added: list[str] = []
    for meta_key, prop_name, prop_unit in _METADATA_PROPERTIES:
        raw = record.get(meta_key)
        if raw is None:
            continue
        p = PropertyNode(
            id=property_id(mpid, prop_name),
            mpid=mpid,
            name=prop_name,
            value=float(raw),
            unit=prop_unit,
            source=PropertySource.MATERIALS_PROJECT,
        )
        G.add_node(p.id, **p.model_dump(mode="json"))
        G.add_edge(mat.id, p.id, key=f"HAS_PROPERTY:{prop_name.value}", type="HAS_PROPERTY")
        properties_added.append(prop_name.value)

    return {
        "mpid": mpid,
        "structure_parsed": struct is not None,
        "crystal_system": sym_info["crystal_system"],
        "space_group_symbol": sym_info["space_group_symbol"],
        "element_counts": counts,
        "properties_added": properties_added,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_graph(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    struct_dir: Path = DEFAULT_STRUCT_DIR,
    clear_cache: bool = False,  # Ignore any existing cache and re-parse every CIF
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]], dict[str, Any]]:
    """Read metadata + CIFs, return (graph, per-material report list, cache stats).

    The graph is built in memory and NOT persisted here -- `main()` owns
    graph persistence. The one side effect of this function is the CIF
    cache: newly parsed structures are written to `cif_cache.pkl` (plus its
    metadata JSON) once, at the end of the run.

    CIF parse failures are recorded in the report and skipped (the
    Material still gets added, just without structure-derived fields).

    Caching:
        - Structures are loaded from cache when present (unless clear_cache)
        - Anything not in the cache is parsed fresh and added to it
        - The merged cache is written once at the end, not per material

    Args:
        metadata_path: Path to metadata.json
        struct_dir: Directory containing CIF files
        clear_cache: If True, ignore the existing cache and parse all CIFs fresh
            (the freshly parsed structures still get written to the cache)

    Returns:
        Tuple of (NetworkX graph, per-material report list, cache statistics)
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    report: list[dict[str, Any]] = []

    # Load cached structures if not clearing cache
    struct_cache: dict[str, Structure] = {}
    if not clear_cache:
        struct_cache = _load_cached_structures()
    cache_at_start = set(struct_cache)

    # Track cache stats
    cache_hits = 0        # served from the pre-existing cache
    cache_misses = 0      # parsed fresh from CIF this run
    cache_failures = 0    # CIF present but failed to parse

    for record in metadata:
        mpid = record["material_id"]
        cif_path = Path(struct_dir) / f"{mpid}.cif"
        struct: Optional[Structure] = None
        cif_relpath: Optional[str] = None

        # POSIX-style relative path so a KG built on Windows resolves
        # on Linux/macOS too. Set regardless of cache hit/miss so cached
        # entries don't silently fall back to a differently-formatted default.
        try:
            cif_relpath = cif_path.resolve().relative_to(REPO_ROOT).as_posix()
        except Exception:
            cif_relpath = f"data/raw/structures/{mpid}.cif"

        # Check cache first
        if mpid in cache_at_start:
            struct = struct_cache[mpid]
            cache_hits += 1
        elif cif_path.exists():
            try:
                struct = Structure.from_file(str(cif_path))
                struct_cache[mpid] = struct  # merged + written once after the loop
                cache_misses += 1
            except Exception as exc:
                # Don't fail the build for one bad CIF; the report flags it.
                report.append({
                    "mpid": mpid, "structure_parsed": False,
                    "error": f"cif parse: {type(exc).__name__}: {exc}",
                    "cache_used": False,
                })
                struct = None
                cache_failures += 1
                continue

        try:
            row = _add_material(G, record, struct, cif_relpath)
        except Exception as exc:
            report.append({
                "mpid": mpid, "structure_parsed": False,
                "error": f"add_material: {type(exc).__name__}: {exc}",
                "cache_used": mpid in cache_at_start,
            })
            continue

        report.append({
            **row,
            "cache_used": mpid in cache_at_start,
        })

    # Single cache write for the whole build.
    _save_cached_structures(struct_cache)

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

def _summary(G: nx.MultiDiGraph, report: list[dict[str, Any]], cache_stats: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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

    # Count property nodes by name, so a build that predates the
    # formation_energy_per_atom field is obvious in the report.
    props_by_name: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        if d.get("type") == NodeType.PROPERTY.value:
            nm = str(d.get("name", "unknown"))
            props_by_name[nm] = props_by_name.get(nm, 0) + 1

    summary = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "nodes_by_type": by_type,
        "properties_by_name": props_by_name,
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
    print(f"Properties: {summary['properties_by_name']}")
    print(f"Materials: {summary['n_materials']}  "
          f"structures parsed: {summary['n_structures_parsed']}  "
          f"failed: {summary['n_structures_failed']}")

    # Display cache statistics if available
    if "cache" in summary and summary["cache"]:
        cache = summary["cache"]
        print(f"\nCache efficiency:")
        print(f"  Served from cache: {cache['hits']}")
        print(f"  Parsed fresh this run: {cache['misses']}")
        print(f"  Parse failures: {cache['failures']}")
        print(f"  Structures now cached: {cache['cached_count']}")

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
