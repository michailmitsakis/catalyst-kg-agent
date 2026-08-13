"""Build the catalyst-kg knowledge graph from data/raw/ artifacts.

Reads:
    data/raw/metadata.json   -- one record per material_id (lightweight fields)
    data/raw/structures/<mpid>.cif  -- crystal structure per material

Writes (via main()):
    data/processed/kg.graphml  -- NetworkX-serialized graph
    data/processed/kg_build_report.json  -- per-material parse status

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
) -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    """Read metadata + CIFs, return (graph, per-material report list).

    Pure function: no files are written. Caller decides where to persist.
    CIF parse failures are recorded in the report and skipped (the
    Material still gets added, just without structure-derived fields).
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    G = nx.MultiDiGraph()
    report: list[dict[str, Any]] = []

    for record in metadata:
        mpid = record["material_id"]
        cif_path = Path(struct_dir) / f"{mpid}.cif"
        struct: Optional[Structure] = None
        cif_relpath: Optional[str] = None
        if cif_path.exists():
            try:
                struct = Structure.from_file(str(cif_path))
                cif_relpath = str(cif_path.relative_to(REPO_ROOT))
            except Exception as exc:
                # Don't fail the build for one bad CIF; the report flags it.
                report.append({
                    "mpid": mpid, "structure_parsed": False,
                    "error": f"cif parse: {type(exc).__name__}: {exc}",
                })
                struct = None
        try:
            row = _add_material(G, record, struct, cif_relpath)
        except Exception as exc:
            report.append({
                "mpid": mpid, "structure_parsed": False,
                "error": f"add_material: {type(exc).__name__}: {exc}",
            })
            continue
        report.append(row)

    return G, report


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

def _summary(G: nx.MultiDiGraph, report: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Unknown")
        by_type[t] = by_type.get(t, 0) + 1
    parsed = sum(1 for r in report if r.get("structure_parsed"))
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "nodes_by_type": by_type,
        "n_materials": len([r for r in report if "mpid" in r]),
        "n_structures_parsed": parsed,
        "n_structures_failed": len([r for r in report if not r.get("structure_parsed")]),
    }


def main() -> None:
    G, report = build_graph()
    summary = _summary(G, report)

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

    print(f"Nodes: {summary['n_nodes']} ({summary['nodes_by_type']})")
    print(f"Edges: {summary['n_edges']}")
    print(f"Materials: {summary['n_materials']}  "
          f"structures parsed: {summary['n_structures_parsed']}  "
          f"failed: {summary['n_structures_failed']}")
    print(f"Wrote: data/processed/kg.json")
    print(gml_msg)
    print(f"Wrote: data/processed/kg_build_report.json")


if __name__ == "__main__":
    main()
