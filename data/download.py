"""Pull a clean-energy/catalyst-relevant subset of Materials Project structures.

Two-stage pipeline:
  1. Metadata pull (light)  -> data/raw/metadata.json
     Fields: material_id, formula_pretty, elements, energy_above_hull,
             formation_energy_per_atom, band_gap
  2. Structure pull (heavy) -> data/raw/structures/{mpid}.cif
     Only for the kept candidates.

Rationale: re-running the full pipeline shouldn't re-serialize hundreds of
Structure objects. Stage 1 is cheap to re-run; stage 2 caches CIFs per-material.

Property usage downstream (see kg/build_graph.py):
  energy_above_hull         -- Critic's stability gate + campaign ranking
  formation_energy_per_atom -- CGCNN training target; the quantity MACE and
                               CGCNN are compared on
  band_gap                  -- ingested to exercise the multi-property schema;
                               not read by the agent loop today

Set MP_API_KEY in .env.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mp_api.client import MPRester

# ChemSys groups: HER-relevant (transition metal + P/S/C), OER-relevant
# (oxides, mixed-oxide spinels), and precious-metal benchmarks.
HER_SYSTEMS = [
    "Ni-P", "Co-P", "Fe-P", "Mo-P", "W-P", "Mn-P",
    "Ni-S", "Co-S", "Fe-S", "Mo-S", "W-S", "Mn-S",
    "Ni-C", "Co-C", "Fe-C", "Mo-C", "W-C", "Mn-C",
]
OER_SYSTEMS = [
    "Ni-O", "Co-O", "Fe-O", "Mn-O",
    "Ni-Fe-O", "Co-Fe-O", "Ni-Co-O",
]
BENCHMARK_SYSTEMS = ["Pt", "Ir-O"]

CHEMSYS_GROUPS = HER_SYSTEMS + OER_SYSTEMS + BENCHMARK_SYSTEMS

# Stability filter per AtomisticSkills workflow: e_above_hull <= 50 meV/atom
# is the conventional "metastable but synthesizable" cutoff.
E_ABOVE_HULL_RANGE = (0.0, 0.05)
NUM_SITES_RANGE = (0, 20)
TOP_N_BY_STABILITY = 300  # sane cap, likely won't be hit

METADATA_FIELDS = [
    "material_id",
    "formula_pretty",
    "elements",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
STRUCT_DIR = RAW_DIR / "structures"
METADATA_PATH = RAW_DIR / "metadata.json"


def pull_metadata() -> list[dict[str, Any]]:
    """Stage 1: pull light metadata for all matching materials, dedupe, rank, cap.

    Returns list of dicts (JSON-serializable). Writes data/raw/metadata.json.
    """
    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY not set. Check .env.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    by_id: dict[str, dict[str, Any]] = {}
    with MPRester(api_key) as mpr:
        for chemsys in CHEMSYS_GROUPS:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                energy_above_hull=E_ABOVE_HULL_RANGE,
                num_sites=NUM_SITES_RANGE,
                fields=METADATA_FIELDS,
            )
            for d in docs:
                by_id[d.material_id] = {
                    "material_id": d.material_id,
                    "formula_pretty": d.formula_pretty,
                    # pymatgen Element objects aren't JSON-serializable; cast to str.
                    "elements": [str(e) for e in d.elements],
                    "energy_above_hull": d.energy_above_hull,
                    # Formation energy relative to elemental references (MP
                    # convention, eV/atom). This is the target both surrogates
                    # are trained/compared on -- see models/ and PROJECT_STATE.
                    "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
                    "band_gap": d.band_gap,
                }

    ranked = sorted(
        by_id.values(),
        key=lambda r: (r["energy_above_hull"] if r["energy_above_hull"] is not None else float("inf")),
    )
    final = ranked[:TOP_N_BY_STABILITY]

    METADATA_PATH.write_text(json.dumps(final, indent=2))

    n_missing_fe = sum(1 for r in final if r.get("formation_energy_per_atom") is None)

    print(f"Unique candidates before cap: {len(by_id)}")
    print(f"Final set size: {len(final)}")
    if n_missing_fe:
        # Surfaced rather than silent: build_graph skips absent properties, so
        # a partial pull would quietly shrink the CGCNN training set.
        print(f"WARNING: {n_missing_fe}/{len(final)} records have no formation_energy_per_atom")
    print(f"Wrote: {METADATA_PATH.relative_to(REPO_ROOT)}")
    return final


def pull_structures(material_ids: list[str]) -> list[Path]:
    """Stage 2: fetch a Structure per material_id, write CIF. Skip if cached.

    Returns list of paths to written (or already-present) CIF files.
    """
    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY not set. Check .env.")

    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    missing: list[str] = []

    # Pass 1: find which ones need fetching
    for mpid in material_ids:
        path = STRUCT_DIR / f"{mpid}.cif"
        if not path.exists():
            missing.append(mpid)
        else:
            written.append(path)

    if not missing:
        print(f"All {len(material_ids)} structures already cached.")
        return written

    # Pass 2: pull only the missing ones, batched
    n_newly_written = 0
    with MPRester(api_key) as mpr:
        # mp-api supports structure fetch by material_id; we do it per-id
        # for clarity, but a batched call exists if this becomes slow.
        for mpid in missing:
            try:
                struct = mpr.get_structure_by_material_id(mpid)
            except Exception as exc:
                print(f"  [skip] {mpid}: {exc}")
                continue
            path = STRUCT_DIR / f"{mpid}.cif"
            struct.to(filename=str(path))
            written.append(path)
            n_newly_written += 1

    # Count actual writes, not len(missing) -- skipped fetches are not writes.
    print(f"Structures cached: {len(written)} (newly written: {n_newly_written})")
    return written


def main() -> None:
    metadata = pull_metadata()
    material_ids = [m["material_id"] for m in metadata]
    pull_structures(material_ids)


if __name__ == "__main__":
    main()
