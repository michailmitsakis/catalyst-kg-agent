"""Compute per-element MACE reference energies for formation-energy calculation.

Formation energy per atom is

    E_f = E_material_per_atom - sum_i( x_i * E_ref_i )

where `x_i` is the atomic fraction of element i in the material and
`E_ref_i` is that element's energy per atom in its reference state. Every
term must come from the SAME MACE checkpoint, or the subtraction mixes
levels of theory and the result is meaningless (this is the same
tier-separation rule the project applies to MP / MACE / UMA energies).

Reference states used here
--------------------------
Solid elements (C, Co, Fe, Ir, Mn, Mo, Ni, P, Pt, S, W):
    The lowest-energy-above-hull elemental crystal from Materials Project
    for that element, relaxed structure taken as-is, energy evaluated with
    MACE. This matches the standard convention of referencing the
    condensed-phase ground state.

Oxygen:
    Referenced against the O2 MOLECULE in a large vacuum box, evaluated
    with the same MACE checkpoint -- not a solid-oxygen crystal. Molecular
    O2 is the standard reference state for oxygen, and the lowest-hull
    "elemental oxygen crystal" in MP is not a physically appropriate
    reference for oxide formation energies.

KNOWN LIMITATION -- oxide offset
--------------------------------
Materials Project's own `formation_energy_per_atom` applies an empirical
anion correction to oxides (and other anion-containing compounds) on top
of raw DFT, fitted to experimental formation enthalpies. See:
    Wang et al., "A framework for quantifying uncertainty in DFT energy
    corrections", Sci Rep 11, 15496 (2021)
    Jain et al., Phys. Rev. B 84, 045115 (2011)  [MP anion corrections]

The MACE-derived formation energies produced with these references apply
NO such correction. Consequently, MACE formation energies for
oxygen-containing compounds carry a roughly systematic offset relative to
MP's corrected values, while non-oxide compounds do not. Any
MACE-vs-MP-target comparison should therefore be reported split by
oxide / non-oxide so the offset is visible rather than averaged away.
See models/ comparison output and the README "Limitations" section.

Usage:
    python models/elemental_references.py                 # build/refresh cache
    python models/elemental_references.py --force         # ignore existing cache
    python models/elemental_references.py --elements Ni P # subset only

Output:
    data/processed/mace_elemental_refs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KG_JSON = REPO_ROOT / "data" / "processed" / "kg.json"
DEFAULT_REFS_PATH = REPO_ROOT / "data" / "processed" / "mace_elemental_refs.json"
DEFAULT_MACE_CHECKPOINT = REPO_ROOT / "models" / "mace-mpa-0-medium.model"

# Elements whose reference state is a molecule in vacuum rather than a
# crystalline solid. Only O appears in the current catalyst corpus; N, H,
# F, Cl are listed so the same code path works if the chemsys groups grow.
DIATOMIC_GAS_REFERENCES = {
    "O": 1.21,   # O2 bond length in Angstrom
    "N": 1.10,   # N2
    "H": 0.74,   # H2
    "F": 1.42,   # F2
    "Cl": 1.99,  # Cl2
}

# Edge length of the cubic vacuum box used for molecular references.
# Must be comfortably larger than the MACE cutoff so periodic images
# do not interact.
VACUUM_BOX_SIZE = 15.0  # Angstrom


# ---------------------------------------------------------------------------
# Element discovery
# ---------------------------------------------------------------------------

def elements_from_kg(kg_path: Path = DEFAULT_KG_JSON) -> list[str]:
    """Read the distinct element symbols present in a built KG.

    Deriving the element list from the graph (rather than hardcoding it)
    keeps the reference set correct if the chemsys groups in
    data/download.py change.

    Args:
        kg_path: Path to kg.json

    Returns:
        Sorted list of element symbols
    """
    blob = json.loads(Path(kg_path).read_text(encoding="utf-8"))
    symbols = {
        n.get("symbol")
        for n in blob.get("nodes", [])
        if n.get("type") == "Element" and n.get("symbol")
    }
    return sorted(symbols)


# ---------------------------------------------------------------------------
# MACE calculator
# ---------------------------------------------------------------------------

def _make_calculator(checkpoint_path: Path, device: str = "cpu"):
    """Instantiate the MACE calculator used for every reference energy.

    A single calculator instance is shared across all elements so the
    references are guaranteed self-consistent.
    """
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=[str(checkpoint_path)], device=device)


def _energy_per_atom(calculator, atoms) -> float:
    """Total energy of an ASE Atoms object divided by its atom count."""
    atoms = atoms.copy()
    atoms.calc = calculator
    total = float(atoms.get_potential_energy())
    return total / len(atoms)


# ---------------------------------------------------------------------------
# Reference structures
# ---------------------------------------------------------------------------

def _diatomic_reference_atoms(symbol: str, bond_length: float):
    """Build a diatomic molecule in a large cubic vacuum box.

    Args:
        symbol: Element symbol (e.g. "O")
        bond_length: Equilibrium bond length in Angstrom

    Returns:
        ASE Atoms object with the molecule centred in the box
    """
    from ase import Atoms

    half = VACUUM_BOX_SIZE / 2.0
    positions = [
        (half - bond_length / 2.0, half, half),
        (half + bond_length / 2.0, half, half),
    ]
    atoms = Atoms(
        symbols=f"{symbol}2",
        positions=positions,
        cell=[VACUUM_BOX_SIZE] * 3,
        pbc=True,
    )
    return atoms


def _fetch_elemental_ground_state(symbol: str, api_key: str):
    """Fetch the most stable elemental crystal for `symbol` from MP.

    Picks the entry with the lowest energy_above_hull in the single-element
    chemical system, which is the condensed-phase ground state by MP's own
    ranking.

    Args:
        symbol: Element symbol
        api_key: Materials Project API key

    Returns:
        Tuple of (pymatgen Structure, mpid string)

    Raises:
        RuntimeError: if no elemental entry is found
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=symbol,
            fields=["material_id", "energy_above_hull", "formula_pretty"],
        )
        if not docs:
            raise RuntimeError(f"No elemental MP entry found for {symbol}")

        best = min(
            docs,
            key=lambda d: (
                d.energy_above_hull if d.energy_above_hull is not None else float("inf")
            ),
        )
        structure = mpr.get_structure_by_material_id(best.material_id)
        return structure, str(best.material_id)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_references(
    elements: list[str],
    checkpoint_path: Path = DEFAULT_MACE_CHECKPOINT,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compute a MACE reference energy per atom for each element.

    Args:
        elements: Element symbols to compute references for
        checkpoint_path: MACE checkpoint (must match agent/predictor.py's)
        device: Torch device string

    Returns:
        Dict suitable for JSON serialization, keyed by element symbol plus
        a `_meta` block recording provenance.
    """
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")

    calculator = _make_calculator(checkpoint_path, device=device)

    refs: dict[str, Any] = {}
    failures: list[str] = []

    for symbol in elements:
        try:
            if symbol in DIATOMIC_GAS_REFERENCES:
                bond = DIATOMIC_GAS_REFERENCES[symbol]
                atoms = _diatomic_reference_atoms(symbol, bond)
                e_per_atom = _energy_per_atom(calculator, atoms)
                refs[symbol] = {
                    "energy_per_atom_eV": e_per_atom,
                    "reference_state": "molecule",
                    "reference_detail": f"{symbol}2 in {VACUUM_BOX_SIZE} A cubic vacuum box, bond {bond} A",
                    "mpid": None,
                }
                print(f"  {symbol}: {e_per_atom:+.4f} eV/atom  (molecular {symbol}2 reference)")
            else:
                if not api_key:
                    raise RuntimeError("MP_API_KEY not set; needed for crystalline references")
                structure, mpid = _fetch_elemental_ground_state(symbol, api_key)
                atoms = structure.to_ase_atoms()
                e_per_atom = _energy_per_atom(calculator, atoms)
                refs[symbol] = {
                    "energy_per_atom_eV": e_per_atom,
                    "reference_state": "crystal",
                    "reference_detail": f"lowest-e_above_hull elemental crystal from MP ({mpid})",
                    "mpid": mpid,
                }
                print(f"  {symbol}: {e_per_atom:+.4f} eV/atom  (crystal {mpid})")

        except Exception as exc:
            print(f"  {symbol}: FAILED -- {type(exc).__name__}: {exc}")
            failures.append(symbol)

    refs["_meta"] = {
        "generated": datetime.now().isoformat(),
        "mace_checkpoint": str(Path(checkpoint_path).name),
        "device": device,
        "vacuum_box_size_A": VACUUM_BOX_SIZE,
        "n_elements": len([k for k in refs if not k.startswith("_")]),
        "failed_elements": failures,
        "note": (
            "Energies are MACE-evaluated and self-consistent with each other. "
            "Oxide formation energies derived from these references carry a "
            "systematic offset vs Materials Project's formation_energy_per_atom, "
            "which applies empirical anion corrections. Report oxide and "
            "non-oxide comparisons separately."
        ),
    }

    return refs


# ---------------------------------------------------------------------------
# Load / apply
# ---------------------------------------------------------------------------

def load_references(refs_path: Path = DEFAULT_REFS_PATH) -> dict[str, float]:
    """Load the cached reference energies as a plain symbol -> eV/atom map.

    Args:
        refs_path: Path to mace_elemental_refs.json

    Returns:
        Dict mapping element symbol to reference energy per atom

    Raises:
        FileNotFoundError: if the cache has not been built yet
    """
    path = Path(refs_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Elemental references not found at {path}. "
            f"Run: python models/elemental_references.py"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {
        k: v["energy_per_atom_eV"]
        for k, v in blob.items()
        if not k.startswith("_")
    }


def formation_energy_per_atom(
    energy_per_atom: float,
    element_counts: dict[str, int],
    references: dict[str, float],
) -> float:
    """Convert a MACE total-energy-per-atom into a formation energy per atom.

        E_f = E_material_per_atom - sum_i( x_i * E_ref_i )

    Args:
        energy_per_atom: MACE total energy per atom for the material (eV/atom)
        element_counts: Per-element atom counts in the cell, e.g. {"Ni": 2, "P": 1}
        references: Symbol -> reference energy per atom (from load_references)

    Returns:
        Formation energy per atom in eV/atom

    Raises:
        KeyError: if an element has no cached reference energy
    """
    total_atoms = sum(element_counts.values())
    if total_atoms == 0:
        raise ValueError("element_counts is empty")

    missing = [s for s in element_counts if s not in references]
    if missing:
        raise KeyError(
            f"No reference energy for element(s): {missing}. "
            f"Rebuild with: python models/elemental_references.py"
        )

    reference_sum = sum(
        (count / total_atoms) * references[symbol]
        for symbol, count in element_counts.items()
    )
    return energy_per_atom - reference_sum


def contains_oxygen(element_counts: dict[str, int]) -> bool:
    """True if the composition contains oxygen.

    Used to split comparison reporting: oxide formation energies carry the
    anion-correction offset described in this module's docstring.
    """
    return "O" in element_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build MACE elemental reference energies for formation-energy calculation"
    )
    parser.add_argument(
        "--elements",
        nargs="*",
        default=None,
        help="Element symbols to compute (default: all elements present in kg.json)",
    )
    parser.add_argument(
        "--kg-path",
        type=str,
        default=str(DEFAULT_KG_JSON),
        help="Path to kg.json, used to discover which elements are needed",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_REFS_PATH),
        help="Output JSON path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_MACE_CHECKPOINT),
        help="MACE checkpoint path (must match agent/predictor.py)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (cpu / cuda)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the output file already exists",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"References already exist at {out_path}. Use --force to recompute.")
        return 0

    if args.elements:
        elements = sorted(args.elements)
    else:
        kg_path = Path(args.kg_path)
        if not kg_path.exists():
            print(f"[ERROR] KG not found at {kg_path}; build it first or pass --elements")
            return 1
        elements = elements_from_kg(kg_path)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[ERROR] MACE checkpoint not found: {checkpoint}")
        return 1

    print("=" * 60)
    print("MACE Elemental Reference Energies")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint.name}")
    print(f"Elements ({len(elements)}): {', '.join(elements)}")
    print()

    refs = build_references(elements, checkpoint_path=checkpoint, device=args.device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(refs, indent=2), encoding="utf-8")

    failed = refs["_meta"]["failed_elements"]
    print()
    print(f"Wrote {out_path}")
    print(f"Computed {refs['_meta']['n_elements']} reference energies")
    if failed:
        print(f"WARNING: failed for {failed} -- materials containing these "
              f"cannot have formation energies computed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
