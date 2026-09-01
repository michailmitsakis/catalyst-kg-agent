"""MACE surrogate predictor for catalyst candidate ranking.

Uses the MACE foundation checkpoint (mace-mpa-0-medium) to score candidate
materials cheaply, and reports a per-material trust signal used by the
Critic to decide whether an expensive high-fidelity check is warranted.

Cost: SURROGATE_COST = 5.0 per prediction.

WHAT THIS PREDICTOR RETURNS
---------------------------
`PredictorResult` carries two energy quantities, both MACE-derived:

    property_value  -- MACE TOTAL energy per atom (eV/atom). Raw model
                       output. Not comparable across compositions.
    formation_energy_per_atom
                    -- The same energy converted to a formation energy using
                       cached MACE elemental references (see
                       models/elemental_references.py):
                           E_f = E_per_atom - sum_i( x_i * E_ref_i )
                       This IS comparable across compositions, and is the
                       quantity the CGCNN baseline is trained on and
                       compared against.

Neither is e_above_hull. A true e_above_hull requires a convex-hull
calculation against competing phases in the chemical system (MP
phase-diagram data), which this predictor does not fetch. The stability
gate in agent/critic.py sources e_above_hull from the KG's MP-derived
value instead.

THE TRUST SIGNAL: max_residual_force
------------------------------------
`max_residual_force` (eV/Angstrom) is the largest force MACE predicts on
any atom in the structure, and it is what drives Critic escalation.

Why this is meaningful: every structure in the corpus is a DFT-relaxed
Materials Project geometry, so by construction DFT's own forces on those
atoms are approximately zero. If MACE predicts a LARGE residual force on
the same geometry, MACE and DFT disagree about where the atoms belong --
a direct, per-material, physically interpretable measure of how far this
structure sits from where the surrogate is reliable. Residual force is the
standard practitioner check for whether an MLIP can be trusted on a given
structure.

    small force  -> MACE agrees with the DFT geometry; trust the estimate
    large force  -> MACE disagrees; escalate to a higher-fidelity check

WHAT THIS REPLACED, AND WHY
---------------------------
Earlier versions claimed a Monte-Carlo-Dropout uncertainty: the same
deterministic MACE forward pass was run N times and the standard deviation
taken. MACE inference has no active dropout, so all N passes returned bit-
identical energies and the reported uncertainty was exactly 0.0 for every
material, always. The Critic's gate could therefore never fire on any
input. That number is gone rather than repaired -- a metric that is
structurally always zero is worse than no metric, because it looks like
evidence.

If the elemental reference cache is missing, predictions still succeed:
`formation_energy_per_atom` is set to None and a one-time warning is
printed. Run `python models/elemental_references.py` to populate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
from pydantic import BaseModel

from kg.schema import MaterialNode
from kg.graph_store import load_graph, rehydrate_node, DEFAULT_KG_JSON


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MACE_CHECKPOINT_PATH = Path("models/mace-mpa-0-medium.model")

# Force magnitude reported when a prediction fails, chosen to sit far above
# any plausible gate so failures escalate rather than pass silently.
FAILED_PREDICTION_FORCE = 99.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictorResult(BaseModel):
    """Structured prediction result.

    property_value: MACE total energy per atom (eV/atom). Raw model output;
        NOT e_above_hull and NOT comparable across compositions.
    formation_energy_per_atom: property_value converted to a formation
        energy via cached MACE elemental references (eV/atom). None if the
        reference cache is unavailable or an element reference is missing.
    max_residual_force: largest per-atom force magnitude MACE predicts on
        this (DFT-relaxed) geometry, in eV/Angstrom. The Critic's escalation
        signal -- see module docstring.
    """
    material_id: str
    property_value: Optional[float]
    formation_energy_per_atom: Optional[float] = None
    max_residual_force: float
    model_used: Literal["mace"]
    prediction_failed: bool


# ---------------------------------------------------------------------------
# MACE surrogate wrapper
# ---------------------------------------------------------------------------

class MACEModel:
    """MACE model inference wrapper."""

    def __init__(self, checkpoint_path: Path = MACE_CHECKPOINT_PATH):
        """Initialize MACE model from checkpoint.

        Args:
            checkpoint_path: Path to MACE model checkpoint file
        """
        self.checkpoint_path = checkpoint_path
        self.calculator = None  # Lazy load to avoid import until first use
        self.device = "cpu"  # Default; can be overridden via env later

    def _load_model(self):
        """Load model lazily on first inference."""
        if self.calculator is not None:
            return

        try:
            from mace.calculators import MACECalculator

            # NOTE: MACE API uses 'model_paths' (list), not 'checkpoint_path'
            self.calculator = MACECalculator(
                model_paths=[str(self.checkpoint_path)],
                device=self.device,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model: {e}")

    def predict_energy_and_forces(self, atoms) -> tuple[float, float]:
        """Evaluate one structure: energy per atom and max residual force.

        Both come from a single MACE evaluation, so the force signal costs
        no extra inference.

        Args:
            atoms: ASE Atoms object (periodic; see PredictorAgent.predict)

        Returns:
            (energy_per_atom in eV/atom, max residual force in eV/Angstrom)
        """
        self._load_model()

        try:
            n_atoms = len(atoms)
            if n_atoms == 0:
                raise ValueError("structure has no atoms")

            atoms = atoms.copy()
            atoms.calc = self.calculator

            total_energy = float(atoms.get_potential_energy())
            # forces: (N, 3) -> per-atom magnitude -> worst atom in the cell
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))

            return total_energy / n_atoms, max_force
        except Exception as e:
            raise RuntimeError(f"MACE prediction failed: {e}")


# ---------------------------------------------------------------------------
# Predictor agent
# ---------------------------------------------------------------------------

class PredictorAgent:
    """MACE-based energy predictor with a residual-force trust signal.

    Returns raw energy per atom (ranking signal), formation energy per atom
    when elemental references are available (comparable across
    compositions), and the max residual force that drives Critic
    escalation. Does NOT compute e_above_hull -- see module docstring.
    """

    # Class-level so the "references missing" warning prints once per process
    # rather than once per material in a 130-material campaign.
    _warned_missing_refs = False

    def __init__(self, checkpoint_path: Path = MACE_CHECKPOINT_PATH):
        """Initialize predictor.

        Args:
            checkpoint_path: Path to MACE model checkpoint
        """
        self.model = MACEModel(checkpoint_path=checkpoint_path)
        self._references: Optional[dict[str, float]] = None
        self._references_loaded = False

    def _get_references(self) -> Optional[dict[str, float]]:
        """Load cached MACE elemental references once, tolerating absence.

        Returns:
            Symbol -> reference energy per atom, or None if unavailable
        """
        if self._references_loaded:
            return self._references

        self._references_loaded = True
        try:
            from models.elemental_references import load_references
            self._references = load_references()
        except Exception as exc:
            if not PredictorAgent._warned_missing_refs:
                print(
                    f"Predictor: elemental references unavailable ({exc}). "
                    f"formation_energy_per_atom will be None. "
                    f"Run: python models/elemental_references.py"
                )
                PredictorAgent._warned_missing_refs = True
            self._references = None
        return self._references

    def _formation_energy(self, energy_per_atom: float, ase_atoms) -> Optional[float]:
        """Convert raw energy per atom to formation energy per atom.

        Args:
            energy_per_atom: MACE total energy per atom (eV/atom)
            ase_atoms: The ASE Atoms the energy was computed on, used for
                the composition

        Returns:
            Formation energy per atom (eV/atom), or None if references are
            missing or incomplete for this composition
        """
        references = self._get_references()
        if not references:
            return None

        try:
            from collections import Counter
            from models.elemental_references import formation_energy_per_atom

            element_counts = dict(Counter(ase_atoms.get_chemical_symbols()))
            return formation_energy_per_atom(
                energy_per_atom=energy_per_atom,
                element_counts=element_counts,
                references=references,
            )
        except KeyError as exc:
            if not PredictorAgent._warned_missing_refs:
                print(f"Predictor: {exc}")
                PredictorAgent._warned_missing_refs = True
            return None
        except Exception:
            return None

    def _failed(self, mpid: str) -> PredictorResult:
        """Uniform failure result.

        max_residual_force is set high so a failed prediction escalates
        rather than slipping through the Critic's gate as trustworthy.
        """
        return PredictorResult(
            material_id=mpid,
            property_value=None,
            formation_energy_per_atom=None,
            max_residual_force=FAILED_PREDICTION_FORCE,
            model_used="mace",
            prediction_failed=True,
        )

    def predict(self, material: MaterialNode) -> PredictorResult:
        """Predict MACE energies and the residual-force trust signal.

        Args:
            material: MaterialNode from KG (structure_id is populated by
                kg.graph_store.rehydrate_node via the HAS_STRUCTURE edge)

        Returns:
            PredictorResult with energy per atom, formation energy per atom
            (when references are available), and max residual force
        """
        if not material.structure_id:
            return self._failed(material.mpid)

        try:
            # structure_id is like "structure:mp-126", not a file path, so
            # resolve it through the canonical KG.
            G = load_graph(DEFAULT_KG_JSON)
            structure_node = rehydrate_node(G, material.structure_id)

            from pymatgen.core import Structure as PMGStructure

            pmg_structure = PMGStructure.from_file(structure_node.cif_path)

            # Convert pymatgen Structure to ASE Atoms (what MACE expects).
            #
            # Do NOT re-wrap into a fresh ase.Atoms here. pymatgen returns
            # MSONAtoms, which IS an ase.Atoms subclass and is accepted by
            # MACECalculator directly. An earlier version rebuilt it as
            #     AseAtoms(symbols=..., positions=..., cell=...)
            # which silently dropped periodic boundary conditions (ASE
            # defaults pbc=False). MACE then evaluated each crystal as an
            # isolated cluster in vacuum: surface atoms are under-
            # coordinated, so energies came out ~1.5 eV/atom too high and
            # formation energies came out POSITIVE for hull-stable
            # compounds. models/elemental_references.py never re-wrapped, so
            # the references stayed correct and the mismatch was the whole
            # error.
            ase_atoms = pmg_structure.to_ase_atoms()

            # Periodicity is required: these are bulk crystals, not clusters.
            if not all(ase_atoms.get_pbc()):
                ase_atoms.set_pbc(True)

            if len(ase_atoms) == 0:
                return self._failed(material.mpid)

        except Exception as e:
            import traceback
            print(f"Predictor error loading structure: {e}")
            traceback.print_exc()
            return self._failed(material.mpid)

        try:
            # One MACE evaluation yields both the energy and the forces.
            energy_per_atom, max_force = self.model.predict_energy_and_forces(ase_atoms)

            # Convert to formation energy using cached MACE elemental
            # references. Both terms come from the same checkpoint, so the
            # subtraction is self-consistent. None if references are missing.
            formation_e = self._formation_energy(energy_per_atom, ase_atoms)

            return PredictorResult(
                material_id=material.mpid,
                property_value=energy_per_atom,
                formation_energy_per_atom=formation_e,
                max_residual_force=max_force,
                model_used="mace",
                prediction_failed=False,
            )

        except Exception:
            return self._failed(material.mpid)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_predictor(checkpoint_path: Path = MACE_CHECKPOINT_PATH) -> PredictorAgent:
    """Factory function.

    Args:
        checkpoint_path: Path to MACE model checkpoint

    Returns:
        Configured PredictorAgent instance
    """
    return PredictorAgent(checkpoint_path=checkpoint_path)


__all__ = [
    "PredictorResult",
    "MACEModel",
    "PredictorAgent",
    "create_predictor",
    "FAILED_PREDICTION_FORCE",
]