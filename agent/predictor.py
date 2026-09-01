"""MACE surrogate predictor for catalyst candidate ranking.

Uses MACE model (mace-mpa-0 foundation checkpoint) with Monte Carlo Dropout
for uncertainty quantification. Single-material inference only (batching added later).

Cost: SURROGATE_COST = 5.0 per prediction.

What this predictor returns
---------------------------
`PredictorResult` carries two energy quantities, both MACE-derived:

    property_value  -- MC-Dropout-mean MACE TOTAL energy per atom (eV/atom).
                       Raw model output. Not comparable across compositions.
    formation_energy_per_atom
                    -- The same energy converted to a formation energy using
                       cached MACE elemental references (see
                       models/elemental_references.py):
                           E_f = E_per_atom - sum_i( x_i * E_ref_i )
                       This IS comparable across compositions, and is the
                       quantity the CGCNN baseline is trained on and compared
                       against.

Neither is e_above_hull. A true e_above_hull requires a convex-hull
calculation against competing phases in the chemical system (MP
phase-diagram data), which this predictor does not fetch. The stability
gate in agent/critic.py correctly sources e_above_hull from the KG's
MP-derived value instead of from this predictor.

`uncertainty` is the MC-Dropout standard deviation of the raw energy per
atom. It is uncertainty in MACE's OWN prediction (i.e. "is this structure
out-of-distribution for the checkpoint?"), not uncertainty about the
material's stability. High variance is what triggers Critic escalation.

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
N_DROPOUT_PASSES = 5  # MC Dropout passes for uncertainty estimate


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictorResult(BaseModel):
    """Structured prediction result.

    property_value: MC-Dropout-mean MACE total energy per atom (eV/atom).
        Raw model output; NOT e_above_hull and NOT comparable across
        compositions -- see module docstring.
    formation_energy_per_atom: property_value converted to a formation
        energy via cached MACE elemental references (eV/atom). None if the
        reference cache is unavailable or an element reference is missing.
    uncertainty: MC-Dropout std dev of the raw energy per atom (eV/atom).
    """
    material_id: str
    property_value: Optional[float]
    formation_energy_per_atom: Optional[float] = None
    uncertainty: float
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

            # Initialize MACE calculator with checkpoint path and device
            # NOTE: MACE API uses 'model_paths' (list) instead of 'checkpoint_path'
            self.calculator = MACECalculator(
                model_paths=[str(self.checkpoint_path)],  # List of checkpoint paths
                device=self.device,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model: {e}")

    def predict_energy_per_atom(self, structure) -> float:
        """Predict total energy per atom for a single structure.

        Args:
            structure: ASE Atoms object

        Returns:
            Predicted total energy per atom in eV/atom
        """
        self._load_model()

        try:
            # Use MACE calculator to compute total energy, divide by num_atoms
            # NOTE: ASE Atoms uses 'num_atoms', pymatgen Structure uses 'num_formulas'
            # MSONAtoms (pymatgen wrapper) uses 'get_number_of_atoms()'
            if hasattr(structure, 'num_atoms'):
                # ASE Atoms object
                n_atoms = structure.num_atoms
            elif hasattr(structure, 'get_number_of_atoms'):
                # pymatgen MSONAtoms or similar wrapper
                n_atoms = structure.get_number_of_atoms()
            elif hasattr(structure, 'num_formulas'):
                # pymatgen Structure object
                n_atoms = structure.num_formulas
            else:
                raise AttributeError(f"Structure has neither num_atoms nor get_number_of_atoms")

            total_energy = self.calculator.get_potential_energy(structure)
            return float(total_energy / n_atoms)
        except Exception as e:
            raise RuntimeError(f"MACE prediction failed: {e}")


# ---------------------------------------------------------------------------
# Predictor agent
# ---------------------------------------------------------------------------

class PredictorAgent:
    """MACE-based energy predictor with MC Dropout uncertainty.

    Returns raw energy per atom (ranking signal + uncertainty carrier) and,
    when elemental references are available, a formation energy per atom
    that is comparable across compositions. Does NOT compute e_above_hull --
    see module docstring.
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
        self.n_dropout_passes = N_DROPOUT_PASSES
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

    def _formation_energy(
        self,
        energy_per_atom: float,
        ase_atoms,
    ) -> Optional[float]:
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
            # An element in this material has no cached reference.
            if not PredictorAgent._warned_missing_refs:
                print(f"Predictor: {exc}")
                PredictorAgent._warned_missing_refs = True
            return None
        except Exception:
            return None

    def predict(self, material: MaterialNode) -> PredictorResult:
        """Predict MACE energies for a single material.

        Args:
            material: MaterialNode from KG (may have structure_id reference to StructureNode)

        Returns:
            PredictorResult with raw energy per atom, formation energy per
            atom (when references are available), and MC-Dropout uncertainty
        """
        # Fetch structure from graph using structure_id field
        # If structure_id is None or empty, prediction will fail gracefully
        if not material.structure_id:
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
                formation_energy_per_atom=None,
                uncertainty=1.0,  # High uncertainty when no structure
                model_used="mace",
                prediction_failed=True,
            )

        try:
            # Step 1: Load the graph and fetch the structure node
            # The structure_id is like "structure:mp-126", not a file path
            # We need to load from the canonical KG location
            G = load_graph(DEFAULT_KG_JSON)
            structure_node = rehydrate_node(G, material.structure_id)

            # Convert StructureNode (pydantic) to pymatgen Structure object
            # The StructureNode has cif_path which points to the actual CIF file
            from pymatgen.core import Structure as PMGSstructure

            pmg_structure = PMGSstructure.from_file(structure_node.cif_path)

            # Convert pymatgen Structure to ASE Atoms (what MACE expects)
            from ase.atoms import Atoms as AseAtoms

            # Handle both pure ASE Atoms and pymatgen's MSONAtoms wrapper
            if hasattr(pmg_structure, 'to_ase_atoms'):
                ase_atoms = pmg_structure.to_ase_atoms()
                # If it's an MSONAtoms (pymatgen wrapper), convert to pure ASE Atoms
                if type(ase_atoms).__name__ == 'MSONAtoms':
                    # MSONAtoms inherits from Atoms, but we need the underlying ASE object
                    ase_atoms = AseAtoms(symbols=ase_atoms.get_chemical_symbols(),
                                       positions=ase_atoms.get_positions(),
                                       cell=ase_atoms.get_cell())
            else:
                # Fallback: try direct conversion
                ase_atoms = pmg_structure.to_ase_atoms()

            if not ase_atoms or len(ase_atoms) == 0:
                return PredictorResult(
                    material_id=material.mpid,
                    property_value=None,
                    formation_energy_per_atom=None,
                    uncertainty=1.0,
                    model_used="mace",
                    prediction_failed=True,
                )

        except Exception as e:
            import traceback
            print(f"Predictor error loading structure: {e}")
            traceback.print_exc()
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
                formation_energy_per_atom=None,
                uncertainty=1.0,
                model_used="mace",
                prediction_failed=True,
            )

        try:
            # Run MC Dropout inference to get energy-per-atom predictions.
            # property_value is the mean of these passes -- MACE total energy
            # per atom (eV/atom). This is NOT e_above_hull (see module
            # docstring): no convex-hull / competing-phase calculation is
            # performed here. Stability is sourced from the KG's MP-derived
            # e_above_hull in agent/critic.py, not from this value.
            energies_per_atom = []
            for _ in range(self.n_dropout_passes):
                e_per_atom = self.model.predict_energy_per_atom(ase_atoms)
                energies_per_atom.append(e_per_atom)

            if not energies_per_atom:
                return PredictorResult(
                    material_id=material.mpid,
                    property_value=None,
                    formation_energy_per_atom=None,
                    uncertainty=1.0,
                    model_used="mace",
                    prediction_failed=True,
                )

            avg_energy_per_atom = float(np.mean(energies_per_atom))
            std_dev = float(np.std(energies_per_atom))

            # Convert to formation energy using cached MACE elemental
            # references. Both terms come from the same checkpoint, so the
            # subtraction is self-consistent. None if references are missing.
            formation_e = self._formation_energy(avg_energy_per_atom, ase_atoms)

            return PredictorResult(
                material_id=material.mpid,
                property_value=avg_energy_per_atom,
                formation_energy_per_atom=formation_e,
                uncertainty=std_dev,
                model_used="mace",
                prediction_failed=False,
            )

        except Exception as e:
            # Handle any unexpected errors
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
                formation_energy_per_atom=None,
                uncertainty=1.0,
                model_used="mace",
                prediction_failed=True,
            )


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
]