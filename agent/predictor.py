"""MACE surrogate predictor for catalyst candidate ranking.

Uses MACE model (mace-mp-0 foundation checkpoint) with Monte Carlo Dropout
for uncertainty quantification. Single-material inference only (batching added later).

Cost: SURROGATE_COST = 5.0 per prediction.

IMPORTANT — what `property_value` actually is:
    This is the MC-Dropout-mean MACE total energy per atom (eV/atom), NOT
    e_above_hull. A true e_above_hull requires a convex-hull calculation
    against competing phases in the chemical system (MP phase-diagram data),
    which this predictor does not fetch. The stability gate in agent/critic.py
    correctly sources e_above_hull from the KG's MP-derived value instead of
    from this predictor.

    This value's role in the pipeline is a cheap relative-ranking signal
    among candidates that already passed the (free, MP-sourced) stability
    gate, plus an uncertainty carrier: high MC-Dropout variance on this
    energy is what triggers Critic escalation (agent/critic.py), not a
    judgment about the material's stability.
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
        NOT e_above_hull -- see module docstring.
    """
    material_id: str
    property_value: Optional[float]  # MACE energy per atom (eV/atom), or None if failed
    uncertainty: float               # MC Dropout std dev (eV/atom)
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
    """MACE-based energy-per-atom predictor with MC Dropout uncertainty.

    Returns a cheap relative-ranking signal (MACE energy per atom) with an
    uncertainty estimate used to trigger Critic escalation. Does NOT compute
    e_above_hull -- see module docstring.
    """

    def __init__(self, checkpoint_path: Path = MACE_CHECKPOINT_PATH):
        """Initialize predictor.

        Args:
            checkpoint_path: Path to MACE model checkpoint
        """
        self.model = MACEModel(checkpoint_path=checkpoint_path)
        self.n_dropout_passes = N_DROPOUT_PASSES

    def predict(self, material: MaterialNode) -> PredictorResult:
        """Predict MACE energy per atom for a single material.

        Args:
            material: MaterialNode from KG (may have structure_id reference to StructureNode)

        Returns:
            PredictorResult with value (energy per atom, eV/atom) + uncertainty
        """
        # Fetch structure from graph using structure_id field
        # If structure_id is None or empty, prediction will fail gracefully
        if not material.structure_id:
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
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
                    uncertainty=1.0,
                    model_used="mace",
                    prediction_failed=True,
                )

            avg_energy_per_atom = float(np.mean(energies_per_atom))
            std_dev = float(np.std(energies_per_atom))

            return PredictorResult(
                material_id=material.mpid,
                property_value=avg_energy_per_atom,
                uncertainty=std_dev,
                model_used="mace",
                prediction_failed=False,
            )

        except Exception as e:
            # Handle any unexpected errors
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
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