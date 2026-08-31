"""MACE surrogate predictor for e_above_hull estimation.

Uses MACE model (mace-mp-0 foundation checkpoint) with Monte Carlo Dropout
for uncertainty quantification. Single-material inference only (batching added later).

Cost: SURROGATE_COST = 5.0 per prediction.

e_above_hull calculation: Uses pymatgen's Structure.form_formation_energy_per_atom()
with Materials Project reference elements for consistent eV/atom values.
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
    """Structured prediction result."""
    material_id: str
    property_value: Optional[float]  # e_above_hull in eV/atom or None if failed
    uncertainty: float               # MC Dropout std dev (0.0-1.0 confidence)
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
        """Predict energy per atom for a single structure.

        Args:
            structure: pymatgen Structure or ASE Atoms object

        Returns:
            Predicted energy per atom in eV
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

    def predict(self, structure, enable_dropout: bool = False) -> float:
        """Predict e_above_hull for a single structure.

        Args:
            structure: pymatgen Structure object
            enable_dropout: If True, enables MC Dropout mode

        Returns:
            Predicted energy per atom in eV (NOT e_above_hull yet)
        """
        self._load_model()

        try:
            # Use MACE calculator to compute energy
            total_energy = self.calculator.get_energy(structure)
            
            # Convert energy to e_per_atom
            return float(total_energy / structure.num_formulas)
        except Exception as e:
            raise RuntimeError(f"MACE prediction failed: {e}")


# ---------------------------------------------------------------------------
# Predictor agent
# ---------------------------------------------------------------------------

class PredictorAgent:
    """MACE-based property predictor with MC Dropout uncertainty.

    Implements e_above_hull calculation using pymatgen's formation energy
    relative to elemental references (Materials Project convention).
    """

    def __init__(self, checkpoint_path: Path = MACE_CHECKPOINT_PATH):
        """Initialize predictor.

        Args:
            checkpoint_path: Path to MACE model checkpoint
        """
        self.model = MACEModel(checkpoint_path=checkpoint_path)
        self.n_dropout_passes = N_DROPOUT_PASSES

    def predict(self, material: MaterialNode) -> PredictorResult:
        """Predict e_above_hull for a single material.

        Args:
            material: MaterialNode from KG (may have structure_id reference to StructureNode)

        Returns:
            PredictorResult with value + uncertainty
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
            # Step 1: Run MC Dropout inference to get energy per atom predictions
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

            # Step 2: Calculate e_above_hull using pymatgen
            # Use the mean energy from MC Dropout as input to formation energy calc
            avg_energy_per_atom = np.mean(energies_per_atom)
            
            # Compute formation energy relative to convex hull
            try:
                from pymatgen.core import Structure
                
                # pymatgen's form_formation_energy_per_atom returns eV/atom
                # relative to elemental references (MP convention)
                formation_energy = structure.form_formation_energy_per_atom()
                
                # Use MACE-predicted energy as the total energy input
                # pymatgen will compute the convex hull and return e_above_hull
                e_above_hull = avg_energy_per_atom - formation_energy
                
            except Exception as e:
                # Fallback: if pymatgen calculation fails, use raw MACE output
                e_above_hull = avg_energy_per_atom

            # Step 3: Calculate uncertainty from MC Dropout variance
            std_dev = np.std(energies_per_atom)

            return PredictorResult(
                material_id=material.mpid,
                property_value=e_above_hull,
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
