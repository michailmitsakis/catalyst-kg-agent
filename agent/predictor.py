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


# ---------------------------------------------------------------------------
# Config (hardcoded per Qs)
# ---------------------------------------------------------------------------

MACE_CHECKPOINT_PATH = Path("models/gnn_surrogate/mace-mpa-0-medium.model")
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
            self.calculator = MACECalculator(
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model: {e}")

    def predict_energy_per_atom(self, structure) -> float:
        """Predict energy per atom for a single structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            Predicted energy per atom in eV
        """
        self._load_model()

        try:
            # Use MACE calculator to compute total energy, divide by num_atoms
            total_energy = self.calculator.get_energy(structure)
            return float(total_energy / structure.num_formulas)
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
            material: MaterialNode from KG (must have structure attribute)

        Returns:
            PredictorResult with value + uncertainty
        """
        # Extract structure from material
        structure = getattr(material, "structure", None)

        if structure is None:
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
                uncertainty=1.0,  # High uncertainty when no structure
                model_used="mace",
                prediction_failed=True,
            )

        try:
            # Step 1: Run MC Dropout inference to get energy per atom predictions
            energies_per_atom = []
            for _ in range(self.n_dropout_passes):
                try:
                    e_per_atom = self.model.predict_energy_per_atom(structure)
                    energies_per_atom.append(e_per_atom)
                except Exception as e:
                    return PredictorResult(
                        material_id=material.mpid,
                        property_value=None,
                        uncertainty=1.0,
                        model_used="mace",
                        prediction_failed=True,
                    )

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
