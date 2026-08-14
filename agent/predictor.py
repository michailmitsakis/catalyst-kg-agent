"""MACE surrogate predictor for e_above_hull estimation.

Uses MACE model (mace-mp-0 foundation checkpoint) with Monte Carlo Dropout
for uncertainty quantification. Single-material inference only (batching added later).

Cost: SURROGATE_COST = 5.0 per prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import numpy as np

from kg.schema import MaterialNode


# ---------------------------------------------------------------------------
# Config (hardcoded per Qs)
# ---------------------------------------------------------------------------

MACE_CHECKPOINT_PATH = Path("models/gnn_surrogate/mace-mp-0.pth")
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
        self.model = None  # Lazy load to avoid import until first use
        self.device = "cpu"  # Default; can be overridden via env later

    def _load_model(self):
        """Load model lazily on first inference."""
        if self.model is not None:
            return

        try:
            from mace_mp.models import build_model, MACEModelWrapper
            from torch.utils.serialization import load

            # Load checkpoint (PyTorch format)
            state_dict = load(str(self.checkpoint_path))

            # Build model architecture
            model = build_model(
                config={
                    "num_tokens": 128,
                    "cutoff": 5.0,
                    "type_map": ["Fe", "Ni", "Cu", "Co", "Mn", "Zn", "Cr",
                                 "Ti", "Al", "V", "Mo", "W", "Pd", "Pt", "Au"],
                    "num_blocks": 5,
                }
            )

            # Load weights
            model.load_state_dict(state_dict)
            model.eval()

            self.model = MACEModelWrapper(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load MACE model: {e}")

    def predict(self, structure, enable_dropout: bool = False) -> float:
        """Predict e_above_hull for a single structure.

        Args:
            structure: pymatgen Structure object
            enable_dropout: If True, enables MC Dropout mode

        Returns:
            Predicted e_above_hull in eV/atom
        """
        self._load_model()

        if not hasattr(self.model, "predict_e_above_hull"):
            raise AttributeError("Model missing predict_e_above_hull method")

        try:
            # Use MACE's native inference API
            pred = self.model.predict_e_above_hull(structure, enable_dropout=enable_dropout)
            return float(pred)
        except Exception as e:
            raise RuntimeError(f"MACE prediction failed: {e}")


# ---------------------------------------------------------------------------
# Predictor agent
# ---------------------------------------------------------------------------

class PredictorAgent:
    """MACE-based property predictor with MC Dropout uncertainty."""

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

        # MC Dropout inference: multiple passes with dropout enabled
        predictions = []
        for _ in range(self.n_dropout_passes):
            try:
                pred = self.model.predict(structure, enable_dropout=True)
                predictions.append(float(pred))
            except Exception as e:
                return PredictorResult(
                    material_id=material.mpid,
                    property_value=None,
                    uncertainty=1.0,
                    model_used="mace",
                    prediction_failed=True,
                )

        if not predictions:
            return PredictorResult(
                material_id=material.mpid,
                property_value=None,
                uncertainty=1.0,
                model_used="mace",
                prediction_failed=True,
            )

        # Compute statistics
        value = np.mean(predictions)
        std_dev = np.std(predictions)  # MC Dropout variance as uncertainty

        return PredictorResult(
            material_id=material.mpid,
            property_value=value,
            uncertainty=std_dev,
            model_used="mace",
            prediction_failed=False,
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
