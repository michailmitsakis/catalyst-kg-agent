"""Test MACE model loading and basic prediction.

Tests that:
1. MACE checkpoint is found at expected path
2. MACE calculator loads successfully
3. Single-material prediction works end-to-end
4. Formation energy is computed from cached elemental references
5. The residual-force trust signal is produced and is physically sane

Note: there is no MC-Dropout uncertainty. MACE inference is deterministic
with no active dropout, so the previous "uncertainty" was identically 0.0
for every material. The escalation signal is max residual force instead --
see agent/predictor.py.

Run with: python tests/test_predictor.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.graph_store import load_graph, rehydrate_node
from agent.predictor import PredictorAgent, MACE_CHECKPOINT_PATH


def test_mace_checkpoint_exists():
    """Test that MACE checkpoint file exists."""
    print("Testing MACE checkpoint existence...")
    
    assert MACE_CHECKPOINT_PATH.exists(), \
        f"MACE checkpoint not found at {MACE_CHECKPOINT_PATH}"
    
    checkpoint_size = MACE_CHECKPOINT_PATH.stat().st_size
    print(f"[PASS] Checkpoint exists: {MACE_CHECKPOINT_PATH.name} ({checkpoint_size:,} bytes)")


def test_predictor_initialization():
    """Test that PredictorAgent initializes successfully."""
    print("\nTesting PredictorAgent initialization...")
    
    try:
        predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
        print(f"[PASS] PredictorAgent initialized with checkpoint: {MACE_CHECKPOINT_PATH}")
    except Exception as e:
        print(f"[FAIL] Failed to initialize PredictorAgent: {e}")
        raise


def test_single_material_prediction():
    """Test prediction on a single material from the KG."""
    print("\nTesting single-material prediction...")
    
    # Load KG and get first material
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    mat_nodes = [nid for nid, data in G.nodes(data=True) 
                 if data.get("type") == "Material"]
    
    assert len(mat_nodes) > 0, "No materials found in KG"
    
    first_mat_id = mat_nodes[0]
    mat_node = rehydrate_node(G, first_mat_id)
    
    print(f"Testing with: {mat_node.formula_pretty} ({mat_node.mpid})")
    print(f"Elements: {mat_node.elements}")
    
    # Check if structure_id is available (updated schema)
    if not hasattr(mat_node, 'structure_id') or mat_node.structure_id is None:
        print("[WARN] Material has no structure_id - skipping prediction test")
        return
    
    # Run prediction
    try:
        predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
        result = predictor.predict(mat_node)
        
        print(f"\nPrediction result:")
        print(f"  Energy per atom      : {result.property_value:.4f} eV/atom")
        if result.formation_energy_per_atom is not None:
            print(f"  Formation energy     : {result.formation_energy_per_atom:+.4f} eV/atom")
        else:
            print(f"  Formation energy     : None (run models/elemental_references.py)")
        print(f"  Max residual force   : {result.max_residual_force:.4f} eV/A")
        print(f"  Model used           : {result.model_used}")
        print(f"  Failed               : {result.prediction_failed}")

        assert not result.prediction_failed, "Prediction failed unexpectedly"
        assert result.property_value is not None, "No energy value returned"
        assert result.max_residual_force >= 0, "Force magnitude must be non-negative"

        print("[PASS] Single-material prediction successful!")
        
    except ImportError as e:
        print(f"\n[FAIL] MACE not installed: {e}")
        print("\nInstall with: pip install mace-torch")
        print("\nOr from source:")
        print("  git clone https://github.com/ACE-Modeling/MACE.git")
        print("  cd MACE && pip install -e .")
        raise
    except Exception as e:
        print(f"\n[FAIL] Error during prediction: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_residual_force_signal():
    """Test that the residual-force trust signal is produced and sane.

    These are DFT-relaxed MP geometries, so DFT's own forces on them are
    ~0. MACE's residual force measures its disagreement with that geometry.
    A value of exactly 0.0 across materials would indicate the signal is not
    actually being computed -- which is the failure mode that made the
    previous MC-Dropout "uncertainty" useless.
    """
    print("\nTesting residual-force trust signal...")
    
    # Load KG and get first material with structure
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    mat_nodes = [nid for nid, data in G.nodes(data=True) 
                 if data.get("type") == "Material"]
    
    first_mat_id = mat_nodes[0]
    mat_node = rehydrate_node(G, first_mat_id)
    
    if not hasattr(mat_node, 'structure_id') or mat_node.structure_id is None:
        print("[WARN] Material has no structure_id - skipping force test")
        return

    predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
    result = predictor.predict(mat_node)

    assert result.max_residual_force >= 0, "Force magnitude must be non-negative"
    assert result.max_residual_force < 50, (
        f"Residual force {result.max_residual_force:.2f} eV/A is implausibly large "
        f"for a relaxed structure -- check that periodic boundary conditions "
        f"survived the pymatgen -> ASE conversion"
    )

    print(f"  Max residual force: {result.max_residual_force:.4f} eV/A")
    print("[PASS] Residual-force signal produced")


def test_formation_energy():
    """Test that formation energy is computed from cached elemental references.

    A formation energy near the raw energy per atom (around -6 eV/atom for
    this corpus) means the reference subtraction did not happen. Correct
    values sit roughly in -2 to +1 eV/atom.
    """
    print("\nTesting formation energy calculation...")

    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)

    mat_nodes = [nid for nid, data in G.nodes(data=True)
                 if data.get("type") == "Material"]
    mat_node = rehydrate_node(G, mat_nodes[0])

    predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
    result = predictor.predict(mat_node)

    if result.formation_energy_per_atom is None:
        print("[SKIP] No elemental references cached.")
        print("       Run: python models/elemental_references.py")
        return

    e_f = result.formation_energy_per_atom
    print(f"  {mat_node.mpid} ({mat_node.formula_pretty})")
    print(f"    energy per atom : {result.property_value:.4f} eV/atom")
    print(f"    formation energy: {e_f:+.4f} eV/atom")

    assert -10.0 < e_f < 5.0, f"Formation energy {e_f} is outside any plausible range"
    if abs(e_f - result.property_value) < 0.1:
        print("[FAIL] Formation energy ~= raw energy: reference subtraction did not happen")
        raise AssertionError("Elemental reference subtraction appears to be a no-op")

    print("[PASS] Formation energy computed from elemental references")


def main():
    """Run all predictor tests."""
    print("="*60)
    print("MACE Predictor Test Suite")
    print("="*60)
    
    # Check if checkpoint exists first
    if not MACE_CHECKPOINT_PATH.exists():
        print(f"\n[FAIL] MACE checkpoint not found at: {MACE_CHECKPOINT_PATH}")
        print("Place mace-mpa-0-medium.model in models/")
        return
    
    print(f"[PASS] Checkpoint found: {MACE_CHECKPOINT_PATH}\n")
    
    # Run tests
    try:
        test_mace_checkpoint_exists()
        test_predictor_initialization()
        test_single_material_prediction()
        test_residual_force_signal()
        test_formation_energy()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed: {e}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()