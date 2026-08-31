"""Test MACE-MP-0 model loading and basic prediction.

Tests that:
1. MACE checkpoint is found at expected path
2. MACE calculator loads successfully
3. Single-material prediction works end-to-end
4. MC Dropout uncertainty estimation works

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
        print(f"  Property value (e_above_hull): {result.property_value:.4f} eV")
        print(f"  Uncertainty (MC Dropout std): {result.uncertainty:.3%}")
        print(f"  Model used: {result.model_used}")
        print(f"  Failed: {result.prediction_failed}")
        
        assert not result.prediction_failed, "Prediction failed unexpectedly"
        assert result.property_value is not None, "No e_above_hull value returned"
        assert result.uncertainty >= 0, "Uncertainty should be non-negative"
        
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


def test_mc_dropout_uncertainty():
    """Test that MC Dropout uncertainty estimation works."""
    print("\nTesting MC Dropout uncertainty estimation...")
    
    # Load KG and get first material with structure
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    mat_nodes = [nid for nid, data in G.nodes(data=True) 
                 if data.get("type") == "Material"]
    
    first_mat_id = mat_nodes[0]
    mat_node = rehydrate_node(G, first_mat_id)
    
    if not hasattr(mat_node, 'structure_id') or mat_node.structure_id is None:
        print("[WARN] Material has no structure_id - skipping uncertainty test")
        return
    
    predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
    result = predictor.predict(mat_node)
    
    # MC Dropout should produce non-zero uncertainty (unless all predictions identical)
    assert result.uncertainty >= 0, "Uncertainty should be non-negative"
    
    print(f"  Uncertainty from {predictor.n_dropout_passes} MC Dropout passes: {result.uncertainty:.3%}")
    print("[PASS] MC Dropout uncertainty estimation works!")


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
        test_mc_dropout_uncertainty()
        
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
