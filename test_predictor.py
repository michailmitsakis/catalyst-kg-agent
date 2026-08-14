"""Test PredictorAgent with a real material from the KG."""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kg.graph_store import load_graph, rehydrate_node
from agent.predictor import PredictorAgent, MACE_CHECKPOINT_PATH


def main():
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    # Pick first material from the KG
    mat_nodes = [nid for nid, data in G.nodes(data=True) 
                 if data.get("type") == "Material"]
    
    if not mat_nodes:
        print("No materials in KG. Run `python kg/build_graph.py` first.")
        return
    
    # Use first material (Ni12P5 - mp-2790)
    first_mat_id = mat_nodes[0]
    mat_node = rehydrate_node(G, first_mat_id)
    
    print(f"Testing with: {mat_node.formula_pretty} ({mat_node.mpid})")
    print(f"Elements: {mat_node.elements}")
    
    # Check if checkpoint exists
    if not MACE_CHECKPOINT_PATH.exists():
        print(f"\nCheckpoint not found at: {MACE_CHECKPOINT_PATH}")
        print("Place mace-mp-0.pth in models/gnn_surrogate/")
        return
    
    print(f"Checkpoint found: {MACE_CHECKPOINT_PATH}")
    
    # Try to load predictor
    try:
        predictor = PredictorAgent(checkpoint_path=MACE_CHECKPOINT_PATH)
        result = predictor.predict(mat_node)
        
        print(f"\nPrediction result:")
        print(f"  Property value (e_above_hull): {result.property_value}")
        print(f"  Uncertainty: {result.uncertainty}")
        print(f"  Model used: {result.model_used}")
        print(f"  Failed: {result.prediction_failed}")
        
    except ImportError as e:
        print(f"\nMACE not installed: {e}")
        print("Install with: pip install mace-torch")
        print("\nOr from source:")
        print("  git clone https://github.com/ACE-Modeling/MACE.git")
        print("  cd MACE && pip install -e .")
    except Exception as e:
        print(f"\nError during prediction: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
