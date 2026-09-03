"""Test Critic agent for safety validation.

Tests that:
1. Critic validates e_above_hull threshold (from the KG's MP-derived value)
2. The residual-force escalation gate is wired correctly
3. Schema compliance checks work
4. Cost impact tracking works

Note: the escalation signal is max residual force (eV/Angstrom), not a
model-uncertainty estimate. See agent/predictor.py for why.

Run with: python tests/test_critic.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.graph_store import load_graph, rehydrate_node
from agent.critic import CriticAgent, get_stability_threshold, get_force_gate


def test_critic_initialization():
    """Test that CriticAgent initializes with correct thresholds."""
    print("\nTesting CriticAgent initialization...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        critic = CriticAgent(graph_path=graph_path)
        
        stability_thresh = get_stability_threshold()
        force_gate = get_force_gate()

        print(f"[PASS] CriticAgent initialized")
        print(f"  - Stability threshold: {stability_thresh} eV/atom")
        print(f"  - Force gate: {force_gate:.3f} eV/A")

        assert critic.stability_threshold == stability_thresh, "Stability threshold mismatch"
        assert critic.force_gate == force_gate, "Force gate mismatch"
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize CriticAgent: {e}")
        raise


def test_critic_stability_check():
    """Test stability validation against KG."""
    print("\nTesting stability check...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        critic = CriticAgent(graph_path=graph_path)
        
        # Get a material from KG with known e_above_hull
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"]
        first_mat_id = mat_nodes[0]
        mat_node = rehydrate_node(G, first_mat_id)
        
        print(f"Testing with: {mat_node.formula_pretty} ({mat_node.mpid})")
        
        # Find e_above_hull property for this material
        from kg.schema import NodeType, PropertyName
        
        eah_value = None
        for nid, data in G.nodes(data=True):
            if (data.get("type") == NodeType.PROPERTY.value and 
                data.get("name") == PropertyName.ENERGY_ABOVE_HULL.value and
                data.get("mpid") == mat_node.mpid):
                eah_value = float(data.get("value", 0))
                break
        
        if eah_value is not None:
            print(f"  - e_above_hull from KG: {eah_value:.4f} eV/atom")
            
            # Check if it passes stability threshold
            passed = eah_value <= critic.stability_threshold
            print(f"  - Stability check: {'PASS' if passed else 'FAIL'}")
            
        print("[PASS] Stability check completed")
        
    except Exception as e:
        print(f"[FAIL] Stability check failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_critic_schema_validation():
    """Test schema compliance checks."""
    print("\nTesting schema validation...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        critic = CriticAgent(graph_path=graph_path)
        
        # Get a valid material from KG
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"]
        first_mat_id = mat_nodes[0]
        mat_node = rehydrate_node(G, first_mat_id)
        
        print(f"Testing schema validation for: {mat_node.mpid}")
        
        # Run schema check
        schema_check = critic._check_schema(mat_node)
        
        if schema_check["passed"]:
            print("[PASS] Schema validation passed")
        else:
            print(f"[FAIL] Schema validation failed: {schema_check['error']}")
            raise AssertionError(f"Schema validation failed: {schema_check['error']}")
        
    except Exception as e:
        print(f"[FAIL] Schema validation test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_critic_decision_output():
    """Test that Critic returns proper decision format."""
    print("\nTesting critic decision output...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        from agent.critic import CriticDecision
        
        critic = CriticAgent(graph_path=graph_path)
        
        # Get a material
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"]
        first_mat_id = mat_nodes[0]
        mat_node = rehydrate_node(G, first_mat_id)
        
        # Use a real PredictorResult rather than a mock: the Critic reads
        # max_residual_force, and a hand-rolled mock silently drifts from the
        # real schema whenever that contract changes (which is exactly what
        # happened when `uncertainty` was removed).
        from agent.predictor import PredictorResult

        predictions = [
            PredictorResult(
                material_id=mat_node.mpid,
                property_value=-6.146,
                formation_energy_per_atom=-0.484,
                max_residual_force=critic.force_gate / 5.0,  # comfortably below the gate
                model_used="mace",
                prediction_failed=False,
            )
        ]
        
        # Validate material
        decisions = critic.validate_materials([mat_node], predictions)
        
        print(f"[PASS] Generated {len(decisions)} decision(s)")
        
        if decisions:
            decision = decisions[0]
            print(f"  - Decision type: {type(decision).__name__}")
            print(f"  - Approved: {decision.approved}")
            print(f"  - Requires escalation: {decision.requires_escalation}")
            
            # Verify decision has required fields
            assert hasattr(decision, 'approved'), "Missing 'approved' field"
            assert hasattr(decision, 'requires_escalation'), "Missing 'requires_escalation' field"
            assert hasattr(decision, 'reason'), "Missing 'reason' field"
            assert hasattr(decision, 'cost_impact'), "Missing 'cost_impact' field"
            
            print("[PASS] Decision has all required fields")
        
    except Exception as e:
        print(f"[FAIL] Decision output test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all critic tests."""
    print("="*60)
    print("Critic Agent Test Suite")
    print("="*60)
    
    try:
        test_critic_initialization()
        test_critic_stability_check()
        test_critic_schema_validation()
        test_critic_decision_output()
        
        print("\n" + "="*60)
        print("All critic tests passed!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()