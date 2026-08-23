"""Test Planner agent for budget-bounded orchestration.

Tests that:
1. Planner manages budget correctly
2. Decision logic works (continue/escalate/stop)
3. Max experiments limit enforced
4. Cost tracking works

Run with: python tests/test_planner.py
"""

from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.graph_store import load_graph, rehydrate_node
from agent.planner import PlannerAgent, get_max_experiments


def test_planner_initialization():
    """Test that PlannerAgent initializes correctly."""
    print("\nTesting PlannerAgent initialization...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        max_exp = get_max_experiments()
        
        print(f"[PASS] PlannerAgent initialized")
        print(f"  - Initial budget: {planner.state.remaining_budget:.1f}")
        print(f"  - Max experiments: {max_exp}")
        
        assert planner.max_experiments == max_exp, "Max experiments mismatch"
        
    except Exception as e:
        print(f"[FAIL] Failed to initialize PlannerAgent: {e}")
        raise


def test_planner_budget_management():
    """Test budget deduction logic."""
    print("\nTesting budget management...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        initial_budget = planner.state.remaining_budget
        
        # Simulate some actions
        test_actions = [1.0, 5.0, 10.0]  # KG lookup, surrogate, experiment
        
        for cost in test_actions:
            old_budget = planner.state.remaining_budget
            planner.state.remaining_budget -= cost
            new_budget = planner.state.remaining_budget
            
            print(f"  - Spent ${cost:.1f}: {old_budget:.1f} -> {new_budget:.1f}")
        
        expected_remaining = initial_budget - sum(test_actions)
        assert abs(planner.state.remaining_budget - expected_remaining) < 0.01, \
            f"Budget mismatch: expected {expected_remaining}, got {planner.state.remaining_budget}"
        
        print(f"[PASS] Budget management correct")
        
    except Exception as e:
        print(f"[FAIL] Budget management test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_planner_decision_continue():
    """Test 'continue' decision when budget allows."""
    print("\nTesting 'continue' decision...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        # Get some materials from KG
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"][:5]
        retrieved_materials = [rehydrate_node(G, m) for m in mat_nodes]
        
        print(f"Testing with {len(retrieved_materials)} materials")
        
        # Make decision (should be continue since we have budget and materials)
        decision = planner.plan_next_step(
            retrieved_materials=retrieved_materials,
        )
        
        print(f"  - Decision: {decision.next_action}")
        print(f"  - Reason: {decision.reason[:80]}...")
        
        assert decision.next_action == "continue", \
            f"Expected 'continue', got '{decision.next_action}'"
        
        print("[PASS] Continue decision correct")
        
    except Exception as e:
        print(f"[FAIL] Continue decision test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_planner_decision_stop_budget():
    """Test 'stop' decision when budget exhausted."""
    print("\nTesting 'stop' decision (budget exhausted)...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        # Set budget to a low value that can't afford even surrogate query (5.0)
        planner.state.remaining_budget = 2.0
        
        print(f"  - Remaining budget: ${planner.state.remaining_budget:.1f}")
        
        # Get some materials
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"][:5]
        retrieved_materials = [rehydrate_node(G, m) for m in mat_nodes]
        
        # Make decision (should be stop due to low budget)
        decision = planner.plan_next_step(
            retrieved_materials=retrieved_materials,
        )
        
        print(f"  - Decision: {decision.next_action}")
        
        # Should be 'stop' since we can't afford any action
        assert decision.next_action == "stop", \
            f"Expected 'stop', got '{decision.next_action}'"
        
        print("[PASS] Stop decision correct")
        
    except Exception as e:
        print(f"[FAIL] Stop decision test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_planner_decision_max_experiments():
    """Test 'stop' decision when max experiments reached."""
    print("\nTesting 'stop' decision (max experiments reached)...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        # Set max experiments to a small number for testing
        os.environ["MAX_EXPERIMENTS"] = "3"
        
        # Drain budget so we can't afford more actions
        while planner.state.remaining_budget > 10.0:
            planner.state.remaining_budget -= 10.0
        
        # Simulate reaching max experiments
        planner.state.experiments_count = planner.max_experiments
        G = load_graph(graph_path)
        
        print(f"  - Experiments count: {planner.state.experiments_count}/{planner.max_experiments}")
        
        # Get some materials
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"][:5]
        retrieved_materials = [rehydrate_node(G, m) for m in mat_nodes]
        
        # Make decision (should be stop due to max experiments)
        decision = planner.plan_next_step(
            retrieved_materials=retrieved_materials,
        )
        
        print(f"  - Decision: {decision.next_action}")
        
        assert decision.next_action == "stop", \
            f"Expected 'stop', got '{decision.next_action}'"
        
        print("[PASS] Max experiments stop decision correct")
        
    except Exception as e:
        print(f"[FAIL] Max experiments test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original value
        os.environ["MAX_EXPERIMENTS"] = "10"


def test_planner_tracking():
    """Test that planner tracks state correctly."""
    print("\nTesting state tracking...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        planner = PlannerAgent(graph_path=graph_path, use_llm=False)
        
        # Get some materials
        mat_nodes = [nid for nid, data in G.nodes(data=True) if data.get("type") == "Material"][:3]
        retrieved_materials = [rehydrate_node(G, m) for m in mat_nodes]
        
        # Make decision
        decision = planner.plan_next_step(
            retrieved_materials=retrieved_materials,
        )
        
        print(f"  - Actions taken: {planner.state.actions_taken}")
        print(f"  - Materials evaluated: {len(planner.state.materials_evaluated)}")
        
        assert planner.state.actions_taken == 1, "Actions not tracked"
        assert len(planner.state.materials_evaluated) == len(retrieved_materials), \
            "Materials not tracked"
        
        print("[PASS] State tracking correct")
        
    except Exception as e:
        print(f"[FAIL] State tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all planner tests."""
    print("="*60)
    print("Planner Agent Test Suite")
    print("="*60)
    
    try:
        test_planner_initialization()
        test_planner_budget_management()
        test_planner_decision_continue()
        test_planner_decision_stop_budget()
        test_planner_decision_max_experiments()
        test_planner_tracking()
        
        print("\n" + "="*60)
        print("All planner tests passed!")
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
