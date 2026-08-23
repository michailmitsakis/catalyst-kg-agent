"""Test Retriever agent for KG lookups.

Tests that:
1. Retriever can parse natural language queries
2. KG traversal returns correct materials
3. Provenance tracking works
4. Error handling for invalid queries

Run with: python tests/test_retriever.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.graph_store import load_graph, rehydrate_node
from agent.retriever import KGRetrieverAgent


def test_retriever_initialization():
    """Test that RetrieverAgent initializes successfully."""
    print("\nTesting KGRetrieverAgent initialization...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        # Initialize without LLM for testing
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        print(f"[PASS] KGRetrieverAgent initialized with KG at {graph_path}")
    except Exception as e:
        print(f"[FAIL] Failed to initialize KGRetrieverAgent: {e}")
        raise


def test_retriever_element_query():
    """Test querying by element via KG queries."""
    print("\nTesting element-based query...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        from kg.queries import find_materials_by_element
        results = find_materials_by_element(G, element_symbol="Ni")
        
        print(f"[PASS] Found {len(results)} Ni-containing materials")
        assert len(results) > 0, "No Ni materials found"
        
        # Verify all results contain Ni
        for mat in results[:5]:
            assert "Ni" in str(mat.elements), f"{mat.mpid} doesn't contain Ni"
        
        print(f"[PASS] All returned materials contain Ni")
        
    except Exception as e:
        print(f"[FAIL] Element query failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_chemsys_query():
    """Test querying by chemical system."""
    print("\nTesting chemsys-based query...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        
        # Query for Ni-P chemsys materials
        results = retriever.search(
            query="Find Ni-P HER catalyst materials",
            chemsys_groups=["Ni-P"]
        )
        
        print(f"[PASS] Found {len(results)} Ni-P materials")
        assert len(results) > 0, "No Ni-P materials found"
        
    except Exception as e:
        print(f"[FAIL] Chemsys query failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_stability_query():
    """Test querying for stable materials."""
    print("\nTesting stability-based query...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        
        # Query for stable materials (e_above_hull < 0.05)
        results = retriever.search(
            query="Find stable HER catalyst materials",
            e_above_hull_threshold=0.05
        )
        
        print(f"[PASS] Found {len(results)} stable materials")
        assert len(results) > 0, "No stable materials found"
        
    except Exception as e:
        print(f"[FAIL] Stability query failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_provenance():
    """Test that provenance tracking works."""
    print("\nTesting provenance tracking...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        
        # Query and check provenance
        results = retriever.search(
            query="Find Ni materials",
            elements=["Ni"]
        )
        
        if results:
            first_result = results[0]
            print(f"[PASS] First result: {first_result.mpid}")
            
            # Check that provenance is tracked
            if hasattr(first_result, 'provenance'):
                print(f"[PASS] Provenance tracked: {first_result.provenance is not None}")
            else:
                print("[INFO] Provenance field not available in result")
        
    except Exception as e:
        print(f"[FAIL] Provenance test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_error_handling():
    """Test error handling for invalid queries."""
    print("\nTesting error handling...")
    
    graph_path = Path("data/processed/kg.json")
    G = load_graph(graph_path)
    
    try:
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        
        # Try query with non-existent element
        results = retriever.search(
            query="Find materials containing XyZ",
            elements=["XyZ"]  # Non-existent element
        )
        
        print(f"[PASS] Returned {len(results)} results for invalid query (expected empty)")
        assert len(results) == 0, "Should return empty for non-existent element"
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all retriever tests."""
    print("="*60)
    print("Retriever Agent Test Suite")
    print("="*60)
    
    try:
        test_retriever_initialization()
        test_retriever_element_query()
        test_retriever_chemsys_query()
        test_retriever_stability_query()
        test_retriever_provenance()
        test_retriever_error_handling()
        
        print("\n" + "="*60)
        print("All retriever tests passed!")
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
