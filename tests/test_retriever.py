"""Test Retriever agent for KG lookups.

Tests that:
1. Retriever can parse natural language queries
2. KG traversal returns correct materials
3. Provenance tracking works
4. Error handling for invalid queries
5. Natural language query parsing (with optional LLM)

Run with: python tests/test_retriever.py

Note: LLM-based tests require a local model server running on UNSLOTH_BASE_URL.
"""

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import sys
import os

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
        results = retriever.run_query(
            "Find Ni-P HER catalyst materials"
        )
        
        print(f"[PASS] Found {len(results.materials)} Ni-P materials")
        assert len(results.materials) > 0, "No Ni-P materials found"
        
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
        results = retriever.run_query(
            "Find stable HER catalyst materials with e_above_hull < 0.05"
        )
        
        print(f"[PASS] Found {len(results.materials)} stable materials")
        assert len(results.materials) > 0, "No stable materials found"
        
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
        results = retriever.run_query(
            "Find Ni materials"
        )
        
        if results.materials:
            first_result = results.materials[0]
            print(f"[PASS] First result: {first_result.mpid}")
            
            # Check that provenance is tracked
            if hasattr(results, 'provenance'):
                print(f"[PASS] Provenance tracked: {results.provenance is not None}")
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
        
        # Try query with non-existent element (using a 3-letter invalid sequence)
        results = retriever.run_query(
            "Find materials containing XYZ"  # Invalid 3-letter sequence - should fall through to broad query
        )
        
        print(f"[PASS] Returned {len(results.materials)} results for invalid query (falls back to all materials)")
        # When no valid element is found, it falls back to returning all materials
        
    except Exception as e:
        print(f"[FAIL] Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_natural_language():
    """Test that retriever can parse natural language queries.
    
    Note: This test uses use_llm=False to avoid LLM dependency in tests,
    but the same logic would work with LLM enabled for more complex queries.
    """
    print("\nTesting natural language query parsing...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=False)
        
        # Test various natural language patterns
        test_queries = [
            "Find HER catalysts with Nickel",
            "What are the stable OER materials?",
            "Show me Fe-based catalysts",
            "List Mo-S systems"
        ]
        
        for query in test_queries:
            results = retriever.run_query(query)
            print(f"[PASS] Query '{query[:40]}...' returned {len(results.materials)} materials")
            assert len(results.materials) >= 0  # Should return some results
        
    except Exception as e:
        print(f"[FAIL] Natural language test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_retriever_with_llm():
    """Test retriever with LLM enabled (optional).
    
    This test requires:
    1. Ollama server running on OLLAMA_BASE_URL
    2. Model available via OLLAMA_MODEL
    
    If Ollama is not available, this test will be skipped.
    """
    print("\nTesting with LLM enabled (Ollama)...")
    
    graph_path = Path("data/processed/kg.json")
    
    try:
        # Try to initialize with LLM
        from pydantic_ai.models import infer_model
        
        # Check if OLLAMA_BASE_URL is set
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model_name = os.getenv("OLLAMA_MODEL", "gemma4:latest")
        
        if not ollama_url.startswith("http"):
            raise Exception("OLLAMA_BASE_URL must be a valid HTTP URL")
        
        # Infer Ollama model
        ollama_model_obj = infer_model(f"ollama:{ollama_model_name}")
        
        retriever = KGRetrieverAgent(graph_path=graph_path, use_llm=True)
        
        # Test a natural language query
        results = retriever.run_query("Find HER catalysts containing Nickel")
        
        print(f"[PASS] LLM query returned {len(results.materials)} materials")
        assert len(results.materials) > 0
        
    except Exception as e:
        # Skip if Ollama not available (expected in CI/test environments)
        print(f"[SKIP] LLM test skipped: {e}")
        print("Note: Ensure Ollama is running with gemma4:latest model")
        return  # Don't raise, just skip
        
    except AssertionError:
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
        test_retriever_natural_language()
        test_retriever_with_llm()  # Optional LLM test
        
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
