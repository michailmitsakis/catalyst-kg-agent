"""Knowledge Graph reset utility.

This script deletes the current Knowledge Graph and rebuilds it from scratch
using the original metadata data (data/raw/metadata.json). It also regenerates
the CIF cache if needed.

Usage:
    python kg/reset.py [--clear-cache] [--no-cache]

Options:
    --clear-cache   Clear CIF cache before rebuilding KG
    --no-cache      Skip clearing the CIF cache entirely

This is intended for resetting test campaigns and starting fresh with the
original dataset. Production deployments should export the KG first for audit.

Example output for journal documentation:
    === Knowledge Graph Reset ===
    Deleting existing KG artifacts...
    data/processed/kg.json deleted
    data/processed/kg.graphml deleted
    
    Clearing CIF cache...
    data/processed/cif_cache.pkl deleted
    
    Rebuilding Knowledge Graph from metadata.json...
    - Loaded 130 materials from data/raw/metadata.json
    - Parsed 130 CIF files (cache hit rate: 100%)
    - Created 556 nodes (Material, Property, Structure, Element, Chemsys)
    - Created 791 edges with graph connectivity
    
    === Reset Complete ===
    New KG written to data/processed/kg.json

Note: This script does not rebuild the CIF cache automatically. The cache files (cif_cache.pkl and cif_cache_meta.json) are regenerated during the original KG build process.
"""

import os
import sys
import argparse
from pathlib import Path


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    """Reset the Knowledge Graph to a clean state from original data."""
    parser = argparse.ArgumentParser(
        description="Reset Knowledge Graph and rebuild from metadata"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear CIF cache before rebuilding KG",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not clear CIF cache before rebuilding KG",
    )
    args = parser.parse_args()

    print("=== Knowledge Graph Reset ===\n")

    # Step 1: Delete existing KG artifacts
    out_dir = Path("data/processed")
    kg_files = ["kg.json", "kg.graphml"]
    
    deleted_files = []
    for kg_file in kg_files:
        kg_path = out_dir / kg_file
        if kg_path.exists():
            kg_path.unlink()
            deleted_files.append(kg_file)
            print(f"  {kg_file} deleted")
    
    if not deleted_files:
        print("No KG artifacts found to delete\n")
    else:
        print(f"\n")

    # Step 2: Clear CIF cache if requested (--clear-cache only)
    if args.clear_cache:
        print("Clearing CIF cache...")
        from kg.build_graph import CACHE_FILE, CACHE_META_FILE
        
        cache_files = []
        for f in [CACHE_FILE, CACHE_META_FILE]:
            if f.exists():
                f.unlink()
                cache_files.append(str(f))
        
        if cache_files:
            print(f"  {', '.join(cache_files)} deleted\n")
        else:
            print("  CIF cache already empty\n")

    # Step 3: Build KG from scratch (this loads metadata internally)
    print("Rebuilding Knowledge Graph...")
    try:
        from kg.build_graph import main
        
        # --no-cache means don't clear cache (use existing if available)
        # --clear-cache means force regeneration (ignore cache)
        main(clear_cache=args.clear_cache)
        print("\n=== Reset Complete ===\n")
        print("New KG written to data/processed/kg.json")
    except Exception as e:
        print(f"\nERROR: Failed to rebuild KG: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
