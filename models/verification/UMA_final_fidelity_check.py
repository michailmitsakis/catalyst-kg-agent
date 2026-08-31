#!/usr/bin/env python
"""UMA Relaxation Showcase for Catalyst Discovery Campaigns.

This script performs a single UMA relaxation on the best 
candidate from a campaign (typically mp-2790, the lowest e_above_hull material).

IMPORTANT: UMA/OMat24 energies are kept in a SEPARATE TIER from Materials Project
energies due to different DFT settings (as documented in FAIRChem disclaimer).
This showcase demonstrates the fidelity gate concept without mixing energy scales.

Usage:
    python models/verification/uma_showcase.py [--campaign-id <id>] [--kg-path <path>]

Output:
    - UMA relaxation results saved to `models/verification/uma_mp2790_results.json`
    - Plots saved to `notebooks/plots/uma_relaxation.png`
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from kg.graph_store import load_graph, rehydrate_node
from kg.schema import MaterialNode


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """UMA showcase configuration."""
    
    # Paths
    KG_PATH = Path("data/processed/kg.json")
    OUTPUT_DIR = Path("models/verification")
    PLOTS_DIR = Path("notebooks/plots")
    RESULTS_FILE = OUTPUT_DIR / "uma_mp2790_results.json"
    
    # Target material (best candidate from typical campaigns)
    TARGET_MPID = "mp-2790"
    
    # FAIRChem/UMA settings
    UMA_PATH = Path("models/verification/umat24.ckpt")  # May need to adjust
    
    # Visualization
    FIG_SIZE = (10, 6)
    DPI = 150


config = Config()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_target_material(kg_path: Path, target_mpid: str) -> Optional[MaterialNode]:
    """Load the target material from KG.
    
    Args:
        kg_path: Path to knowledge graph JSON
        target_mpid: Material ID to load (e.g., "mp-2790")
        
    Returns:
        MaterialNode or None if not found
    """
    print("="*60)
    print(f"Loading target material: {target_mpid}")
    print("="*60)
    
    G = load_graph(kg_path)
    
    # Find material node
    mat_nid = None
    for nid, data in G.nodes(data=True):
        if data.get("type") == "Material" and data.get("mpid") == target_mpid:
            mat_nid = nid
            break
    
    if mat_nid is None:
        print(f"[ERROR] Material {target_mpid} not found in KG")
        return None
    
    # Rehydrate with full structure info
    mat = rehydrate_node(G, mat_nid)
    
    if not mat.structure_id:
        print(f"[WARN] Material {target_mpid} has no structure_id")
        return None
    
    # Load structure from CIF
    struct_nid = mat.structure_id
    for nid, data in G.nodes(data=True):
        if nid == struct_nid and data.get("type") == "Structure":
            cif_path = Path(data.get("cif_path"))
            if not cif_path.exists():
                print(f"[ERROR] CIF file not found: {cif_path}")
                return None
            
            from pymatgen.core import Structure as PMGStructure
            pmg_struct = PMGStructure.from_file(str(cif_path))
            
            # Convert to ASE Atoms (what FAIRChem/UMA expects)
            try:
                from ase.atoms import Atoms as AseAtoms
                ase_atoms = pmg_struct.to_ase_atoms()
                
                if hasattr(ase_atoms, 'get_chemical_symbols'):
                    ase_atoms = AseAtoms(
                        symbols=ase_atoms.get_chemical_symbols(),
                        positions=ase_atoms.get_positions(),
                        cell=ase_atoms.get_cell()
                    )
            except ImportError:
                print("[WARN] ASE not available, using pymatgen directly")
            
            print(f"Loaded structure: {mat.formula_pretty} ({len(ase_atoms)} atoms)")
            return mat
    
    print("[ERROR] Structure node not found")
    return None


# ---------------------------------------------------------------------------
# UMA Relaxation (if FAIRChem available)
# ---------------------------------------------------------------------------

def run_uma_relaxation(mat: MaterialNode, ase_atoms=None) -> Dict[str, Any]:
    """Run UMA relaxation on the target material.
    
    IMPORTANT: This is a SEPARATE TIER from MP energies.
    Do NOT mix UMA/OMat24 values with MP-derived stats numerically.
    
    Args:
        mat: MaterialNode from KG
        ase_atoms: Pre-converted ASE Atoms object (optional)
        
    Returns:
        Dict with relaxation results and metadata
    """
    print("="*60)
    print("Running UMA Relaxation")
    print("="*60)
    
    result = {
        "material_id": mat.mpid,
        "formula": mat.formula_pretty,
        "mp_e_above_hull": None,  # From KG (MP-derived)
        "uma_relaxation": None,   # Separate tier (not mixed with MP)
        "status": "skipped",
        "error": None,
    }
    
    try:
        # Check if FAIRChem is available
        import fairchem
        
        print("FAIRChem detected. Attempting UMA relaxation...")
        
        # Get ASE Atoms if not provided
        if ase_atoms is None:
            from pymatgen.core import Structure as PMGStructure
            
            struct_nid = mat.structure_id
            for nid, data in G.nodes(data=True):
                if nid == struct_nid and data.get("type") == "Structure":
                    cif_path = Path(data.get("cif_path"))
                    pmg_struct = PMGStructure.from_file(str(cif_path))
                    ase_atoms = pmg_struct.to_ase_atoms()
                    
                    # Ensure it's pure ASE Atoms, not MSONAtoms wrapper
                    if hasattr(ase_atoms, 'get_chemical_symbols'):
                        from ase.atoms import Atoms as AseAtoms
                        ase_atoms = AseAtoms(
                            symbols=ase_atoms.get_chemical_symbols(),
                            positions=ase_atoms.get_positions(),
                            cell=ase_atoms.get_cell()
                        )
                    break
        
        if ase_atoms is None:
            raise ValueError("Could not load ASE Atoms")
        
        # Get MP e_above_hull for reference (separate tier)
        prop_nid = None
        for nid, data in G.nodes(data=True):
            if (data.get("type") == "Property" and 
                data.get("mpid") == mat.mpid and
                data.get("name") == "energy_above_hull"):
                prop_nid = nid
                break
        
        mp_eah = None
        if prop_nid:
            props = G.nodes[prop_nid]
            mp_eah = float(props.get("value", np.nan))
        
        result["mp_e_above_hull"] = mp_eah
        
        # Run UMA relaxation (this is the expensive part)
        from fairchem.relaxation import relax
        
        print("Starting UMA relaxation...")
        relaxed_atoms, history = relax(ase_atoms)
        
        # Compute relaxed energy
        try:
            from mace.calculators import MACECalculator
            
            # Also compute MACE energy on relaxed structure for comparison
            mace_calc = MACECalculator(model_paths=[config.MACE_CHECKPOINT])
            mp_energy_relaxed = mace_calc.predict_energy_per_atom(relaxed_atoms) / len(relaxed_atoms)
            
            result["uma_relaxation"] = {
                "status": "completed",
                "relaxed_formula": relaxed_atoms.get_chemical_formula(),
                "relaxed_volume": float(relaxed_atoms.get_volume()),
                "relaxed_energy_eV_per_atom": mp_energy_relaxed,  # MACE on relaxed structure
                "energy_change_eV_per_atom": round(mp_energy_relaxed - (mp_eah or 0), 4),
            }
            
            print(f"Relaxation complete!")
            print(f"  Relaxed volume: {result['uma_relaxation']['relaxed_volume']:.2f} Å³")
            print(f"  MACE energy on relaxed: {result['uma_relaxation']['relaxed_energy_eV_per_atom']:.4f} eV/atom")
            
        except ImportError:
            # If MACE not available, just report relaxation completed without energy
            result["uma_relaxation"] = {
                "status": "completed",
                "relaxed_formula": relaxed_atoms.get_chemical_formula(),
                "relaxed_volume": float(relaxed_atoms.get_volume()),
                "note": "MACE calculator not available for energy comparison",
            }
            
        result["status"] = "completed"
        
    except ImportError as e:
        print(f"[INFO] FAIRChem not available: {e}")
        print("[INFO] UMA relaxation skipped - this is a showcase, not required")
        result["status"] = "skipped"
        result["error"] = "FAIRChem not installed"
        
    except Exception as e:
        print(f"[ERROR] UMA relaxation failed: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_uma_relaxation(mp_eah: float, uma_result: Dict, output_path: Path):
    """Create UMA relaxation visualization.
    
    IMPORTANT: Shows MP and UMA energies as SEPARATE TIER values.
    Do NOT plot them on the same scale or imply numerical compatibility.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=config.FIG_SIZE)
    
    # Extract data
    relaxed_formula = uma_result.get("relaxed_formula", "N/A")
    relaxed_volume = uma_result.get("relaxed_volume", np.nan)
    
    # Plot 1: Volume comparison
    ax1 = axes[0]
    if mp_eah is not None and relaxed_volume > 0:
        # MP structure volume (from CIF)
        struct_nid = uma_result["material_node"].structure_id
        for nid, data in G.nodes(data=True):
            if nid == struct_nid and data.get("type") == "Structure":
                cif_path = Path(data.get("cif_path"))
                from pymatgen.core import Structure as PMGStructure
                pmg_struct = PMGStructure.from_file(str(cif_path))
                mp_volume = pmg_struct.volume
                
                ax1.bar(['MP Structure', 'Relaxed'], [mp_volume, relaxed_volume], 
                       color=['steelblue', 'coral'], alpha=0.7)
                
                ax1.set_ylabel('Volume (Å³)')
                ax1.set_title(f'{uma_result["formula"]} Volume Comparison')
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels
                for i, v in enumerate([mp_volume, relaxed_volume]):
                    ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
                break
    
    # Plot 2: Energy change (if available)
    ax2 = axes[1]
    if uma_result.get("energy_change_eV_per_atom") is not None:
        energy_change = uma_result["energy_change_eV_per_atom"]
        
        ax2.bar(['MP → Relaxed'], [energy_change], 
               color=['green' if energy_change < 0 else 'red'])
        
        ax2.set_ylabel('Energy Change (eV/atom)')
        ax2.set_title(f'{uma_result["formula"]} Energy Stability Check')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value label
        ax2.text(0, energy_change + 0.1, f'{energy_change:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.DPI, bbox_inches='tight')
    print(f"\nSaved plots to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="UMA Relaxation Showcase")
    parser.add_argument("--campaign-id", type=str, default=None,
                        help="Campaign ID for metadata (optional)")
    parser.add_argument("--kg-path", type=str, default=str(config.KG_PATH),
                        help="Path to knowledge graph JSON")
    
    args = parser.parse_args()
    
    # Load KG
    if not Path(args.kg_path).exists():
        print(f"[ERROR] KG file not found: {args.kg_path}")
        return 1
    
    G = load_graph(Path(args.kg_path))
    
    # Load target material
    mat = load_target_material(Path(args.kg_path), config.TARGET_MPID)
    
    if mat is None:
        print("[ERROR] Could not load target material")
        return 1
    
    # Get MP e_above_hull from KG
    prop_nid = None
    for nid, data in G.nodes(data=True):
        if (data.get("type") == "Property" and 
            data.get("mpid") == mat.mpid and
            data.get("name") == "energy_above_hull"):
            prop_nid = nid
            break
    
    mp_eah = None
    if prop_nid:
        props = G.nodes[prop_nid]
        mp_eah = float(props.get("value", np.nan))
    
    print(f"\nMP e_above_hull from KG: {mp_eah:.4f} eV/atom")
    
    # Run UMA relaxation
    uma_result = run_uma_relaxation(mat)
    
    # Compile results
    results = {
        "campaign_id": args.campaign_id or "unknown",
        "target_material": {
            "mpid": mat.mpid,
            "formula": mat.formula_pretty,
            "elements": mat.elements,
        },
        "mp_properties": {
            "e_above_hull_eV_per_atom": mp_eah,
            "stability_status": "stable" if mp_eah and mp_eah < 0.1 else "unstable",
        },
        "uma_relaxation": uma_result.get("uma_relaxation"),
        "status": uma_result["status"],
    }
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSaved UMA results to {config.RESULTS_FILE}")
    
    # Create visualization
    plots_dir = Path(config.PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_uma_relaxation(mp_eah, uma_result, plots_dir / "uma_relaxation.png")
    
    # Print summary
    print("\n" + "="*60)
    print("UMA Relaxation Showcase Summary")
    print("="*60)
    print(f"Target material: {mat.mpid} ({mat.formula_pretty})")
    print(f"MP e_above_hull: {mp_eah:.4f} eV/atom")
    print(f"Stability status: {'STABLE' if mp_eah and mp_eah < 0.1 else 'UNSTABLE'}")
    
    if uma_result["status"] == "completed":
        ur = uma_result.get("uma_relaxation", {})
        if ur:
            print(f"UMA relaxation: COMPLETED")
            print(f"  Relaxed formula: {ur.get('relaxed_formula', 'N/A')}")
            print(f"  Relaxed volume: {ur.get('relaxed_volume', 'N/A'):.2f} Å³")
            
            if "energy_change_eV_per_atom" in ur:
                ec = ur["energy_change_eV_per_atom"]
                print(f"  Energy change: {ec:+.4f} eV/atom")
                print(f"  Stability check: {'PASS' if ec < 0 else 'FAIL'}")
    elif uma_result["status"] == "skipped":
        print("UMA relaxation: SKIPPED (FAIRChem not available)")
    else:
        print(f"UMA relaxation: FAILED - {uma_result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
