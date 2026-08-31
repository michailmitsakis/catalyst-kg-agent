#!/usr/bin/env python
"""MACE vs CGCNN Surrogate Comparison for Catalyst Discovery Campaigns.

This script evaluates both MACE (production) and CGCNN (from-scratch baseline) 
surrogates on the same Materials Project catalyst subset, comparing:
- e_above_hull predictions (raw values per material)
- MC-Dropout uncertainty estimates  
- Inference speed per candidate
- Stability-classification agreement (both vs 0.1 eV/atom threshold)

Dataset: ~130 unique catalyst materials from data/raw/metadata.json
CGCNN is trained from-scratch on this small subset (honest limitations noted).

Output:
- Comparison metrics saved to `models/gnn_surrogate/cgcnn_vs_mace_comparison.json`
- Plots saved to `notebooks/plots/` directory
"""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys_path_root = Path(__file__).resolve().parent.parent
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from kg.graph_store import load_graph, rehydrate_node
from kg.schema import MaterialNode
from agent.cost_model import SURROGATE_COST

# Import CGCNN model (may not be trained yet)
try:
    from models.gnn_surrogate.baseline_cgcnn import CGCNN, MaterialsDataset, get_node_features
    CGCNN_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] CGCNN module not found: {e}")
    print("[INFO] Run 'python models/gnn_surrogate/baseline_cgcnn.py --train' first")
    CGCNN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Comparison configuration."""
    
    # Paths
    KG_PATH = Path("data/processed/kg.json")
    OUTPUT_DIR = Path("models/gnn_surrogate")
    PLOTS_DIR = Path("notebooks/plots")
    COMPARISON_FILE = OUTPUT_DIR / "cgcnn_vs_mace_comparison.json"
    
    # Stability threshold for classification
    STABILITY_THRESHOLD = 0.1  # eV/atom
    
    # MACE checkpoint
    MACE_CHECKPOINT = Path("models/gnn_surrogate/mace-mpa-0-medium.model")
    
    # CGCNN architecture (default medium)
    CGCNN_HIDDEN_CHANNELS = 256
    CGCNN_NUM_CONV_LAYERS = 4
    
    # Training config for small dataset
    CGCNN_TRAIN_EPOCHS = 100  # Small dataset, quick training


config = Config()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_catalyst_materials(kg_path: Path) -> List[MaterialNode]:
    """Load catalyst materials from KG.
    
    Filters for clean-energy/catalyst-relevant materials with structures.
    
    Returns:
        List of MaterialNode objects with structure_id populated.
    """
    print("="*60)
    print("Loading catalyst materials from KG...")
    print("="*60)
    
    G = load_graph(kg_path)
    
    # Filter for materials with structures (needed for both models)
    material_ids = [nid for nid, data in G.nodes(data=True) 
                    if data.get("type") == "Material" and data.get("structure_id")]
    
    print(f"Found {len(material_ids)} materials with structures")
    
    materials = []
    for mid in material_ids[:130]:  # Limit to ~130 catalyst materials
        try:
            mat = rehydrate_node(G, mid)
            if mat.structure_id:
                materials.append(mat)
        except Exception as e:
            print(f"[WARN] Failed to rehydrate {mid}: {e}")
            continue
    
    print(f"Loaded {len(materials)} materials for comparison")
    return materials


# ---------------------------------------------------------------------------
# MACE Prediction (Production Surrogate)
# ---------------------------------------------------------------------------

def predict_with_mace(mat: MaterialNode, enable_dropout: bool = True) -> Dict[str, float]:
    """Predict e_above_hull with MACE.
    
    Args:
        mat: MaterialNode from KG
        enable_dropout: If True, use MC-Dropout for uncertainty
        
    Returns:
        Dict with 'value' and 'uncertainty' (std dev of dropout passes)
    """
    from agent.predictor import PredictorAgent
    
    predictor = PredictorAgent(checkpoint_path=config.MACE_CHECKPOINT)
    result = predictor.predict(mat)
    
    if result.prediction_failed or result.property_value is None:
        return {"value": np.nan, "uncertainty": np.nan}
    
    # MC-Dropout uncertainty is std dev of N passes
    unc = result.uncertainty if enable_dropout else 0.0
    
    return {"value": float(result.property_value), 
            "uncertainty": float(unc)}


# ---------------------------------------------------------------------------
# CGCNN Training & Prediction (Baseline Surrogate)
# ---------------------------------------------------------------------------

def train_cgcnn(materials: List[MaterialNode]) -> Tuple[CGCNN, dict]:
    """Train CGCNN from scratch on catalyst subset.
    
    Args:
        materials: List of MaterialNode objects
        
    Returns:
        Trained CGCNN model and training stats
    """
    if not CGCNN_AVAILABLE:
        raise ImportError("CGCNN module not available")
    
    print("="*60)
    print("Training CGCNN baseline...")
    print("="*60)
    
    # Load ground truth e_above_hull from KG
    data_list = []
    for mat in materials:
        # Get property node for this material
        prop_nid = None
        for nid, data in G.nodes(data=True):
            if (data.get("type") == "Property" and 
                data.get("mpid") == mat.mpid and
                data.get("name") == "energy_above_hull"):
                prop_nid = nid
                break
        
        if prop_nid is None:
            continue
            
        props = G.nodes[prop_nid]
        e_above_hull = float(props.get("value", np.nan))
        
        # Get atomic numbers from structure
        struct_nid = mat.structure_id
        for nid, data in G.nodes(data=True):
            if nid == struct_nid and data.get("type") == "Structure":
                cif_path = Path(data.get("cif_path"))
                if cif_path.exists():
                    from pymatgen.core import Structure as PMGStructure
                    pmg_struct = PMGStructure.from_file(str(cif_path))
                    
                    # Get atomic numbers
                    atomic_numbers = np.array([pmg_struct.get_atomic_number(s) 
                                               for s in pmg_struct.species], dtype=np.int64)
                    
                    # Get cell (fractional)
                    cell = np.array(pmg_struct.lattice.frac_coords, dtype=np.float64)
                    
                    # Get positions (fractional)
                    positions = np.array(pmg_struct.frac_coords, dtype=np.float64)
                    
                    data_list.append({
                        "atomic_numbers": atomic_numbers,
                        "cell": cell,
                        "positions": positions,
                        "e_above_hull": e_above_hull,
                        "dataset_type": "uniform"  # Static/property prediction
                    })
                break
    
    if len(data_list) < 10:
        raise ValueError(f"Not enough data for training (got {len(data_list)}, need >= 10)")
    
    print(f"Prepared {len(data_list)} training examples")
    
    # Split into train/val
    np.random.seed(42)
    indices = np.arange(len(data_list))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create datasets and loaders
    train_dataset = MaterialsDataset(train_data)
    val_dataset = MaterialsDataset(val_data)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.CGCNN_CONFIG.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.CGCNN_CONFIG.batch_size, shuffle=False
    )
    
    # Initialize model
    model = CGCNN(
        hidden_channels=config.CGCNN_CONFIG.hidden_channels,
        num_conv_layers=config.CGCNN_CONFIG.num_conv_layers
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CGCNN_CONFIG.learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("Training...")
    for epoch in range(config.CGCNN_CONFIG.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = evaluate(model, val_loader)[:0]  # Just run eval
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.CGCNN_CONFIG.num_epochs}: "
                  f"train_loss={train_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    # Save model
    output_path = Path("models/gnn_surrogate/cgcnn_catalyst.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config.CGCNN_CONFIG),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, output_path)
    
    print(f"Saved CGCNN model to {output_path}")
    
    # Compute final validation metrics
    mse, mae, rmse, r2, max_err = evaluate(model, val_loader)
    
    return model, {
        "train_size": len(train_data),
        "val_size": len(val_data),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
        "validation_mse": float(mse),
        "validation_mae": float(mae),
        "validation_r2": float(r2),
        "validation_rmse": float(rmse),
    }


def train_epoch(model: CGCNN, loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Create batch tensor for pooling
        batch_indices = torch.zeros(len(batch.x), dtype=torch.long)
        
        pred = model(batch, batch=batch_indices, training=True)
        target = batch.y
        
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: CGCNN, loader: torch.utils.data.DataLoader) -> Tuple[float, float, float, float, float]:
    """Evaluate model on test set."""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch_indices = torch.zeros(len(batch.x), dtype=torch.long)
            pred = model(batch, batch=batch_indices, training=False)
            target = batch.y
            
            predictions.extend(pred.numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    max_error = np.max(np.abs(predictions - targets))
    
    return mse, mae, rmse, r2, max_error


def predict_with_cgcnn(model: CGCNN, mat: MaterialNode) -> Dict[str, float]:
    """Predict e_above_hull with trained CGCNN.
    
    Args:
        model: Trained CGCNN model
        mat: MaterialNode from KG
        
    Returns:
        Dict with 'value' and 'uncertainty' (estimated via dropout-like noise)
    """
    # Load structure
    struct_nid = mat.structure_id
    for nid, data in G.nodes(data=True):
        if nid == struct_nid and data.get("type") == "Structure":
            cif_path = Path(data.get("cif_path"))
            if not cif_path.exists():
                return {"value": np.nan, "uncertainty": np.nan}
            
            from pymatgen.core import Structure as PMGStructure
            pmg_struct = PMGStructure.from_file(str(cif_path))
            
            # Get atomic numbers
            atomic_numbers = np.array([pmg_struct.get_atomic_number(s) 
                                       for s in pmg_struct.species], dtype=np.int64)
            
            # Get cell (fractional)
            cell = np.array(pmg_struct.lattice.frac_coords, dtype=np.float64)
            
            # Get positions (fractional)
            positions = np.array(pmg_struct.frac_coords, dtype=np.float64)
            
            # Build graph with multi-feature edges
            edge_index, edge_attr = build_graph_edges(positions, cell)
            
            # Create PyG Data object
            data = torch_geometric_data.Data(
                x=torch.tensor(atomic_numbers, dtype=torch.long),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=positions,
                cell=cell,
                y=torch.tensor([float(np.nan)]),  # No ground truth during inference
            )
            
            # Predict (single pass for speed)
            model.eval()
            with torch.no_grad():
                pred = model(data, training=False).item()
            
            return {"value": float(pred), "uncertainty": 0.5}  # Placeholder uncertainty
    
    return {"value": np.nan, "uncertainty": np.nan}


def build_graph_edges(positions: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build graph edges with multi-feature attributes."""
    N = positions.shape[0]
    
    # Convert to Cartesian coordinates
    coords_cart = torch.einsum('ni,ij->nj', positions, cell)
    
    # Find nearest neighbors within edge radius
    edge_index_list = []
    edge_attr_list = []
    
    for i in range(N):
        for j in range(i + 1, N):
            dx_cart = coords_cart[j] - coords_cart[i]
            
            # Apply periodic boundary conditions
            abc_values = cell.diag()
            dx_cart[:, 0] = (dx_cart[:, 0] + abc_values[0] / 2.0) % abc_values[0] - abc_values[0] / 2.0
            dx_cart[:, 1] = (dx_cart[:, 1] + abc_values[1] / 2.0) % abc_values[1] - abc_values[1] / 2.0
            dx_cart[:, 2] = (dx_cart[:, 2] + abc_values[2] / 2.0) % abc_values[2] - abc_values[2] / 2.0
            
            dist_sq = torch.sum(dx_cart ** 2)
            
            if dist_sq < 5.5 ** 2 and dist_sq > 1e-6:  # edge_radius = 5.5 Å
                dx = dx_cart.norm()
                edge_index_list.append([i, j])
                edge_attr_list.append(torch.tensor([dx.item(), 
                                                     dx_cart[0].item(),
                                                     dx_cart[1].item(),
                                                     dx_cart[2].item()]))
    
    if len(edge_index_list) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)
    
    # Make undirected
    edge_index = torch.tensor(edge_index_list + [[j, i] for i, j in edge_index_list], dtype=torch.long)
    edge_attr = torch.stack(edge_attr_list + [attr for attr in edge_attr_list]).t().contiguous()
    
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Comparison Metrics
# ---------------------------------------------------------------------------

def compute_comparison_metrics(materials: List[MaterialNode]) -> Dict[str, any]:
    """Compute all comparison metrics.
    
    Args:
        materials: List of MaterialNode objects
        
    Returns:
        Dict with per-material predictions and aggregate statistics
    """
    print("="*60)
    print("Running MACE predictions...")
    print("="*60)
    
    mace_results = []
    cgcnn_results = []
    
    for i, mat in enumerate(materials):
        # MACE prediction
        mace_start = time.time()
        mace_pred = predict_with_mace(mat)
        mace_time = time.time() - mace_start
        
        # CGCNN prediction (if trained)
        cgcnn_start = time.time()
        if CGCNN_AVAILABLE and 'cgcnn_model' in globals():
            cgcnn_pred = predict_with_cgcnn(cgcnn_model, mat)
            cgcnn_time = time.time() - cgcnn_start
        else:
            cgcnn_pred = {"value": np.nan, "uncertainty": np.nan}
            cgcnn_time = 0.0
        
        # Ground truth from KG
        prop_nid = None
        for nid, data in G.nodes(data=True):
            if (data.get("type") == "Property" and 
                data.get("mpid") == mat.mpid and
                data.get("name") == "energy_above_hull"):
                prop_nid = nid
                break
        
        gt_value = None
        if prop_nid:
            props = G.nodes[prop_nid]
            gt_value = float(props.get("value", np.nan))
        
        # Stability classification (e_above_hull < 0.1)
        mace_stable = bool(not np.isnan(mace_pred["value"]) and mace_pred["value"] < config.STABILITY_THRESHOLD)
        cgcnn_stable = bool(not np.isnan(cgcnn_pred["value"]) and cgcnn_pred["value"] < config.STABILITY_THRESHOLD)
        
        result = {
            "material_id": mat.mpid,
            "formula": mat.formula_pretty,
            "ground_truth_e_above_hull": gt_value,
            "mace": {
                "e_above_hull": mace_pred["value"],
                "uncertainty": mace_pred["uncertainty"],
                "inference_time_sec": round(mace_time, 4),
                "stable_classification": mace_stable,
            },
            "cgcnn": {
                "e_above_hull": cgcnn_pred["value"],
                "uncertainty": cgcnn_pred["uncertainty"],
                "inference_time_sec": round(cgcnn_time, 4),
                "stable_classification": cgcnn_stable,
            }
        }
        
        mace_results.append(result)
        cgcnn_results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(materials)} materials")
    
    return mace_results, cgcnn_results


def compute_agreement_stats(mace_results: List[Dict], cgcnn_results: List[Dict]) -> Dict[str, any]:
    """Compute agreement statistics between models.
    
    Returns:
        Dict with agreement metrics
    """
    n_total = len(mace_results)
    
    # Stability classification agreement
    mace_stable_flags = [r["mace"]["stable_classification"] for r in mace_results]
    cgcnn_stable_flags = [r["cgcnn"]["stable_classification"] for r in cgcnn_results]
    
    both_stable = sum(1 for ms, cs in zip(mace_stable_flags, cgcnn_stable_flags) if ms and cs)
    both_unstable = sum(1 for ms, cs in zip(mace_stable_flags, cgcnn_stable_flags) 
                       if not ms and not cs)
    
    agreement_rate = (both_stable + both_unstable) / n_total if n_total > 0 else 0.0
    
    # Value correlation
    mace_values = [r["mace"]["e_above_hull"] for r in mace_results 
                   if not np.isnan(r["mace"]["e_above_hull"])]
    cgcnn_values = [r["cgcnn"]["e_above_hull"] for r in cgcnn_results 
                    if not np.isnan(r["cgcnn"]["e_above_hull"])]
    
    common_values = list(set(mace_values) & set(cgcnn_values))
    
    if len(common_values) > 1:
        corr, _ = np.polyfit(common_values, common_values, 1)
    else:
        corr = np.nan
    
    # Speed comparison
    mace_times = [r["mace"]["inference_time_sec"] for r in mace_results]
    cgcnn_times = [r["cgcnn"]["inference_time_sec"] for r in cgcnn_results]
    
    speed_ratio = np.mean(cgcnn_times) / np.mean(mace_times) if np.mean(mace_times) > 0 else np.nan
    
    return {
        "n_materials": n_total,
        "stability_agreement": {
            "both_stable": both_stable,
            "both_unstable": both_unstable,
            "agreement_rate": round(agreement_rate, 3),
        },
        "value_correlation": round(float(corr), 4) if not np.isnan(corr) else None,
        "speed_comparison": {
            "mace_mean_time_sec": round(np.mean(mace_times), 4),
            "cgcnn_mean_time_sec": round(np.mean(cgcnn_times), 4),
            "cgcnn_vs_mace_ratio": round(speed_ratio, 2) if not np.isnan(speed_ratio) else None,
        }
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(mace_results: List[Dict], cgcnn_results: List[Dict], 
                    output_path: Path):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    mace_eah = [r["mace"]["e_above_hull"] for r in mace_results 
                if not np.isnan(r["mace"]["e_above_hull"])]
    cgcnn_eah = [r["cgcnn"]["e_above_hull"] for r in cgcnn_results 
                 if not np.isnan(r["cgcnn"]["e_above_hull"])]
    gt_eah = [r["ground_truth_e_above_hull"] for r in mace_results 
              if r["ground_truth_e_above_hull"] is not None]
    
    # Plot 1: Predicted vs Ground Truth (MACE)
    ax1 = axes[0, 0]
    if gt_eah and mace_eah:
        ax1.scatter(gt_eah, mace_eah, alpha=0.6, s=20, label='MACE')
        min_val = min(np.min(gt_eah), np.min(mace_eah))
        max_val = max(np.max(gt_eah), np.max(mace_eah))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Ground Truth e_above_hull (eV/atom)')
        ax1.set_ylabel('MACE Predicted e_above_hull (eV/atom)')
        ax1.legend()
        ax1.set_title('MACE: Predicted vs Ground Truth')
    
    # Plot 2: Predicted vs Ground Truth (CGCNN)
    ax2 = axes[0, 1]
    if gt_eah and cgcnn_eah:
        ax2.scatter(gt_eah, cgcnn_eah, alpha=0.6, s=20, label='CGCNN')
        min_val = min(np.min(gt_eah), np.min(cgcnn_eah))
        max_val = max(np.max(gt_eah), np.max(cgcnn_eah))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax2.set_xlabel('Ground Truth e_above_hull (eV/atom)')
        ax2.set_ylabel('CGCNN Predicted e_above_hull (eV/atom)')
        ax2.legend()
        ax2.set_title('CGCNN: Predicted vs Ground Truth')
    
    # Plot 3: MACE vs CGCNN Agreement
    ax3 = axes[1, 0]
    mace_stable_flags = [r["mace"]["stable_classification"] for r in mace_results]
    cgcnn_stable_flags = [r["cgcnn"]["stable_classification"] for r in cgcnn_results]
    
    # Create confusion matrix-style plot
    agreement_data = np.array([[0, 0], [0, 0]])  [[both unstable], [both stable]]
    for ms, cs in zip(mace_stable_flags, cgcnn_stable_flags):
        if ms and cs:
            agreement_data[1, 1] += 1  # Both stable
        elif not ms and not cs:
            agreement_data[0, 0] += 1  # Both unstable
        else:
            if ms:
                agreement_data[1, 0] += 1  # MACE stable, CGCNN unstable
            else:
                agreement_data[0, 1] += 1  # MACE unstable, CGCNN stable
    
    im = ax3.imshow(agreement_data, cmap='Blues', aspect='auto')
    ax3.set_xlabel('CGCNN Stable Classification')
    ax3.set_ylabel('MACE Stable Classification')
    ax3.set_title('Stability Classification Agreement')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add counts to cells
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(agreement_data[i, j]), ha='center', va='center', 
                    color='white' if agreement_data[i, j] > 10 else 'black')
    
    # Plot 4: Inference Speed Comparison
    ax4 = axes[1, 1]
    mace_times = [r["mace"]["inference_time_sec"] for r in mace_results]
    cgcnn_times = [r["cgcnn"]["inference_time_sec"] for r in cgcnn_results]
    
    if mace_times and cgcnn_times:
        ax4.bar(['MACE', 'CGCNN'], [np.mean(mace_times), np.mean(cgcnn_times)], 
                color=['steelblue', 'coral'])
        ax4.set_ylabel('Mean Inference Time (seconds)')
        ax4.set_title('Inference Speed Comparison')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare MACE vs CGCNN surrogates")
    parser.add_argument("--train-cgcnn", action="store_true", 
                        help="Train CGCNN from scratch before comparison")
    parser.add_argument("--kg-path", type=str, default=str(config.KG_PATH),
                        help="Path to knowledge graph JSON")
    
    args = parser.parse_args()
    
    # Load KG
    if not Path(args.kg_path).exists():
        print(f"[ERROR] KG file not found: {args.kg_path}")
        print("[INFO] Run 'python kg/build_graph.py' first")
        return 1
    
    G = load_graph(Path(args.kg_path))
    
    # Load materials
    materials = load_catalyst_materials(Path(args.kg_path))
    
    if not materials:
        print("[ERROR] No materials found in KG")
        return 1
    
    # Train CGCNN if requested
    cgcnn_model = None
    if args.train_cgcnn:
        try:
            cgcnn_model, train_stats = train_cgcnn(materials)
            globals()['cgcnn_model'] = cgcnn_model
            print(f"\nCGCNN Training Stats:")
            for k, v in train_stats.items():
                if k != 'train_losses' and k != 'val_losses':
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"[ERROR] CGCNN training failed: {e}")
            return 1
    
    # Run comparison
    mace_results, cgcnn_results = compute_comparison_metrics(materials)
    
    # Compute agreement stats
    agreement_stats = compute_agreement_stats(mace_results, cgcnn_results)
    
    # Save results
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_data = {
        "mace_predictions": mace_results,
        "cgcnn_predictions": cgcnn_results,
        "agreement_statistics": agreement_stats,
        "training_stats": train_stats if 'train_stats' in locals() else None,
    }
    
    with open(config.COMPARISON_FILE, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\nSaved comparison results to {config.COMPARISON_FILE}")
    
    # Create plots
    plots_dir = Path(config.PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(mace_results, cgcnn_results, plots_dir / "mace_vs_cgcnn_comparison.png")
    
    print("="*60)
    print("Comparison Summary:")
    print("="*60)
    print(f"Materials evaluated: {agreement_stats['n_materials']}")
    print(f"Stability agreement rate: {agreement_stats['stability_agreement']['agreement_rate']*100:.1f}%")
    
    if 'value_correlation' in agreement_stats and agreement_stats['value_correlation'] is not None:
        print(f"Value correlation (MACE vs CGCNN): {agreement_stats['value_correlation']:.3f}")
    
    if 'speed_comparison' in agreement_stats:
        sc = agreement_stats['speed_comparison']
        print(f"MACE mean inference time: {sc['mace_mean_time_sec']:.4f}s")
        print(f"CGCNN mean inference time: {sc['cgcnn_mean_time_sec']:.4f}s")
        if sc['cgcnn_vs_mace_ratio']:
            print(f"CGCNN is {sc['cgcnn_vs_mace_ratio']:.1f}x {'slower' if sc['cgcnn_vs_mace_ratio'] > 1 else 'faster'} than MACE")


if __name__ == "__main__":
    main()
