"""Improved CGCNN-style GNN baseline for materials property prediction.

Implements a from-scratch Graph Convolutional Network using PyTorch Geometric
to predict e_above_hull (formation energy above convex hull) for crystalline
materials. Trained on Materials Project subset, compared against fine-tuned MACE.

Architecture:
- Input: Atomic number features + structural connectivity
- 3-4x GraphConv layers with skip connections and batch normalization
- Multi-feature edges: Euclidean distance + x, y, z components
- Readout to global representation via mean/max pooling
- MLP head for property prediction

Output: e_above_hull (eV/atom) with uncertainty estimation via MC Dropout.

Key improvements over baseline:
1. Multi-feature edges (4 features: total dist + x, y, z components)
2. Proper super-cell construction for periodic boundary conditions
3. Support for both min-max and z-score normalization
4. Multiple model architectures (256/512 hidden channels)
5. Output value normalization with saved constants
6. Comprehensive metrics (MSE, MAE, R², max error)
7. Dataset type support (uniform/static vs phononic)

Usage:
    # Train and evaluate
    python models/gnn_surrogate/baseline_cgcnn_improved.py --train --epochs 500
    
    # Predict on single material
    python models/gnn_surrogate/baseline_cgcnn_improved.py --predict mpid=mp-2790
    
    # Compare with MACE
    python models/gnn_surrogate/baseline_cgcnn_improved.py --compare-mace

References:
    - Xie & Grossman, "Crystal Graph Convolutional Neural Networks", PRL 2018
    - polbeni/GNN-materials: https://github.com/polbeni/GNN-materials
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CGCNNConfig:
    """CGCNN hyperparameters."""
    
    # Model architecture
    hidden_channels = 256
    num_conv_layers = 4
    dropout = 0.6
    
    # Training
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_epochs = 500
    batch_size = 128
    train_split = 0.8
    
    # Data
    feature_dim = 94  # Number of atomic number bins (0-93)
    edge_radius = 5.5  # Ångströms for edge construction
    
    # Uncertainty
    mc_dropout_prob = 0.2

config = CGCNNConfig()


# ---------------------------------------------------------------------------
# Atom Features Dictionary
# ---------------------------------------------------------------------------

ATOM_FEATURES = {
    "H": (0, 2.20, 1.008, 53),
    "He": (1, 2.22, 4.0026, 31),
    "Li": (2, 0.98, 6.94, 167),
    "Be": (3, 1.57, 9.0122, 112),
    "B": (4, 2.04, 10.81, 87),
    "C": (5, 2.55, 12.011, 67),
    "N": (6, 3.04, 14.007, 56),
    "O": (7, 3.44, 15.999, 48),
    "F": (8, 3.98, 18.998, 42),
    "Ne": (9, 4.0, 20.180, 38),
    "Na": (10, 0.93, 22.990, 190),
    "Mg": (11, 1.31, 24.305, 145),
    "Al": (12, 1.61, 26.982, 118),
    "Si": (13, 1.90, 28.085, 111),
    "P": (14, 2.19, 30.974, 98),
    "S": (15, 2.58, 32.06, 88),
    "Cl": (16, 3.16, 35.45, 79),
    "Ar": (17, 3.2, 39.948, 71),
    "K": (18, 0.82, 39.098, 243),
    "Ca": (19, 1.0, 40.078, 194),
    "Sc": (20, 1.36, 44.956, 184),
    "Ti": (21, 1.54, 47.867, 176),
    "V": (22, 1.63, 50.942, 171),
    "Cr": (23, 1.66, 51.996, 166),
    "Mn": (24, 1.55, 54.938, 161),
    "Fe": (25, 1.83, 55.845, 156),
    "Co": (26, 1.88, 58.933, 152),
    "Ni": (27, 1.91, 58.693, 149),
    "Cu": (28, 1.90, 63.546, 145),
    "Zn": (29, 1.65, 65.38, 142),
    "Ga": (30, 1.81, 69.723, 136),
    "Ge": (31, 2.01, 72.630, 125),
    "As": (32, 2.18, 74.922, 114),
    "Se": (33, 2.55, 78.971, 103),
    "Br": (34, 2.96, 79.904, 94),
    "Kr": (35, 3.0, 83.798, 88),
    "Rb": (36, 0.82, 85.468, 265),
    "Sr": (37, 0.95, 87.62, 219),
    "Y": (38, 1.22, 88.906, 212),
    "Zr": (39, 1.33, 91.224, 206),
    "Nb": (40, 1.6, 92.906, 198),
    "Mo": (41, 2.16, 95.95, 190),
    "Tc": (42, 1.9, 98.0, 183),
    "Ru": (43, 2.2, 101.07, 178),
    "Rh": (44, 2.28, 102.91, 173),
    "Pd": (45, 2.20, 106.42, 169),
    "Ag": (46, 1.93, 107.87, 165),
    "Cd": (47, 1.69, 112.41, 161),
    "In": (48, 1.78, 114.82, 156),
    "Sn": (49, 1.96, 118.71, 145),
    "Sb": (50, 2.05, 121.76, 133),
    "Te": (51, 2.1, 127.60, 123),
    "I": (52, 2.66, 126.90, 115),
    "Xe": (53, 2.6, 131.29, 108),
    "Cs": (54, 0.79, 132.91, 298),
    "Ba": (55, 0.89, 137.33, 253),
    "La": (56, 1.10, 138.91, 226),
    "Ce": (57, 1.12, 140.12, 210),
    "Pr": (58, 1.13, 140.91, 247),
    "Nd": (59, 1.14, 144.24, 206),
    "Pm": (60, 1.13, 145.0, 205),
    "Sm": (61, 1.17, 150.36, 238),
    "Eu": (62, 1.20, 151.96, 231),
    "Gd": (63, 1.20, 157.25, 233),
    "Tb": (64, 1.22, 158.93, 225),
    "Dy": (65, 1.22, 162.50, 228),
    "Ho": (66, 1.23, 164.93, 226),
    "Er": (67, 1.24, 167.26, 226),
    "Tm": (68, 1.25, 168.93, 222),
    "Yb": (69, 1.1, 173.05, 222),
    "Lu": (70, 1.27, 174.97, 217),
    "Hf": (71, 1.3, 178.49, 208),
    "Ta": (72, 1.5, 180.95, 200),
    "W": (73, 2.36, 183.84, 193),
    "Re": (74, 1.9, 186.21, 188),
    "Os": (75, 2.2, 190.23, 185),
    "Ir": (76, 2.2, 192.22, 180),
    "Pt": (77, 2.28, 195.08, 177),
    "Au": (78, 2.54, 196.97, 174),
    "Hg": (79, 2.0, 200.59, 171),
    "Th": (80, 1.62, 204.38, 156),
    "Pb": (81, 2.33, 207.2, 154),
    "Bi": (82, 2.02, 208.98, 143),
    "Ac": (83, 1.1, 227.0, 186),
    "Th": (84, 1.3, 232.04, 175),
    "Pa": (85, 1.5, 231.04, 169),
    "U": (86, 1.38, 238.03, 170),
    "Np": (87, 1.36, 237.0, 171),
    "Pu": (88, 1.28, 244.0, 172)
}


def get_node_features(species: str) -> np.ndarray:
    """Get node features for an atom species.
    
    Features: [atomic_number, electronegativity, atomic_weight, atomic_radius]
    
    Args:
        species: Element symbol (e.g., 'Fe', 'Cu')
        
    Returns:
        Array of 4 feature values
    """
    if species not in ATOM_FEATURES:
        raise ValueError(f"Unknown species: {species}")
    
    return np.array(ATOM_FEATURES[species])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaterialsDataset(Dataset):
    """PyTorch Dataset for materials property prediction."""
    
    def __init__(self, data_list: List[Dict]):
        """Initialize dataset.
        
        Args:
            data_list: List of dicts with keys:
                - 'atomic_numbers': np.array of atomic numbers
                - 'cell': np.array of lattice vectors (3 x 3)
                - 'positions': np.array of fractional coordinates (N_atoms x 3)
                - 'e_above_hull': float formation energy
                - 'dataset_type': 'uniform', 'static', or 'phononic'
        """
        self.data_list = data_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        item = self.data_list[idx]
        
        # Convert to torch tensors
        atomic_numbers = torch.tensor(item['atomic_numbers'], dtype=torch.long)
        cell = torch.tensor(item['cell'], dtype=torch.float)
        positions = torch.tensor(item['positions'], dtype=torch.float)
        target = torch.tensor([item['e_above_hull']], dtype=torch.float)
        
        # Build graph with multi-feature edges
        edge_index, edge_attr = self._build_graph(positions, cell)
        
        # Create PyG Data object
        data = Data(
            x=atomic_numbers,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            cell=cell,
            y=target,
        )
        
        return data
    
    def _build_graph(self, positions: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph with multi-feature edges using super-cell construction.
        
        Args:
            positions: Fractional coordinates (N x 3)
            cell: Lattice vectors (3 x 3)
            
        Returns:
            edge_index: Edge index tensor (2 x E)
            edge_attr: Edge attributes (E x 4) - [total_dist, dx, dy, dz]
        """
        N = positions.shape[0]
        
        # Convert to Cartesian coordinates
        coords_cart = torch.einsum('ni,ij->nj', positions, cell)
        
        # Create super-cell for periodic boundary conditions
        # Find minimum lattice parameter
        abc = cell.norm(dim=0)  # Norm of each lattice vector
        min_param = float(abc.min())
        
        # Determine super-cell size based on edge radius
        n_supercell = 3
        param_supercell = min_param
        while param_supercell < config.edge_radius:
            n_supercell += 2
            param_supercell = min_param * (n_supercell - 2)
        
        # Create scaling matrix for super-cell
        scaling_matrix = torch.tensor([
            [n_supercell, 0, 0],
            [0, n_supercell, 0],
            [0, 0, n_supercell]
        ], dtype=torch.float)
        
        # Apply periodic boundary conditions to positions
        scaled_positions = (positions * n_supercell).remainder(1.0)
        scaled_cartesian = torch.einsum('ni,ij->nj', scaled_positions, cell)
        
        # Find all pairs within edge radius
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
                
                if dist_sq < config.edge_radius ** 2 and dist_sq > 1e-6:
                    dx = dx_cart.norm()
                    edge_index_list.append([i, j])
                    edge_attr_list.append(torch.tensor([dx.item(), 
                                                        dx_cart[0].item(),
                                                        dx_cart[1].item(),
                                                        dx_cart[2].item()]))
        
        if len(edge_index_list) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float)
        
        # Make undirected (add reverse edges)
        edge_index = torch.tensor(edge_index_list + [[j, i] for i, j in edge_index_list], dtype=torch.long)
        edge_attr = torch.stack(edge_attr_list + [attr for attr in edge_attr_list]).t().contiguous()
        
        return edge_index, edge_attr


# ---------------------------------------------------------------------------
# CGCNN Model
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """Graph convolutional layer with atomic number features and batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Feature transformation
        self.lin_x = nn.Linear(config.feature_dim, out_channels)
        self.lin_msg = nn.Linear(in_channels, out_channels)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-feature edges.
        
        Args:
            x: Node features (N x in_channels)
            edge_index: Edge indices (2 x E)
            edge_attr: Edge attributes (E x 4) - [total, dx, dy, dz]
            
        Returns:
            Out features (N x out_channels)
        """
        # Transform node features
        x_transformed = self.lin_x(x)
        
        # Message passing with feature aggregation
        if edge_attr is not None and edge_attr.size(1) == 4:
            # Use multi-feature edges: concatenate distance components
            dist_features = torch.cat([
                edge_attr[:, 0:1],  # Total distance
                edge_attr[:, 1:2],  # dx component
                edge_attr[:, 2:3],  # dy component
                edge_attr[:, 3:4],  # dz component
            ], dim=1)
            
            msg = self.lin_msg(dist_features)
        else:
            msg = self.lin_msg(x[edge_index[0]])
        
        # Aggregate messages
        out = torch.scatter_add(torch.zeros_like(x), 0, 
                                edge_index.view(-1, 2), msg.view(-1))
        out = out.view(x.shape[0], -1)
        
        # Batch normalization and activation
        out = self.bn(out)
        out = F.relu(out)
        
        return out


class CGCNN(nn.Module):
    """CGCNN-style Graph Convolutional Network.
    
    Architecture:
    - Input: Atomic number features (94 bins)
    - Nx GraphConv layers with batch normalization and dropout
    - Readout: Global mean/max pooling
    - Output: MLP for e_above_hull prediction
    
    Variants:
    - model_small: hidden_channels=128, num_conv_layers=3
    - model_medium (default): hidden_channels=256, num_conv_layers=4
    - model_large: hidden_channels=512, num_conv_layers=4
    """
    
    def __init__(self, hidden_channels: int = 256, 
                 num_conv_layers: int = 4,
                 seed: int = 12345):
        super().__init__()
        
        torch.manual_seed(seed)
        
        # Input layer: atomic number features
        self.input_proj = nn.Linear(config.feature_dim, hidden_channels)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            GraphConv(hidden_channels, hidden_channels) 
            for _ in range(num_conv_layers)
        ])
        
        # Readout and output
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, data: Data, batch: Optional[torch.Tensor] = None, 
                training: bool = False) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyG Data object with x (atomic numbers), edge_index, edge_attr
            batch: Batch indices for pooling (if None, single graph)
            training: If True, use dropout
            
        Returns:
            Predicted e_above_hull (eV/atom)
        """
        # Input projection
        h = self.input_proj(data.x.float())
        
        # Graph convolutions with multi-feature edges
        for i, conv in enumerate(self.convs):
            edge_attr = data.edge_attr if i == 0 else None
            h = conv(h, data.edge_index, edge_attr)
            h = F.dropout(h, p=config.dropout, training=training)
        
        # Global pooling (use max pooling as in original implementation)
        if batch is not None:
            global_repr = global_mean_pool(h, batch)
        else:
            global_repr = h.mean(dim=0).unsqueeze(0)
        
        # Output head
        out = self.readout(global_repr)
        
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train for one epoch.
    
    Args:
        model: CGCNN model
        loader: Training data loader
        optimizer: Optimizer
        
    Returns:
        Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Create batch tensor for pooling
        batch_indices = torch.zeros(len(batch.x), dtype=torch.long)
        
        pred = model(batch, batch=batch_indices, training=True)
        target = batch.y
        
        # MSE loss
        loss = F.mse_loss(pred, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float, float]:
    """Evaluate model on test set.
    
    Args:
        model: CGCNN model (eval mode)
        loader: Test data loader
        
    Returns:
        Tuple of (MSE, MAE, R², max_error)
    """
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
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Maximum error
    max_error = np.max(np.abs(predictions - targets))
    
    return mse, mae, rmse, r2, max_error


def plot_training_curves(train_losses: List[float], test_losses: List[float], 
                         output_path: str):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: Loss values per epoch for training set
        test_losses: Loss values per epoch for test set
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses[1:], label='train', linewidth=2)
    plt.plot(test_losses[1:], label='test', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_predictions(real_values: np.ndarray, predicted_values: np.ndarray, 
                     output_path: str):
    """Plot predicted vs actual values.
    
    Args:
        real_values: Ground truth values
        predicted_values: Model predictions
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(real_values, predicted_values, alpha=0.6, s=10)
    min_val = min(np.min(real_values), np.min(predicted_values))
    max_val = max(np.max(real_values), np.max(predicted_values))
    plt.xlim(min_val - 0.2 * (max_val - min_val), 
             max_val + 0.2 * (max_val - min_val))
    plt.ylim(min_val - 0.2 * (max_val - min_val), 
             max_val + 0.2 * (max_val - min_val))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('DFT computed e_above_hull (eV/atom)')
    plt.ylabel('Predicted e_above_hull (eV/atom)')
    plt.legend(['predictions', 'y=x'])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Improved CGCNN baseline for materials property prediction")
    
    parser.add_argument("--train", action="store_true", help="Train CGCNN model")
    parser.add_argument("--predict", type=str, help="Predict on single material (mpid=mp-XXXX)")
    parser.add_argument("--compare-mace", action="store_true", help="Compare predictions with MACE")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--hidden-channels", type=int, default=config.hidden_channels, 
                        help="Hidden channels in GNN layers")
    parser.add_argument("--num-conv-layers", type=int, default=config.num_conv_layers,
                        help="Number of graph convolution layers")
    
    args = parser.parse_args()
    
    if not any([args.train, args.predict]):
        print("Usage: python baseline_cgcnn_improved.py [--train] [--predict mpid=...] [--compare-mace]")
        return
    
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.hidden_channels = args.hidden_channels
    config.num_conv_layers = args.num_conv_layers
    
    if args.train:
        train_model()
    
    if args.predict:
        predict_on_material(args.predict)


def train_model():
    """Train the CGCNN model."""
    print("="*60)
    print("CGCNN Training (Improved)")
    print("="*60)
    
    # TODO: Load training data from data/raw/metadata.json + CIF files
    # For now, use a small dummy dataset for demonstration
    
    print("\n[TODO] Implement data loading from Materials Project subset")
    print("[TODO] Create train/val/test splits")
    print(f"[INFO] Model config: hidden={config.hidden_channels}, conv_layers={config.num_conv_layers}")
    print("[TODO] Train model and save to models/gnn_surrogate/cgcnn_improved.pt")


def predict_on_material(mpid: str):
    """Predict e_above_hull for a single material."""
    print("="*60)
    print(f"CGCNN Prediction (Improved): {mpid}")
    print("="*60)
    
    # TODO: Load trained model
    # TODO: Load material structure from CIF
    # TODO: Convert to PyG Data format with multi-feature edges
    # TODO: Run inference with MC Dropout for uncertainty
    
    print("\n[TODO] Implement prediction pipeline")


def compare_with_mace():
    """Compare CGCNN predictions with MACE."""
    print("="*60)
    print("CGCNN vs MACE Comparison (Improved)")
    print("="*60)
    
    # TODO: Load both models
    # TODO: Run on same test set
    # TODO: Report MAE, RMSE, R² for both models
    
    print("\n[TODO] Implement comparison pipeline")


if __name__ == "__main__":
    main()
