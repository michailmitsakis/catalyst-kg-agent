"""CGCNN-style GNN baseline for materials property prediction.

Implements a from-scratch Graph Convolutional Network using PyTorch Geometric
to predict e_above_hull (formation energy above convex hull) for crystalline
materials. Trained on Materials Project subset, compared against fine-tuned MACE.

Architecture:
- Input: Atomic number features + structural connectivity
- 3x GraphConv layers with skip connections
- Readout to global representation
- MLP head for property prediction

Output: e_above_hull (eV/atom) with uncertainty estimation via MC Dropout.

Usage:
    # Train and evaluate
    python models/gnn_surrogate/baseline_cgcnn.py --train --epochs 100
    
    # Predict on single material
    python models/gnn_surrogate/baseline_cgcnn.py --predict mpid=mp-2790
    
    # Compare with MACE
    python models/gnn_surrogate/baseline_cgcnn.py --compare-mace

References:
    - Klicpera et al., "Combining Graph Convolutional Recurrent Networks for
      Molecular Property Prediction", ICLR 2019
    - polbeni/GNN-materials: https://github.com/polbeni/GNN-materials
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CGCNNConfig:
    """CGCNN hyperparameters."""
    
    # Model architecture
    hidden_channels = 128
    num_layers = 3
    dropout = 0.2
    
    # Training
    learning_rate = 0.001
    weight_decay = 1e-5
    epochs = 100
    batch_size = 32
    
    # Data
    feature_dim = 94  # Number of atomic number bins (0-93)
    
    # Uncertainty
    mc_dropout_prob = 0.1  # Dropout probability for MC Dropout


config = CGCNNConfig()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaterialsDataset(Dataset):
    """PyTorch Dataset for materials property prediction."""
    
    def __init__(self, data_list: List[dict]):
        """Initialize dataset.
        
        Args:
            data_list: List of dicts with keys:
                - 'atomic_numbers': np.array of atomic numbers
                - 'cell': np.array of lattice vectors (3x3)
                - 'positions': np.array of fractional coordinates (N_atoms x 3)
                - 'e_above_hull': float formation energy
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
        
        # Create PyG Data object
        data = Data(
            x=atomic_numbers,
            edge_index=self._build_edge_index(positions, cell),
            pos=positions,
            cell=cell,
            y=target,
        )
        
        return data
    
    def _build_edge_index(self, positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Build nearest-neighbor edge indices.
        
        Args:
            positions: Fractional coordinates (N x 3)
            cell: Lattice vectors (3 x 3)
            
        Returns:
            Edge index tensor (2 x E)
        """
        # Compute Cartesian distances with periodic boundary conditions
        N = positions.shape[0]
        coords_cart = torch.einsum('ni,ij->nj', positions, cell)
        
        # Create all pairs
        dists_sq = torch.cdist(coords_cart, coords_cart, p=2) ** 2
        
        # Exclude self-edges and keep nearest neighbors (max 12)
        mask = (dists_sq > 1e-6) & (dists_sq < 5.0)  # Within 2.23 Å
        edges = torch.where(mask)[0].view(-1, 2)
        
        if len(edges) == 0:
            # Return empty edge index
            return torch.empty((2, 0), dtype=torch.long)
        
        # Sort by distance and keep nearest neighbors
        dists_sorted = torch.cdist(coords_cart[edges[:, 0]], 
                                   coords_cart[edges[:, 1]], p=2).squeeze()
        _, topk_edges = torch.topk(dists_sorted, min(12, len(edges)), largest=False)
        
        return edges[topk_edges]


# ---------------------------------------------------------------------------
# CGCNN Model
# ---------------------------------------------------------------------------

class GraphConv(MessagePassing):
    """Graph convolutional layer with atomic number features."""
    
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add")
        
        self.lin_x = nn.Linear(config.feature_dim, out_channels)
        self.lin_msg = nn.Linear(in_channels, out_channels)
        self.lin_bias = nn.Linear(1, out_channels) if bias else None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Transform node features
        x_transformed = self.lin_x(x)
        
        # Message passing
        out = self.propagate(edge_index, size=(x.shape[0], x.shape[0]),
                            x=x_transformed, size_x=x_transformed)
        
        # Add bias if present
        if self.lin_bias is not None:
            out = out + self.lin_bias(torch.ones_like(out))
        
        return out
    
    def message(self, x_j: torch.Tensor, size_x: torch.Tensor) -> torch.Tensor:
        """Message function with feature transformation."""
        return self.lin_msg(size_x)


class CGCNN(nn.Module):
    """CGCNN-style Graph Convolutional Network.
    
    Architecture:
    - Input: Atomic number features (94 bins)
    - 3x GraphConv layers with skip connections
    - Readout: Global mean pooling
    - Output: MLP for e_above_hull prediction
    """
    
    def __init__(self):
        super().__init__()
        
        # Input layer: atomic number features
        self.input_proj = nn.Linear(config.feature_dim, config.hidden_channels)
        
        # Graph convolution layers
        self.conv1 = GraphConv(config.hidden_channels, config.hidden_channels)
        self.conv2 = GraphConv(config.hidden_channels, config.hidden_channels)
        self.conv3 = GraphConv(config.hidden_channels, config.hidden_channels)
        
        # Skip connections
        self.skip1 = nn.Linear(config.hidden_channels, config.hidden_channels)
        self.skip2 = nn.Linear(config.hidden_channels, config.hidden_channels)
        self.skip3 = nn.Linear(config.hidden_channels, config.hidden_channels)
        
        # Readout and output
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, data: Data, training: bool = False) -> torch.Tensor:
        """Forward pass with optional MC Dropout.
        
        Args:
            data: PyG Data object with x (atomic numbers) and edge_index
            training: If True, use dropout for uncertainty estimation
            
        Returns:
            Predicted e_above_hull (eV/atom)
        """
        # Input projection
        h = self.input_proj(data.x)
        
        # Layer 1 with skip connection
        h1 = self.conv1(h, data.edge_index)
        h = h + self.skip1(h1)
        h = F.relu(h)
        h = F.dropout(h, p=config.dropout, training=training)
        
        # Layer 2 with skip connection
        h2 = self.conv2(h, data.edge_index)
        h = h + self.skip2(h2)
        h = F.relu(h)
        h = F.dropout(h, p=config.dropout, training=training)
        
        # Layer 3 (final) without skip
        h3 = self.conv3(h, data.edge_index)
        
        # Global readout
        global_repr = global_mean_pool(h3, batch=data.batch if hasattr(data, 'batch') else torch.zeros_like(data.x))
        
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
        
        pred = model(batch, training=True)
        target = batch.y
        
        # MSE loss
        loss = F.mse_loss(pred, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, float]:
    """Evaluate model on test set.
    
    Args:
        model: CGCNN model (eval mode)
        loader: Test data loader
        
    Returns:
        Tuple of (MAE, RMSE, R²)
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            pred = model(batch, training=False)
            target = batch.y
            
            predictions.extend(pred.numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CGCNN baseline for materials property prediction")
    
    parser.add_argument("--train", action="store_true", help="Train CGCNN model")
    parser.add_argument("--predict", type=str, help="Predict on single material (mpid=mp-XXXX)")
    parser.add_argument("--compare-mace", action="store_true", help="Compare predictions with MACE")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size")
    
    args = parser.parse_args()
    
    if not any([args.train, args.predict, args.compare_mace]):
        print("Usage: python baseline_cgcnn.py [--train] [--predict mpid=...] [--compare-mace]")
        return
    
    # Set batch size
    config.batch_size = args.batch_size
    
    if args.train:
        train_model()
    
    if args.predict:
        predict_on_material(args.predict)
    
    if args.compare_mace:
        compare_with_mace()


def train_model():
    """Train the CGCNN model."""
    print("="*60)
    print("CGCNN Training")
    print("="*60)
    
    # TODO: Load training data from data/raw/metadata.json + CIF files
    # For now, use a small dummy dataset for demonstration
    
    print("\n[TODO] Implement data loading from Materials Project subset")
    print("[TODO] Create train/val/test splits")
    print("[TODO] Train model and save to models/gnn_surrogate/cgcnn.pt")


def predict_on_material(mpid: str):
    """Predict e_above_hull for a single material."""
    print("="*60)
    print(f"CGCNN Prediction: {mpid}")
    print("="*60)
    
    # TODO: Load trained model
    # TODO: Load material structure from CIF
    # TODO: Convert to PyG Data format
    # TODO: Run inference with MC Dropout for uncertainty
    
    print("\n[TODO] Implement prediction pipeline")


def compare_with_mace():
    """Compare CGCNN predictions with MACE."""
    print("="*60)
    print("CGCNN vs MACE Comparison")
    print("="*60)
    
    # TODO: Load both models
    # TODO: Run on same test set
    # TODO: Report MAE, RMSE, R² for both models
    
    print("\n[TODO] Implement comparison pipeline")


if __name__ == "__main__":
    main()
