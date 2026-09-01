"""CGCNN baseline for materials formation-energy prediction.

A from-scratch Crystal Graph Convolutional Neural Network (Xie & Grossman,
PRL 120, 145301 (2018)) trained on the project's Materials Project catalyst
subset. Serves as an offline baseline against the MACE foundation-model
surrogate used in the live campaign loop.

TARGET QUANTITY
---------------
`formation_energy_per_atom` (eV/atom), read from the KG's MP-derived
PropertyNodes. This is deliberately the SAME quantity that
agent/predictor.py reports for MACE (via models/elemental_references.py),
so the two models are compared on like for like. Neither model predicts
e_above_hull -- see agent/predictor.py's docstring.

The KG must therefore have been built from metadata containing
formation_energy_per_atom:
    python data/download.py
    python kg/build_graph.py --clear-cache

ARCHITECTURE
------------
- Atom features: learned embedding over atomic number (Z -> hidden dim)
- Edge features: Gaussian-expanded interatomic distance. Distances are
  rotation-invariant; raw Cartesian components are not, so only the scalar
  distance is used.
- Neighbour finding: pymatgen `Structure.get_all_neighbors(cutoff)`, which
  handles periodic boundary conditions correctly (a hand-rolled supercell
  is easy to get subtly wrong).
- N x CGConv-style message-passing layers with batch norm and residual
  connections
- Global mean pooling -> MLP head -> scalar
- MC Dropout at inference for an uncertainty estimate, mirroring the
  MACE predictor's uncertainty treatment

HONEST LIMITATIONS
------------------
- The training set is ~130 materials. That is very small for a GNN trained
  from scratch; the published CGCNN was trained on ~10^4-10^5 structures.
  Expect the foundation model (MACE) to win comfortably. The point of this
  baseline is the comparison itself, not to be competitive.
- With a set this small, a single train/test split is noisy. `--k-folds`
  runs k-fold cross-validation so reported metrics come with a spread
  rather than a single number.
- Target normalisation statistics are computed on the training fold only
  and saved with the checkpoint, so evaluation is not contaminated.

Usage:
    python models/baseline_cgcnn.py --train
    python models/baseline_cgcnn.py --train --k-folds 5
    python models/baseline_cgcnn.py --predict mp-2790
    python models/baseline_cgcnn.py --evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg.graph_store import load_graph, rehydrate_node, DEFAULT_KG_JSON
from kg.schema import NodeType, PropertyName


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "cgcnn_catalyst.pt"
DEFAULT_METRICS_PATH = REPO_ROOT / "models" / "cgcnn_training_metrics.json"

TARGET_PROPERTY = PropertyName.FORMATION_ENERGY_PER_ATOM.value

# Highest atomic number the embedding table covers (Pu = 94 in the corpus's
# widest plausible range). Index 0 is unused padding.
MAX_ATOMIC_NUMBER = 95


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CGCNNConfig:
    """CGCNN hyperparameters."""

    # Model architecture
    hidden_channels: int = 128
    num_conv_layers: int = 3
    dropout: float = 0.2

    # Graph construction
    cutoff_radius: float = 5.0     # Angstrom
    max_neighbors: int = 12        # keep the N nearest within the cutoff
    n_gaussians: int = 40          # edge feature dimension

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 300
    batch_size: int = 16
    train_split: float = 0.8
    seed: int = 42

    # Uncertainty
    mc_dropout_passes: int = 20


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class GaussianExpansion:
    """Expand a scalar distance into a smooth Gaussian basis.

    Standard CGCNN edge featurisation: a single distance becomes an
    `n_gaussians`-dim vector of RBF activations, which gives the network a
    smooth, differentiable notion of "how far" rather than one raw number.
    """

    def __init__(self, d_min: float = 0.0, d_max: float = 5.0, n_gaussians: int = 40):
        self.centers = np.linspace(d_min, d_max, n_gaussians)
        self.width = (d_max - d_min) / n_gaussians

    def __call__(self, distances: np.ndarray) -> np.ndarray:
        """Args: distances (E,) -> returns (E, n_gaussians)."""
        diff = distances[:, None] - self.centers[None, :]
        return np.exp(-(diff ** 2) / (self.width ** 2))


def structure_to_graph(
    structure,
    target: Optional[float],
    config: CGCNNConfig,
    expansion: GaussianExpansion,
    mpid: str = "",
) -> Optional[Data]:
    """Convert a pymatgen Structure into a PyG Data object.

    Neighbour finding uses pymatgen's `get_all_neighbors`, which resolves
    periodic images correctly. Edges are directed both ways (i->j and j->i)
    so message passing is symmetric.

    Args:
        structure: pymatgen Structure
        target: formation energy per atom (eV/atom), or None at inference
        config: hyperparameters (cutoff, max_neighbors)
        expansion: Gaussian basis for edge distances
        mpid: material id, carried on the Data object for reporting

    Returns:
        PyG Data object, or None if the structure yields no edges
    """
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    n_atoms = len(atomic_numbers)

    all_neighbors = structure.get_all_neighbors(config.cutoff_radius)

    src: list[int] = []
    dst: list[int] = []
    dists: list[float] = []

    for i, neighbors in enumerate(all_neighbors):
        # Nearest-first, capped: keeps graphs a consistent density and
        # stops dense structures from dominating the edge count.
        neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[: config.max_neighbors]
        for nbr in neighbors:
            j = nbr.index
            d = float(nbr.nn_distance)
            if d < 1e-8:
                continue
            src.append(i)
            dst.append(j)
            dists.append(d)

    if not src:
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)  # (2, E) as PyG expects
    edge_attr = torch.tensor(
        expansion(np.array(dists, dtype=np.float64)), dtype=torch.float
    )

    data = Data(
        x=torch.tensor(atomic_numbers, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=n_atoms,
    )
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float)
    data.mpid = mpid
    return data


def load_dataset(
    config: CGCNNConfig,
    kg_path: Path = DEFAULT_KG_JSON,
    require_target: bool = True,
) -> list[Data]:
    """Build the dataset from the KG plus its referenced CIF files.

    Args:
        config: hyperparameters
        kg_path: path to kg.json
        require_target: if True, skip materials without the target property

    Returns:
        List of PyG Data objects
    """
    from pymatgen.core import Structure

    G = load_graph(kg_path)
    expansion = GaussianExpansion(0.0, config.cutoff_radius, config.n_gaussians)

    # mpid -> formation energy, from MP-sourced PropertyNodes
    targets: dict[str, float] = {}
    for _nid, data in G.nodes(data=True):
        if data.get("type") == NodeType.PROPERTY.value and data.get("name") == TARGET_PROPERTY:
            mpid = data.get("mpid")
            value = data.get("value")
            if mpid is not None and value is not None:
                targets[mpid] = float(value)

    # mpid -> cif path, from StructureNodes
    cif_paths: dict[str, str] = {}
    for _nid, data in G.nodes(data=True):
        if data.get("type") == NodeType.STRUCTURE.value:
            mpid = data.get("mpid")
            cif = data.get("cif_path")
            if mpid and cif:
                cif_paths[mpid] = cif

    material_mpids = [
        data.get("mpid")
        for _nid, data in G.nodes(data=True)
        if data.get("type") == NodeType.MATERIAL.value and data.get("mpid")
    ]

    dataset: list[Data] = []
    n_missing_target = 0
    n_missing_cif = 0
    n_failed = 0

    for mpid in sorted(material_mpids):
        target = targets.get(mpid)
        if require_target and target is None:
            n_missing_target += 1
            continue

        cif = cif_paths.get(mpid)
        if not cif:
            n_missing_cif += 1
            continue

        cif_path = Path(cif)
        if not cif_path.is_absolute():
            cif_path = REPO_ROOT / cif_path
        if not cif_path.exists():
            n_missing_cif += 1
            continue

        try:
            structure = Structure.from_file(str(cif_path))
            graph = structure_to_graph(structure, target, config, expansion, mpid=mpid)
            if graph is None:
                n_failed += 1
                continue
            dataset.append(graph)
        except Exception as exc:
            print(f"  [skip] {mpid}: {type(exc).__name__}: {exc}")
            n_failed += 1

    print(f"Dataset: {len(dataset)} graphs")
    if n_missing_target:
        print(
            f"  {n_missing_target} materials skipped: no '{TARGET_PROPERTY}' in KG "
            f"(rebuild after adding it to data/download.py)"
        )
    if n_missing_cif:
        print(f"  {n_missing_cif} materials skipped: CIF missing")
    if n_failed:
        print(f"  {n_failed} materials skipped: structure/graph build failed")

    return dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CGConvLayer(MessagePassing):
    """CGCNN convolution (Xie & Grossman eq. 5).

    For each edge (i, j) the concatenation [h_i, h_j, e_ij] is passed
    through a gate (sigmoid) and a filter (softplus); their product is the
    message. Messages are summed over neighbours and added residually to
    h_i, which keeps deep stacks trainable.
    """

    def __init__(self, hidden_channels: int, edge_channels: int):
        super().__init__(aggr="add")
        self.lin_gate = nn.Linear(2 * hidden_channels + edge_channels, hidden_channels)
        self.lin_filter = nn.Linear(2 * hidden_channels + edge_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.softplus(x + self.bn(out))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return torch.sigmoid(self.lin_gate(z)) * F.softplus(self.lin_filter(z))


class CGCNN(nn.Module):
    """Crystal Graph Convolutional Neural Network.

    Atomic numbers are embedded (not one-hot), edges carry Gaussian-expanded
    distances, and the readout is a mean pool over atoms followed by an MLP.
    Mean pooling (rather than sum) makes the output intensive, which matches
    the per-atom target.
    """

    def __init__(self, config: CGCNNConfig):
        super().__init__()
        torch.manual_seed(config.seed)

        self.config = config
        self.embedding = nn.Embedding(MAX_ATOMIC_NUMBER, config.hidden_channels)
        self.convs = nn.ModuleList(
            [
                CGConvLayer(config.hidden_channels, config.n_gaussians)
                for _ in range(config.num_conv_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.Softplus(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Args: PyG Data/Batch -> returns (batch_size,) predictions."""
        h = self.embedding(data.x)

        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)

        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(
            data.x.size(0), dtype=torch.long, device=data.x.device
        )
        pooled = global_mean_pool(h, batch)
        return self.readout(pooled).squeeze(-1)

    @torch.no_grad()
    def predict_with_uncertainty(self, data: Data, n_passes: int = 20) -> tuple[float, float]:
        """MC-Dropout prediction.

        Dropout is kept active across `n_passes` forward passes; the spread
        of those passes is the uncertainty estimate. This mirrors how
        agent/predictor.py estimates MACE uncertainty, so the two models'
        uncertainties are at least methodologically comparable.

        Returns:
            (mean, std) in normalised units -- de-normalise before use
        """
        was_training = self.training
        self.eval()
        # Re-enable only the dropout modules.
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        preds = [float(self(data).item()) for _ in range(n_passes)]

        if was_training:
            self.train()
        return float(np.mean(preds)), float(np.std(preds))


# ---------------------------------------------------------------------------
# Target normalisation
# ---------------------------------------------------------------------------

class Normalizer:
    """Standardise targets using TRAINING-fold statistics only.

    Fitting the mean/std on the full dataset would leak test information
    into training; these stats are saved with the checkpoint so inference
    de-normalises identically.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std) if std > 1e-8 else 1.0

    @classmethod
    def from_targets(cls, targets: list[float]) -> "Normalizer":
        arr = np.asarray(targets, dtype=np.float64)
        return cls(mean=arr.mean(), std=arr.std())

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denorm(self, x: torch.Tensor | float) -> torch.Tensor | float:
        return x * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: CGCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    normalizer: Normalizer,
    device: torch.device,
) -> float:
    """One training epoch. Returns mean loss in normalised units."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        target = normalizer.norm(batch.y)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: CGCNN,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate in ORIGINAL units (eV/atom), not normalised units.

    Returns:
        Dict with mae, rmse, mse, r2, max_error
    """
    model.eval()
    preds: list[float] = []
    targets: list[float] = []

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch)
        preds.extend(np.atleast_1d(normalizer.denorm(pred_norm.cpu().numpy())))
        targets.extend(np.atleast_1d(batch.y.cpu().numpy()))

    preds_arr = np.asarray(preds, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)

    if preds_arr.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mse": float("nan"),
                "r2": float("nan"), "max_error": float("nan"), "n": 0}

    errors = preds_arr - targets_arr
    mse = float(np.mean(errors ** 2))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((targets_arr - targets_arr.mean()) ** 2))

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
        "max_error": float(np.max(np.abs(errors))),
        "n": int(preds_arr.size),
    }


def train_single_split(
    dataset: list[Data],
    config: CGCNNConfig,
    device: torch.device,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    verbose: bool = True,
) -> tuple[CGCNN, Normalizer, dict[str, Any]]:
    """Train one model on one split. Returns (model, normalizer, metrics)."""
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    # Normalisation fitted on the training fold ONLY.
    normalizer = Normalizer.from_targets([float(d.y.item()) for d in train_data])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    model = CGCNN(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, normalizer, device)
        if verbose and (epoch + 1) % 25 == 0:
            metrics = evaluate(model, test_loader, normalizer, device)
            print(
                f"  epoch {epoch + 1:4d}/{config.num_epochs}  "
                f"train_loss={train_loss:.4f}  test_mae={metrics['mae']:.4f} eV/atom"
            )
            history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})

    final_train = evaluate(model, train_loader, normalizer, device)
    final_test = evaluate(model, test_loader, normalizer, device)

    return model, normalizer, {
        "train": final_train,
        "test": final_test,
        "history": history,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }


def baseline_metrics(dataset: list[Data], test_idx: np.ndarray, train_idx: np.ndarray) -> dict[str, float]:
    """Metrics for a mean-predictor baseline.

    A GNN that cannot beat "always predict the training mean" has learned
    nothing; reporting this alongside the model keeps the comparison honest.
    """
    train_targets = np.array([float(dataset[i].y.item()) for i in train_idx])
    test_targets = np.array([float(dataset[i].y.item()) for i in test_idx])
    pred = train_targets.mean()
    errors = pred - test_targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def train_model(
    config: CGCNNConfig,
    k_folds: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> int:
    """Train the CGCNN, optionally with k-fold cross-validation.

    Args:
        config: hyperparameters
        k_folds: 0 for a single split, >= 2 for k-fold CV
        model_path: where to save the final checkpoint
        metrics_path: where to save metrics JSON

    Returns:
        Process exit code
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("CGCNN Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Target: {TARGET_PROPERTY} (eV/atom)")
    print()

    dataset = load_dataset(config)
    if len(dataset) < 10:
        print(
            f"[ERROR] Only {len(dataset)} usable graphs. Need >= 10.\n"
            f"        Check that the KG contains '{TARGET_PROPERTY}' properties:\n"
            f"          python data/download.py\n"
            f"          python kg/build_graph.py --clear-cache"
        )
        return 1

    targets = np.array([float(d.y.item()) for d in dataset])
    print(
        f"Target stats: mean={targets.mean():+.4f}  std={targets.std():.4f}  "
        f"min={targets.min():+.4f}  max={targets.max():+.4f} eV/atom"
    )
    print()

    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(dataset))

    results: dict[str, Any] = {
        "config": asdict(config),
        "target_property": TARGET_PROPERTY,
        "n_graphs": len(dataset),
        "target_stats": {
            "mean": float(targets.mean()),
            "std": float(targets.std()),
            "min": float(targets.min()),
            "max": float(targets.max()),
        },
    }

    if k_folds and k_folds >= 2:
        folds = np.array_split(indices, k_folds)
        fold_metrics: list[dict[str, Any]] = []
        best_model = None
        best_normalizer = None
        best_mae = float("inf")

        for fold_i in range(k_folds):
            print(f"--- Fold {fold_i + 1}/{k_folds} ---")
            test_idx = folds[fold_i]
            train_idx = np.concatenate([f for j, f in enumerate(folds) if j != fold_i])

            model, normalizer, metrics = train_single_split(
                dataset, config, device, train_idx, test_idx, verbose=False
            )
            metrics["mean_predictor_baseline"] = baseline_metrics(dataset, test_idx, train_idx)
            fold_metrics.append(metrics)

            print(
                f"  test MAE={metrics['test']['mae']:.4f}  "
                f"RMSE={metrics['test']['rmse']:.4f}  "
                f"R2={metrics['test']['r2']:.3f}  "
                f"(mean-predictor MAE={metrics['mean_predictor_baseline']['mae']:.4f})"
            )

            if metrics["test"]["mae"] < best_mae:
                best_mae = metrics["test"]["mae"]
                best_model, best_normalizer = model, normalizer

        maes = [m["test"]["mae"] for m in fold_metrics]
        rmses = [m["test"]["rmse"] for m in fold_metrics]
        r2s = [m["test"]["r2"] for m in fold_metrics]
        baseline_maes = [m["mean_predictor_baseline"]["mae"] for m in fold_metrics]

        print()
        print("=" * 60)
        print(f"{k_folds}-fold cross-validation")
        print("=" * 60)
        print(f"  MAE  : {np.mean(maes):.4f} +/- {np.std(maes):.4f} eV/atom")
        print(f"  RMSE : {np.mean(rmses):.4f} +/- {np.std(rmses):.4f} eV/atom")
        print(f"  R2   : {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
        print(f"  mean-predictor MAE: {np.mean(baseline_maes):.4f} eV/atom")
        if np.mean(maes) >= np.mean(baseline_maes):
            print("  NOTE: the model does not beat a constant mean predictor.")

        results["cross_validation"] = {
            "k_folds": k_folds,
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
            "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
            "mean_predictor_mae": float(np.mean(baseline_maes)),
            "folds": fold_metrics,
        }
        model, normalizer = best_model, best_normalizer

    else:
        split = int(config.train_split * len(dataset))
        train_idx, test_idx = indices[:split], indices[split:]
        print(f"Split: {len(train_idx)} train / {len(test_idx)} test")
        model, normalizer, metrics = train_single_split(
            dataset, config, device, train_idx, test_idx, verbose=True
        )
        metrics["mean_predictor_baseline"] = baseline_metrics(dataset, test_idx, train_idx)

        print()
        print("=" * 60)
        print("Final metrics (eV/atom)")
        print("=" * 60)
        for split_name in ("train", "test"):
            m = metrics[split_name]
            print(
                f"  {split_name:5s}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
                f"R2={m['r2']:.3f}  max_err={m['max_error']:.4f}  (n={m['n']})"
            )
        print(f"  mean-predictor baseline MAE={metrics['mean_predictor_baseline']['mae']:.4f}")
        if metrics["test"]["mae"] >= metrics["mean_predictor_baseline"]["mae"]:
            print("  NOTE: the model does not beat a constant mean predictor.")

        results["single_split"] = metrics

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer.to_dict(),
            "target_property": TARGET_PROPERTY,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(f"Saved model to   {model_path}")
    print(f"Saved metrics to {metrics_path}")
    return 0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_trained_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> tuple[CGCNN, Normalizer, CGCNNConfig]:
    """Load a trained checkpoint plus its normalisation statistics.

    Raises:
        FileNotFoundError: if the checkpoint does not exist
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained CGCNN at {path}. Run: python models/baseline_cgcnn.py --train"
        )

    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = CGCNNConfig(**checkpoint["config"])
    model = CGCNN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_dict = checkpoint["normalizer"]
    normalizer = Normalizer(mean=norm_dict["mean"], std=norm_dict["std"])
    return model, normalizer, config


def predict_material(
    mpid: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    kg_path: Path = DEFAULT_KG_JSON,
) -> Optional[dict[str, Any]]:
    """Predict formation energy per atom for one material from the KG.

    Returns:
        Dict with prediction, uncertainty, and (if present) ground truth
    """
    from pymatgen.core import Structure

    model, normalizer, config = load_trained_model(model_path)
    expansion = GaussianExpansion(0.0, config.cutoff_radius, config.n_gaussians)

    G = load_graph(kg_path)

    cif = None
    truth = None
    for _nid, data in G.nodes(data=True):
        if data.get("mpid") != mpid:
            continue
        if data.get("type") == NodeType.STRUCTURE.value:
            cif = data.get("cif_path")
        elif data.get("type") == NodeType.PROPERTY.value and data.get("name") == TARGET_PROPERTY:
            truth = data.get("value")

    if not cif:
        print(f"[ERROR] No structure for {mpid} in KG")
        return None

    cif_path = Path(cif)
    if not cif_path.is_absolute():
        cif_path = REPO_ROOT / cif_path
    if not cif_path.exists():
        print(f"[ERROR] CIF not found: {cif_path}")
        return None

    structure = Structure.from_file(str(cif_path))
    graph = structure_to_graph(structure, None, config, expansion, mpid=mpid)
    if graph is None:
        print(f"[ERROR] Could not build graph for {mpid}")
        return None

    mean_norm, std_norm = model.predict_with_uncertainty(graph, n_passes=config.mc_dropout_passes)

    # De-normalise: the mean shifts and scales, the spread only scales.
    prediction = float(normalizer.denorm(mean_norm))
    uncertainty = float(std_norm * normalizer.std)

    result = {
        "material_id": mpid,
        "formula": structure.composition.reduced_formula,
        "predicted_formation_energy_per_atom": prediction,
        "uncertainty": uncertainty,
        "ground_truth_formation_energy_per_atom": float(truth) if truth is not None else None,
        "model": "cgcnn",
    }
    if truth is not None:
        result["absolute_error"] = abs(prediction - float(truth))

    print("=" * 60)
    print(f"CGCNN prediction: {mpid} ({result['formula']})")
    print("=" * 60)
    print(f"  Predicted : {prediction:+.4f} eV/atom")
    print(f"  Uncertainty (MC dropout, {config.mc_dropout_passes} passes): {uncertainty:.4f} eV/atom")
    if truth is not None:
        print(f"  MP value  : {float(truth):+.4f} eV/atom")
        print(f"  Abs error : {result['absolute_error']:.4f} eV/atom")

    return result


def evaluate_saved_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    kg_path: Path = DEFAULT_KG_JSON,
) -> int:
    """Evaluate a saved checkpoint over the whole dataset.

    Note: this reports metrics on ALL materials, which for a
    single-split checkpoint includes its own training data. Use the
    cross-validation numbers from --train --k-folds for an honest
    generalisation estimate.
    """
    model, normalizer, config = load_trained_model(model_path)
    device = torch.device("cpu")

    dataset = load_dataset(config, kg_path)
    if not dataset:
        print("[ERROR] Empty dataset")
        return 1

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    metrics = evaluate(model, loader, normalizer, device)

    print("=" * 60)
    print("CGCNN evaluation (whole dataset)")
    print("=" * 60)
    print(f"  MAE      : {metrics['mae']:.4f} eV/atom")
    print(f"  RMSE     : {metrics['rmse']:.4f} eV/atom")
    print(f"  R2       : {metrics['r2']:.3f}")
    print(f"  max error: {metrics['max_error']:.4f} eV/atom")
    print(f"  n        : {metrics['n']}")
    print()
    print("NOTE: includes training data for a single-split checkpoint.")
    print("      Use --train --k-folds 5 for a generalisation estimate.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CGCNN baseline for formation-energy prediction"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, metavar="MPID",
                        help="Predict for one material, e.g. mp-2790")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate a saved checkpoint over the dataset")
    parser.add_argument("--k-folds", type=int, default=0,
                        help="k-fold cross-validation (0 = single split)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--num-conv-layers", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=None,
                        help="Neighbour cutoff radius in Angstrom")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not any([args.train, args.predict, args.evaluate]):
        parser.print_help()
        return 0

    config = CGCNNConfig()
    for arg_name, field in [
        ("epochs", "num_epochs"),
        ("batch_size", "batch_size"),
        ("hidden_channels", "hidden_channels"),
        ("num_conv_layers", "num_conv_layers"),
        ("cutoff", "cutoff_radius"),
        ("seed", "seed"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, field, value)

    if args.train:
        code = train_model(config, k_folds=args.k_folds)
        if code != 0:
            return code

    if args.predict:
        predict_material(args.predict)

    if args.evaluate:
        return evaluate_saved_model()

    return 0


if __name__ == "__main__":
    sys.exit(main())
