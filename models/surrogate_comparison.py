"""MACE vs CGCNN surrogate comparison on formation energy per atom.

Evaluates both surrogates against the same Materials Project ground truth
on the same catalyst subset, reporting accuracy, uncertainty behaviour and
inference cost.

WHAT IS COMPARED
----------------
Both models predict `formation_energy_per_atom` (eV/atom), and are scored
against the MP-derived value stored in the KG:

    MACE  -- foundation checkpoint (mace-mpa-0-medium), zero-shot on this
             corpus. Raw energy per atom is converted to a formation energy
             using MACE elemental references (models/elemental_references.py)
             so that both terms come from the same checkpoint.
    CGCNN -- trained from scratch on this corpus
             (models/baseline_cgcnn.py), predicting the target directly.

This is a like-for-like comparison. Neither model predicts e_above_hull;
that quantity is read from MP and used only by the Critic's stability gate.

THE OXIDE OFFSET -- READ BEFORE INTERPRETING RESULTS
----------------------------------------------------
MP's `formation_energy_per_atom` includes empirical anion corrections
fitted to experimental formation enthalpies (Jain et al., PRB 84, 045115
(2011); Wang et al., Sci Rep 11, 15496 (2021)). The MACE-derived formation
energies apply NO such correction, and oxygen is referenced against
molecular O2 rather than a corrected gas-phase reference.

MACE errors on oxygen-containing compounds therefore carry a roughly
systematic offset that non-oxides do not. Averaging over the whole set
would hide this, so every metric here is reported three ways: overall,
oxides only, and non-oxides only. The non-oxide numbers are the cleaner
measure of MACE's accuracy; the oxide-vs-non-oxide gap is itself the
evidence for the offset.

CGCNN is unaffected: it is trained directly on MP's corrected values, so
it learns whatever correction is present in the targets.

Usage:
    python models/surrogate_comparison.py
    python models/surrogate_comparison.py --limit 30        # quick run
    python models/surrogate_comparison.py --skip-mace       # CGCNN only

Output:
    models/mace_vs_cgcnn_comparison.json
    notebooks/plots/mace_vs_cgcnn_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg.graph_store import load_graph, rehydrate_node, DEFAULT_KG_JSON
from kg.schema import MaterialNode, NodeType, PropertyName


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "models" / "mace_vs_cgcnn_comparison.json"
DEFAULT_PLOT_DIR = REPO_ROOT / "notebooks" / "plots"

TARGET_PROPERTY = PropertyName.FORMATION_ENERGY_PER_ATOM.value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_evaluation_set(
    kg_path: Path = DEFAULT_KG_JSON,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Collect materials that have both a structure and a ground-truth target.

    Args:
        kg_path: path to kg.json
        limit: optional cap on the number of materials (for quick runs)

    Returns:
        List of dicts with mpid, material node, ground truth and composition
    """
    G = load_graph(kg_path)

    targets: dict[str, float] = {}
    for _nid, data in G.nodes(data=True):
        if data.get("type") == NodeType.PROPERTY.value and data.get("name") == TARGET_PROPERTY:
            mpid, value = data.get("mpid"), data.get("value")
            if mpid is not None and value is not None:
                targets[mpid] = float(value)

    entries: list[dict[str, Any]] = []
    for nid, data in G.nodes(data=True):
        if data.get("type") != NodeType.MATERIAL.value:
            continue
        mpid = data.get("mpid")
        if mpid is None or mpid not in targets:
            continue

        material = rehydrate_node(G, nid)
        if not material.structure_id:
            continue

        entries.append({
            "mpid": mpid,
            "material": material,
            "formula": material.formula_pretty,
            "elements": list(material.elements),
            "ground_truth": targets[mpid],
            "is_oxide": "O" in material.elements,
        })

    entries.sort(key=lambda e: e["mpid"])
    if limit:
        entries = entries[:limit]

    n_oxide = sum(1 for e in entries if e["is_oxide"])
    print(f"Evaluation set: {len(entries)} materials "
          f"({n_oxide} oxides, {len(entries) - n_oxide} non-oxides)")

    if not entries:
        print(f"  No materials carry '{TARGET_PROPERTY}'. Rebuild the KG:")
        print("    python data/download.py")
        print("    python kg/build_graph.py --clear-cache")

    return entries


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_mace(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Run the MACE predictor over the evaluation set.

    Returns:
        mpid -> {value, uncertainty, seconds} (value may be None on failure)
    """
    from agent.predictor import create_predictor

    print("\nRunning MACE predictions...")
    predictor = create_predictor()
    results: dict[str, dict[str, Any]] = {}

    for i, entry in enumerate(entries, 1):
        start = time.perf_counter()
        try:
            prediction = predictor.predict(entry["material"])
            elapsed = time.perf_counter() - start
            results[entry["mpid"]] = {
                # The formation energy, NOT the raw energy per atom -- this is
                # what makes the comparison like-for-like.
                "value": prediction.formation_energy_per_atom,
                "raw_energy_per_atom": prediction.property_value,
                "uncertainty": prediction.uncertainty,
                "seconds": elapsed,
                "failed": prediction.prediction_failed,
            }
        except Exception as exc:
            results[entry["mpid"]] = {
                "value": None, "raw_energy_per_atom": None, "uncertainty": None,
                "seconds": time.perf_counter() - start, "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if i % 20 == 0:
            print(f"  {i}/{len(entries)}")

    n_ok = sum(1 for r in results.values() if r["value"] is not None)
    print(f"  MACE: {n_ok}/{len(entries)} succeeded")
    if n_ok == 0:
        print("  (formation energies are None -- run: python models/elemental_references.py)")
    return results


def run_cgcnn(entries: list[dict[str, Any]], kg_path: Path) -> dict[str, dict[str, Any]]:
    """Run the trained CGCNN over the evaluation set.

    Returns:
        mpid -> {value, uncertainty, seconds}
    """
    import torch
    from pymatgen.core import Structure
    from models.baseline_cgcnn import (
        GaussianExpansion,
        load_trained_model,
        structure_to_graph,
    )

    print("\nRunning CGCNN predictions...")
    try:
        model, normalizer, config = load_trained_model()
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")
        return {}

    expansion = GaussianExpansion(0.0, config.cutoff_radius, config.n_gaussians)
    G = load_graph(kg_path)

    cif_paths = {
        data.get("mpid"): data.get("cif_path")
        for _nid, data in G.nodes(data=True)
        if data.get("type") == NodeType.STRUCTURE.value and data.get("mpid")
    }

    results: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(entries, 1):
        mpid = entry["mpid"]
        start = time.perf_counter()
        try:
            cif = cif_paths.get(mpid)
            if not cif:
                raise FileNotFoundError(f"no cif_path for {mpid}")
            cif_path = Path(cif)
            if not cif_path.is_absolute():
                cif_path = REPO_ROOT / cif_path

            structure = Structure.from_file(str(cif_path))
            graph = structure_to_graph(structure, None, config, expansion, mpid=mpid)
            if graph is None:
                raise ValueError("empty graph")

            mean_norm, std_norm = model.predict_with_uncertainty(
                graph, n_passes=config.mc_dropout_passes
            )
            results[mpid] = {
                "value": float(normalizer.denorm(mean_norm)),
                # Spread scales with the normaliser but does not shift.
                "uncertainty": float(std_norm * normalizer.std),
                "seconds": time.perf_counter() - start,
                "failed": False,
            }
        except Exception as exc:
            results[mpid] = {
                "value": None, "uncertainty": None,
                "seconds": time.perf_counter() - start, "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if i % 20 == 0:
            print(f"  {i}/{len(entries)}")

    n_ok = sum(1 for r in results.values() if r["value"] is not None)
    print(f"  CGCNN: {n_ok}/{len(entries)} succeeded")
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Error metrics for (prediction, ground_truth) pairs, in eV/atom.

    `mean_signed_error` is reported alongside MAE because a systematic
    offset (such as the oxide anion correction) shows up as a large signed
    error while MAE alone would not reveal its direction.
    """
    if not pairs:
        return {"n": 0}

    preds = np.array([p for p, _ in pairs], dtype=np.float64)
    truth = np.array([t for _, t in pairs], dtype=np.float64)
    errors = preds - truth

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))

    metrics = {
        "n": len(pairs),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_signed_error": float(np.mean(errors)),
        "max_error": float(np.max(np.abs(errors))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
    }

    if len(pairs) > 1 and preds.std() > 1e-12 and truth.std() > 1e-12:
        metrics["pearson_r"] = float(np.corrcoef(preds, truth)[0, 1])
    else:
        metrics["pearson_r"] = float("nan")

    return metrics


def metrics_by_subset(
    entries: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute metrics overall and split by oxide / non-oxide.

    The split is the point: see the module docstring on the anion-correction
    offset. Reporting only the overall number would average the offset away.
    """
    buckets: dict[str, list[tuple[float, float]]] = {"overall": [], "oxides": [], "non_oxides": []}

    for entry in entries:
        prediction = predictions.get(entry["mpid"])
        if not prediction or prediction.get("value") is None:
            continue
        pair = (float(prediction["value"]), float(entry["ground_truth"]))
        buckets["overall"].append(pair)
        buckets["oxides" if entry["is_oxide"] else "non_oxides"].append(pair)

    return {name: compute_metrics(pairs) for name, pairs in buckets.items()}


def timing_stats(predictions: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Mean and median seconds per successful prediction."""
    times = [r["seconds"] for r in predictions.values() if not r.get("failed")]
    if not times:
        return {"mean_seconds": float("nan"), "median_seconds": float("nan"), "n": 0}
    return {
        "mean_seconds": float(np.mean(times)),
        "median_seconds": float(np.median(times)),
        "n": len(times),
    }


def uncertainty_stats(predictions: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Summary of each model's uncertainty estimates.

    Note the two uncertainties are NOT the same construct: MACE's is the
    spread of MC-Dropout passes over a frozen foundation model, CGCNN's is
    the spread over dropout in a model trained on this corpus. Compare them
    qualitatively, not as calibrated equivalents.
    """
    values = [
        r["uncertainty"] for r in predictions.values()
        if not r.get("failed") and r.get("uncertainty") is not None
    ]
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    entries: list[dict[str, Any]],
    mace: dict[str, dict[str, Any]],
    cgcnn: dict[str, dict[str, Any]],
    output_path: Path,
) -> Optional[Path]:
    """Parity plots and an error-distribution panel.

    Oxides and non-oxides are drawn as separate series so the offset is
    visible rather than buried in an aggregate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(predictions: dict[str, dict[str, Any]], oxide: bool):
        xs, ys = [], []
        for entry in entries:
            if entry["is_oxide"] != oxide:
                continue
            prediction = predictions.get(entry["mpid"])
            if not prediction or prediction.get("value") is None:
                continue
            xs.append(entry["ground_truth"])
            ys.append(prediction["value"])
        return np.array(xs), np.array(ys)

    panels = [(name, preds) for name, preds in (("MACE", mace), ("CGCNN", cgcnn)) if preds]
    if not panels:
        print("No predictions to plot.")
        return None

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 5))
    axes = np.atleast_1d(axes)

    for ax, (name, preds) in zip(axes, panels):
        for oxide, colour, label in [(False, "steelblue", "non-oxide"), (True, "coral", "oxide")]:
            xs, ys = series(preds, oxide)
            if xs.size:
                ax.scatter(xs, ys, alpha=0.65, s=28, c=colour, label=label)

        all_x, all_y = [], []
        for oxide in (False, True):
            xs, ys = series(preds, oxide)
            all_x.extend(xs); all_y.extend(ys)
        if all_x:
            lo = min(min(all_x), min(all_y))
            hi = max(max(all_x), max(all_y))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.5, label="y = x")

        ax.set_xlabel("MP formation energy (eV/atom)")
        ax.set_ylabel(f"{name} prediction (eV/atom)")
        ax.set_title(f"{name} vs MP ground truth")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    ax_err = axes[-1]
    for name, preds, colour in (("MACE", mace, "steelblue"), ("CGCNN", cgcnn, "coral")):
        if not preds:
            continue
        errors = [
            preds[e["mpid"]]["value"] - e["ground_truth"]
            for e in entries
            if preds.get(e["mpid"]) and preds[e["mpid"]].get("value") is not None
        ]
        if errors:
            ax_err.hist(errors, bins=25, alpha=0.55, label=name, color=colour)
    ax_err.axvline(0, color="black", lw=1)
    ax_err.set_xlabel("Signed error (eV/atom)")
    ax_err.set_ylabel("Count")
    ax_err.set_title("Error distribution")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(name: str, subsets: dict[str, Any]) -> None:
    """Print one model's metrics, split by subset."""
    print(f"\n{name}")
    print("-" * 60)
    for subset in ("overall", "non_oxides", "oxides"):
        m = subsets.get(subset, {})
        if not m.get("n"):
            print(f"  {subset:11s} (no data)")
            continue
        print(
            f"  {subset:11s} n={m['n']:3d}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  "
            f"signed={m['mean_signed_error']:+.4f}  R2={m['r2']:.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MACE and CGCNN surrogates on formation energy per atom"
    )
    parser.add_argument("--kg-path", type=str, default=str(DEFAULT_KG_JSON))
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N materials")
    parser.add_argument("--skip-mace", action="store_true")
    parser.add_argument("--skip-cgcnn", action="store_true")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    kg_path = Path(args.kg_path)
    if not kg_path.exists():
        print(f"[ERROR] KG not found: {kg_path}")
        print("        Run: python kg/build_graph.py")
        return 1

    print("=" * 60)
    print("MACE vs CGCNN -- formation energy per atom (eV/atom)")
    print("=" * 60)

    entries = load_evaluation_set(kg_path, limit=args.limit)
    if not entries:
        return 1

    mace = {} if args.skip_mace else run_mace(entries)
    cgcnn = {} if args.skip_cgcnn else run_cgcnn(entries, kg_path)

    if not mace and not cgcnn:
        print("\n[ERROR] Neither model produced predictions.")
        return 1

    mace_subsets = metrics_by_subset(entries, mace) if mace else {}
    cgcnn_subsets = metrics_by_subset(entries, cgcnn) if cgcnn else {}

    print()
    print("=" * 60)
    print("Results (eV/atom)")
    print("=" * 60)
    if mace_subsets:
        print_report("MACE (zero-shot foundation model)", mace_subsets)
    if cgcnn_subsets:
        print_report("CGCNN (trained on this corpus)", cgcnn_subsets)

    # The oxide/non-oxide MAE gap is the anion-correction offset made visible.
    if mace_subsets:
        ox = mace_subsets.get("oxides", {})
        nox = mace_subsets.get("non_oxides", {})
        if ox.get("n") and nox.get("n"):
            print()
            print("Oxide offset check (MACE):")
            print(f"  non-oxide signed error {nox['mean_signed_error']:+.4f} eV/atom")
            print(f"  oxide     signed error {ox['mean_signed_error']:+.4f} eV/atom")
            print(f"  gap                    {ox['mean_signed_error'] - nox['mean_signed_error']:+.4f} eV/atom")
            print("  A large gap is expected: MP applies empirical anion corrections")
            print("  to oxides that the MACE-reference formation energies do not.")

    print()
    print("Inference cost (seconds per material)")
    print("-" * 60)
    for name, preds in (("MACE", mace), ("CGCNN", cgcnn)):
        if preds:
            t = timing_stats(preds)
            print(f"  {name:6s} mean={t['mean_seconds']:.4f}  median={t['median_seconds']:.4f}  n={t['n']}")

    print()
    print("Uncertainty estimates (eV/atom, not cross-calibrated)")
    print("-" * 60)
    for name, preds in (("MACE", mace), ("CGCNN", cgcnn)):
        if preds:
            u = uncertainty_stats(preds)
            if u["n"]:
                print(f"  {name:6s} mean={u['mean']:.4f}  median={u['median']:.4f}  max={u['max']:.4f}")

    payload = {
        "target_property": TARGET_PROPERTY,
        "n_materials": len(entries),
        "n_oxides": sum(1 for e in entries if e["is_oxide"]),
        "mace": {
            "metrics": mace_subsets,
            "timing": timing_stats(mace) if mace else None,
            "uncertainty": uncertainty_stats(mace) if mace else None,
        },
        "cgcnn": {
            "metrics": cgcnn_subsets,
            "timing": timing_stats(cgcnn) if cgcnn else None,
            "uncertainty": uncertainty_stats(cgcnn) if cgcnn else None,
        },
        "per_material": [
            {
                "mpid": e["mpid"],
                "formula": e["formula"],
                "is_oxide": e["is_oxide"],
                "ground_truth": e["ground_truth"],
                "mace": mace.get(e["mpid"], {}).get("value"),
                "cgcnn": cgcnn.get(e["mpid"], {}).get("value"),
            }
            for e in entries
        ],
        "caveats": [
            "MACE formation energies use MACE elemental references with molecular O2 "
            "for oxygen; MP applies empirical anion corrections. Oxide errors carry a "
            "systematic offset, which is why metrics are split oxide/non-oxide.",
            "CGCNN is trained on this ~130-material corpus and evaluated on the same "
            "corpus here; use the cross-validation metrics from baseline_cgcnn.py "
            "--train --k-folds for a generalisation estimate.",
            "MACE is zero-shot on this corpus and was never fitted to these targets.",
            "The two uncertainty estimates are not cross-calibrated.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")

    if not args.no_plot:
        plot_comparison(entries, mace, cgcnn, DEFAULT_PLOT_DIR / "mace_vs_cgcnn_comparison.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
