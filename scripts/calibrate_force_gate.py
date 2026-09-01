#!/usr/bin/env python
"""Measure the MACE residual-force distribution across the whole corpus.

Purpose: choose FORCE_GATE_EV_PER_ANG from data instead of guessing it.

Every structure in the KG is a DFT-relaxed Materials Project geometry, so
DFT's own forces on those atoms are approximately zero. Whatever residual
force MACE reports is therefore MACE-vs-DFT geometric disagreement. This
script computes that value for every material and prints the distribution,
so the Critic's escalation gate can be set where the tail actually is.

Read the output honestly. If the distribution is tight and no material
approaches any sensible threshold, the correct conclusion is "MACE agrees
with DFT geometry across this corpus and escalation does not fire" -- NOT
"lower the gate until something trips". A gate tuned downwards to
manufacture escalations would be reporting an artefact of the threshold,
not a property of the materials.

Usage:
    python scripts/calibrate_force_gate.py
    python scripts/calibrate_force_gate.py --limit 20      # quick check
    python scripts/calibrate_force_gate.py --out data/processed/force_dist.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Ensure project root is in Python path for relative imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from kg.graph_store import load_graph, rehydrate_node, DEFAULT_KG_JSON
from kg.schema import NodeType


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "force_distribution.json"

# Candidate gates to report hit-rates for. 0.05 eV/A is a common tight
# convergence criterion in DFT relaxation; 0.1 is a common looser one.
CANDIDATE_GATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the Critic's residual-force escalation gate"
    )
    parser.add_argument("--kg-path", type=str, default=str(DEFAULT_KG_JSON))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N materials")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    kg_path = Path(args.kg_path)
    if not kg_path.exists():
        print(f"[ERROR] KG not found: {kg_path}")
        return 1

    from agent.predictor import create_predictor

    G = load_graph(kg_path)
    material_nids = [
        nid for nid, data in G.nodes(data=True)
        if data.get("type") == NodeType.MATERIAL.value
    ]
    material_nids.sort()
    if args.limit:
        material_nids = material_nids[: args.limit]

    print("=" * 64)
    print("MACE residual-force distribution")
    print("=" * 64)
    print(f"Materials: {len(material_nids)}")
    print("(all are DFT-relaxed MP geometries, so DFT forces ~ 0 by construction)")
    print()

    predictor = create_predictor()
    records: list[dict[str, Any]] = []

    for i, nid in enumerate(material_nids, 1):
        material = rehydrate_node(G, nid)
        result = predictor.predict(material)
        records.append({
            "mpid": result.material_id,
            "formula": material.formula_pretty,
            "elements": list(material.elements),
            "is_oxide": "O" in material.elements,
            "max_residual_force": result.max_residual_force,
            "formation_energy_per_atom": result.formation_energy_per_atom,
            "failed": result.prediction_failed,
        })
        if i % 10 == 0:
            print(f"  {i}/{len(material_nids)}")

    ok = [r for r in records if not r["failed"]]
    failed = [r for r in records if r["failed"]]

    if not ok:
        print("\n[ERROR] No successful predictions.")
        return 1

    forces = np.array([r["max_residual_force"] for r in ok], dtype=np.float64)

    print()
    print("=" * 64)
    print("Distribution of max residual force (eV/Angstrom)")
    print("=" * 64)
    print(f"  n            : {forces.size}   (failed: {len(failed)})")
    print(f"  min          : {forces.min():.4f}")
    print(f"  25th pct     : {np.percentile(forces, 25):.4f}")
    print(f"  median       : {np.median(forces):.4f}")
    print(f"  75th pct     : {np.percentile(forces, 75):.4f}")
    print(f"  90th pct     : {np.percentile(forces, 90):.4f}")
    print(f"  95th pct     : {np.percentile(forces, 95):.4f}")
    print(f"  max          : {forces.max():.4f}")
    print(f"  mean +/- std : {forces.mean():.4f} +/- {forces.std():.4f}")

    print()
    print("Escalation rate at candidate gates:")
    gate_table = {}
    for gate in CANDIDATE_GATES:
        n_over = int(np.sum(forces > gate))
        pct = 100.0 * n_over / forces.size
        gate_table[str(gate)] = {"n_escalated": n_over, "percent": round(pct, 1)}
        print(f"  > {gate:5.2f} eV/A : {n_over:4d} / {forces.size}  ({pct:5.1f}%)")

    print()
    print("Highest-force materials (escalation candidates):")
    for r in sorted(ok, key=lambda x: -x["max_residual_force"])[:10]:
        print(f"  {r['mpid']:>12s}  {r['formula']:<12s}  {r['max_residual_force']:.4f} eV/A")

    # Oxide split: oxides are the systems where MP's own corrections and
    # MACE's training distribution differ most, so it is worth knowing
    # whether force disagreement tracks that split too.
    ox = np.array([r["max_residual_force"] for r in ok if r["is_oxide"]])
    nox = np.array([r["max_residual_force"] for r in ok if not r["is_oxide"]])
    if ox.size and nox.size:
        print()
        print("Oxide split:")
        print(f"  oxides     n={ox.size:3d}  median={np.median(ox):.4f}  max={ox.max():.4f}")
        print(f"  non-oxides n={nox.size:3d}  median={np.median(nox):.4f}  max={nox.max():.4f}")

    if failed:
        print()
        print(f"Failed predictions ({len(failed)}):")
        for r in failed[:10]:
            print(f"  {r['mpid']} ({r['formula']})")

    print()
    print("-" * 64)
    print("Choosing a gate: pick a value that isolates a genuine tail. If no")
    print("material approaches any sensible threshold, report that escalation")
    print("does not fire on this corpus -- do not lower the gate to force it.")
    print("-" * 64)

    payload = {
        "n_materials": len(records),
        "n_successful": len(ok),
        "n_failed": len(failed),
        "statistics": {
            "min": float(forces.min()),
            "p25": float(np.percentile(forces, 25)),
            "median": float(np.median(forces)),
            "p75": float(np.percentile(forces, 75)),
            "p90": float(np.percentile(forces, 90)),
            "p95": float(np.percentile(forces, 95)),
            "max": float(forces.max()),
            "mean": float(forces.mean()),
            "std": float(forces.std()),
        },
        "escalation_rate_by_gate": gate_table,
        "per_material": records,
        "note": (
            "All structures are DFT-relaxed MP geometries; DFT forces on them "
            "are ~0 by construction, so these values are MACE-vs-DFT geometric "
            "disagreement, used as the Critic's escalation signal."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
