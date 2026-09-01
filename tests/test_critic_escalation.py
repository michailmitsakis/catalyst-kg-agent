#!/usr/bin/env python
"""Manual test suite for Critic escalation logic with the residual-force gate.

This standalone script tests the Critic's escalation behaviour without
requiring pytest fixtures. It verifies that materials whose MACE residual
force exceeds FORCE_GATE_EV_PER_ANG are flagged for **individual**
escalation (not batch -- each material is evaluated independently).

WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------
This test previously injected a synthetic `uncertainty` value into
PredictorResult. That field is gone: it held an MC-Dropout standard
deviation that was structurally always exactly 0.0, because MACE inference
is deterministic and no dropout is active. The gate could therefore never
fire on real data -- this test only passed because it supplied the values
by hand.

The escalation signal is now `max_residual_force` (eV/Angstrom): the
largest force MACE predicts on a DFT-relaxed MP geometry, where DFT's own
forces are ~0 by construction. See agent/predictor.py for the reasoning.

The forces used below are synthetic, chosen to sit either side of the gate.
For the real distribution across the corpus, run:
    python scripts/calibrate_force_gate.py

Usage:
    python tests/test_critic_escalation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from kg.graph_store import load_graph, rehydrate_node
from kg.schema import MaterialNode, NodeType
from agent.critic import CriticAgent, get_force_gate, get_stability_threshold
from agent.predictor import PredictorResult


def make_prediction(mpid: str, force: float) -> PredictorResult:
    """Build a synthetic PredictorResult with a chosen residual force.

    Energy values are realistic for this corpus (raw MACE energy per atom
    around -6 eV/atom, formation energy around -0.5 eV/atom) so the object
    is not obviously fabricated if it is ever printed.
    """
    return PredictorResult(
        material_id=mpid,
        property_value=-6.146,
        formation_energy_per_atom=-0.484,
        max_residual_force=force,
        model_used="mace",
        prediction_failed=False,
    )


def main():
    """Run escalation logic tests manually."""

    print("=" * 70)
    print("Critic Escalation Logic - Manual Test Suite")
    print("=" * 70)

    graph_path = Path("data/processed/kg.json")
    graph = load_graph(graph_path)
    critic = CriticAgent(graph_path=graph_path)

    force_gate = get_force_gate()
    stability_threshold = get_stability_threshold()

    print(f"\nConfiguration:")
    print(f"  Force gate         : {force_gate:.3f} eV/A")
    print(f"  Stability threshold: {stability_threshold:.4f} eV/atom")

    # Load a few real materials from the KG. rehydrate_node populates
    # structure_id via the HAS_STRUCTURE edge, which MaterialNode(**data)
    # would not do.
    material_nids = [
        nid for nid, data in graph.nodes(data=True)
        if data.get("type") == NodeType.MATERIAL.value
    ][:5]
    test_materials = [rehydrate_node(graph, nid) for nid in material_nids]

    if not test_materials:
        print("\n[ERROR] No materials found in KG!")
        return 1

    print(f"\nLoaded {len(test_materials)} materials from KG")

    failures = 0

    # -- Test 1: force below the gate should be approved -------------------
    low_force = force_gate / 5.0
    print("\n" + "-" * 70)
    print(f"Test 1: Low residual force ({low_force:.3f} eV/A, below gate)")
    print("-" * 70)

    decision_low = critic.validate_materials(
        [test_materials[0]], [make_prediction(test_materials[0].mpid, low_force)]
    )[0]

    print(f"Material: {test_materials[0].mpid} ({test_materials[0].formula_pretty})")
    print(f"  Result           : {'APPROVED' if decision_low.approved else 'REJECTED'}")
    print(f"  Escalation needed: {decision_low.requires_escalation}")

    if decision_low.approved and not decision_low.requires_escalation:
        print("[PASS] Low residual force correctly approved")
    else:
        print(f"[FAIL] Expected approval. Reason given: {decision_low.reason}")
        failures += 1

    # -- Test 2: force above the gate should escalate ----------------------
    high_force = force_gate * 2.0
    print("\n" + "-" * 70)
    print(f"Test 2: High residual force ({high_force:.3f} eV/A, above gate)")
    print("-" * 70)

    decision_high = critic.validate_materials(
        [test_materials[0]], [make_prediction(test_materials[0].mpid, high_force)]
    )[0]

    print(f"Material: {test_materials[0].mpid} ({test_materials[0].formula_pretty})")
    print(f"  Result           : {'APPROVED' if decision_high.approved else 'REJECTED'}")
    print(f"  Escalation needed: {decision_high.requires_escalation}")
    print(f"  Cost impact      : {decision_high.cost_impact}")

    if decision_high.requires_escalation:
        print("[PASS] High residual force correctly escalated")
    else:
        print(f"[FAIL] Expected escalation. Reason given: {decision_high.reason}")
        failures += 1

    # -- Test 3: exactly at the gate should NOT escalate --------------------
    # The check is `max_force <= gate`, so the boundary is inclusive.
    # Boundary behaviour is worth pinning down explicitly: an off-by-one
    # here silently changes the escalation rate across a whole campaign.
    print("\n" + "-" * 70)
    print(f"Test 3: Residual force exactly at the gate ({force_gate:.3f} eV/A)")
    print("-" * 70)

    decision_edge = critic.validate_materials(
        [test_materials[0]], [make_prediction(test_materials[0].mpid, force_gate)]
    )[0]

    print(f"  Escalation needed: {decision_edge.requires_escalation}")
    if not decision_edge.requires_escalation:
        print("[PASS] Gate boundary is inclusive (force == gate does not escalate)")
    else:
        print("[FAIL] Force exactly at the gate should not escalate")
        failures += 1

    # -- Test 4: per-material independence ---------------------------------
    print("\n" + "-" * 70)
    print("Test 4: Independent evaluation (one prediction per material)")
    print("-" * 70)

    if len(test_materials) < 2:
        print("[SKIP] Need at least 2 materials in the KG")
    else:
        test_mats = test_materials[:2]
        predictions = [
            make_prediction(test_mats[0].mpid, force_gate * 3.0),   # should escalate
            make_prediction(test_mats[1].mpid, force_gate / 10.0),  # should not
        ]

        decisions = critic.validate_materials(test_mats, predictions)

        print(f"\nProcessing {len(decisions)} materials:")
        for i, (mat, pred, dec) in enumerate(zip(test_mats, predictions, decisions), 1):
            status = "APPROVED" if dec.approved else "REJECTED"
            esc = "ESCALATE" if dec.requires_escalation else ""
            print(f"\n  Material {i}: {mat.mpid} ({mat.formula_pretty})")
            print(f"    Residual force: {pred.max_residual_force:.3f} eV/A")
            print(f"    Status        : {status} {esc}")

        escalations = sum(1 for d in decisions if d.requires_escalation)
        print(f"\nTotal escalations: {escalations}/{len(decisions)}")

        if escalations == 1:
            print("[PASS] Only the high-force material escalated")
        else:
            print(f"[FAIL] Expected 1 escalation, got {escalations}")
            failures += 1

    # -- Test 5: failed predictions must escalate, not pass silently -------
    print("\n" + "-" * 70)
    print("Test 5: Failed prediction escalates rather than passing")
    print("-" * 70)

    from agent.predictor import FAILED_PREDICTION_FORCE

    failed_pred = PredictorResult(
        material_id=test_materials[0].mpid,
        property_value=None,
        formation_energy_per_atom=None,
        max_residual_force=FAILED_PREDICTION_FORCE,
        model_used="mace",
        prediction_failed=True,
    )
    decision_failed = critic.validate_materials([test_materials[0]], [failed_pred])[0]

    print(f"  Sentinel force   : {FAILED_PREDICTION_FORCE}")
    print(f"  Escalation needed: {decision_failed.requires_escalation}")
    if decision_failed.requires_escalation:
        print("[PASS] Failed prediction escalates")
    else:
        print("[FAIL] A failed prediction must not be treated as trustworthy")
        failures += 1

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Test Complete" if failures == 0 else f"Test Complete - {failures} FAILURE(S)")
    print("=" * 70)
    print("\nNotes:")
    print(f"  - Escalation signal: max residual force, gate {force_gate:.3f} eV/A")
    print("  - Forces here are synthetic, chosen either side of the gate.")
    print("    For the real corpus distribution: python scripts/calibrate_force_gate.py")
    print("  - Stability is checked separately against the KG's MP-derived")
    print(f"    e_above_hull (threshold {stability_threshold:.4f} eV/atom).")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())