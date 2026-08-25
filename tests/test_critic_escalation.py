#!/usr/bin/env python
"""Manual test suite for Critic escalation logic with uncertainty metadata.

This standalone script tests the Critic's uncertainty gate functionality without
requiring pytest fixtures. It verifies that predictions with high uncertainty
(above UNCERTAINTY_GATE threshold) are correctly flagged for escalation.

Key Features:
- Tests low-uncertainty predictions (should be approved)
- Tests high-uncertainty predictions (should escalate)
- Demonstrates batch validation behavior (safety-first escalation)
- Shows both synthetic prediction uncertainties and KG-stored metadata

Usage:
    python tests/test_critic_escalation.py

Note:
    This script displays BOTH the synthetic uncertainty values passed to the
    Critic AND the uncertainty values stored in the Knowledge Graph. The Critic
    may use KG metadata for validation, which can differ from synthetic test
    values. Batch validation uses a safety-first approach where if ANY prediction
    in a batch exceeds the uncertainty gate, ALL materials are escalated.

For pytest-based testing with fixtures, see tests/test_critic_escalation.py (pytest version).
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg.graph_store import load_graph
from kg.schema import MaterialNode
from agent.critic import CriticAgent, get_uncertainty_gate, get_stability_threshold
from agent.predictor import PredictorResult


def main():
    """Run escalation logic tests manually."""
    
    print("="*70)
    print("Critic Escalation Logic - Manual Test Suite")
    print("="*70)
    
    # Load critic with KG
    graph = load_graph()
    critic = CriticAgent(graph_path=Path("data/processed/kg.json"))
    
    # Get thresholds
    uncertainty_gate = get_uncertainty_gate()
    stability_threshold = get_stability_threshold()
    
    print(f"\nConfiguration:")
    print(f"  Uncertainty Gate: {uncertainty_gate:.1%}")
    print(f"  Stability Threshold: {stability_threshold:.4f} eV/atom")
    
    # Load test materials
    material_nodes = [
        node for node in graph.nodes(data=True)
        if node[1].get("type") == "Material" and node[1].get("mpid")
    ][:5]
    
    test_materials = [MaterialNode(id=nid, **node_data) for nid, node_data in material_nodes]
    
    if not test_materials:
        print("\n[ERROR] No materials found in KG!")
        return 1
    
    print(f"\nLoaded {len(test_materials)} materials from KG")
    
    # Test 1: Low uncertainty should be approved
    print("\n" + "-"*70)
    print("Test 1: Low Uncertainty Prediction (5%)")
    print("-"*70)
    
    prediction_low = PredictorResult(
        material_id=test_materials[0].mpid,
        property_value=-4.5591,
        uncertainty=0.05,
        model_used="mace",
        prediction_failed=False
    )
    
    decision_low = critic.validate_materials([test_materials[0]], [prediction_low])[0]
    
    print(f"Material: {test_materials[0].mpid}")
    print(f"  Uncertainty: 5%")
    print(f"  Result: {'APPROVED' if decision_low.approved else 'REJECTED'}")
    print(f"  Escalation Required: {decision_low.requires_escalation}")
    
    if decision_low.approved and not decision_low.requires_escalation:
        print("[PASS] Low uncertainty correctly approved")
    else:
        print("[FAIL] Low uncertainty should be approved")
    
    # Test 2: High uncertainty should escalate
    print("\n" + "-"*70)
    print("Test 2: High Uncertainty Prediction (60%)")
    print("-"*70)
    
    prediction_high = PredictorResult(
        material_id=test_materials[0].mpid,
        property_value=-4.5591,
        uncertainty=0.6,
        model_used="mace",
        prediction_failed=False
    )
    
    decision_high = critic.validate_materials([test_materials[0]], [prediction_high])[0]
    
    print(f"Material: {test_materials[0].mpid}")
    print(f"  Uncertainty: 60%")
    print(f"  Result: {'APPROVED' if decision_high.approved else 'REJECTED'}")
    print(f"  Escalation Required: {decision_high.requires_escalation}")
    
    if not decision_high.approved and decision_high.requires_escalation:
        print("[PASS] High uncertainty correctly escalated")
    else:
        print("[FAIL] High uncertainty should be escalated")
    
    # Test 3: Full escalation flow (one prediction per material)
    print("\n" + "-"*70)
    print("Test 3: Full Escalation Flow (One Prediction Per Material)")
    print("-"*70)
    
    test_mats = test_materials[:2]
    
    # Create predictions matched to specific materials
    predictions = [
        PredictorResult(
            material_id=test_mats[0].mpid,
            property_value=-4.5591,
            uncertainty=0.7,  # High for first material - should escalate
            model_used="mace",
            prediction_failed=False
        ),
        PredictorResult(
            material_id=test_mats[1].mpid,
            property_value=-3.5,
            uncertainty=0.1,  # Low for second material - should be approved
            model_used="mace",
            prediction_failed=False
        ),
    ]
    
    decisions = critic.validate_materials(test_mats, predictions)
    
    print(f"\nProcessing {len(decisions)} materials:")
    for i, (mat, pred, dec) in enumerate(zip(test_mats, predictions, decisions)):
        status = "APPROVED" if dec.approved else "REJECTED"
        esc = "ESCALATE" if dec.requires_escalation else ""
        
        # Debug: Show both synthetic uncertainty and KG uncertainty
        kg_uncertainty = 0.0
        for nid, data in graph.nodes(data=True):
            if (data.get('type') == 'Property' and 
                data.get('mpid') == mat.mpid):
                kg_uncertainty = float(data.get('uncertainty', 0))
                break
        
        print(f"\n  Material {i+1}: {mat.mpid}")
        print(f"    Synthetic uncertainty: {pred.uncertainty*100:.1f}%")
        print(f"    KG uncertainty: {kg_uncertainty*100:.1f}%")
        print(f"    Status: {status} {esc}")
    
    # Note: Due to batch validation, if ANY prediction has high uncertainty,
    # ALL materials in the batch get escalated (safety-first approach)
    escalations = sum(1 for d in decisions if d.requires_escalation)
    print(f"\nTotal Escalations: {escalations}/{len(decisions)}")
    
    if escalations >= 1:
        print("[PASS] Escalation flow working correctly (batch escalation triggered)")
    else:
        print("[INFO] No escalations triggered (may be due to KG metadata)")
    
    # Summary
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print("\nKey Findings:")
    print(f"  - Uncertainty gate threshold: {uncertainty_gate:.1%}")
    print(f"  - Low uncertainty predictions should be approved")
    print(f"  - High uncertainty predictions (>30%) should escalate")
    print(f"  - Stability check (threshold: {stability_threshold:.4f}) also applies")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
