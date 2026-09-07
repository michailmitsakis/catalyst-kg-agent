#!/usr/bin/env python
"""Query MLflow database to extract best_candidate_e_above_hull from campaigns."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

def query_mlflow_best_candidates() -> List[Dict[str, Any]]:
    """Extract best_candidate_e_above_hull metrics from all campaign runs.
    
    Returns:
        List of dicts with experiment info and metrics
    """
    db_path = Path("mlflow.db")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    results = []
    
    # Get all catalyst_discovery experiments
    cursor.execute("""
        SELECT experiment_id, name 
        FROM experiments 
        WHERE name LIKE 'catalyst_discovery/%'
    """)
    experiments = cursor.fetchall()
    
    print("="*60)
    print("MLflow Database Query: Extract Best Candidate e_above_hull")
    print("="*60)
    print(f"\nFound {len(experiments)} catalyst_discovery experiments\n")
    
    for exp_id, exp_name in experiments:
        # Get all runs for this experiment
        cursor.execute("SELECT run_uuid FROM runs WHERE experiment_id = ?", (exp_id,))
        run_uuids = [row[0] for row in cursor.fetchall()]
        
        if not run_uuids:
            continue
            
        print(f"{'='*60}")
        print(f"Experiment: {exp_name} (id={exp_id})")
        print('='*60)
        
        # Check each run for best_candidate_e_above_hull metric
        for run_uuid in run_uuids:
            cursor.execute("SELECT key, value FROM metrics WHERE run_uuid = ?", (run_uuid,))
            metrics = cursor.fetchall()
            
            # Look for best_candidate_e_above_hull specifically
            for metric_key, metric_value in metrics:
                if 'best_candidate' in metric_key.lower() and 'e_above' in metric_key.lower():
                    print(f"\n  *** FOUND: {metric_key} = {metric_value}")
                    
                    # Also get other relevant metrics
                    cursor.execute("SELECT key, value FROM metrics WHERE run_uuid = ?", (run_uuid,))
                    all_metrics = cursor.fetchall()
                    
                    print(f"\n  All metrics for this run:")
                    for m_key, m_val in all_metrics:
                        val_str = str(m_val)[:100] if isinstance(m_val, str) else str(m_val)
                        print(f"    {m_key}: {val_str}" if len(str(val_str)) > 100 else f"    {m_key}: {val_str}")
                    
                    # Add to results
                    results.append({
                        'experiment_id': exp_id,
                        'experiment_name': exp_name,
                        'run_uuid': run_uuid[:16] + '...',
                        'best_candidate_e_above_hull': metric_value,
                        'total_cost': next((m for m in all_metrics if m[0] == 'total_cost'), None),
                        'materials_evaluated': next((m for m in all_metrics if m[0] == 'materials_evaluated'), None),
                    })
    
    conn.close()
    return results


def main():
    """Main entry point."""
    best_candidates = query_mlflow_best_candidates()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Total experiments checked: {len(best_candidates)}")
    print(f"Campaigns with best_candidate_e_above_hull: {sum(1 for c in best_candidates if c['best_candidate_e_above_hull'] is not None)}")
    
    # Show best candidates sorted by e_above_hull
    valid_candidates = [c for c in best_candidates if c['best_candidate_e_above_hull'] is not None]
    if valid_candidates:
        print(f"\nBest candidates (sorted by e_above_hull):")
        for i, candidate in enumerate(sorted(valid_candidates, key=lambda x: x['best_candidate_e_above_hull']), 1):
            print(f"\n{i}. {candidate['experiment_name']} - {candidate['run_uuid']}")
            print(f"   e_above_hull: {candidate['best_candidate_e_above_hull']} eV/atom")
            if candidate['total_cost']:
                print(f"   Total cost: {candidate['total_cost']} credits")
            if candidate['materials_evaluated']:
                print(f"   Materials evaluated: {candidate['materials_evaluated']}")


if __name__ == "__main__":
    main()
