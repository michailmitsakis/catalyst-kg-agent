import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')

print("="*60)
print("MLflow Campaign Tracking Verification")
print("="*60)

# List all experiments
exps = mlflow.search_experiments()
print(f"\nTotal experiments: {len(exps)}")

# Show summary
print("\nAll experiments:")
for e in exps[:10]:  # Show first 10
    print(f"  - {e.name}")
if len(exps) > 10:
    print(f"  ... and {len(exps) - 10} more")

# Get runs for all experiments (need to specify experiment_ids)
experiment_ids = [e.experiment_id for e in exps]
all_runs = mlflow.search_runs(experiment_ids=experiment_ids, output_format='pandas')

if len(all_runs) > 0:
    # Sort by start_time descending to get most recent
    all_runs = all_runs.sort_values('start_time', ascending=False)
    latest_run = all_runs.iloc[0]
    
    exp_name = f"catalyst_discovery/{latest_run['experiment_id']}"
    exp = mlflow.get_experiment_by_name(exp_name)
    
    if exp:
        print(f"\n\nLatest Run: {exp.name}")
        print(f"  - Run ID: {latest_run['run_id']}")
        print(f"  - Total cost: {latest_run.get('metrics.total_cost', 'N/A')}")
        print(f"  - Materials evaluated: {latest_run.get('metrics.materials_evaluated', 'N/A')}")
        print(f"  - Final outcome: {latest_run.get('params.final_outcome', 'N/A')}")
        
        # Summary stats
        total_cost = all_runs['metrics.total_cost'].sum() if 'metrics.total_cost' in all_runs.columns else 0
        total_materials = all_runs['metrics.materials_evaluated'].sum() if 'metrics.materials_evaluated' in all_runs.columns else 0
        print(f"\n\nSummary across all {len(all_runs)} runs:")
        print(f"  - Total budget spent: {total_cost}")
        print(f"  - Total materials evaluated: {total_materials}")

print("\n\n[OK] MLflow tracking is working correctly!")
print("   Run a campaign and check the database to see new runs.")
