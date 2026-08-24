import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')

# List all experiments
exps = mlflow.search_experiments()
print(f"Total experiments: {len(exps)}")

# Get runs with metrics for a specific experiment
exp = mlflow.get_experiment_by_name('catalyst_discovery/YOUR_CAMPAIGN_ID')
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format='pandas')
print(runs[['run_id', 'metrics.total_cost', 'metrics.materials_evaluated']])
