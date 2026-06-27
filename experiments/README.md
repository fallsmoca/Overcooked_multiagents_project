# Experiment Commands

The project workflow and full commands are documented in the repository
[`README.md`](../README.md).

Main entry points:

```powershell
# Train or evaluate one PPO pair
python experiments/run_overcooked_experiment.py --help

# Run the five-layout, three-seed PPO baseline
powershell -ExecutionPolicy Bypass -File experiments/run_full_ppo_baseline.ps1

# Plot completed evaluations
python experiments/plot_results.py --runs-dir runs/baseline_full --out runs/baseline_full/eval_bar.png

# Run a saved pair in the terminal
python experiments/demo_overcooked_policy.py --help
```

All run directories contain `metadata.json`, saved models, TensorBoard events,
and evaluation results. Completed matrix runs are skipped automatically.
