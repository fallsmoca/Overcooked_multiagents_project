# Cooperative Overcooked MARL Project

This project trains two independently optimized PPO agents to cooperate in the
five standard Overcooked layouts. It includes reproducible training,
evaluation, plotting, TensorBoard logs, saved models, and a browser demo.

## Setup

Run from the repository root in PowerShell:

```powershell
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai
pip install -e .
```

## Layouts

| Standard name | Training argument |
|---|---|
| Cramped Room | `simple` |
| Asymmetric Advantages | `unident_s` |
| Coordination Ring | `random1` |
| Forced Coordination | `random0` |
| Counter Circuit | `random3` |

Each layout is trained separately. The baseline uses the same PPO parameters
and default reward shaping on every layout.

## Quick Validation

Use this command to verify training, model saving, and evaluation:

```powershell
python experiments/run_overcooked_experiment.py `
  --layout simple `
  --reward-profile default `
  --seed 0 `
  --timesteps 2048 `
  --horizon 100 `
  --n-steps 512 `
  --batch-size 256 `
  --eval-episodes 2 `
  --ego-config experiments/configs/ppo_baseline.json `
  --alt-config experiments/configs/ppo_baseline.json `
  --output-dir runs/smoke
```

## Train One PPO Baseline

Example for Cramped Room:

```powershell
python experiments/run_overcooked_experiment.py `
  --layout simple `
  --reward-profile default `
  --ego PPO `
  --alt PPO `
  --seed 0 `
  --timesteps 500000 `
  --horizon 400 `
  --n-steps 2048 `
  --batch-size 256 `
  --eval-episodes 50 `
  --ego-config experiments/configs/ppo_baseline.json `
  --alt-config experiments/configs/ppo_baseline.json `
  --output-dir runs/baseline_full
```

```powershell
python experiments/run_overcooked_experiment.py `
  --layout random1 `
  --reward-profile default `
  --ego PPO `
  --alt PPO `
  --seed 0 `
  --timesteps 1000000 `
  --horizon 400 `
  --n-steps 2048 `
  --batch-size 256 `
  --eval-episodes 50 `
  --ego-config experiments/configs/ppo_baseline.json `
  --alt-config experiments/configs/ppo_baseline.json `
  --output-dir runs/baseline_full
```



Use `1000000` timesteps for `unident_s`, `random1`, `random0`, and `random3`.

## Train All Five Layouts

The launcher runs all five layouts with seeds `0`, `1`, and `2`. Completed
runs are skipped, so the command can be used to resume an interrupted matrix.

```powershell
powershell -ExecutionPolicy Bypass -File experiments/run_full_ppo_baseline.ps1
```

To run selected seeds or use another output directory:

```powershell
powershell -ExecutionPolicy Bypass -File experiments/run_full_ppo_baseline.ps1 `
  -Seeds 0,1,2 `
  -OutputDir runs/baseline_full
```

## Results

Each run writes:

```text
runs/baseline_full/<layout>_ppo-ppo_<reward-profile>_seed<seed>/
|-- metadata.json
|-- models/
|   |-- ego.zip
|   `-- alt.zip
|-- tensorboard/
`-- results/
    |-- eval_summary.json
    `-- eval_episodes.csv
```

View training curves:

```powershell
tensorboard --logdir runs/baseline_full
```

Then open <http://127.0.0.1:6006>.

Generate the evaluation summary plot:

```powershell
python experiments/plot_results.py `
  --runs-dir runs/baseline_full `
  --out runs/baseline_full/eval_bar.png
```

## Evaluate Saved Models

```powershell
python experiments/run_overcooked_experiment.py `
  --eval-only `
  --layout simple `
  --reward-profile default `
  --eval-episodes 50 `
  --ego-load runs/baseline_full/simple_ppo-ppo_default_seed0/models/ego.zip `
  --alt-load runs/baseline_full/simple_ppo-ppo_default_seed0/models/alt.zip `
  --output-dir runs/reevaluation
```

Use mean sparse return as the primary metric. One delivered soup contributes
20 sparse reward, so `mean_sparse_return / 20` is the mean delivery count.

## Browser Demo

Start the trained Cramped Room policy pair:

```powershell
python overcookedgym/overcooked-flask/app.py `
  --modelpath_p0 runs/baseline_full/simple_ppo-ppo_default_seed0/models/ego.zip `
  --modelpath_p1 runs/baseline_full/simple_ppo-ppo_default_seed0/models/alt.zip `
  --layout_name simple `
  --port 5000
```

```powershell
python overcookedgym/overcooked-flask/app.py `
  --modelpath_p0 runs/IPPO_random0_default/random0_ppo-ppo_default_seed0/models/ego.zip `
  --modelpath_p1 runs/IPPO_random0_default/random0_ppo-ppo_default_seed0/models/alt.zip `
  --layout_name random0 `
  --port 5000
```



Open <http://127.0.0.1:5000>, keep both players set to **AI agent**, select
**Cramped Room (simple)**, set the game length, deselect the form controls, and
press Enter. The server uses deterministic policy actions by default; add
`--stochastic` to sample actions.

For another map, change the model paths, `--layout_name`, and browser layout
using the layout table above.

## Terminal Demo

```powershell
python experiments/demo_overcooked_policy.py `
  --layout simple `
  --reward-profile default `
  --ego-load runs/baseline_full/simple_ppo-ppo_default_seed0/models/ego.zip `
  --alt-load runs/baseline_full/simple_ppo-ppo_default_seed0/models/alt.zip `
  --max-steps 400 `
  --sleep 0.15
```

## Project Files

- `experiments/run_overcooked_experiment.py`: train and evaluate one policy pair.
- `experiments/configs/ppo_baseline.json`: PPO hyperparameters.
- `experiments/run_full_ppo_baseline.ps1`: five-layout baseline launcher.
- `experiments/plot_results.py`: evaluation summary plot.
- `experiments/demo_overcooked_policy.py`: terminal visualization.
- `overcookedgym/overcooked-flask/app.py`: browser visualization server.
- `report_outline.md`: report structure and experiment checklist.
