param(
    [int[]]$Seeds = @(0, 1, 2),
    [string]$OutputDir = "runs/baseline_full",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ppoConfig = "experiments/configs/ppo_baseline.json"
$experiments = @(
    @{ Layout = "simple";  Reward = "default"; Timesteps = 500000 },
    @{ Layout = "unident_s"; Reward = "default"; Timesteps = 1000000 },
    @{ Layout = "random1"; Reward = "default"; Timesteps = 1000000 },
    @{ Layout = "random0"; Reward = "default"; Timesteps = 1000000 },
    @{ Layout = "random3"; Reward = "default"; Timesteps = 1000000 }
)

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($seed in $Seeds) {
    foreach ($exp in $experiments) {
        $runName = "{0}_ppo-ppo_{1}_seed{2}" -f $exp.Layout, $exp.Reward, $seed
        $summaryPath = Join-Path $OutputDir "$runName/results/eval_summary.json"

        if ((Test-Path $summaryPath) -and -not $Force) {
            Write-Host "Skipping completed run: $runName"
            continue
        }

        Write-Host "Starting run: $runName ($($exp.Timesteps) timesteps)"
        python experiments/run_overcooked_experiment.py `
            --layout $exp.Layout `
            --reward-profile $exp.Reward `
            --ego PPO `
            --alt PPO `
            --seed $seed `
            --timesteps $exp.Timesteps `
            --horizon 400 `
            --n-steps 2048 `
            --batch-size 256 `
            --eval-episodes 50 `
            --ego-config $ppoConfig `
            --alt-config $ppoConfig `
            --output-dir $OutputDir

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $runName with exit code $LASTEXITCODE"
        }
    }
}

python experiments/plot_results.py `
    --runs-dir $OutputDir `
    --out (Join-Path $OutputDir "eval_bar.png")

if ($LASTEXITCODE -ne 0) {
    throw "Plot generation failed with exit code $LASTEXITCODE"
}

Write-Host "Full PPO baseline matrix completed."
