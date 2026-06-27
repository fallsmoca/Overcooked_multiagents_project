param(
    [int[]]$Seeds = @(0, 1, 2),
    [int]$Timesteps = 100000,
    [int]$EvalEpisodes = 20,
    [int]$NSteps = 512,
    [int]$BatchSize = 256
)

$ErrorActionPreference = "Stop"

$experiments = @(
    @{ Layout = "simple";  Reward = "default"; Timesteps = [Math]::Min($Timesteps, 50000) },
    @{ Layout = "random0"; Reward = "default"; Timesteps = $Timesteps },
    @{ Layout = "random0"; Reward = "sparse";  Timesteps = $Timesteps },
    @{ Layout = "random0"; Reward = "dense";   Timesteps = $Timesteps },
    @{ Layout = "random1"; Reward = "default"; Timesteps = $Timesteps }
)

foreach ($seed in $Seeds) {
    foreach ($exp in $experiments) {
        python experiments/run_overcooked_experiment.py `
            --layout $exp.Layout `
            --reward-profile $exp.Reward `
            --seed $seed `
            --timesteps $exp.Timesteps `
            --eval-episodes $EvalEpisodes `
            --n-steps $NSteps `
            --batch-size $BatchSize
    }
}

python experiments/plot_results.py --runs-dir runs --out runs/eval_bar.png
