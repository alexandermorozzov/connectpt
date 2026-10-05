param(
    [Parameter(Mandatory=$true)][string]$ComparisonRoot,
    [int]$TransitionBudget = 6144000,
    [int]$BatchSize = 256,
    [int]$Horizon = 120,
    [string]$ConfigName = 'runs/train_construction',
    [string]$Algorithms = 'ppo,decomposed_ppo,d3po',
    [int]$Seed = 0,
    [string[]]$Overrides = @(),
    [int]$MaxSecondsPerAlgorithm = 0,
    [int]$IterationLimit = 100000
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$comparison = [IO.Path]::GetFullPath((Join-Path $repo $ComparisonRoot))
$artifacts = [IO.Path]::GetFullPath((Join-Path $repo 'artifacts/comparisons'))
if (-not $comparison.StartsWith($artifacts + [IO.Path]::DirectorySeparatorChar)) {
    throw 'ComparisonRoot must be a new directory under artifacts/comparisons'
}
New-Item -ItemType Directory -Force -Path (Join-Path $comparison 'logs') | Out-Null
$rolloutSize = $BatchSize * $Horizon
$algorithmNames = @($Algorithms.Split(',') | ForEach-Object { $_.Trim() })
$trainerGroups = @{ppo='ppo_construction'; decomposed_ppo='decomposed_construction'; d3po='d3po_construction'}
if ($algorithmNames.Count -lt 2 -or @($algorithmNames | Select-Object -Unique).Count -ne $algorithmNames.Count -or
    @($algorithmNames | Where-Object { $_ -notin @('ppo', 'decomposed_ppo', 'd3po') }).Count -gt 0) {
    throw 'Algorithms must contain at least two unique supported algorithm names, separated by commas'
}
if ($BatchSize -lt 2 -or $Horizon -lt 1 -or $MaxSecondsPerAlgorithm -lt 0 -or $IterationLimit -lt 1) {
    throw 'Invalid batch, horizon, time budget or iteration limit'
}
if ($MaxSecondsPerAlgorithm -eq 0 -and ($TransitionBudget -lt $rolloutSize -or
    $TransitionBudget % $rolloutSize -ne 0)) {
    throw 'TransitionBudget must be a positive multiple of BatchSize * Horizon; BatchSize >= 2'
}
$iterations = if ($MaxSecondsPerAlgorithm -gt 0) { $IterationLimit } else { [int]($TransitionBudget / $rolloutSize) }
$maxSecondsOverride = if ($MaxSecondsPerAlgorithm -gt 0) { "$MaxSecondsPerAlgorithm" } else { 'null' }
$budgetMode = if ($MaxSecondsPerAlgorithm -gt 0) { 'wall_time' } else { 'transitions' }
$manifestPath = Join-Path $comparison 'protected_weights.json'
if (-not (Test-Path -LiteralPath $manifestPath)) {
    $protected = @(foreach ($relative in @('artifacts/model_weights', 'artifacts/model_weights_edit')) {
        $folder = Join-Path $repo $relative
        if (Test-Path -LiteralPath $folder) {
            Get-ChildItem -LiteralPath $folder -Recurse -File -Filter '*.pt' | ForEach-Object {
                @{path=$_.FullName; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
            }
        }
    })
    ConvertTo-Json -InputObject $protected | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}
$statePath = Join-Path $comparison 'queue_status.json'
function Save-QueueState($state, $algorithm, $message) {
    @{state=$state; algorithm=$algorithm; updated=(Get-Date).ToString('o'); message=$message; queue_pid=$PID} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
}
function Check-ProtectedWeights {
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $comparison 'protected_weights.json') | ConvertFrom-Json
    foreach ($entry in $manifest) {
        if ((Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash -ne $entry.sha256) {
            throw "Protected weight changed: $($entry.path)"
        }
    }
}
try {
    Check-ProtectedWeights
    foreach ($algorithm in $algorithmNames) {
        $statusFile = Join-Path $comparison "runs/$algorithm/status.json"
        if (Test-Path -LiteralPath $statusFile) {
            throw "Run already exists: $algorithm. Use a new ComparisonRoot."
        }
        Save-QueueState 'training' $algorithm "budget=$budgetMode, max_seconds=$maxSecondsOverride, iteration_limit=$iterations, batch=$BatchSize horizon=$Horizon"
        & (Join-Path $repo 'scripts/run_training.ps1') --wait --config-name $ConfigName `
            @Overrides "run.name=$algorithm" "trainer=$($trainerGroups[$algorithm])" "run.seed=$Seed" `
            "train_loop.n_iterations=$iterations" "train_loop.max_seconds=$maxSecondsOverride" `
            "batch_size=$BatchSize" "ppo.horizon=$Horizon" `
            --weights-dir "$ComparisonRoot/weights" --runs-dir "$ComparisonRoot/runs" `
            --log "$ComparisonRoot/logs/$algorithm.log"
        $status = Get-Content -Raw -LiteralPath $statusFile | ConvertFrom-Json
        if ($status.state -ne 'completed') { throw "$algorithm did not complete successfully" }
        if ($MaxSecondsPerAlgorithm -eq 0 -and $status.total_active_steps -ne $TransitionBudget) {
            throw "$algorithm did not reach the interaction budget"
        }
        if ($MaxSecondsPerAlgorithm -gt 0 -and $status.stop_reason -ne 'time_budget') {
            throw "$algorithm stopped before the time budget"
        }
        Check-ProtectedWeights
    }
    Save-QueueState 'evaluating' 'all' 'Computing HV, conditioned/archive EU and pairwise coverage'
    $env:CONNECTPT_COMPARISON_ROOT = $comparison
    $env:CONNECTPT_COMPARISON_ALGORITHMS = $algorithmNames -join ','
    $env:CONNECTPT_COMPARISON_BUDGET = $budgetMode
    $reportCode = "import os; from connectpt.routes_generator.training.construction_comparison import compare_runs; [compare_runs(os.environ['CONNECTPT_COMPARISON_ROOT'], names=tuple(os.environ['CONNECTPT_COMPARISON_ALGORITHMS'].split(',')), checkpoint=c, mode=m, budget_mode=os.environ['CONNECTPT_COMPARISON_BUDGET']) for c in ('last','best') for m in ('greedy','stochastic')]"
    & (Join-Path $repo '.venv/Scripts/python.exe') -c $reportCode
    if ($LASTEXITCODE -ne 0) { throw 'Comparison report failed' }
    Check-ProtectedWeights
    Save-QueueState 'completed' 'all' 'Selected runs and JSON comparisons completed; protected weights verified'
} catch {
    Save-QueueState 'failed' $algorithm $_.Exception.Message
    throw
}
