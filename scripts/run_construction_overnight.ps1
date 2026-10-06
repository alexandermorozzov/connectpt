param(
    [string]$ComparisonRoot = ('artifacts/comparisons/construction_overnight_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [double]$Hours = 0,
    [int]$Seed = 0,
    [string[]]$Overrides = @(),
    [switch]$DryRun,
    [switch]$Wait
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$py = Join-Path $repo '.venv/Scripts/python.exe'
$script = Join-Path $repo 'scripts/run_training.py'
$comparison = [IO.Path]::GetFullPath((Join-Path $repo $ComparisonRoot))
$comparisonsDir = [IO.Path]::GetFullPath((Join-Path $repo 'artifacts/comparisons'))
if (-not $comparison.StartsWith($comparisonsDir + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'ComparisonRoot must be a new folder under artifacts/comparisons'
}
if (Test-Path -LiteralPath $comparison) { throw "Output folder already exists: $comparison" }
$sharedOverrides = @("run.seed=$Seed") + $Overrides
if ($Hours -ne 0) {
    if ($Hours -le 1) { throw 'Hours must exceed the one-hour allowance for final evaluation' }
    $trainingSeconds = [int][math]::Floor(($Hours - 1) * 3600)
    $sharedOverrides += "train_loop.max_seconds=$trainingSeconds"
}
$commands = @{}
foreach ($name in @('ppo', 'd3po')) {
    $tokens = @('--config-name', 'runs/train_construction_overnight') + $sharedOverrides + @(
        "trainer=${name}_construction", "run.name=$name",
        '--weights-dir', (Join-Path $comparison 'weights'),
        '--runs-dir', (Join-Path $comparison 'runs'),
        '--log', (Join-Path $comparison "logs/$name.log")
    )
    $commands[$name] = $tokens
    # Validate both ordinary training commands before starting either worker.
    $checkRoot = Join-Path $repo "artifacts/audits/direct_launcher/$([guid]::NewGuid().ToString('N'))"
    $checkTokens = @($tokens)
    $checkTokens[[Array]::IndexOf($checkTokens, '--weights-dir') + 1] = Join-Path $checkRoot 'weights'
    $checkTokens[[Array]::IndexOf($checkTokens, '--runs-dir') + 1] = Join-Path $checkRoot 'runs'
    $checkTokens[[Array]::IndexOf($checkTokens, '--log') + 1] = Join-Path $checkRoot "$name.log"
    $checkTokens += '--dry-run'
    & $py $script @checkTokens
    if ($LASTEXITCODE -ne 0) { throw "Config validation failed: $name" }
}
if ($DryRun) {
    ConvertTo-Json -InputObject @{ launch_mode = 'direct'; commands = $commands } -Depth 5
    return
}
New-Item -ItemType Directory -Path (Join-Path $comparison 'logs') -Force | Out-Null
$processes = @{}
$requests = @{}
foreach ($name in @('ppo', 'd3po')) {
    # JSON preserves argument boundaries, including paths containing spaces.
    $request = Join-Path $comparison "logs/launch_$name.json"
    ConvertTo-Json -InputObject @($commands[$name]) | Set-Content -LiteralPath $request -Encoding UTF8
    $arguments = "-u `"$script`" --launch-request `"$request`""
    $proc = Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $comparison "logs/${name}_console.log") `
        -RedirectStandardError (Join-Path $comparison "logs/${name}_stderr.log")
    $proc.Handle | Out-Null
    $processes[$name] = $proc
    $requests[$name] = $request
    Write-Host "$name launched directly, PID $($proc.Id)"
}
$launch = @{
    launch_mode = 'direct'; launched_at = (Get-Date -Format o); seed = $Seed; requests = $requests
    worker_pids = @{ ppo = $processes['ppo'].Id; d3po = $processes['d3po'].Id }
    commands = $commands
}
ConvertTo-Json -InputObject $launch -Depth 5 | Set-Content -LiteralPath (Join-Path $comparison 'launch.json') -Encoding UTF8
Write-Host "Outputs: $comparison"
Write-Host 'Status: runs/ppo/status.json and runs/d3po/status.json'
Write-Host 'TensorBoard: use the existing server with --logdir artifacts'
if ($Wait) {
    foreach ($name in @('ppo', 'd3po')) { $processes[$name].WaitForExit() }
    foreach ($name in @('ppo', 'd3po')) {
        if ($processes[$name].ExitCode -ne 0) { throw "Training failed: $name; see $comparison/logs" }
    }
}
