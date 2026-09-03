<#
Launch edit-model training DETACHED on Windows. Windows twin of run_training.sh.

Every token is passed straight through to scripts/run_training.py, so there is
no param() block on purpose -- `--config-name training/edit_scratch` and Hydra
overrides such as `run.seed=3` must reach $args untouched.

Logging is done by run_training.py itself (`--log`), which is added only when
the caller did not supply one: Start-Process cannot send stdout and stderr to a
single file, so PowerShell-side redirection is deliberately NOT used.

  .\scripts\run_training.ps1
  .\scripts\run_training.ps1 --config-name training/edit_scratch
  .\scripts\run_training.ps1 --dry-run
  .\scripts\run_training.ps1 run.seed=3

  follow: Get-Content -Wait -Tail 20 artifacts\cli_logs\train_<stamp>.log
  stop:   Stop-Process -Id <PID>
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "project venv not found at $py; create it with 'python -m venv .venv' first"
}

$logDir = Join-Path $root 'artifacts\cli_logs'
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$passthru = @($args)
$log = $null
if ($passthru -notcontains '--log') {
    $log = Join-Path $logDir ("train_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $passthru += @('--log', $log)
}

$argList = @('-u', 'scripts/run_training.py') + $passthru
$proc = Start-Process -FilePath $py -ArgumentList $argList `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

Write-Host "Training started (PID $($proc.Id))"
if ($log) { Write-Host "  log:    $log" }
Write-Host "  follow: Get-Content -Wait -Tail 20 '$log'"
Write-Host "  stop:   Stop-Process -Id $($proc.Id)"
Write-Host "  board:  .venv/bin/tensorboard --logdir artifacts/runs"
