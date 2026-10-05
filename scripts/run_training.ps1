<#
Launch route-policy training DETACHED on Windows. Windows twin of run_training.sh.

Every token is passed straight through to scripts/run_training.py, so there is
no param() block on purpose -- `--config-name train` and Hydra
overrides such as `run.seed=3` must reach $args untouched.

Logging is done by run_training.py itself (`--log`), which is added only when
the caller did not supply one: Start-Process cannot send stdout and stderr to a
single file, so PowerShell-side redirection is deliberately NOT used.

  .\scripts\run_training.ps1
  .\scripts\run_training.ps1 --config-name train
  .\scripts\run_training.ps1 --dry-run
  .\scripts\run_training.ps1 run.seed=3
  .\scripts\run_training.ps1 --wait --config-name runs/train_construction trainer=d3po_construction

  follow: Get-Content -Wait -Tail 20 artifacts\logs\train_<stamp>.log
  stop:   Stop-Process -Id <PID>
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "project venv not found at $py; create it with 'python -m venv .venv' first"
}

$waitForExit = @($args) -contains '--wait'
$passthru = @($args | Where-Object { $_ -ne '--wait' })
$log = $null
if ($passthru -contains '--log') {
    $log = $passthru[[Array]::IndexOf($passthru, '--log') + 1]
}
if ($passthru -notcontains '--log') {
    $logDir = Join-Path $root 'artifacts\logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
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
$boardDir = 'artifacts/runs'
if ($passthru -contains '--runs-dir') {
    $boardDir = $passthru[[Array]::IndexOf($passthru, '--runs-dir') + 1]
}
Write-Host "  board:  .venv/Scripts/tensorboard.exe --logdir $boardDir"
if ($waitForExit) {
    $proc.Handle | Out-Null
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) { throw "Training failed (exit $($proc.ExitCode)): $log" }
}
