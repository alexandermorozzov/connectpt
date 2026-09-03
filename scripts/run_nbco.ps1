<#
Launch an NBCO experiment DETACHED on Windows. Windows twin of run_nbco.sh.

Every token is passed straight through to scripts/run_nbco.py, so there is no
param() block on purpose -- `--cities Mumford1` and `-p n_iterations=200` must
reach $args untouched instead of being bound as PowerShell parameters.

Logging is done by run_nbco.py itself (`--log`), which is added only when the
caller did not supply one: Start-Process cannot send stdout and stderr to a
single file, so PowerShell-side redirection is deliberately NOT used.

  .\scripts\run_nbco.ps1 table3_nbco_vs_our --suite suite_rerun
  .\scripts\run_nbco.ps1 table5_fig5_5model --cities Mumford1
  .\scripts\run_nbco.ps1 ekb_case_study --suite suite_smoke

  follow: Get-Content -Wait -Tail 20 artifacts\cli_logs\nbco_<stamp>.log
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
    $log = Join-Path $logDir ("nbco_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $passthru += @('--log', $log)
}

# -u: unbuffered, so the log updates live. -WindowStyle Hidden detaches the
# child from this console; it survives closing the window.
$argList = @('-u', 'scripts/run_nbco.py') + $passthru
$proc = Start-Process -FilePath $py -ArgumentList $argList `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

Write-Host "NBCO run started (PID $($proc.Id))"
if ($log) { Write-Host "  log:    $log" }
Write-Host "  follow: Get-Content -Wait -Tail 20 '$log'"
Write-Host "  stop:   Stop-Process -Id $($proc.Id)"
Write-Host "  board:  .venv\Scripts\tensorboard --logdir artifacts\runs"
