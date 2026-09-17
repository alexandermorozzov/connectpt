<# Launch a config-driven bee-colony search detached on Windows. #>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "project venv not found at $py; create it with 'python -m venv .venv' first"
}

$logDir = Join-Path $root 'artifacts\logs'
if (-not (Test-Path -LiteralPath $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$passthru = @($args)
$log = $null
if ($passthru -notcontains '--log') {
    $log = Join-Path $logDir ("search_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $passthru += @('--log', $log)
}

$argList = @('-u', 'scripts/run_search.py') + $passthru
$proc = Start-Process -FilePath $py -ArgumentList $argList `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

Write-Host "Search started (PID $($proc.Id))"
if ($log) { Write-Host "  log:    $log" }
Write-Host "  follow: Get-Content -Wait -Tail 20 '$log'"
Write-Host "  stop:   Stop-Process -Id $($proc.Id)"
