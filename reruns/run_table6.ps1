# Table 6 (4 BCO configurations, Mumford1 by default) -> artifacts/reruns/.
#   .\reruns\run_table6.ps1
#   .\reruns\run_table6.ps1 -Profile smoke
#   .\reruns\run_table6.ps1 -Profile full -Cities Mumford0
#   .\reruns\run_table6.ps1 -Profile full -Cities Mandl,Mumford0,Mumford1
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full",
    [string[]]$Cities = @()
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"

if ($Profile -eq "smoke") {
    $suiteArgs = @("--suite-smoke", "suite_rerun_smoke")
} else {
    $suiteArgs = @("--suite", "suite_rerun")
}
$cityArgs = @()
if ($Cities.Count -gt 0) {
    $cityArgs = @("--cities") + $Cities
}

Write-Host "Table 6 rerun starting (profile=$Profile) -> artifacts/reruns/"
Write-Host "TB: tensorboard --logdir artifacts/runs/table6_bco_comparison"

# The Python runner writes its complete log under artifacts/reruns/. Keep native
# stderr out of the PowerShell pipeline because PS 5.1 wraps tqdm output as
# NativeCommandError records.
$ErrorActionPreference = "Continue"
& $py scripts\run_table6.py --profile $Profile @suiteArgs @cityArgs `
    2> artifacts\cli_logs\rerun_table6.log
