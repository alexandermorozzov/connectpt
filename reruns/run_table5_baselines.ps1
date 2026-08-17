# Table-5-style SA/GA/HH baselines with LC init on benchmark graphs.
#   .\reruns\run_table5_baselines.ps1
#   .\reruns\run_table5_baselines.ps1 -Profile smoke
#   .\reruns\run_table5_baselines.ps1 -Profile full -Cities Mumford2,Mumford3
#   .\reruns\run_table5_baselines.ps1 -Profile full -BudgetMode scaled
# Default full budget mode is eval20k: comparable to Table 5 BCO's 200*10*5*2
# candidate evaluations. Other modes: paper40k, scaled.
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full",
    [ValidateSet("eval20k", "paper40k", "scaled")]
    [string]$BudgetMode = "eval20k",
    [string[]]$Cities = @(),
    [string[]]$Methods = @(),
    [double[]]$Alphas = @()
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"
if ($Profile -eq "smoke") { $suiteArgs = @("--suite-smoke", "suite_rerun_smoke") } else { $suiteArgs = @("--suite", "suite_rerun") }
$cityArgs = @()
if ($Cities.Count -gt 0) { $cityArgs = @("--cities") + $Cities }
$methodArgs = @()
if ($Methods.Count -gt 0) { $methodArgs = @("--methods") + $Methods }
$alphaArgs = @()
if ($Alphas.Count -gt 0) { $alphaArgs = @("--alphas") + ($Alphas | ForEach-Object { "$_" }) }
Write-Host "Table 5 baselines (SA/GA/HH, LC init) starting (profile=$Profile, budget=$BudgetMode) -> artifacts/reruns/"
$ErrorActionPreference = "Continue"
& $py scripts\run_table5_baselines.py --profile $Profile --budget-mode $BudgetMode @suiteArgs @cityArgs @methodArgs @alphaArgs 2> artifacts\cli_logs\rerun_table5_baselines.log
