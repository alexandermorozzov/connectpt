# Run Figure 5 / Table 5 (five operator combinations, Mumford1) on Windows.
#   .\run_figure5_pareto.ps1                 # full paper budget
#   .\run_figure5_pareto.ps1 -Profile smoke  # fast check
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"
Write-Host "Figure 5 / Table 5 starting (profile=$Profile). Logs also stream below."
Write-Host "TB: tensorboard --logdir artifacts/runs/table5_fig5_5model"
& $py scripts\run_figure5_pareto.py --profile $Profile 2>&1 |
    Tee-Object -FilePath artifacts\cli_logs\figure5_pareto.log
