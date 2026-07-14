# Table 5 / Figure 5 (5 operator combos, Mumford1) rerun -> artifacts/reruns/.
#   .\reruns\run_table5.ps1                 # full paper budget
#   .\reruns\run_table5.ps1 -Profile smoke  # fast plumbing check
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"
if ($Profile -eq "smoke") { $suiteArgs = @("--suite-smoke", "suite_rerun_smoke") } else { $suiteArgs = @("--suite", "suite_rerun") }
Write-Host "Table 5 / Figure 5 (5 operator combos, Mumford1) rerun starting (profile=$Profile) -> artifacts/reruns/"
Write-Host "TB: tensorboard --logdir artifacts/runs/table5_fig5_5model"
& $py scripts\run_figure5_pareto.py --profile $Profile @suiteArgs 2>&1 |
    Tee-Object -FilePath artifacts\cli_logs\rerun_table5.log
