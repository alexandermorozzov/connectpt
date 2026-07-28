# Table 5 / Figure 5 (5 operator combos, Mumford1 by default) rerun -> artifacts/reruns/.
#   .\reruns\run_table5.ps1                 # full paper budget
#   .\reruns\run_table5.ps1 -Profile smoke  # fast plumbing check
#   .\reruns\run_table5.ps1 -Profile full -Cities Mandl,Mumford0,Mumford1,Mumford2,Mumford3
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full",
    [string[]]$Cities = @()
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"
if ($Profile -eq "smoke") { $suiteArgs = @("--suite-smoke", "suite_rerun_smoke") } else { $suiteArgs = @("--suite", "suite_rerun") }
$cityArgs = @()
if ($Cities.Count -gt 0) { $cityArgs = @("--cities") + $Cities }
Write-Host "Table 5 / Figure 5 (5 operator combos) rerun starting (profile=$Profile) -> artifacts/reruns/"
Write-Host "TB: tensorboard --logdir artifacts/runs/table5_fig5_5model"
# The script writes its own full log to artifacts/reruns/<stem>_run.log. Do NOT
# merge native stderr into the PowerShell pipeline (2>&1) -- PS 5.1 wraps each
# tqdm/log line as a NativeCommandError. Stream stdout to console; send stderr to
# a file (plain redirect, no wrapping) for a combined log.
$ErrorActionPreference = "Continue"
& $py scripts\run_figure5_pareto.py --profile $Profile @suiteArgs @cityArgs 2> artifacts\cli_logs\rerun_table5.log
