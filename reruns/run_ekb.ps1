# EKB case study (Improved NBCO, alpha sweep) rerun -> artifacts/reruns/.
#   .\reruns\run_ekb.ps1                 # full paper budget
#   .\reruns\run_ekb.ps1 -Profile smoke  # fast plumbing check
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null
$py = ".venv\Scripts\python.exe"
if ($Profile -eq "smoke") { $suiteArgs = @("--suite-smoke", "suite_rerun_smoke") } else { $suiteArgs = @("--suite", "suite_rerun") }
Write-Host "EKB case study (Improved NBCO, alpha sweep) rerun starting (profile=$Profile) -> artifacts/reruns/"
Write-Host "TB: tensorboard --logdir artifacts/runs/ekb_case_study/tb"
& $py scripts\run_ekb_case_study.py --profile $Profile @suiteArgs 2>&1 |
    Tee-Object -FilePath artifacts\cli_logs\rerun_ekb.log
