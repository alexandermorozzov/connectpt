# EA vs RSL-EA vs NEA-Edit, alpha 0..1 step 0.1, adjustment off.
# Full mode runs Mandl and Mumford0-3 by default and writes to artifacts/reruns/.
#
#   .\reruns\run_edit_operator_ablation.ps1
#   .\reruns\run_edit_operator_ablation.ps1 -Profile smoke
#   .\reruns\run_edit_operator_ablation.ps1 -Profile full -Cities Mumford1
#   .\reruns\run_edit_operator_ablation.ps1 -Profile full -Cities Mandl,Mumford0
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

Write-Host "Edit-operator ablation starting (profile=$Profile) -> artifacts/reruns/"
Write-Host "Methods: EA, RSL-EA, NEA-Edit; alpha: 0.0..1.0; iterations: 400"
Write-Host "TB: tensorboard --logdir artifacts/runs/edit_operator_ablation"

# The Python runner writes its full INFO log under artifacts/reruns/.  Keep
# native stderr separate because Windows PowerShell 5.1 wraps tqdm output as
# NativeCommandError records when it is merged into the pipeline.
$ErrorActionPreference = "Continue"
& $py scripts\run_edit_operator_ablation.py --profile $Profile `
    @suiteArgs @cityArgs `
    2> artifacts\cli_logs\rerun_edit_operator_ablation.log
