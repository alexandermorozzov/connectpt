# All five EA/NEA methods on Mandl + Mumford0-3, alpha 0..1, seeds 0..9.
#
#   .\reruns\run_all_methods_alpha_seed_sweep.ps1
#   .\reruns\run_all_methods_alpha_seed_sweep.ps1 -Profile smoke
#   .\reruns\run_all_methods_alpha_seed_sweep.ps1 -Profile full -Cities Mandl,Mumford0
#   .\reruns\run_all_methods_alpha_seed_sweep.ps1 -Profile full -Methods EA,NEA-Combined
param(
    [ValidateSet("smoke", "full")]
    [string]$Profile = "full",
    [string[]]$Cities = @("Mandl", "Mumford0", "Mumford1", "Mumford2", "Mumford3"),
    [string[]]$Methods = @("EA", "NEA", "NEA-Edit", "NEA-Combined", "RSL-EA"),
    [int[]]$Seeds = (0..9),
    [double[]]$Alphas = @(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    [int]$NIterations = 0,
    [switch]$NoResume,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force artifacts\cli_logs | Out-Null

$windowsPython = ".venv\Scripts\python.exe"
$unixPython = ".venv/bin/python"
$py = if (Test-Path $windowsPython) { $windowsPython } else { $unixPython }

if ($Profile -eq "smoke") {
    $suiteArgs = @("--suite-smoke", "suite_rerun_smoke")
} else {
    $suiteArgs = @("--suite", "suite_rerun")
}
$iterationArgs = if ($NIterations -gt 0) {
    @("--n-iterations", $NIterations)
} else { @() }
$resumeArgs = if ($NoResume) { @("--no-resume") } else { @() }
$cpuArgs = if ($Cpu) { @("--cpu") } else { @() }

if (-not $env:PYTORCH_ENABLE_MPS_FALLBACK) {
    $env:PYTORCH_ENABLE_MPS_FALLBACK = "1"
}

Write-Host "All-method alpha/seed sweep starting (profile=$Profile)"
Write-Host "cities=$($Cities -join ','); methods=$($Methods -join ',')"
Write-Host "seeds=$($Seeds -join ','); alpha=$($Alphas -join ',')"
Write-Host "results: artifacts/reruns/all_methods_alpha_seed_sweep/"
Write-Host "TB: tensorboard --logdir artifacts/runs/all_methods_alpha_seed_sweep"

$ErrorActionPreference = "Continue"
& $py scripts\run_all_methods_alpha_seed_sweep.py --profile $Profile @suiteArgs `
    --cities @Cities --methods @Methods --seeds @Seeds --alphas @Alphas `
    @iterationArgs @resumeArgs @cpuArgs `
    2> artifacts\cli_logs\run_all_methods_alpha_seed_sweep.log
