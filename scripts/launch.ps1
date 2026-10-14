<#
Launch any scripts/*.py in the background with a log. Windows twin of launch.sh.

The first argument is the script; every other token goes to it unchanged. When
the tokens have no --log, the launcher adds --log <log-dir>/<script>_<stamp>.log
(<log-dir> = --log-dir or artifacts/logs). Console output goes next to the log:
<log>.out and <log>.err. --wait blocks until the run ends and fails on a
non-zero exit code. There is no param() block: tokens such as --config-name and
Hydra overrides must reach the script untouched.

  .\scripts\launch.ps1 run.py --config-name runs/train_construction training/algorithm=ppo
  .\scripts\launch.ps1 run.py --config-name runs/train_construction training/algorithm=d3po
  .\scripts\launch.ps1 run.py --config-name runs/search --methods ea nea --instances mandl --alpha 0 1
  .\scripts\launch.ps1 --wait run.py --config-name runs/train_edit --dry-run

  follow: Get-Content -Wait -Tail 20 <log>
  stop:   Stop-Process -Id <PID>
#>
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    throw "project venv not found at $py; create it with 'python -m venv .venv' first"
}

$wait = @($args) -contains '--wait'
$tokens = @($args | Where-Object { $_ -ne '--wait' })
if ($tokens.Count -eq 0 -or -not ($tokens[0] -like '*.py')) {
    throw 'usage: launch.ps1 [--wait] <script>.py [arguments]'
}
$script = Join-Path 'scripts' $tokens[0]
if (-not (Test-Path -LiteralPath $script)) { throw "no script $script" }
$passthru = @($tokens | Select-Object -Skip 1)

if ($passthru -contains '--log') {
    $log = $passthru[[Array]::IndexOf($passthru, '--log') + 1]
} else {
    $logDir = 'artifacts\logs'
    if ($passthru -contains '--log-dir') {
        $logDir = $passthru[[Array]::IndexOf($passthru, '--log-dir') + 1]
    }
    $stem = [IO.Path]::GetFileNameWithoutExtension($script)
    $log = Join-Path $logDir ("{0}_{1}.log" -f $stem, (Get-Date -Format 'yyyyMMdd_HHmmss'))
    $passthru += @('--log', $log)
}
$logPath = if ([IO.Path]::IsPathRooted($log)) { $log } else { Join-Path $root $log }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null

# Quote every token: Start-Process joins the list with spaces.
$quoted = @('-u', $script) + $passthru | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
$proc = Start-Process -FilePath $py -ArgumentList ($quoted -join ' ') -WorkingDirectory $root `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
$proc.Handle | Out-Null

Write-Host "Started $($tokens[0]) (PID $($proc.Id))"
Write-Host "  log:    $log"
Write-Host "  follow: Get-Content -Wait -Tail 20 '$log'"
Write-Host "  stop:   Stop-Process -Id $($proc.Id)"
Write-Host '  board:  the existing TensorBoard server (--logdir artifacts)'
if ($wait) {
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) { throw "$($tokens[0]) failed (exit $($proc.ExitCode)): $log.err" }
}
