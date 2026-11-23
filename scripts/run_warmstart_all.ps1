# Warm-start study (CHANGELOG #19, #21): continue baseline_s<k> policies for 150k steps.
#   -Set main (default), CHANGELOG #19, from baseline_s{0,1,2}:
#     (a) filter mode            -> filter_ws_s<k>
#     (b) penalty mode, w = 0.05 -> penalty_w0.05_ws_s<k>
#     (c) penalty mode, w = 0.1  -> penalty_w0.1_ws_s<k>
#     (d) baseline mode (control: same extra steps, no constraint) -> baseline_ws_s<k>
#     seed 0 of every method finishes first, then seed 1, then seed 2
#   -Set penalty_extra, CHANGELOG #21, from baseline_s0 only:
#     penalty mode, w = 0.2 -> penalty_w0.2_ws_s0;  w = 0.5 -> penalty_w0.5_ws_s0
# One training run at a time, seed k = the source run's seed. A run whose model.zip exists is
# skipped, so the script can be restarted after an interruption (an interrupted run's folder
# must be deleted by hand first: train.py refuses to overwrite a model but not a partial log).
#   powershell -ExecutionPolicy Bypass -File scripts\run_warmstart_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_warmstart_all.ps1 -Set penalty_extra
# Then evaluate: powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1 -Set warmstart
param([ValidateSet("main", "penalty_extra")] [string]$Set = "main")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$steps = 150000
# name, mode, extra arguments (names must match busaware/runs.run_name)
if ($Set -eq "main") {
    $seeds = 0, 1, 2
    $methods = @(
        , @("filter_ws", "filter", @())
        , @("penalty_w0.05_ws", "penalty", @("--penalty_w", "0.05"))
        , @("penalty_w0.1_ws", "penalty", @("--penalty_w", "0.1"))
        , @("baseline_ws", "baseline", @())
    )
} else {
    $seeds = , 0
    $methods = @(
        , @("penalty_w0.2_ws", "penalty", @("--penalty_w", "0.2"))
        , @("penalty_w0.5_ws", "penalty", @("--penalty_w", "0.5"))
    )
}
$t0 = Get-Date
foreach ($s in $seeds) {
    foreach ($m in $methods) {
        $name, $mode, $extra = $m[0], $m[1], $m[2]
        $run = "$($name)_s$s"
        if (Test-Path "results\$run\model.zip") { Write-Host "skip $run (model.zip exists)"; continue }
        if (Test-Path "results\$run") { throw "results\$run exists without model.zip (interrupted run?); delete it first" }
        Write-Host ("[{0:HH:mm:ss}] {1}: {2} mode, {3} steps from baseline_s{4}" -f (Get-Date), $run, $mode, $steps, $s)
        & $py scripts\train.py --mode $mode --steps $steps --seed $s --init "baseline_s$s" @extra
        if ($LASTEXITCODE -ne 0) { throw "train.py failed for $run (exit $LASTEXITCODE)" }
        if (-not (Test-Path "results\$run\model.zip")) { throw "results\$run\model.zip missing: run name mismatch?" }
    }
}
Write-Host ("warm-start runs ({0}) finished in {1:N1} h" -f $Set, ((Get-Date) - $t0).TotalHours)
