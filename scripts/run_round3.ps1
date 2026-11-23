# Round 3 (CHANGELOG #22): warm starts from baseline_s<k>, 150k steps each, one at a time:
#   lagrangian_ws_s0, penalty_w2.0_ws_s0, lagrangian_ws_s1, penalty_w0.5_ws_s1,
#   lagrangian_ws_s2, penalty_w0.5_ws_s2
# SAC-Lagrangian: cost = fraction of substeps with undervoltage or overcurrent, limit d = 0.001,
# multiplier log in results/<run>/lagrangian.csv (busaware/lagrangian.py).
# A run whose model.zip exists is skipped; a folder without model.zip (interrupted run) stops
# the script and must be deleted by hand.
#   powershell -ExecutionPolicy Bypass -File scripts\run_round3.ps1
# Then evaluate: powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1 -Set round3
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$steps = 150000
# run name (must match busaware/runs.run_name), seed, mode, extra arguments
$runs = @(
    , @("lagrangian_ws_s0", 0, "lagrangian", @())
    , @("penalty_w2.0_ws_s0", 0, "penalty", @("--penalty_w", "2.0"))
    , @("lagrangian_ws_s1", 1, "lagrangian", @())
    , @("penalty_w0.5_ws_s1", 1, "penalty", @("--penalty_w", "0.5"))
    , @("lagrangian_ws_s2", 2, "lagrangian", @())
    , @("penalty_w0.5_ws_s2", 2, "penalty", @("--penalty_w", "0.5"))
)
$t0 = Get-Date
foreach ($r in $runs) {
    $run, $s, $mode, $extra = $r[0], $r[1], $r[2], $r[3]
    if (Test-Path "results\$run\model.zip") { Write-Host "skip $run (model.zip exists)"; continue }
    if (Test-Path "results\$run") { throw "results\$run exists without model.zip (interrupted run?); delete it first" }
    Write-Host ("[{0:HH:mm:ss}] {1}: {2} mode, {3} steps from baseline_s{4}" -f (Get-Date), $run, $mode, $steps, $s)
    & $py scripts\train.py --mode $mode --steps $steps --seed $s --init "baseline_s$s" @extra
    if ($LASTEXITCODE -ne 0) { throw "train.py failed for $run (exit $LASTEXITCODE)" }
    if (-not (Test-Path "results\$run\model.zip")) { throw "results\$run\model.zip missing: run name mismatch?" }
}
Write-Host ("round 3 finished in {0:N1} h" -f ((Get-Date) - $t0).TotalHours)
