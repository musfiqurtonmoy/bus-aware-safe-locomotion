# Steps 4 and 5(a)-(b), and the warm-start runs (CHANGELOG #19): evaluate trained policies,
# one process at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1                  # the nine from-scratch runs
#   powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1 -Set warmstart   # the fourteen *_ws runs
#     (12 from CHANGELOG #19 incl. baseline_ws_s* with the filter at test time, + penalty_w0.2/0.5_ws_s0)
#   powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1 -Set round3      # CHANGELOG #22
#     lagrangian_ws_s0..2 (own mode = baseline env, and with the filter at test time),
#     penalty_w2.0_ws_s0, penalty_w0.5_ws_s1, penalty_w0.5_ws_s2 (penalty mode and filter at test)
#   powershell -ExecutionPolicy Bypass -File scripts\run_eval_all.ps1 -Set all         # CHANGELOG #27
#     all three sets above: full re-evaluation with the current code
# 20 episodes at SOC 1.0, 0.5 and 0.2 each, fixed reset seeds 1000..1019 (scripts/evaluate.py).
#   policies trained without the filter (baseline*, penalty*, lagrangian*): in their own mode and
#                                                              with the filter at test time
#   policies trained with the filter (filter*):                in filter mode
# Writes results/<run>/eval.csv. Stops at the first failure. Then: python scripts/aggregate.py
# Restartable: evaluate.py --skip_done skips a run whose eval.csv already has every requested
# episode and the proof-assumption counter columns (written by the current code, CHANGELOG #25);
# an eval.csv written before the counters is re-evaluated and overwritten.
param([ValidateSet("original", "warmstart", "round3", "all")] [string]$Set = "original")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$jobs = @()
foreach ($s in 0, 1, 2) {
    if ($Set -eq "original" -or $Set -eq "all") {
        $jobs += , @("baseline_s$s", @("baseline", "filter"))
        $jobs += , @("penalty_s$s", @("penalty", "filter"))
        $jobs += , @("filter_s$s", @("filter"))
    }
    if ($Set -eq "warmstart" -or $Set -eq "all") {
        $jobs += , @("filter_ws_s$s", @("filter"))
        $jobs += , @("penalty_w0.05_ws_s$s", @("penalty", "filter"))
        $jobs += , @("penalty_w0.1_ws_s$s", @("penalty", "filter"))
        $jobs += , @("baseline_ws_s$s", @("baseline", "filter"))   # filter at test = equal-budget comparison
    }
}
if ($Set -eq "warmstart" -or $Set -eq "all") {
    # CHANGELOG #21: extra penalty weights, seed 0 only
    $jobs += , @("penalty_w0.2_ws_s0", @("penalty", "filter"))
    $jobs += , @("penalty_w0.5_ws_s0", @("penalty", "filter"))
}
if ($Set -eq "round3" -or $Set -eq "all") {
    foreach ($s in 0, 1, 2) { $jobs += , @("lagrangian_ws_s$s", @("lagrangian", "filter")) }
    $jobs += , @("penalty_w2.0_ws_s0", @("penalty", "filter"))
    $jobs += , @("penalty_w0.5_ws_s1", @("penalty", "filter"))
    $jobs += , @("penalty_w0.5_ws_s2", @("penalty", "filter"))
}
$missing = @($jobs | Where-Object { -not (Test-Path ("results\" + $_[0] + "\model.zip")) } | ForEach-Object { $_[0] })
if ($missing.Count -gt 0) { throw ("no model.zip for: " + ($missing -join ", ") + " (train them first)") }
$t0 = Get-Date
foreach ($j in $jobs) {
    $run, $modes = $j[0], $j[1]
    Write-Host ("[{0:HH:mm:ss}] {1}: test modes {2}" -f (Get-Date), $run, ($modes -join " "))
    & $py scripts\evaluate.py --run $run --episodes 20 --socs 1.0 0.5 0.2 --test_modes @modes --skip_done
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $run (exit $LASTEXITCODE)" }
}
Write-Host ("{0} runs evaluated in {1:N1} min" -f $jobs.Count, ((Get-Date) - $t0).TotalMinutes)
