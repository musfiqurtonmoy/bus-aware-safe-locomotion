# CHANGELOG #38 evaluation batch: joint-level limiter and margin ablation. Evaluation only, no
# training. One process at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\run_perjoint_margin_batch.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_perjoint_margin_batch.ps1 -List   # print the jobs only
#
# 10 episodes per seed at SOC 1.0, 0.5, 0.2 (reset seeds 1000..1009, the same initial states as
# every other setting file).
# 1. Joint-level limiter (test mode perjoint): the filter's power budget split equally between the
#    two drives, each joint limited on its own (busaware/safety_filter.project_per_joint), for
#    baseline_ws_s0..2 and filter_ws_s0..2: 6 evaluations, files eval_ctl-perjoint.csv.
# 2. Margin ablation for filter_ws_s0..2 in filter mode, one margin changed at a time:
#    --margin_i 0, --margin_i 1.0 (delta_V stays 0.2 V); --margin_v 0, --margin_v 0.5 (delta_I
#    stays 0.5 A); and both margins zero together (--margin_v 0 --margin_i 0): 15 evaluations,
#    files eval_mi0.0.csv, eval_mi1.0.csv, eval_mv0.0.csv, eval_mv0.5.csv, eval_mv0.0_mi0.0.csv.
#    The defaults (0.2 V, 0.5 A) are the base rows (eval.csv, episodes 0..9).
# 21 evaluations in total. Restartable: evaluate.py --skip_done skips finished evaluations. Stops at
# the first failure. Afterwards: python scripts/aggregate_ablations.py (groups controller, margin).
param([switch]$List)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$socs = @("1.0", "0.5", "0.2")
$n_ep = 10

$jobs = @()   # each: run, test mode, extra evaluate.py arguments
foreach ($r in "baseline_ws", "filter_ws") {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("${r}_s$s", "perjoint", @())
    }
}
$margins = @(@("--margin_i", "0"), @("--margin_i", "1.0"), @("--margin_v", "0"), @("--margin_v", "0.5"),
             @("--margin_v", "0", "--margin_i", "0"))
foreach ($m in $margins) {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("filter_ws_s$s", "filter", $m)
    }
}

$missing = @($jobs | ForEach-Object { $_[0] } | Sort-Object -Unique | Where-Object { -not (Test-Path ("results\" + $_ + "\model.zip")) })
if ($missing.Count -gt 0) { throw ("no model.zip for: " + ($missing -join ", ")) }

if ($List) {
    $i = 0
    foreach ($j in $jobs) { $i++; Write-Host ("{0,3} {1,-15} {2,-9} {3}" -f $i, $j[0], $j[1], ($j[2] -join " ")) }
    Write-Host ("{0} evaluations" -f $jobs.Count)
    exit 0
}

$t0 = Get-Date
$i = 0
foreach ($j in $jobs) {
    $i++
    $run, $mode, $extra = $j[0], $j[1], $j[2]
    Write-Host ("[{0:HH:mm:ss}] {1}/{2} {3}: {4} {5}" -f (Get-Date), $i, $jobs.Count, $run, $mode, ($extra -join " "))
    & $py scripts\evaluate.py --run $run --episodes $n_ep --socs @socs --test_modes $mode @extra --skip_done
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $run $mode $($extra -join ' ') (exit $LASTEXITCODE)" }
}
Write-Host ("Per-joint and margin batch: {0} evaluations in {1:N1} min" -f $jobs.Count, ((Get-Date) - $t0).TotalMinutes)
