# CHANGELOG #55 evaluation batch: stress tests of the warm-started penalty policies with the filter
# added at test time. Evaluation only, no training. One process at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\run_penalty_stress.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_penalty_stress.ps1 -List   # print the jobs only
#
# Policy: penalty_w0.5_ws_s0..2. Of the penalty weights trained with 3 seeds (0.05, 0.1, 0.5),
# w = 0.5 has the highest mean max height with the test-time filter at every state of charge in the
# main evaluation (results/summary/per_seed.csv, method penalty_w0.5_ws+filter@test: 0.231, 0.212,
# 0.203 m at SOC 1.0, 0.5, 0.2; Table S8), with no violation and no fall.
# Settings: the same as the existing stress tests of baseline_ws and filter_ws
# (scripts\run_final_batch.ps1, part (a)):
#   - the 10 Params variants, plant and filter alike (current lag and gear efficiency plant only);
#   - filter barrier rate gamma 0.1, 0.6 and 1.0 (0.3 is the base evaluation, eval.csv);
#   - test mode filter only (the filter added at test time), 10 episodes per seed (reset seeds
#     1000..1009, evaluate.py SEED_BASE + episode), states of charge 1.0, 0.5 and 0.2, 3 seeds.
# 39 evaluations. Files: results\penalty_w0.5_ws_s<k>\eval_var-<variant>.csv and eval_g<gamma>.csv.
# Restartable: evaluate.py --skip_done skips any evaluation whose file is already complete and
# written by the current code. Stops at the first failure.
# Estimated run time (from the file times of the earlier stress tests, one filter-mode evaluation of
# 3 x 10 episodes: median 163 s, range 129-204 s): about 1.8 h (1.4-2.2 h).
# Afterwards: scripts\aggregate_ablations.py reads only baseline_ws and filter_ws (RUNS); it must be
# extended to the penalty runs before these files enter any table.
param([switch]$List)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$socs = @("1.0", "0.5", "0.2")
$n_ep = 10
$run_prefix = "penalty_w0.5_ws"

$jobs = @()   # each: run, extra evaluate.py arguments
$variants = @("current_lag", "cold", "vmin_datasheet", "wiring_0mohm", "wiring_20mohm",
              "paux_0w", "paux_10w", "rphase_x0.5", "rphase_x1.5", "gear_eff_0.9")
foreach ($v in $variants) {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("${run_prefix}_s$s", @("--variant", $v))
    }
}
foreach ($g in "0.1", "0.6", "1.0") {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("${run_prefix}_s$s", @("--gamma", $g))
    }
}

$missing = @($jobs | ForEach-Object { $_[0] } | Sort-Object -Unique | Where-Object { -not (Test-Path ("results\" + $_ + "\model.zip")) })
if ($missing.Count -gt 0) { throw ("no model.zip for: " + ($missing -join ", ")) }

if ($List) {
    $i = 0
    foreach ($j in $jobs) { $i++; Write-Host ("{0,3} {1,-20} filter {2}" -f $i, $j[0], ($j[1] -join " ")) }
    Write-Host ("{0} evaluations, test mode filter, {1} episodes per seed and state of charge" -f $jobs.Count, $n_ep)
    exit 0
}

$t0 = Get-Date
$i = 0
foreach ($j in $jobs) {
    $i++
    $run, $extra = $j[0], $j[1]
    Write-Host ("[{0:HH:mm:ss}] {1}/{2} {3}: filter at test {4}" -f (Get-Date), $i, $jobs.Count, $run, ($extra -join " "))
    & $py scripts\evaluate.py --run $run --episodes $n_ep --socs @socs --test_modes filter @extra --skip_done
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $run $($extra -join ' ') (exit $LASTEXITCODE)" }
}
Write-Host ("Penalty stress batch: {0} evaluations in {1:N1} min" -f $jobs.Count, ((Get-Date) - $t0).TotalMinutes)
