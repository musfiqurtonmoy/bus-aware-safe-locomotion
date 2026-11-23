# Step 8 evaluation batch (CHANGELOG #30): the barrier filter with the battery EKF. One process at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\run_ekf_batch.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_ekf_batch.ps1 -List   # print the jobs only
#
# filter_ws_s0..2 in filter mode, 10 episodes per seed at SOC 1.0, 0.5, 0.2 (reset seeds 1000..1009),
# estimator x condition:
#   estimators: EKF with fixed margins (--estimator ekf), EKF with confidence-scaled margins
#               (--estimator ekf --margin confidence)
#   conditions: nominal; --est_noise 0.05 (here: on the EKF's two measurements); --r0_scale 2.0
# 2 x 3 x 3 = 18 evaluations. The true-state rows are not rerun: they are the existing files
# eval.csv (episodes 0..9, the same reset seeds), eval_r1.0_n0.05.csv and eval_r2.0_n0.0.csv of
# filter_ws_s0..2 (CHANGELOG #29). Files: eval_est-ekf*.csv (busaware/runs.eval_filename).
# Then (CHANGELOG #31) --r0_scale 0.75: plant R0 smaller than the filter's model (the unsafe
# direction for the current barrier), for the true-state filter (eval_r0.75_n0.0.csv) and both
# EKF settings: 3 x 3 = 9 evaluations. 27 in total.
# Restartable: evaluate.py --skip_done skips finished evaluations. Stops at the first failure.
# Afterwards: python scripts/aggregate_ablations.py (group "estimator").
param([switch]$List)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$socs = @("1.0", "0.5", "0.2")
$n_ep = 10

$jobs = @()   # each: run, extra evaluate.py arguments
$conditions = @(@(), @("--est_noise", "0.05"), @("--r0_scale", "2.0"))
$estimators = @(@("--estimator", "ekf"), @("--estimator", "ekf", "--margin", "confidence"))
foreach ($c in $conditions) {
    foreach ($e in $estimators) {
        foreach ($s in 0, 1, 2) {
            $jobs += , @("filter_ws_s$s", @($e + $c))
        }
    }
}
# r0_scale 0.75: true state, EKF fixed margin, EKF confidence margin
$est075 = @(, @("--estimator", "true")) + $estimators     # unary comma: keep the inner array whole
foreach ($e in $est075) {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("filter_ws_s$s", @($e + @("--r0_scale", "0.75")))
    }
}

$missing = @($jobs | ForEach-Object { $_[0] } | Sort-Object -Unique | Where-Object { -not (Test-Path ("results\" + $_ + "\model.zip")) })
if ($missing.Count -gt 0) { throw ("no model.zip for: " + ($missing -join ", ")) }

if ($List) {
    $i = 0
    foreach ($j in $jobs) { $i++; Write-Host ("{0,3} {1,-14} {2}" -f $i, $j[0], ($j[1] -join " ")) }
    Write-Host ("{0} evaluations" -f $jobs.Count)
    exit 0
}

$t0 = Get-Date
$i = 0
foreach ($j in $jobs) {
    $i++
    $run, $extra = $j[0], $j[1]
    Write-Host ("[{0:HH:mm:ss}] {1}/{2} {3}: filter {4}" -f (Get-Date), $i, $jobs.Count, $run, ($extra -join " "))
    & $py scripts\evaluate.py --run $run --episodes $n_ep --socs @socs --test_modes filter @extra --skip_done
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $run $($extra -join ' ') (exit $LASTEXITCODE)" }
}
Write-Host ("Step 8 batch: {0} evaluations in {1:N1} min" -f $jobs.Count, ((Get-Date) - $t0).TotalMinutes)
