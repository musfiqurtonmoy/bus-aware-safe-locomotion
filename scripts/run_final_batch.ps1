# Final evaluation batch (CHANGELOG #27), ordered by priority. One process at a time.
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_batch.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_final_batch.ps1 -List   # print the jobs only
#
# (a) Step 9 sensitivity, 10 episodes per seed at SOC 1.0, 0.5, 0.2 (reset seeds 1000..1009):
#     every Params variant (cold + Params.VARIANTS) for baseline_ws_s0..2 (baseline mode and
#     filter at test time) and filter_ws_s0..2 (filter mode); one variant at a time, all seeds.
#     Then filter gamma 0.1, 0.6, 1.0 for baseline_ws_s0..2 with the filter at test time and
#     filter_ws_s0..2 (gamma acts only in filter mode, so baseline mode is not repeated).
# (b) Estimation error, filter_ws_s0..2 in filter mode, 10 episodes per seed: est_noise 0.02 and
#     0.05; plant r0_scale 1.5 and 2.0.
# (c) Last: scripts\run_eval_all.ps1 -Set all, the full re-evaluation of every trained run with
#     the current code (20 episodes per seed, as the existing tables).
#
# Each evaluation writes its own file in results\<run>\ (busaware/runs.eval_filename), e.g.
# eval_var-cold.csv, eval_g0.6.csv, eval_r1.0_n0.02.csv, eval_r1.5_n0.0.csv; (c) writes eval.csv.
# Restartable: evaluate.py --skip_done skips any evaluation whose file is already complete and
# written by the current code, so after an interruption just run the same command again.
# Stops at the first failure. Afterwards: python scripts/aggregate.py (ablations: Step 9 script).
param([switch]$List)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$socs = @("1.0", "0.5", "0.2")
$n_ep = 10

$jobs = @()   # each: run, test modes, extra evaluate.py arguments
# (a) Params variants, in priority order
$variants = @("current_lag", "cold", "vmin_datasheet", "wiring_0mohm", "wiring_20mohm",
              "paux_0w", "paux_10w", "rphase_x0.5", "rphase_x1.5", "gear_eff_0.9")
foreach ($v in $variants) {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("baseline_ws_s$s", @("baseline", "filter"), @("--variant", $v))
        $jobs += , @("filter_ws_s$s", @("filter"), @("--variant", $v))
    }
}
# (a) filter gamma (default 0.3 is the base evaluation)
foreach ($g in "0.1", "0.6", "1.0") {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("baseline_ws_s$s", @("filter"), @("--gamma", $g))
        $jobs += , @("filter_ws_s$s", @("filter"), @("--gamma", $g))
    }
}
# (b) estimation error and aged pack, filter policies
foreach ($e in @(@("--est_noise", "0.02"), @("--est_noise", "0.05"), @("--r0_scale", "1.5"), @("--r0_scale", "2.0"))) {
    foreach ($s in 0, 1, 2) {
        $jobs += , @("filter_ws_s$s", @("filter"), $e)
    }
}

$missing = @($jobs | ForEach-Object { $_[0] } | Sort-Object -Unique | Where-Object { -not (Test-Path ("results\" + $_ + "\model.zip")) })
if ($missing.Count -gt 0) { throw ("no model.zip for: " + ($missing -join ", ")) }

if ($List) {
    $i = 0
    foreach ($j in $jobs) { $i++; Write-Host ("{0,3} {1,-16} {2,-17} {3}" -f $i, $j[0], ($j[1] -join " "), ($j[2] -join " ")) }
    Write-Host ("{0} evaluations in (a)-(b), then scripts\run_eval_all.ps1 -Set all" -f $jobs.Count)
    exit 0
}

$t0 = Get-Date
$i = 0
foreach ($j in $jobs) {
    $i++
    $run, $modes, $extra = $j[0], $j[1], $j[2]
    Write-Host ("[{0:HH:mm:ss}] {1}/{2} {3}: test modes {4} {5}" -f (Get-Date), $i, $jobs.Count, $run, ($modes -join " "), ($extra -join " "))
    & $py scripts\evaluate.py --run $run --episodes $n_ep --socs @socs --test_modes @modes @extra --skip_done
    if ($LASTEXITCODE -ne 0) { throw "evaluate.py failed for $run $($extra -join ' ') (exit $LASTEXITCODE)" }
}
Write-Host ("(a)-(b): {0} evaluations in {1:N1} min" -f $jobs.Count, ((Get-Date) - $t0).TotalMinutes)

# (c) full re-evaluation with the current code
& powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_eval_all.ps1") -Set all
if ($LASTEXITCODE -ne 0) { throw "run_eval_all.ps1 -Set all failed (exit $LASTEXITCODE)" }
Write-Host ("final batch finished in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
