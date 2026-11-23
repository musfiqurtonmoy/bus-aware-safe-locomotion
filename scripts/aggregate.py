"""Steps 4 and 5(c): merge every results/<method>_s<seed>/eval.csv into one summary.

    python scripts/aggregate.py

Method = run name without the seed (baseline, filter, filter_ws, penalty_w0.05_ws, ...), plus
"+filter@test" when a policy is evaluated in another mode than its training mode (from
busaware/runs.run_config). Episodes are averaged within a seed first (falls as the fraction of
episodes that fell); the table then gives mean and standard deviation (ddof=1) over seeds.
v_min and i_max are per-episode extremes averaged the same way. Times are seconds of simulated
time per 4 s episode.

Height (CHANGELOG #20): the primary height metric is height_max, the maximum torso height above
the start height during the episode, defined for every episode. Mean hop apex is secondary: it
exists only for episodes with at least one hop, so a seed's value averages those episodes and
the over-seed value averages the seeds that have one; apex_n_seeds gives how many.
eval.csv files written before z0/height_max were recorded: height_max = z_max - z0 with z0 = 0,
because reset puts the rail at the keyframe (0 m) and perturbs only the joint angles
(tests/test_rollout.py::test_reset_starts_the_rail_at_the_keyframe_for_every_seed).

Writes to results/summary/:
  per_seed.csv        every metric, per method, SOC and seed
  per_seed_table.csv  falls (count of episodes), hops, max height, violation times per seed
  violations.csv      mean and std over seeds per method and SOC
  proof_counters.csv  per method and SOC: substeps summed over episodes for each proof-assumption
                      counter (busaware/rollout.PROOF_COUNTERS) and the number of episodes that
                      record it (0 for evaluations written before the counter existed)
and prints the table and the filter compute time from filter_timing.json (time_filter.py).
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from busaware.rollout import PROOF_COUNTERS  # noqa: E402
from busaware.runs import run_config  # noqa: E402

METRICS = ["uv_s", "oc_s", "collapse_s", "ov_s", "uv_frac", "oc_frac", "collapse_frac",
           "filt_frac", "infeas_s", "infeas_frac", "v_min", "i_max", "n_hops", "height_max",
           "apex_mean", "apex_max", "z_max", "fell", "ret", "e_elec"] + [k + "_n" for k in PROOF_COUNTERS]
# proof-assumption counters (A3, Remark 1: CHANGELOG #23; A4, A6: #25); eval.csv files written
# before a counter existed have no column for it (NaN, reported as not recorded)
PER_SEED_TABLE = ["method", "train", "test_mode", "soc", "seed", "n_episodes", "n_fell", "n_hops",
                  "height_max", "uv_s", "oc_s", "collapse_s"]
ORDER = ["baseline", "baseline+filter@test", "penalty", "penalty+filter@test", "filter",
         # warm starts from baseline_s* (CHANGELOG #19)
         "baseline_ws", "baseline_ws+filter@test", "penalty_w0.05_ws", "penalty_w0.05_ws+filter@test",
         "penalty_w0.1_ws", "penalty_w0.1_ws+filter@test",
         "penalty_w0.2_ws", "penalty_w0.2_ws+filter@test",     # seed 0 only (CHANGELOG #21)
         "penalty_w0.5_ws", "penalty_w0.5_ws+filter@test",
         "penalty_w2.0_ws", "penalty_w2.0_ws+filter@test",     # seed 0 only (CHANGELOG #22)
         "lagrangian_ws", "lagrangian_ws+filter@test",         # CHANGELOG #22
         "filter_ws",
         "lagrangian", "lagrangian+filter@test", "perjoint"]


def method_name(train, train_mode, test_mode):
    return train if test_mode == train_mode else f"{train}+{test_mode}@test"


def load(results=ROOT / "results"):
    frames = []
    for f in sorted(results.glob("*/eval.csv")):
        m = re.fullmatch(r"(.+)_s(\d+)", f.parent.name)
        if m:
            d = pd.read_csv(f)
            d["train"], d["seed"] = m.group(1), int(m.group(2))
            d["train_mode"] = run_config(f.parent)["mode"]
            for k in [c + "_n" for c in PROOF_COUNTERS]:
                if k not in d:
                    d[k] = float("nan")              # not recorded in this evaluation
            if "height_max" not in d:            # written before z0 was recorded: z0 = 0 (see above)
                d["z0"] = 0.0
                d["height_max"] = d["z_max"] - d["z0"]
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["method"] = [method_name(t, tm, m) for t, tm, m in zip(df["train"], df["train_mode"], df["test_mode"])]
    return df


def summarise(df):
    keys = ["method", "train", "test_mode", "soc", "seed"]
    per_seed = (df.groupby(keys)
                  .agg(n_episodes=("episode", "size"), n_fell=("fell", "sum"),
                       **{k: (k, "mean") for k in METRICS})
                  .reset_index())
    g = per_seed.groupby(["method", "train", "test_mode", "soc"])
    out = g.agg(n_seeds=("seed", "size"), n_episodes=("n_episodes", "sum"),
                apex_n_seeds=("apex_mean", "count")).reset_index()
    for k in METRICS:
        out[f"{k}_mean"] = g[k].mean().values
        out[f"{k}_std"] = g[k].std(ddof=1).values
    rank = {m: i for i, m in enumerate(ORDER)}
    for t in (per_seed, out):
        t["_r"] = [rank.get(m, len(ORDER)) for m in t["method"]]
    out = out.sort_values(["_r", "soc"], ascending=[True, False]).drop(columns="_r")
    per_seed = per_seed.sort_values(["_r", "soc", "seed"], ascending=[True, False, True]).drop(columns="_r")
    return per_seed.reset_index(drop=True), out.reset_index(drop=True)


def proof_counters(df):
    """Per method, test mode and SOC: for each proof-assumption counter, the substeps summed over
    the episodes that record it, and how many episodes those are."""
    rows = []
    for (meth, tm, soc), g in df.groupby(["method", "test_mode", "soc"], sort=False):
        row = dict(method=meth, test_mode=tm, soc=soc, n_episodes=len(g))
        for c in PROOF_COUNTERS:
            rec = g[c + "_n"].dropna()
            row[f"{c}_episodes"] = len(rec)
            row[f"{c}_substeps"] = int(rec.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load()
    if df is None:
        print("no results/*/eval.csv found")
        return
    per_seed, out = summarise(df)
    sd = ROOT / "results" / "summary"
    sd.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(sd / "per_seed.csv", index=False)
    per_seed[PER_SEED_TABLE].to_csv(sd / "per_seed_table.csv", index=False)
    out.to_csv(sd / "violations.csv", index=False)
    proof_counters(df).to_csv(sd / "proof_counters.csv", index=False)

    pm = lambda r, k, f: f"{r[k + '_mean']:{f}} +/- {r[k + '_std']:{f}}"
    print("mean +/- std over seeds; times in s per 4 s episode; height = max torso height above start")
    hdr = (f"{'method':22s} {'SOC':>4s} {'n':>2s} {'below 16.8 V (s)':>17s} {'above I_lim (s)':>17s} "
           f"{'collapse (s)':>17s} {'V_min (V)':>13s} {'I_max (A)':>13s} {'hops':>11s} "
           f"{'max height (m)':>15s} {'mean apex (m) [seeds]':>21s} {'falls':>11s} {'return':>15s}")
    print(hdr)
    for _, r in out.iterrows():
        print(f"{r.method:22s} {r.soc:4.1f} {int(r.n_seeds):2d} {pm(r, 'uv_s', '7.4f'):>17s} "
              f"{pm(r, 'oc_s', '7.4f'):>17s} {pm(r, 'collapse_s', '7.4f'):>17s} {pm(r, 'v_min', '5.2f'):>13s} "
              f"{pm(r, 'i_max', '5.1f'):>13s} {pm(r, 'n_hops', '4.1f'):>11s} {pm(r, 'height_max', '6.3f'):>15s} "
              f"{pm(r, 'apex_mean', '6.3f') + f' [{int(r.apex_n_seeds)}]':>21s} {pm(r, 'fell', '4.2f'):>11s} "
              f"{pm(r, 'ret', '6.1f'):>15s}")
    print("\nper seed: falls (episodes), hops, max height (m), below 16.8 V / above I_lim / collapse (s)")
    for _, r in per_seed.iterrows():
        print(f"{r.method:22s} {r.soc:4.1f} s{int(r.seed)} {int(r.n_fell):2d}/{int(r.n_episodes)} "
              f"{r.n_hops:5.2f} {r.height_max:7.4f} {r.uv_s:7.4f} {r.oc_s:7.4f} {r.collapse_s:7.4f}")
    filt = out[out["test_mode"] == "filter"]
    if len(filt):
        print(f"filter active (fraction of substeps): {filt['filt_frac_mean'].min():.3f}"
              f"-{filt['filt_frac_mean'].max():.3f}; infeasible time max {filt['infeas_s_mean'].max():.4f} s")
    print(f"proof-assumption counters, substeps summed over all recorded episodes (of {len(df)}):")
    names = dict(a3_fail="A3 physical bus state fails", rem1_fail="Remark 1 bound fails",
                 a4_fail="A4 plant root below x*", a6_empty="A6 empty circle/box intersection")
    for c in PROOF_COUNTERS:
        rec = df.dropna(subset=[c + "_n"])
        print(f"  {names[c]:34s} {int(rec[c + '_n'].sum()):8d}  ({len(rec)} episodes record it)")
        if len(rec):
            by = rec.groupby("method")[c + "_n"].sum()
            for meth, n in by[by > 0].items():
                print(f"      {meth:30s} {int(n)}")
    tf = sd / "filter_timing.json"
    if tf.exists():
        t = json.loads(tf.read_text())
        print(f"filter compute time per call: mean {t['total_us']['mean']:.1f} us, "
              f"p99 {t['total_us']['p99']:.1f} us ({t['n_calls']} calls)")
    print("wrote", sd / "violations.csv", sd / "per_seed.csv", sd / "per_seed_table.csv",
          sd / "proof_counters.csv")


if __name__ == "__main__":
    main()
