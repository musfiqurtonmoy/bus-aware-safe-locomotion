"""Step 9: merge the evaluation-setting files of the warm-start runs into one ablation table.

    python scripts/aggregate_ablations.py

Reads results/<run>/eval_*.csv written by scripts/evaluate.py with --variant, --gamma,
--r0_scale or --est_noise (file names: busaware/runs.eval_filename), for the runs baseline_ws_s*
and filter_ws_s*, plus a reference "base" setting taken from the same runs' eval.csv restricted
to the episodes whose reset seeds the setting files use (1000..1009, i.e. the first 10 episodes;
evaluate.py gives episode ep the reset seed 1000 + ep in every file), so every row compares the
same initial states.

Groups (column `ablation`):
  variant     Params variant applied to plant and filter model alike (cold, Params.VARIANTS);
              current_lag and gear_eff act on the plant only (busaware/runs.py)
  gamma       filter barrier rate; 0.3 is the default (= base)
  model_error the filter's model differs from the plant: est_noise perturbs its EMF and R0
              estimates, r0_scale scales the plant's R0 only. The only settings where the
              filter's model is wrong.
  estimator   Step 8, filter_ws only: the filter's EMF and R0 from the true state ("true"), the
              battery EKF with fixed margins ("ekf") or the EKF with confidence-scaled margins
              ("ekfconf"), each under nominal, est_noise=0.05, r0_scale=2.0 and r0_scale=0.75. The "true" rows
              are copies of the base and model_error rows. est_noise acts on the E and R0
              estimates for "true" and on the EKF's two measurements for the EKF rows
              (busaware/leg_env.py). EKF columns: ekf_e_abs_mean (mean |E_hat - E|, V),
              ekf_r0_rel_final (R0_hat / plant R0 at the episode end), ekf_conf_mean,
              margin_v_mean (V) and margin_i_mean (A).
  controller  CHANGELOG #38: test mode perjoint, the joint-level limiter (the filter's budget split
              equally between the drives, each joint limited on its own), for baseline_ws and
              filter_ws (methods <run>+perjoint@test). Compare with the base rows.
  margin      CHANGELOG #38: filter_ws with one fixed margin changed (delta_V or delta_I); the
              other keeps its default (env margin_v 0.2 V, margin_i 0.5 A).
Method = run name without the seed, plus "+filter@test" for a policy evaluated in another mode
than its training mode (as scripts/aggregate.py). Episodes are averaged within a seed first,
then mean and standard deviation (ddof=1) over seeds; n_fell is the total over seeds.
i_max (peak battery discharge current of an episode, A) is averaged like the others; i_max_peak is
the largest over all episodes of the row. EKF rows also give n_r0_diverged, the number of episodes
whose final R0 estimate is outside R0_DIV_BOUNDS x the plant's R0, and ekf_conf_ep_min/max, the
smallest and largest episode-mean confidence over the row's episodes (NaN for true-state rows).
Times are seconds of simulated time per 4 s episode. uv_s counts time below the setting's own
v_min (12 V under vmin_datasheet). n_ep_below_base_vmin counts episodes whose minimum bus
voltage is below the base v_min (Params().v_min = 16.8 V); in every other episode the time below
16.8 V is zero. Proof-assumption counters (_n) are substeps summed over the setting's episodes.

Writes results/summary/ablations.csv and prints it; for vmin_datasheet also the time below
16.8 V (results/summary/ablation_vmin_datasheet_below_16v8.csv, see vmin_datasheet_below_base)
and, per filter-mode evaluation, whether every episode equals the base episode
(ablation_vmin_datasheet_identity.csv); for est_noise the first-order current-barrier rule
against the observed current overshoot (est_noise_sensitivity.csv, see est_noise_sensitivity).
CHANGELOG #56: also the EXTRA_RUNS (penalty_w0.5_ws_s*) with the filter at test time under the
variants and barrier rates, with their base rows, summarised like the other rows into
results/summary/ablations_penalty.csv (see load_extra); ablations.csv is unchanged by them.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from busaware.params import Params  # noqa: E402
from busaware.rollout import PROOF_COUNTERS  # noqa: E402
from busaware.runs import EVAL_VARIANTS, run_config  # noqa: E402

RUNS = ("baseline_ws", "filter_ws")
SEEDS = (0, 1, 2)
RESULTS = ROOT / "results"
# CHANGELOG #56: warm-started penalty policies evaluated with the filter added at test time under the
# variants and barrier rates (scripts/run_penalty_stress.ps1). Only these groups and this test mode
# are read; their base rows are the runs' eval.csv rows of the same test mode and reset seeds. They
# are summarised separately (results/summary/ablations_penalty.csv), so ablations.csv and every
# number taken from it are unchanged.
EXTRA_RUNS = {"penalty_w0.5_ws": ("variant", "gamma")}
EXTRA_TEST_MODE = "filter"
FILE_RE = re.compile(r"eval(?:_var-(?P<var>[^_]+(?:_[^_]+)*?))?(?:_g(?P<g>[0-9.]+))?"
                     r"(?:_est-(?P<est>ekf|ekfconf))?(?:_ctl-(?P<ctl>perjoint))?"
                     r"(?:_mv(?P<mv>[0-9.]+))?(?:_mi(?P<mi>[0-9.]+))?"
                     r"(?:_r(?P<r>[0-9.]+)_n(?P<n>[0-9.]+))?\.csv")
MEAN_METRICS = ["uv_s", "oc_s", "ov_s", "collapse_s", "infeas_s", "filt_frac", "fell", "n_hops",
                "height_max", "v_min", "i_max", "ret",
                "ekf_e_abs_mean", "ekf_r0_rel_final", "ekf_conf_mean", "margin_v_mean", "margin_i_mean"]
EST_CONDITIONS = {("base", "base"): "nominal", ("model_error", "est_noise=0.05"): "est_noise=0.05",
                  ("model_error", "r0_scale=2.0"): "r0_scale=2.0",
                  ("model_error", "r0_scale=0.75"): "r0_scale=0.75"}
COUNTERS = [c + "_n" for c in PROOF_COUNTERS]
# EKF R0 estimate counted as diverged when its final value is outside these multiples of the plant's
# R0 (a reporting threshold, not used by the controller; CHANGELOG #34)
R0_DIV_BOUNDS = (0.5, 2.0)
BASE_VMIN = Params().v_min
# columns that describe the evaluation setting, not the episode (absent from files written before
# they were added): excluded when episodes are compared with their base episode
SETTING_COLS = {"variant", "gamma", "r0_scale", "est_noise", "estimator", "margin_mode", "ablation",
                "setting", "method", "train_mode", "run", "reset_seed", "margin_v_fixed", "margin_i_fixed"}


def setting_of(name):
    """(ablation, setting) for an eval file name, or None for files that are not Step 9 settings
    (e.g. eval_pre_eps.csv)."""
    m = FILE_RE.fullmatch(name)
    if not m or name == "eval.csv":
        return None
    var, g, r, n, est = m.group("var"), m.group("g"), m.group("r"), m.group("n"), m.group("est")
    ctl, mv, mi = m.group("ctl"), m.group("mv"), m.group("mi")
    # CHANGELOG #38: joint-level limiter and fixed-margin ablation, each only on its own
    if ctl or mv or mi:
        if var or g or est or r:
            return None
        if ctl and not (mv or mi):
            return "controller", ctl
        if not ctl and (mv or mi):
            return "margin", ", ".join(([f"delta_V={float(mv)}"] if mv else []) + ([f"delta_I={float(mi)}"] if mi else []))
        return None
    if est:
        if var or g:
            return None
        if not r:
            return "estimator", f"{est}, nominal"
        r, n = float(r), float(n)
        if r == 1.0 and n != 0.0:
            return "estimator", f"{est}, est_noise={n}"
        if n == 0.0 and r != 1.0:
            return "estimator", f"{est}, r0_scale={r}"
        return None
    if var and not g and not r and var in EVAL_VARIANTS:
        return "variant", var
    if g and not var and not r:
        return "gamma", f"gamma={float(g)}"
    if r and not var and not g:
        r, n = float(r), float(n)
        if n == 0.0 and r != 1.0:
            return "model_error", f"r0_scale={r}"
        if r == 1.0 and n != 0.0:
            return "model_error", f"est_noise={n}"
    return None


def load(runs=RUNS, groups=None, test_mode=None, results=None):
    """Setting files and base rows of `runs` (seeds SEEDS) under `results` (default results/).
    groups: ablation groups to keep (None: all); test_mode: keep only this test mode (None: all).
    With groups or test_mode given (the extra runs), base rows are added only for the run seeds
    that have at least one kept setting file; otherwise for every run seed (eval.csv required)."""
    results = pathlib.Path(results) if results is not None else RESULTS
    restricted = groups is not None or test_mode is not None
    frames, n_ep, found = [], set(), set()
    for run in runs:
        for s in SEEDS:
            d_run = results / f"{run}_s{s}"
            if restricted and not d_run.is_dir():
                continue
            train_mode = run_config(d_run)["mode"]
            for f in sorted(d_run.glob("eval_*.csv")):
                st = setting_of(f.name)
                if st is None or (groups is not None and st[0] not in groups):
                    continue
                d = pd.read_csv(f)
                if test_mode is not None:
                    d = d[d["test_mode"] == test_mode]
                if d.empty:
                    continue
                d["ablation"], d["setting"] = st
                n_ep.add(int(d["episode"].max()) + 1)
                frames.append(d.assign(train=run, seed=s, train_mode=train_mode))
                found.add((run, s))
    if not frames:
        return None
    if len(n_ep) != 1:
        raise ValueError(f"setting files have different episode counts: {sorted(n_ep)}")
    n = n_ep.pop()
    # base reference: the same runs' eval.csv, episodes with the setting files' reset seeds
    for run in runs:
        for s in SEEDS:
            if restricted and (run, s) not in found:
                continue
            d_run = results / f"{run}_s{s}"
            d = pd.read_csv(d_run / "eval.csv")
            d = d[d["episode"] < n]
            if test_mode is not None:
                d = d[d["test_mode"] == test_mode]
            frames.append(d.assign(ablation="base", setting="base", train=run, seed=s,
                                   train_mode=run_config(d_run)["mode"]))
    df = pd.concat(frames, ignore_index=True)
    # Step 8: the true-state reference rows of the estimator group (filter_ws, filter mode)
    if (df["ablation"] == "estimator").any():
        ref = [df[(df["ablation"] == a) & (df["setting"] == st) & (df["train"] == "filter_ws")]
               .assign(ablation="estimator", setting=f"true, {cond}")
               for (a, st), cond in EST_CONDITIONS.items()]
        df = pd.concat([df] + ref, ignore_index=True)
    df["method"] = [t if tm == trm else f"{t}+{tm}@test"
                    for t, tm, trm in zip(df["train"], df["test_mode"], df["train_mode"])]
    df["below_base_vmin"] = (df["v_min"] < BASE_VMIN).astype(int)
    for k in MEAN_METRICS:              # EKF columns exist only in EKF evaluations
        if k not in df:
            df[k] = float("nan")
    lo, hi = R0_DIV_BOUNDS
    r0 = df["ekf_r0_rel_final"]
    df["r0_div"] = ((r0 < lo) | (r0 > hi)).astype(float).where(r0.notna())
    return df, n


def load_extra(results=None):
    """CHANGELOG #56: the EXTRA_RUNS (penalty_w0.5_ws) with the filter at test time under their
    groups (variant, gamma) plus base rows; methods <run>+filter@test, the same metrics as the
    baseline_ws+filter@test rows of ablations.csv. None if no setting file exists yet."""
    frames, ns = [], set()
    for run, groups in EXTRA_RUNS.items():
        res = load(runs=(run,), groups=groups, test_mode=EXTRA_TEST_MODE, results=results)
        if res is not None:
            frames.append(res[0])
            ns.add(res[1])
    if not frames:
        return None
    if len(ns) != 1:
        raise ValueError(f"extra runs have different episode counts: {sorted(ns)}")
    return pd.concat(frames, ignore_index=True), ns.pop()


def summarise(df):
    keys = ["ablation", "setting", "method", "soc"]
    per_seed = (df.groupby(keys + ["seed"])
                  .agg(n_episodes=("episode", "size"), n_fell=("fell", "sum"),
                       n_ep_below_base_vmin=("below_base_vmin", "sum"),
                       i_max_peak=("i_max", "max"),
                       n_r0_diverged=("r0_div", lambda x: x.sum(min_count=1)),
                       ekf_conf_ep_min=("ekf_conf_mean", "min"), ekf_conf_ep_max=("ekf_conf_mean", "max"),
                       **{k: (k, "mean") for k in MEAN_METRICS},
                       **{k: (k, "sum") for k in COUNTERS})
                  .reset_index())
    g = per_seed.groupby(keys)
    out = g.agg(n_seeds=("seed", "size"), n_episodes=("n_episodes", "sum"), n_fell=("n_fell", "sum"),
                n_ep_below_base_vmin=("n_ep_below_base_vmin", "sum"),
                i_max_peak=("i_max_peak", "max"),
                n_r0_diverged=("n_r0_diverged", lambda x: x.sum(min_count=1)),
                ekf_conf_ep_min=("ekf_conf_ep_min", "min"), ekf_conf_ep_max=("ekf_conf_ep_max", "max"),
                **{k: (k, "sum") for k in COUNTERS}).reset_index()
    for k in MEAN_METRICS:
        out[f"{k}_mean"] = g[k].mean().values
        out[f"{k}_std"] = g[k].std(ddof=1).values
    out["filter_model_wrong"] = out["ablation"] == "model_error"
    order_a = {a: i for i, a in enumerate(["base", "variant", "gamma", "model_error", "estimator", "controller",
                                           "margin"])}
    order_s = {v: i for i, v in enumerate(("base",) + EVAL_VARIANTS)}
    order_m = {m: i for i, m in enumerate(["baseline_ws", "baseline_ws+filter@test", "baseline_ws+perjoint@test",
                                           "filter_ws", "filter_ws+perjoint@test"])}
    est_rank = {e: i for i, e in enumerate(["true", "ekf", "ekfconf"])}
    cond_rank = lambda st: next((i for i, c in enumerate(EST_CONDITIONS.values()) if st.endswith(c)), 9)
    out["_k"] = [(order_a[a], order_s.get(s, 99),
                  (cond_rank(s), est_rank.get(s.split(",")[0], 9)) if a == "estimator" else s,
                  order_m.get(m, 9), -soc)
                 for a, s, m, soc in zip(out["ablation"], out["setting"], out["method"], out["soc"])]
    return out.sort_values("_k").drop(columns="_k").reset_index(drop=True)


def vmin_datasheet_below_base(df):
    """Time below the base v_min (16.8 V) under vmin_datasheet, where uv_s counts time below 12 V.
    Not recorded directly; obtained per (method, SOC) from the episodes:
      - every episode with minimum bus voltage >= 16.8 V has zero time below 16.8 V;
      - an episode whose row equals the base episode (same run, test mode, SOC, reset seed) in
        every recorded column except uv_s/uv_frac has the same trajectory, so its time below
        16.8 V is the base episode's uv_s (the base v_min is 16.8 V).
    Rows where neither holds for some episode are reported as not determined (NaN)."""
    v = df[(df["ablation"] == "variant") & (df["setting"] == "vmin_datasheet")]
    b = df[df["ablation"] == "base"]
    key = ["train", "seed", "test_mode", "soc", "episode"]
    skip = set(key) | {"uv_s", "uv_frac", "below_base_vmin"} | SETTING_COLS
    cols = [c for c in v.columns if c not in skip and pd.api.types.is_numeric_dtype(v[c])]
    m = v.merge(b, on=key, suffixes=("", "_base"))
    if len(m) != len(v):
        raise ValueError("vmin_datasheet episodes without a base episode")
    same = pd.Series(True, index=m.index)
    for c in cols:
        same &= (m[c] == m[c + "_base"]) | (m[c].isna() & m[c + "_base"].isna())
    t = pd.Series(float("nan"), index=m.index)
    t[m["v_min"] >= BASE_VMIN] = 0.0
    t[same] = m.loc[same, "uv_s_base"]
    m["below_base_vmin_s"], m["how"] = t, ["v_min>=16.8" if vm >= BASE_VMIN else ("identical to base" if s else "undetermined")
                                           for vm, s in zip(m["v_min"], same)]
    per_seed = m.groupby(["method", "soc", "seed"]).agg(t=("below_base_vmin_s", "mean")).reset_index()
    g = per_seed.groupby(["method", "soc"])["t"]
    out = pd.DataFrame({"below_base_vmin_s_mean": g.mean(), "below_base_vmin_s_std": g.std(ddof=1)}).reset_index()
    how = m.groupby(["method", "soc"])["how"].agg(lambda s: "; ".join(f"{k} {n}" for k, n in s.value_counts().items()))
    return out.merge(how.reset_index(), on=["method", "soc"]).sort_values(["method", "soc"], ascending=[True, False])


def vmin_datasheet_identity(df):
    """Per filter-mode evaluation (run, test mode): is every vmin_datasheet episode equal to the
    base episode in every recorded column except uv_s/uv_frac? If so, the 16.8 V barrier never
    set the budget at a substep where the projection was active (with 12 V that budget would
    have been larger and the commanded currents different)."""
    v = df[(df["ablation"] == "variant") & (df["setting"] == "vmin_datasheet") & (df["test_mode"] == "filter")]
    b = df[(df["ablation"] == "base") & (df["test_mode"] == "filter")]
    key = ["train", "seed", "test_mode", "soc", "episode"]
    skip = set(key) | {"uv_s", "uv_frac", "below_base_vmin"} | SETTING_COLS
    cols = [c for c in v.columns if c not in skip and pd.api.types.is_numeric_dtype(v[c])
            and v[c].notna().any()]
    m = v.merge(b, on=key, suffixes=("", "_base"))
    same = pd.Series(True, index=m.index)
    for c in cols:
        same &= (m[c] == m[c + "_base"]) | (m[c].isna() & m[c + "_base"].isna())
    m["same"] = same
    return (m.groupby(["method", "train", "seed"])
             .agg(n_episodes=("same", "size"), n_identical=("same", "sum"),
                  max_abs_dheight=("height_max", lambda x: float((x - m.loc[x.index, "height_max_base"]).abs().max())))
             .reset_index().assign(identical=lambda t: t["n_identical"] == t["n_episodes"]))


def est_noise_sensitivity(df):
    """First-order sensitivity of the current barrier (paper, remark after Corollary 1) against
    the observed overshoot, filter_ws in filter mode under est_noise (true-state filter).
    The filter's estimates are E (1 + s n1) and R0 (1 + s n2), n ~ N(0, 1) independent per
    substep, so a one-standard-deviation error is |dE| = s E, |dR0| = s R0, and the rule
    dI = |dE|/R0 + |E - V| |dR0| / R0^2 with |E - V| = R0 I_target gives
    rule_1sd = s (E/R0 + I_target). E: the pack OCV at the episode's start SOC (E <= OCV while
    discharging); R0 = Params().r0; I_target = i_bat_max - margin_i, the current the filter aims at.
    Observed: overshoot = per-episode peak battery current - I_target; mean over seeds of the
    per-seed episode mean, std over seeds, and the largest episode."""
    from busaware.battery import Battery
    from busaware.leg_env import HoppingLegEnv
    p, env = Params(), HoppingLegEnv()
    i_target = p.i_bat_max - env.margin_i
    e = df[(df["ablation"] == "model_error") & df["setting"].str.startswith("est_noise=")
           & (df["method"] == "filter_ws")].copy()
    rows = []
    for (st, soc), g in e.groupby(["setting", "soc"]):
        sig = float(st.split("=")[1])
        over = g["i_max"] - i_target
        per_seed = over.groupby(g["seed"]).mean()
        ocv = Battery(p, soc).ocv()
        rule_e, rule_r = sig * ocv / p.r0, sig * i_target
        rows.append(dict(est_noise=sig, soc=soc, e_ocv=ocv, r0=p.r0, i_target=i_target,
                         margin_i=env.margin_i, rule_e=rule_e, rule_r=rule_r, rule_1sd=rule_e + rule_r,
                         over_mean=per_seed.mean(), over_std=per_seed.std(ddof=1), over_max=over.max(),
                         n_seeds=len(per_seed), n_episodes=len(g), oc_s_mean=g.groupby("seed")["oc_s"].mean().mean(),
                         ratio_max_to_rule=over.max() / (rule_e + rule_r)))
    return pd.DataFrame(rows).sort_values(["est_noise", "soc"], ascending=[True, False])


def main():
    res = load()
    if res is None:
        print("no Step 9 evaluation files found")
        return
    df, n = res
    out = summarise(df)
    sd = ROOT / "results" / "summary"
    sd.mkdir(parents=True, exist_ok=True)
    out.to_csv(sd / "ablations.csv", index=False)
    pm = lambda r, k, f: f"{r[k + '_mean']:{f}}+/-{r[k + '_std']:{f}}"
    print(f"mean +/- std over seeds, {n} episodes per seed per SOC (reset seeds 1000..{999 + n}); "
          f"base = eval.csv episodes 0..{n - 1}; times in s per 4 s episode")
    print(f"{'ablation':11s} {'setting':16s} {'method':24s} {'SOC':>3s} {'uv (s)':>15s} {'oc (s)':>15s} "
          f"{'ov (s)':>15s} {'collapse (s)':>15s} {'infeas (s)':>15s} {'falls':>6s} {'hops':>9s} "
          f"{'max h (m)':>13s} {'<16.8V ep':>9s} {'A3 R1 A4 A6':>13s}")
    for _, r in out.iterrows():
        print(f"{r.ablation:11s} {r.setting:16s} {r.method:24s} {r.soc:3.1f} {pm(r, 'uv_s', '6.4f'):>15s} "
              f"{pm(r, 'oc_s', '6.4f'):>15s} {pm(r, 'ov_s', '6.4f'):>15s} {pm(r, 'collapse_s', '6.4f'):>15s} "
              f"{pm(r, 'infeas_s', '6.4f'):>15s} {int(r.n_fell):3d}/{int(r.n_episodes):<2d} "
              f"{pm(r, 'n_hops', '3.1f'):>9s} {pm(r, 'height_max', '5.3f'):>13s} "
              f"{int(r.n_ep_below_base_vmin):>9d} "
              f"{' '.join(str(int(r[c])) for c in COUNTERS):>13s}")
    est = out[out["ablation"] == "estimator"]
    if len(est):
        print(f"\nStep 8 estimator (filter_ws): peak battery current (A; mean +/- std over seeds of the "
              f"episode peak, and largest; I_lim {Params().i_bat_max} A), EKF |E err| (V), R0_hat/R0 at end, "
              f"confidence, margins (V, A)")
        for _, r in est.iterrows():
            print(f"  {r.setting:26s} soc {r.soc:.1f}: i_max {pm(r, 'i_max', '5.2f')} (largest "
                  f"{r.i_max_peak:5.2f}) |E err| {r.ekf_e_abs_mean_mean:.4f} R0 ratio "
                  f"{r.ekf_r0_rel_final_mean:.3f} (diverged {r.n_r0_diverged:.0f}/{int(r.n_episodes)}) conf "
                  f"{r.ekf_conf_mean_mean:.2f} [{r.ekf_conf_ep_min:.2f}, {r.ekf_conf_ep_max:.2f}] margins "
                  f"{r.margin_v_mean_mean:.2f}/{r.margin_i_mean_mean:.2f}")
    filt = out[out["method"] != "baseline_ws"]
    viol = filt[(filt[["uv_s_mean", "oc_s_mean", "ov_s_mean", "collapse_s_mean"]] > 0).any(axis=1)]
    fell = filt[(filt["method"] == "filter_ws") & (filt["n_fell"] > 0)]
    print(f"\nfilter-mode rows with a limit violation (uv, oc, ov or collapse > 0): {len(viol)}")
    for _, r in viol.iterrows():
        print(f"  {r.ablation} {r.setting} {r.method} soc {r.soc}: uv {r.uv_s_mean:.4f} oc {r.oc_s_mean:.4f} "
              f"ov {r.ov_s_mean:.4f} collapse {r.collapse_s_mean:.4f} s")
    print(f"filter_ws rows with a fall: {len(fell)}")
    for _, r in fell.iterrows():
        print(f"  {r.ablation} {r.setting} soc {r.soc}: {int(r.n_fell)}/{int(r.n_episodes)}")
    vd = vmin_datasheet_below_base(df)
    vd.to_csv(sd / "ablation_vmin_datasheet_below_16v8.csv", index=False)
    print(f"\nvmin_datasheet: time below the base v_min {BASE_VMIN} V (s per episode), mean +/- std over seeds")
    for _, r in vd.iterrows():
        print(f"  {r.method:24s} soc {r.soc:.1f}: {r.below_base_vmin_s_mean:.4f} +/- {r.below_base_vmin_s_std:.4f}"
              f"   (episodes: {r.how})")
    vi = vmin_datasheet_identity(df)
    vi.to_csv(sd / "ablation_vmin_datasheet_identity.csv", index=False)
    print("\nvmin_datasheet, filter-mode evaluations identical to base in every episode "
          f"(except uv): {int(vi['identical'].sum())} of {len(vi)}")
    for _, r in vi.iterrows():
        print(f"  {r.method:24s} s{int(r.seed)}: {int(r.n_identical)}/{int(r.n_episodes)} episodes identical, "
              f"max |d height| {r.max_abs_dheight:.2e} m")
    se = est_noise_sensitivity(df)
    se.to_csv(sd / "est_noise_sensitivity.csv", index=False)
    print("\nest_noise: first-order current-barrier rule (1 sd) vs observed overshoot over I_lim - margin_i (A)")
    for _, r in se.iterrows():
        print(f"  est_noise {r.est_noise:.2f} soc {r.soc:.1f}: rule {r.rule_1sd:.2f} (E term {r.rule_e:.2f}, R0 term "
              f"{r.rule_r:.2f}); observed mean {r.over_mean:.2f} +/- {r.over_std:.2f}, largest {r.over_max:.2f} "
              f"(ratio {r.ratio_max_to_rule:.2f}); above I_lim {r.oc_s_mean:.4f} s")
    print("wrote", sd / "ablations.csv", sd / "ablation_vmin_datasheet_below_16v8.csv",
          sd / "ablation_vmin_datasheet_identity.csv", sd / "est_noise_sensitivity.csv")
    # CHANGELOG #56: penalty policies with the filter at test time, variants and barrier rates
    ex = load_extra()
    if ex is None:
        print("\nno penalty stress-test files (scripts/run_penalty_stress.ps1) found")
        return
    edf, en = ex
    if en != n:
        raise ValueError(f"penalty stress tests have {en} episodes per seed and SOC, the others {n}")
    eout = summarise(edf)
    eout.to_csv(sd / "ablations_penalty.csv", index=False)
    print(f"\npenalty policies with the filter at test time (mean +/- std over seeds, {en} episodes per seed per SOC)")
    for _, r in eout.iterrows():
        print(f"  {r.ablation:8s} {r.setting:16s} {r.method:30s} {r.soc:3.1f} seeds {int(r.n_seeds)} "
              f"uv {pm(r, 'uv_s', '6.4f')} oc {pm(r, 'oc_s', '6.4f')} collapse {pm(r, 'collapse_s', '6.4f')} "
              f"infeas {pm(r, 'infeas_s', '6.4f')} falls {int(r.n_fell)}/{int(r.n_episodes)} "
              f"hops {pm(r, 'n_hops', '3.1f')} max h {pm(r, 'height_max', '5.3f')} "
              f"filter active {pm(r, 'filt_frac', '5.3f')}")
    print("wrote", sd / "ablations_penalty.csv")


if __name__ == "__main__":
    main()
