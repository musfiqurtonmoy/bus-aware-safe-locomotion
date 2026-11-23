"""Turn results/ into LaTeX inputs under paper/generated/, then build the Overleaf zip.

  python scripts/export_paper.py
Reads results/summary/violations.csv (scripts/aggregate.py: episodes averaged within a seed,
then mean and std over seeds) and results/summary/filter_timing.json. Writes:
  paper/generated/results.tex         \resdef{key}{value} macros
  paper/generated/tab_violations.tex  main results table (from-scratch runs)
  paper/generated/tab_warmstart.tex   warm-started runs (*_ws), if evaluated
  paper/generated/tab_per_seed.tex    supplementary: per-seed falls, hops, max height, violation
                                      times (results/summary/per_seed_table.csv); *_warmstart for *_ws
  paper/generated/tab_params.tex      parameters, straight from busaware/params.py
  paper/generated/figures/*.pdf       copied from results/figures/
  paper_overleaf.zip                  upload this to Overleaf (New Project > Upload)
Missing inputs are skipped; the manuscript shows red [pending] markers for them.
"""
import sys, pathlib, re, shutil, zipfile, datetime
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from busaware.params import Params

GEN = ROOT / "paper" / "generated"; (GEN / "figures").mkdir(parents=True, exist_ok=True)
SHORT = {"baseline": "base", "filter": "filt", "penalty": "pen", "lagrangian": "lag", "perjoint": "pj"}
macros, lines = {}, []

SUMMARY = ROOT / "results" / "summary"
LABEL = {"baseline": "Baseline", "baseline+filter@test": "Baseline + filter at test",
         "penalty": "Penalty", "penalty+filter@test": "Penalty + filter at test",
         "filter": "Filter (trained with)",
         "baseline_ws": "Baseline, warm start", "baseline_ws+filter@test": "Baseline, warm start + filter at test",
         "penalty_w0.05_ws": "Penalty $w{=}0.05$, warm start",
         "penalty_w0.05_ws+filter@test": "Penalty $w{=}0.05$, warm start + filter at test",
         "penalty_w0.1_ws": "Penalty $w{=}0.1$, warm start",
         "penalty_w0.1_ws+filter@test": "Penalty $w{=}0.1$, warm start + filter at test",
         "penalty_w0.2_ws": "Penalty $w{=}0.2$, warm start",
         "penalty_w0.2_ws+filter@test": "Penalty $w{=}0.2$, warm start + filter at test",
         "penalty_w0.5_ws": "Penalty $w{=}0.5$, warm start",
         "penalty_w0.5_ws+filter@test": "Penalty $w{=}0.5$, warm start + filter at test",
         "penalty_w2.0_ws": "Penalty $w{=}2$, warm start",
         "penalty_w2.0_ws+filter@test": "Penalty $w{=}2$, warm start + filter at test",
         "lagrangian_ws": "SAC-Lagrangian, warm start",
         "lagrangian_ws+filter@test": "SAC-Lagrangian, warm start + filter at test",
         "filter_ws": "Filter, warm start"}
KEYS = {"baseline": "base", "baseline+filter@test": "base_filttest", "penalty": "pen",
        "penalty+filter@test": "pen_filttest", "filter": "filt",
        "baseline_ws": "basews", "baseline_ws+filter@test": "basews_filttest",
        "penalty_w0.05_ws": "penwAws", "penalty_w0.05_ws+filter@test": "penwAws_filttest",
        "penalty_w0.1_ws": "penwBws", "penalty_w0.1_ws+filter@test": "penwBws_filttest",
        "penalty_w0.2_ws": "penwCws", "penalty_w0.2_ws+filter@test": "penwCws_filttest",
        "penalty_w0.5_ws": "penwDws", "penalty_w0.5_ws+filter@test": "penwDws_filttest",
        "penalty_w2.0_ws": "penwEws", "penalty_w2.0_ws+filter@test": "penwEws_filttest",
        "lagrangian_ws": "lagws", "lagrangian_ws+filter@test": "lagws_filttest",
        "filter_ws": "filtws",
        "baseline_ws+perjoint@test": "basews_pjtest", "filter_ws+perjoint@test": "filtws_pjtest"}
# table columns: (summary column, macro name, format). Height (CHANGELOG #20): max torso height
# above the start height is the primary metric (every episode); mean hop apex is secondary and
# carries the number of seeds that had hops ([n]).
COLS = [("uv_s", "uv_s", ".3f"), ("oc_s", "oc_s", ".3f"), ("collapse_s", "collapse_s", ".3f"),
        ("v_min", "vmin", ".2f"), ("i_max", "imax", ".1f"), ("n_hops", "hops", ".1f"),
        ("height_max", "hmax", ".3f"), ("apex_mean", "apex", ".3f"), ("fell", "falls", ".2f"),
        ("ret", "ret", ".1f")]
def H(*lines):
    """A table header cell on several lines (keeps columns narrow)."""
    # "\\{}" so that a line starting with "[" is not read as the optional skip of "\\"
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\{}".join(lines) + "\\end{tabular}"


def tlabel(method, warm=False):
    """Method label for a table; inside the warm-start tables ", warm start" is implied by the caption."""
    s = LABEL.get(method, method)
    return s.replace(", warm start", "") if warm else s


HEAD = ("\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrrrrrrr}\\toprule\n"
        "Method & SOC & " + " & ".join([H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"),
                                        H("Collapse", "(s)"), H("$\\Vb$ min", "(V)"), H("Peak $\\Ib$", "(A)"),
                                        "Hops", H("Max", "height (m)"), H("Mean apex (m)", "[seeds]"), "Falls",
                                        "Return", "Seeds"]) + " \\\\\\midrule\n")
TABLES = {"tab_violations": lambda m: "_ws" not in m,      # main table: from-scratch runs
          "tab_warmstart": lambda m: "_ws" in m}           # warm starts from baseline_s* (CHANGELOG #19)
vio = SUMMARY / "violations.csv"
out = pd.read_csv(vio) if vio.exists() else None   # written by scripts/aggregate.py; mean/std over seeds
fmt = lambda x, f: "--" if pd.isna(x) else format(x, f)
for tab, keep in TABLES.items():
    rows, prev_m = [], None
    for _, r in (out.iterrows() if out is not None else []):
        if not keep(r.method):
            continue
        k = KEYS.get(r.method, r.method.replace("+", "_").replace("@", "").replace(".", ""))
        key = lambda name: f"{k}_{name}_soc{r.soc:g}"
        for col, name, f in COLS:
            macros[key(name)] = fmt(r[f"{col}_mean"], f)
            macros[key(name + "_std")] = fmt(r[f"{col}_std"], f)
        macros[key("uv_pct")] = fmt(100 * r.uv_frac_mean, ".2f")
        macros[key("oc_pct")] = fmt(100 * r.oc_frac_mean, ".2f")
        macros[key("zmax")] = fmt(r.z_max_mean, ".3f")
        macros[key("nseeds")] = str(int(r.n_seeds))
        macros[key("apex_nseeds")] = str(int(r.apex_n_seeds))
        cell = lambda c, f: f"{fmt(r[c + '_mean'], f)} $\\pm$ {fmt(r[c + '_std'], f)}" + (
            f" [{int(r.apex_n_seeds)}]" if c == "apex_mean" else "")
        cells = " & ".join(cell(c, f) for c, _, f in COLS)
        if prev_m is not None and r.method != prev_m:
            rows.append("\\addlinespace")
        lab = tlabel(r.method, tab == "tab_warmstart") if r.method != prev_m else ""
        prev_m = r.method
        rows.append(f"{lab} & {r.soc:g} & {cells} & {int(r.n_seeds)} \\\\")
    if rows:
        (GEN / f"{tab}.tex").write_text(HEAD + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n")
        print(f"{tab}.tex: {sum(not x.startswith('\\') for x in rows)} rows from {vio.relative_to(ROOT)}")
    else:
        (GEN / f"{tab}.tex").unlink(missing_ok=True)   # never leave a stale table
        print(f"{tab}.tex: no rows (run scripts/aggregate.py): left pending")

# ---- CHANGELOG #51: condensed main-text versions of the main tables (full ones go to the
# supplementary material). Means over seeds only (std in the full tables); key columns; the
# warm-start table keeps the methods with three seeds.
SHORT_HEAD = ("{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.21\\linewidth}lrrrrrrr}\\toprule\n"
              "Method & SOC & " + " & ".join([H("Below", "$V_{\\min}$ (s)"), H("Above", "$I_{\\lim}$ (s)"),
                                              H("Collapse", "(s)"), H("Peak", "$\\Ib$ (A)"), "Hops",
                                              H("Max", "height (m)"), "Falls"]) + " \\\\\\midrule\n")
SHORT_KEEP = {"tab_violations_short": lambda m: "_ws" not in m,
              "tab_warmstart_short": lambda m: m in ("baseline_ws", "baseline_ws+filter@test", "penalty_w0.05_ws",
                                                     "penalty_w0.1_ws", "penalty_w0.5_ws", "lagrangian_ws", "filter_ws")}
for tab, keep in SHORT_KEEP.items():
    rows, prev_m = [], None
    for _, r in (out.iterrows() if out is not None else []):
        if not keep(r.method):
            continue
        if prev_m is not None and r.method != prev_m:
            rows.append("\\addlinespace")
        lab = tlabel(r.method, "warmstart" in tab) if r.method != prev_m else ""
        prev_m = r.method
        rows.append(f"{lab} & {r.soc:g} & {fmt(r.uv_s_mean, '.3f')} & {fmt(r.oc_s_mean, '.3f')} & "
                    f"{fmt(r.collapse_s_mean, '.3f')} & {fmt(r.i_max_mean, '.1f')} & {fmt(r.n_hops_mean, '.1f')} & "
                    f"{fmt(r.height_max_mean, '.3f')} & {fmt(r.fell_mean, '.2f')} \\\\")
    if rows:
        (GEN / f"{tab}.tex").write_text(SHORT_HEAD + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n")
        print(f"{tab}.tex: {sum(not x.startswith('\\') for x in rows)} rows")

# supplementary: per-seed table (results/summary/per_seed_table.csv, written by aggregate.py)
PS_HEAD = ("\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrrr}\\toprule\n"
           "Method & SOC & Seed & Falls & Hops & " + " & ".join([
               H("Max", "height (m)"), H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"),
               H("Collapse", "(s)")]) + " \\\\\\midrule\n")


def ps_render(items, warm):
    """Per-seed rows; method and SOC printed when they change (and on a part's first row)."""
    out_rows, pm, psoc = [], None, None
    for r in items:
        if pm is not None and r.method != pm:
            out_rows.append("\\addlinespace")
        lab = tlabel(r.method, warm) if r.method != pm else ""
        soc = f"{r.soc:g}" if (r.method, r.soc) != (pm, psoc) else ""
        pm, psoc = r.method, r.soc
        out_rows.append(f"{lab} & {soc} & {int(r.seed)} & {int(r.n_fell)}/{int(r.n_episodes)} & "
                        f"{r.n_hops:.2f} & {r.height_max:.3f} & {r.uv_s:.3f} & {r.oc_s:.3f} & {r.collapse_s:.3f} \\\\")
    return out_rows
PS_TABLES = {"tab_per_seed": lambda m: "_ws" not in m, "tab_per_seed_warmstart": lambda m: "_ws" in m}
PS_PART_MAX = 40
pst = SUMMARY / "per_seed_table.csv"
ps = pd.read_csv(pst) if pst.exists() else None
for tab, keep in PS_TABLES.items():
    items = [r for r in (ps.itertuples() if ps is not None else []) if keep(r.method)]
    warm = tab == "tab_per_seed_warmstart"
    rows = items
    for old in GEN.glob(f"{tab}_part*.tex"):
        old.unlink()                       # never leave a stale part
    if rows:
        (GEN / f"{tab}.tex").write_text(PS_HEAD + "\n".join(ps_render(items, warm)) + "\n\\bottomrule\\end{tabular}}\n")
        # the same rows in parts of at most PS_PART_MAX rows, each fitting one page (appendix)
        n_parts = -(-len(rows) // PS_PART_MAX)
        size = -(-len(rows) // n_parts)
        for i in range(n_parts):
            (GEN / f"{tab}_part{i + 1}.tex").write_text(
                PS_HEAD + "\n".join(ps_render(items[i * size:(i + 1) * size], warm)) + "\n\\bottomrule\\end{tabular}}\n")
        macros[f"{tab.replace('_', '')}_nparts"] = str(n_parts)
        print(f"{tab}.tex: {len(rows)} rows from {pst.relative_to(ROOT)} ({n_parts} parts)")
    else:
        (GEN / f"{tab}.tex").unlink(missing_ok=True)
        print(f"{tab}.tex: no rows: left pending")
tf = SUMMARY / "filter_timing.json"
if tf.exists():
    import json
    t = json.loads(tf.read_text())
    macros["filt_us_mean"] = f"{t['total_us']['mean']:.0f}"
    macros["filt_us_p99"] = f"{t['total_us']['p99']:.0f}"

# ---- CHANGELOG #31: est_noise against the first-order current-barrier rule (filter.tex remark),
# A6 occurrences (discussion.tex), vmin_datasheet filter rows (results.tex) ----
sens = SUMMARY / "est_noise_sensitivity.csv"          # scripts/aggregate_ablations.py
if sens.exists():
    se = pd.read_csv(sens)
    rows = []
    for _, r in se.iterrows():
        k = f"n{r.est_noise:g}_soc{r.soc:g}"
        macros.update({f"sens_rule_{k}": f"{r.rule_1sd:.2f}", f"sens_rulee_{k}": f"{r.rule_e:.2f}",
                       f"sens_ruler_{k}": f"{r.rule_r:.2f}", f"sens_over_{k}": f"{r.over_mean:.2f}",
                       f"sens_over_std_{k}": f"{r.over_std:.2f}", f"sens_overmax_{k}": f"{r.over_max:.2f}",
                       f"sens_ratio_{k}": f"{r.ratio_max_to_rule:.2f}", f"sens_eocv_{k}": f"{r.e_ocv:.2f}"})
        rows.append(f"{r.est_noise:g} & {r.soc:g} & {r.e_ocv:.2f} & {r.rule_e:.2f} & {r.rule_r:.2f} & "
                    f"{r.rule_1sd:.2f} & {r.over_mean:.2f} $\\pm$ {r.over_std:.2f} & {r.over_max:.2f} & "
                    f"{r.ratio_max_to_rule:.2f} & {r.oc_s_mean:.4f} \\\\")
    worst = se.loc[se["ratio_max_to_rule"].idxmax()]
    macros.update({"sens_ncases": str(len(se)),
                   "sens_nwithinone": str(int((se["ratio_max_to_rule"] <= 1.0).sum())),
                   "sens_nwithintwo": str(int((se["ratio_max_to_rule"] <= 2.0).sum())),
                   "sens_ratio_max": f"{worst.ratio_max_to_rule:.2f}", "sens_worst_sigma": f"{worst.est_noise:g}",
                   "sens_worst_soc": f"{worst.soc:g}", "sens_ratio_min": f"{se['ratio_max_to_rule'].min():.2f}",
                   "sens_itarget": f"{se['i_target'].iloc[0]:g}", "sens_r0": f"{se['r0'].iloc[0]:.3f}",
                   "sens_nruleabovemargin": str(int((se["rule_1sd"] > se["margin_i"]).sum())),
                   # smallest multiple k of the noise scale for which the rule k sigma (E/R0 + I_lim - delta_I)
                   # bounds every largest observed overshoot; rounded up, so it remains a bound
                   "sens_kall": f"{np.ceil(100 * se['ratio_max_to_rule'].max()) / 100:.2f}",
                   "sens_noc": str(int((se["oc_s_mean"] > 0).sum()))})
    (GEN / "tab_est_noise_rule.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{rrrrrrrrrr}\\toprule\n"
        "$\\sigma$ & SOC & $E$ (V) & $\\sigma E/R_0$ (A) & $\\sigma (I_{\\lim}-\\delta_I)$ (A) & Rule, $\\kappa{=}1$ (A) & "
        "Overshoot (A) & Largest (A) & $\\kappa$ needed & $t(\\Ib>I_{\\lim})$ (s) \\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_est_noise_rule.tex: {len(rows)} rows from {sens.relative_to(ROOT)}")
# A6: substeps with an empty voltage-circle / current-box intersection, main evaluations
# (results/<run>/eval.csv), by test mode; fraction of all substeps simulated (steps x substeps)
a6f = [pd.read_csv(f) for f in sorted((ROOT / "results").glob("*_s[0-9]/eval.csv"))]
a6f = [d for d in a6f if "a6_empty_n" in d]
if a6f:
    a6 = pd.concat(a6f, ignore_index=True)
    from busaware.leg_env import HoppingLegEnv as _E
    n_sub = _E().substeps
    for tm, key in (("baseline", "base"), ("penalty", "pen"), ("filter", "filt"), ("lagrangian", "lag")):
        g = a6[a6["test_mode"] == tm]
        if len(g):
            macros[f"asix_{key}_sub"] = str(int(g["a6_empty_n"].sum()))
            macros[f"asix_{key}_ep"] = str(len(g))
            macros[f"asix_{key}_ppm"] = f"{1e6 * g['a6_empty_n'].sum() / (g['steps'].sum() * n_sub):.0f}"
abl = SUMMARY / "ablations.csv"
if abl.exists():
    ab = pd.read_csv(abl)
    var = ab[(ab["ablation"] == "variant") & (ab["method"] == "baseline_ws")]
    macros["asix_var_sub"] = str(int(var["a6_empty_n"].sum()))
    fv = ab[(ab["ablation"] == "variant") & (ab["method"] != "baseline_ws")]
    macros["asix_varfilt_sub"] = str(int(fv["a6_empty_n"].sum()))
    # vmin_datasheet against base, filter-mode rows (mean over seeds per method and SOC)
    vm = ab[(ab["setting"] == "vmin_datasheet") & (ab["method"] != "baseline_ws")].set_index(["method", "soc"])
    bs = ab[(ab["setting"] == "base") & (ab["method"] != "baseline_ws")].set_index(["method", "soc"]).loc[vm.index]
    dmax = lambda c: float((vm[c] - bs[c]).abs().max())
    macros.update({"vmin_dfalls": str(int(dmax("n_fell"))), "vmin_dhops": f"{dmax('n_hops_mean'):.2f}",
                   "vmin_dhmax": f"{dmax('height_max_mean'):.4f}",
                   "vmin_dviol": f"{max(dmax(c) for c in ('uv_s_mean', 'oc_s_mean', 'ov_s_mean', 'collapse_s_mean')):g}"})
vid = SUMMARY / "ablation_vmin_datasheet_identity.csv"
if vid.exists():
    vi = pd.read_csv(vid)
    macros.update({"vmin_ident_evals": str(int(vi["identical"].sum())), "vmin_evals": str(len(vi)),
                   "vmin_ident_ep": str(int(vi["n_identical"].sum())), "vmin_ep": str(int(vi["n_episodes"].sum())),
                   "vmin_dh_ep_max": f"{vi['max_abs_dheight'].max():.3f}"})
    vlab = {"baseline_ws+filter@test": "Baseline + filter at test", "filter_ws": "Filter-trained"}
    (GEN / "tab_vmin_identity.tex").write_text(
        "\\begin{tabular}{lrrr}\\toprule\nMethod & Seed & " + H("Identical", "episodes") + " & "
        + H("Largest $|\\Delta|$", "max height (m)") + " \\\\\\midrule\n"
        + "\n".join(f"{vlab.get(r.method, r.method)} & {int(r.seed)} & {int(r.n_identical)}/{int(r.n_episodes)} & "
                    f"{r.max_abs_dheight:.4f} \\\\" for r in vi.itertuples())
        + "\n\\bottomrule\\end{tabular}\n")
    print(f"tab_vmin_identity.tex: {len(vi)} rows from {vid.relative_to(ROOT)}")
# A6 timing check: scripts/rollout_plot.py on penalty_w0.05_ws_s1 (the main-table run with the most
# A6 substeps), reset seed 1000 (= evaluation episode 0), SOC 1.0, 0.5, 0.2
chk = ROOT / "results" / "penalty_w0.05_ws_s1" / "rollout" / "summary_penalty.json"
if chk.exists():
    import json
    c = json.loads(chk.read_text())
    if c.get("seed") == 1000 and all("a6_after_collapse_substeps" in v for v in c["socs"].values()):
        tot = lambda k: str(sum(v[k] for v in c["socs"].values()))
        macros.update({"asixchk_ep": str(len(c["socs"])), "asixchk_sub": tot("a6_substeps"),
                       "asixchk_same": tot("a6_collapse_substeps"), "asixchk_after": tot("a6_after_collapse_substeps"),
                       "asixchk_uv": tot("a6_below_vmin_substeps")})

# number of automated tests (setup.tex): collected by pytest from tests/ (collection only; the
# suite itself is run separately, CLAUDE.md rule 7)
import subprocess
_col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT,
                      capture_output=True, text=True)
_m = re.search(r"(\d+) tests? collected", _col.stdout)
if _col.returncode == 0 and _m:
    macros["n_tests"] = _m.group(1)
else:
    print("n_tests: pytest collection failed; left pending")

# setup values used in the prose (model.tex, filter.tex): from params.py, the environment's
# defaults (an instance, so the MJCF timestep is the real one) and the validation results
from busaware.leg_env import HoppingLegEnv
_p, _env = Params(), HoppingLegEnv()
macros.update({
    "par_vmin": f"{_p.v_min:g}", "par_vmax": f"{_p.v_max:g}", "par_ibatmax": f"{_p.i_bat_max:g}",
    "par_ichg": f"{_p.i_chg_max:g}", "par_cbus_mF": f"{1e3 * _p.c_bus:.3f}",
    "par_vchopoff": f"{_p.v_chop_off:g}", "par_vchop": f"{_p.v_chop:g}", "par_rchop": f"{_p.r_chop:g}",
    "par_paux": f"{_p.p_aux:g}", "par_pidle": f"{_p.p_idle:g}", "par_gear": f"{_p.gear:g}", "par_ilim": f"{_p.i_lim:.1f}",
    "par_vmin_ds": f"{Params.variant('vmin_datasheet').v_min:g}",
    "par_imax": f"{_p.i_max:g}", "par_taumax": f"{_p.tau_max_out:g}", "par_nseries": f"{_p.n_series:d}",
    "env_dtsim_ms": f"{1e3 * _env.dt_sim:g}", "env_substeps": f"{_env.substeps:d}",
    "env_dtctrl_ms": f"{1e3 * _env.dt_sim * _env.substeps:g}",
    "env_episode_s": f"{_env.max_steps * _env.dt_sim * _env.substeps:g}",
    "env_gamma": f"{_env.gamma:g}", "env_marginv": f"{_env.margin_v:g}", "env_margini": f"{_env.margin_i:g}",
})
vf = ROOT / "results" / "validation" / "metrics.json"
if vf.exists():
    import json
    v = json.loads(vf.read_text())
    sw = v["switching"]
    macros.update({"ref_dt_us": f"{1e6 * sw['dt_s']:g}", "ref_fpwm_khz": f"{1e-3 * sw['f_pwm_Hz']:g}",
                   "ref_deadtime_ns": f"{1e9 * sw['dead_time_s']:g}", "ref_ibw_hz": f"{sw['current_bw_Hz']:g}",
                   "val_dtavg_ms": f"{1e3 * v['averaged_dt_s']:g}", "val_tend_ms": f"{1e3 * v['t_end_s']:g}",
                   "val_tstep_ms": f"{1e3 * v['t_step_s']:g}", "val_tsteady_ms": f"{1e3 * v['t_steady_from_s']:g}",
                   "val_soc": f"{v['soc']:g}"})
    sc = {"S1_standstill_step": "sone", "S2_high_speed_step": "stwo", "S3_braking": "sthree"}
    sig = {"v_bus": "vbus", "i_bat": "ibat", "p_elec": "pelec"}
    steady = []
    for sname, sk in sc.items():
        mm = v["scenarios"][sname]["metrics"]
        for s, k in sig.items():
            macros[f"val_{sk}_{k}_trpk_pct"] = f"{mm[s + '_transient']['peak_pct']:.1f}"
            macros[f"val_{sk}_{k}_stpk_pct"] = f"{mm[s + '_steady']['peak_pct']:.2f}"
            steady.append(mm[s + "_steady"]["peak_pct"])
    macros["val_steady_maxpk_pct"] = f"{max(steady):.2f}"

# ---- CHANGELOG #34: setup values (setup.tex) from the code, the run configs and the saved models;
# ablation, estimator and Lagrangian tables; per-row and per-seed macros for the results text ----
import json, zipfile
import busaware.leg_env as LE
import busaware.ekf as EKFM
from busaware.rollout import MIN_FLIGHT_S
from busaware.runs import run_config, DEFAULT_PENALTY_W
from busaware.params import R0_CELLS_25C

RESULTS = ROOT / "results"
num = lambda x: f"{x:g}"
thin = lambda n: f"{int(n):,}".replace(",", "\\,")          # 200000 -> 200\,000


def join_and(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def only(values, what):
    """The single value of a collection that must not vary (warns and takes the first otherwise)."""
    vs = sorted(set(values))
    if len(vs) != 1:
        print(f"WARNING: {what} differs: {vs}")
    return vs[0]


# environment and task: an instance gives the defaults that every training run used (train.py
# passes only mode, penalty_w, w_energy and seed)
macros.update({
    "env_kp": num(_env.kp), "env_kd": num(_env.kd), "env_actscale": num(_env.act_scale),
    "env_qstand_hip": num(LE.Q_STAND[0]), "env_qstand_knee": num(LE.Q_STAND[1]),
    "env_resetnoise": num(LE.RESET_JOINT_NOISE), "env_soclo": num(_env.soc_range[0]),
    "env_sochi": num(_env.soc_range[1]), "env_obsdim": str(_env.observation_space.shape[0]),
    "env_actdim": str(_env.action_space.shape[0]), "env_maxsteps": thin(_env.max_steps),
    "env_fall_m": num(-LE.FALL_HEIGHT), "rew_height_w": num(LE.REWARD_HEIGHT_W),
    "rew_height_min": num(LE.REWARD_HEIGHT_MIN), "rew_alive": num(LE.REWARD_ALIVE),
    "rew_smooth_w": num(LE.REWARD_SMOOTH_W), "rew_wfilter": num(_env.w_filter),
    "obs_qvel_scale": num(LE.OBS_QVEL_SCALE), "obs_v_scale": num(LE.OBS_V_SCALE),
    "par_vobsref": num(_p.v_obs_ref), "hop_minflight_ms": num(1e3 * MIN_FLIGHT_S),
    "pen_w_scratch": num(DEFAULT_PENALTY_W), "env_marginvmax": num(_env.margin_v_max),
    "env_marginimax": num(_env.margin_i_max),
    "env_marginratio": num(only([_env.margin_v_max / _env.margin_v, _env.margin_i_max / _env.margin_i],
                                "largest-to-fixed margin ratio")),
})
# training: run configs (busaware/runs.run_config) and the hyperparameters stored in each model.zip
runs = sorted(d for d in RESULTS.glob("*_s[0-9]") if (d / "model.zip").exists())
if runs:
    cfgs = {d.name: run_config(d) for d in runs}
    zd = {d.name: json.loads(zipfile.ZipFile(d / "model.zip").read("data")) for d in runs}
    SAC_KEYS = ("learning_rate", "gamma", "tau", "buffer_size", "batch_size", "learning_starts",
                "ent_coef", "target_entropy", "gradient_steps", "target_update_interval")
    hp = {k: only([str(z.get(k)) for z in zd.values()], f"SAC {k}") for k in SAC_KEYS}
    arch = only([tuple(z["policy_kwargs"]["net_arch"]) for z in zd.values()], "net_arch")
    scratch = [n for n, c in cfgs.items() if not c.get("init")]
    warm = [n for n, c in cfgs.items() if c.get("init")]
    macros.update({
        "sac_lr": hp["learning_rate"], "sac_discount": hp["gamma"], "sac_tau": hp["tau"],
        "sac_buffer": thin(float(hp["buffer_size"])), "sac_batch": hp["batch_size"],
        "sac_learnstarts": thin(float(hp["learning_starts"])), "sac_entcoef": hp["ent_coef"],
        "sac_targetent": hp["target_entropy"], "sac_gradsteps": hp["gradient_steps"],
        "sac_targetint": hp["target_update_interval"], "sac_nlayers": str(len(arch)),
        "sac_width": str(only(arch, "hidden layer width")),
        "train_steps_scratch": thin(only([zd[n]["num_timesteps"] for n in scratch], "from-scratch steps")),
        "train_steps_ws": thin(only([zd[n]["num_timesteps"] for n in warm], "warm-start steps")),
        "sb3_version": only([c["sb3"] for c in cfgs.values() if c.get("sb3")], "SB3 version"),
        "rew_wenergy": num(only([float(c["w_energy"]) for c in cfgs.values()], "w_energy")),
        "ws_init_src": only([re.sub(r"_s\d+$", "", c["init"]) for c in cfgs.values() if c.get("init")],
                            "warm-start source"),
    })
    seeds_of = {}
    for n, c in cfgs.items():
        seeds_of.setdefault(re.sub(r"_s\d+$", "", n), set()).add(int(c["seed"]))
    all_seeds = sorted(set().union(*seeds_of.values()))
    macros.update({"seed_list": join_and([str(s) for s in all_seeds]), "n_seeds": str(len(all_seeds))})
    pw = {}
    for n, c in cfgs.items():
        if c["mode"] == "penalty" and c.get("init"):
            pw.setdefault(float(c["penalty_w"]), set()).add(int(c["seed"]))
    full = sorted(w for w, s in pw.items() if len(s) == len(all_seeds))
    part = sorted(w for w, s in pw.items() if len(s) < len(all_seeds))
    macros.update({"pen_ws_w_full": join_and([num(w) for w in full]),
                   "pen_ws_w_part": join_and([num(w) for w in part]),
                   "pen_ws_part_seed": join_and(sorted({str(s) for w in part for s in pw[w]})),
                   **{f"pen_ws_w{chr(ord('A') + i)}": num(w) for i, w in enumerate(sorted(pw))}})
    lagc = [c["lagrangian"] for c in cfgs.values() if c.get("lagrangian")]
    if lagc:
        macros.update({"lag_lr": num(only([l["lag_lr"] for l in lagc], "lag_lr")),
                       "lag_d": num(only([l["cost_limit"] for l in lagc], "cost limit")),
                       "lag_init": num(only([l["lag_init"] for l in lagc], "lag_init"))})
# evaluation protocol: episodes per seed and SOC, reset seeds, SOC levels
if out is not None:
    macros["eval_eps_main"] = str(int(only((out.n_episodes / out.n_seeds).tolist(), "episodes per seed")))
    macros["eval_socs"] = join_and([f"{s:.1f}" for s in sorted(out.soc.unique(), reverse=True)])
    e0 = pd.read_csv(RESULTS / "filter_ws_s0" / "eval.csv")
    macros["eval_seed_first"] = str(int(e0.reset_seed.min()))
    macros["eval_seed_last"] = str(int(e0.reset_seed.max()))
    # falls as counts (main evaluations), per method and SOC, from the per-seed table
    if ps is not None:
        for (m, soc), g in ps.groupby(["method", "soc"]):
            k = KEYS.get(m, m.replace("+", "_").replace("@", "").replace(".", ""))
            macros[f"{k}_fallsn_soc{soc:g}"] = str(int(g.n_fell.sum()))
            macros[f"{k}_neps_soc{soc:g}"] = str(int(g.n_episodes.sum()))
        for _, r in ps.iterrows():
            k = KEYS.get(r.method, r.method.replace("+", "_").replace("@", "").replace(".", ""))
            b = f"ps_{k}_s{int(r.seed)}_soc{r.soc:g}"
            macros.update({b + "_fell": str(int(r.n_fell)), b + "_neps": str(int(r.n_episodes)),
                           b + "_hops": f"{r.n_hops:.1f}", b + "_hmax": f"{r.height_max:.3f}",
                           b + "_uv": f"{r.uv_s:.3f}", b + "_oc": f"{r.oc_s:.3f}"})

# SAC-Lagrangian training: multiplier and episode cost from results/lagrangian_ws_s*/lagrangian.csv
LAG_LAST = 20                      # training episodes averaged at the end of training
lag_rows, lag_vals = [], []
for f in sorted(RESULTS.glob("lagrangian_ws_s[0-9]/lagrangian.csv")):
    s = int(re.search(r"_s(\d+)", f.parent.name).group(1))
    d = pd.read_csv(f)
    last = d.tail(LAG_LAST)
    v = dict(seed=s, nep=len(d), lamfin=d.lam_new.iloc[-1], lammax=d.lam_new.max(),
             cost=last.episode_cost.mean(), length=last.episode_len.mean())
    lag_vals.append(v)
    macros.update({f"lag_s{s}_nep": thin(v["nep"]), f"lag_s{s}_lamfin": f"{v['lamfin']:.2f}",
                   f"lag_s{s}_cost": f"{v['cost']:.3f}", f"lag_s{s}_len": f"{v['length']:.0f}"})
    lag_rows.append(f"{s} & {v['nep']} & {v['lamfin']:.2f} & {v['lammax']:.2f} & {v['cost']:.3f} & "
                    f"{v['length']:.0f} \\\\")
if lag_rows:
    lv = pd.DataFrame(lag_vals)
    macros.update({"lag_last": str(LAG_LAST), "lag_nseeds": str(len(lv)),
                   "lag_nfinalmax": str(int((lv.lamfin >= lv.lammax).sum())),
                   "lag_ncostabove": str(int((lv.cost > float(macros.get("lag_d", "nan"))).sum())),
                   "lag_costmin": f"{lv.cost.min():.3f}", "lag_costmax": f"{lv.cost.max():.3f}"})
    (GEN / "tab_lagrangian.tex").write_text(
        "\\begin{tabular}{rrrrrr}\\toprule\n"
        "Seed & Episodes & Final $\\mu$ & Largest $\\mu$ & Cost $\\bar\\ell$ & Length (steps) \\\\\\midrule\n" + "\n".join(lag_rows) + "\n\\bottomrule\\end{tabular}\n")
    print(f"tab_lagrangian.tex: {len(lag_rows)} rows from results/lagrangian_ws_s*/lagrangian.csv")

# ablations (scripts/aggregate_ablations.py): stress-test table and estimator table
if abl.exists():
    ab = pd.read_csv(abl)
    # CHANGELOG #57: warm-started penalty policies with the filter at test time under the variants and
    # barrier rates (scripts/aggregate_ablations.py -> ablations_penalty.csv, same metrics and episodes)
    _abp_f = SUMMARY / "ablations_penalty.csv"
    abp = pd.read_csv(_abp_f) if _abp_f.exists() else ab.iloc[:0]
    PEN_M = only(abp.method, "penalty stress-test method") if len(abp) else None
    pen_w = float(re.fullmatch(r"penalty_w([0-9.]+)_ws\+filter@test", PEN_M).group(1)) if PEN_M else None
    if PEN_M:
        only((pd.concat([ab, abp]).n_episodes / pd.concat([ab, abp]).n_seeds).tolist(), "setting episodes (with penalty)")
    macros["eval_eps_set"] = str(int(only((ab.n_episodes / ab.n_seeds).tolist(), "setting episodes")))
    macros["eval_seed_set_last"] = str(int(macros["eval_seed_first"]) + int(macros["eval_eps_set"]) - 1)
    skey = lambda s: re.sub(r"[^A-Za-z0-9.]", "", s.replace("r0_scale=", "r").replace("est_noise=", "n")
                            .replace("gamma=", "g"))
    for _, r in pd.concat([ab, abp], ignore_index=True).iterrows():
        mk = KEYS.get(r.method, r.method)
        b = f"abl_{skey(r.setting)}_{mk}_soc{r.soc:g}"
        macros.update({b + "_uv": f"{r.uv_s_mean:.3f}", b + "_oc": f"{r.oc_s_mean:.3f}",
                       b + "_col": f"{r.collapse_s_mean:.3f}", b + "_falls": str(int(r.n_fell)),
                       b + "_neps": str(int(r.n_episodes)), b + "_hops": f"{r.n_hops_mean:.1f}",
                       b + "_hmax": f"{r.height_max_mean:.3f}", b + "_filt": f"{100 * r.filt_frac_mean:.1f}",
                       b + "_infeas": f"{r.infeas_s_mean:.3f}", b + "_imax": f"{r.i_max_mean:.2f}",
                       b + "_imaxpk": f"{r.i_max_peak:.2f}"})
    # stress-test table: parameter variants, barrier rate and (true-state) model error
    v0 = {name: Params.variant(name) for name in Params.VARIANTS}
    wiring = lambda q: f"{1e3 * (q.r0 - R0_CELLS_25C) / 4:.0f}"     # per pack, mOhm
    macros.update({"var_wiring_base": wiring(_p), "var_wiring_lo": wiring(v0["wiring_0mohm"]),
                   "var_wiring_hi": wiring(v0["wiring_20mohm"]), "var_paux_lo": num(v0["paux_0w"].p_aux),
                   "var_paux_hi": num(v0["paux_10w"].p_aux),
                   "var_rphase_lo": num(v0["rphase_x0.5"].r_phase / _p.r_phase),
                   "var_rphase_hi": num(v0["rphase_x1.5"].r_phase / _p.r_phase),
                   "var_lag_hz": num(v0["current_lag"].i_loop_bw), "var_geareff": num(v0["gear_eff_0.9"].gear_eff),
                   "var_cold_r0": f"{Params.cold().r0:.3f}", "par_r0": f"{_p.r0:.3f}",
                   "var_cold_ichg": num(Params.cold().i_chg_max),
                   # constant-load change of the P_aux variants against the drives' total idle power
                   "var_paux_delta": num(only([abs(v0["paux_0w"].p_aux - _p.p_aux),
                                               abs(v0["paux_10w"].p_aux - _p.p_aux)], "P_aux change")),
                   "par_pidle_total": num(_env.action_space.shape[0] * _p.p_idle)})
    SLABEL = {"base": "Base", "cold": "Cold battery (0\\,$^\\circ$C)",
              "vmin_datasheet": f"$V_{{\\min}}={v0['vmin_datasheet'].v_min:g}$~V",
              "wiring_0mohm": f"Wiring {wiring(v0['wiring_0mohm'])}~m$\\Omega$",
              "wiring_20mohm": f"Wiring {wiring(v0['wiring_20mohm'])}~m$\\Omega$",
              "paux_0w": f"$P_{{\\mathrm{{aux}}}}={v0['paux_0w'].p_aux:g}$~W",
              "paux_10w": f"$P_{{\\mathrm{{aux}}}}={v0['paux_10w'].p_aux:g}$~W",
              "rphase_x0.5": f"$R\\times{v0['rphase_x0.5'].r_phase / _p.r_phase:g}$",
              "rphase_x1.5": f"$R\\times{v0['rphase_x1.5'].r_phase / _p.r_phase:g}$",
              "current_lag": f"Current lag {v0['current_lag'].i_loop_bw:g}~Hz",
              "gear_eff_0.9": f"Gear efficiency {v0['gear_eff_0.9'].gear_eff:g}"}

    def slabel(s):
        if s in SLABEL:
            return SLABEL[s]
        k, val = s.split("=")
        return {"gamma": f"$\\gamma={float(val):g}$", "est_noise": f"Noise $\\sigma={float(val):g}$",
                "r0_scale": f"Plant $R_0\\times{float(val):g}$"}[k]
    MLAB = {"baseline_ws": "Baseline", "baseline_ws+filter@test": "Baseline + filter at test",
            "filter_ws": "Filter-trained"}
    if PEN_M:
        MLAB[PEN_M] = f"Penalty $w={pen_w:g}$ + filter at test"
    st = ab[ab.ablation.isin(["base", "variant", "gamma", "model_error"])]
    # full table: the penalty rows inserted per setting, before the filter-trained row
    _so = {k: i for i, k in enumerate(dict.fromkeys(zip(st.ablation, st.setting)))}
    _mo = {m: i for i, m in enumerate(["baseline_ws", "baseline_ws+filter@test", PEN_M, "filter_ws"])}
    st_full = pd.concat([st, abp[abp.ablation.isin(["base", "variant", "gamma"])]], ignore_index=True)
    st_full = (st_full.assign(_a=[_so[k] for k in zip(st_full.ablation, st_full.setting)],
                              _m=st_full.method.map(_mo), _s=-st_full.soc)
               .sort_values(["_a", "_m", "_s"], kind="stable").drop(columns=["_a", "_m", "_s"]))
    st_rows, agg = [], {}
    for (a_, s, m), g in st_full.groupby(["ablation", "setting", "method"], sort=False):
        g = g.sort_values("soc", ascending=False)
        h = " / ".join(f"{x:.3f}" for x in g.height_max_mean)
        v = dict(uv=g.uv_s_mean.max(), oc=g.oc_s_mean.max(), col=g.collapse_s_mean.max(),
                 falls=int(g.n_fell.sum()), neps=int(g.n_episodes.sum()), hops=g.n_hops_mean.mean(),
                 filt=100 * g.filt_frac_mean.mean(), inf=g.infeas_s_mean.max())
        agg[(a_, s, m)] = v
        mk = KEYS.get(m, m)
        b = f"abl_{skey(s)}_{mk}"
        macros.update({b + "_uvmax": f"{v['uv']:.3f}", b + "_ocmax": f"{v['oc']:.3f}",
                       b + "_colmax": f"{v['col']:.3f}", b + "_falls": str(v["falls"]),
                       b + "_neps": str(v["neps"]), b + "_hops": f"{v['hops']:.1f}",
                       b + "_filt": f"{v['filt']:.1f}", b + "_infmax": f"{v['inf']:.3f}"})
        filt_cells = "--" if m == "baseline_ws" else f"{v['filt']:.1f}"
        st_rows.append(f"{MLAB.get(m, m)} & {v['uv']:.3f} & {v['oc']:.3f} & {v['col']:.3f} & "
                       f"{v['falls']}/{v['neps']} & {v['hops']:.1f} & {h} & {filt_cells} & {v['inf']:.3f} \\\\")
    # setting label once per setting; rule between groups, space between settings
    rows_out, prev = [], None
    for (a_, s, m), row in zip(agg.keys(), st_rows):
        if prev is not None and a_ != prev[0]:
            rows_out.append("\\midrule")
        elif prev is not None and s != prev[1]:
            rows_out.append("\\addlinespace")
        rows_out.append(("" if prev is not None and s == prev[1] else slabel(s)) + " & " + row)
        prev = (a_, s)
    (GEN / "tab_ablation.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrcrr}\\toprule\n"
        "Setting & Method & " + " & ".join([
            H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"), H("Collapse", "(s)"), "Falls", "Hops",
            H("Max height (m)", "SOC 1.0 / 0.5 / 0.2"), H("Filter", "active (\\%)"), H("Infeasible", "(s)")])
        + " \\\\\\midrule\n" + "\n".join(rows_out) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_ablation.tex: {len(st_rows)} rows from {abl.relative_to(ROOT)}")
    # group-level counts used in the text
    fm = {k: v for k, v in agg.items() if k[0] in ("variant", "gamma") and k[2] != "baseline_ws"}
    viol = lambda v: v["uv"] > 0 or v["oc"] > 0 or v["col"] > 0
    fws = [v for k, v in fm.items() if k[2] == "filter_ws"]
    bft = {k[1]: v for k, v in agg.items() if k[0] == "variant" and k[2] == "baseline_ws+filter@test"}
    bws = {k[1]: v for k, v in agg.items() if k[0] == "variant" and k[2] == "baseline_ws"}
    macros.update({"abl_fm_nrows": str(len(fm)), "abl_fm_nviol": str(sum(viol(v) for v in fm.values())),
                   "abl_fws_falls": str(sum(v["falls"] for v in fws)), "abl_fws_neps": str(sum(v["neps"] for v in fws)),
                   "abl_nvar": str(len(bws)), "abl_bws_nviol": str(sum(viol(v) for v in bws.values())),
                   "abl_bft_falls_min": str(min(v["falls"] for v in bft.values())),
                   "abl_bft_falls_max": str(max(v["falls"] for v in bft.values())),
                   "abl_bft_neps": str(only([v["neps"] for v in bft.values()], "variant episodes")),
                   "abl_fws_infmax": f"{max(v['inf'] for v in fws):.3f}"})
    # CHANGELOG #61: the filter-mode rows as settings x policy groups (Section 7.6 wording)
    _fm_set = sorted({k[1] for k in fm}); _fm_grp = sorted({k[2] for k in fm})
    assert len(fm) == len(_fm_set) * len(_fm_grp), "filter-mode rows are not a full settings x groups grid"
    macros.update({"abl_fm_nset": str(len(_fm_set)), "abl_fm_ngrp": str(len(_fm_grp))})
    print(f"filter-mode rows: {len(_fm_set)} settings x {len(_fm_grp)} groups {_fm_grp}")
    # model error, true-state filter (filter_ws): settings with any violation
    me = {k[1]: v for k, v in agg.items() if k[0] == "model_error"}
    macros["abl_me_nviol"] = str(sum(viol(v) for v in me.values()))
    macros["abl_me_n"] = str(len(me))
    # ---- CHANGELOG #57: penalty policies with the filter at test time against the filter-trained
    # policies over the variants and barrier rates (Sections 7.2 and 8, abstract, highlights,
    # conclusion). Rows: (setting, SOC), means over seeds; falls: counts over seeds and SOCs.
    if PEN_M:
        VG = ("variant", "gamma")
        _pv = abp[abp.ablation.isin(VG)].set_index(["setting", "soc"]).sort_index()
        _fv = ab[ab.ablation.isin(VG) & (ab.method == "filter_ws")].set_index(["setting", "soc"]).loc[_pv.index]
        _vc = ["uv_s_mean", "oc_s_mean", "ov_s_mean", "collapse_s_mean"]
        _dh = _pv.height_max_mean - _fv.height_max_mean
        _fa_p = 100 * _pv.groupby(level="setting").filt_frac_mean.mean()      # per setting, mean over SOC
        _fa_f = 100 * _fv.groupby(level="setting").filt_frac_mean.mean()
        # where the falls are: per-episode rows of the penalty stress tests
        import importlib.util as _ilu7
        _sp7 = _ilu7.spec_from_file_location("aggabl_pen", ROOT / "scripts" / "aggregate_ablations.py")
        _ag7 = _ilu7.module_from_spec(_sp7); _sp7.loader.exec_module(_ag7)
        _pe = _ag7.load_extra()[0]
        _pf = _pe[_pe.ablation.isin(VG) & (_pe.fell > 0)]
        _per = _pf.groupby(["setting", "seed", "soc"]).size()
        _fall_sets = sorted(_pf.setting.unique())
        macros.update({
            "pst_w": num(pen_w), "pst_nset": str(_pv.index.get_level_values("setting").nunique()),
            "pst_nrows": str(len(_pv)), "pst_nviol": str(int((_pv[_vc] > 0).any(axis=1).sum())),
            "pst_falls": str(int(_pv.n_fell.sum())), "pst_neps": str(int(_pv.n_episodes.sum())),
            "pst_nfallset": str(len(_fall_sets)),
            "pst_fall_seed": str(int(only(_pf.seed, "seed of the penalty stress-test falls"))),
            "pst_fall_soc": num(only(_pf.soc, "SOC of the penalty stress-test falls")),
            "pst_fall_k": str(int(only(_per.values, "falls per setting"))),
            "pst_nhigher": str(int((_dh > 0).sum())), "pst_dhmax": f"{_dh.max():.3f}",
            "pst_filt_min": f"{_fa_p.min():.1f}", "pst_filt_max": f"{_fa_p.max():.1f}",
            "pst_fws_filt_min": f"{_fa_f.min():.1f}", "pst_fws_filt_max": f"{_fa_f.max():.1f}",
            "pst_nfiltmore": str(int((_fa_p > _fa_f).sum())),
        })
        for _soc in sorted(_pv.index.get_level_values("soc").unique()):
            _hp = _pv.xs(_soc, level="soc").height_max_mean
            _hf = _fv.xs(_soc, level="soc").height_max_mean
            macros.update({f"pst_h_soc{_soc:g}_min": f"{_hp.min():.3f}", f"pst_h_soc{_soc:g}_max": f"{_hp.max():.3f}",
                           f"pst_fws_h_soc{_soc:g}_min": f"{_hf.min():.3f}", f"pst_fws_h_soc{_soc:g}_max": f"{_hf.max():.3f}"})
        # (j) statements of the text, checked: no violation or collapse in any penalty row; the
        #     filter-trained policies fell in none of the same episodes; the penalty falls are one seed
        #     at one SOC, in the cold, phase-resistance x1.5 and gear-efficiency settings, the same
        #     count in each; the penalty mean max height is below filter_ws exactly in the rows with falls;
        #     the main evaluation of the penalty policies with the filter at test has no violation or fall.
        _lo = set(_dh[_dh <= 0].index)
        _fr = set(_pv[_pv.n_fell > 0].index)
        macros["pst_nlower"] = str(len(_lo))
        _ps7 = pd.read_csv(SUMMARY / "per_seed.csv")
        _pm6 = _ps7[_ps7.method == PEN_M]
        # Section 7.2: w is, of the penalty weights with every seed, the one whose mean max height with
        # the filter at test is largest at every SOC (main evaluation)
        _nsd = _ps7.groupby("method").seed.nunique()
        _pw7 = _ps7[_ps7.method.str.fullmatch(r"penalty_w[0-9.]+_ws\+filter@test")
                    & _ps7.method.map(_nsd).eq(_nsd.max())]
        _best7 = {s_: x_.groupby("method").height_max.mean().idxmax() for s_, x_ in _pw7.groupby("soc")}
        _jc = {
            "no penalty row with a violation or collapse": macros["pst_nviol"] == "0",
            "filter-trained: no fall in the same settings": macros["abl_fws_falls"] == "0",
            "penalty falls: one seed, one SOC, cold / R x1.5 / gear eff": (
                _pf.seed.nunique() == 1 and _pf.soc.nunique() == 1 and _per.nunique() == 1
                and _fall_sets == sorted(["cold", "rphase_x1.5", "gear_eff_0.9"])),
            "penalty lower than filter_ws exactly where it fell": _lo == _fr and len(_fr) > 0,
            "penalty + test filter, main evaluation: no violation or fall": (
                not _pm6.empty and (_pm6.uv_s + _pm6.oc_s + _pm6.collapse_s).sum() == 0 and _pm6.n_fell.sum() == 0),
            "weight equals pen_ws_wD": macros.get("pen_ws_wD") == num(pen_w),
            "w: highest all-seed penalty weight with the filter at test, every SOC": (
                len(_best7) == 3 and set(_best7.values()) == {PEN_M}),
            "rows with falls = settings with falls (one SOC)": len(_fr) == len(_fall_sets),
        }
        for _k, _v in _jc.items():
            if not _v:
                print("   claim (j) fails:", _k)
        print("claim (j) (penalty w with the filter at test, stress tests, against filter-trained):",
              "OK" if all(_jc.values()) else "FAILS")
    # evaluated setting values, from the setting names in the summary
    vals = lambda pre: sorted(float(s.split("=")[1]) for s in st.setting.unique() if s.startswith(pre + "="))
    macros.update({"set_gammas": join_and([num(g) for g in vals("gamma")]),
                   "set_r0s": join_and([num(r) for r in vals("r0_scale")]),
                   "set_noises": join_and([num(n) for n in vals("est_noise")]),
                   "set_rsmall": num(min(vals("r0_scale"))), "set_rbig": num(max(vals("r0_scale"))),
                   "set_rmid": num(vals("r0_scale")[1]), "set_noise_lo": num(min(vals("est_noise"))),
                   "set_noise_hi": num(max(vals("est_noise"))),
                   **{f"set_g{i + 1}": num(g) for i, g in enumerate(vals("gamma"))},
                   "set_ntotal": str(st.setting.nunique() - 1)})

    # ---- CHANGELOG #40: joint-level limiter (controller group) and margin ablation (margin group),
    # warm-started policies, against the base rows of the same episodes (episodes 0..eval_eps_set-1)
    mv0, mi0 = _env.margin_v, _env.margin_i
    PM_METHODS = {"base": ["baseline_ws+filter@test", "filter_ws"],
                  "controller": ["baseline_ws+perjoint@test", "filter_ws+perjoint@test"],
                  "margin": ["filter_ws"]}
    PM_MLAB = {"baseline_ws+filter@test": "Baseline + filter at test", "filter_ws": "Filter-trained",
               "baseline_ws+perjoint@test": "Baseline + per-joint limit at test",
               "filter_ws+perjoint@test": "Filter-trained, per-joint limit at test"}

    def pm_slabel(a_, s):
        if a_ == "base":
            return f"Joint filter, $\\delta_V={mv0:g}$~V, $\\delta_I={mi0:g}$~A"
        if a_ == "controller":
            return "Per-joint limit (same margins)"
        parts = dict(p.split("=") for p in s.split(", "))
        dv = float(parts.get("delta_V", mv0))
        di = float(parts.get("delta_I", mi0))
        return f"Joint filter, $\\delta_V={dv:g}$~V, $\\delta_I={di:g}$~A"

    margin_order = sorted(ab[ab.ablation == "margin"].setting.unique(),
                          key=lambda s: (s.count("delta") > 1, s.split("=")[0], float(s.split("=")[1].split(",")[0])))
    pm_keys = ([("base", "base", m) for m in PM_METHODS["base"]]
               + [("controller", "perjoint", m) for m in PM_METHODS["controller"]]
               + [("margin", s, "filter_ws") for s in margin_order])
    pm_rows, pm_agg, prev = [], {}, None
    for a_, s, m in pm_keys:
        g = ab[(ab.ablation == a_) & (ab.setting == s) & (ab.method == m)].sort_values("soc", ascending=False)
        assert len(g) == 3, (a_, s, m, len(g))
        v = dict(uv=g.uv_s_mean.max(), oc=g.oc_s_mean.max(), col=g.collapse_s_mean.max(),
                 imax=g.i_max_mean.max(), imaxpk=g.i_max_peak.max(), falls=int(g.n_fell.sum()),
                 neps=int(g.n_episodes.sum()), hops=g.n_hops_mean.mean(), filt=100 * g.filt_frac_mean.mean(),
                 inf=g.infeas_s_mean.max(), h=list(g.height_max_mean),
                 falls_soc=[int(x) for x in g.n_fell], neps_soc=int(only(list(g.n_episodes), "pm episodes")))
        pm_agg[(a_, s, m)] = v
        b = f"pm_{skey(s)}_{KEYS.get(m, m)}"
        macros.update({b + "_uvmax": f"{v['uv']:.3f}", b + "_ocmax": f"{v['oc']:.3f}",
                       b + "_colmax": f"{v['col']:.3f}", b + "_imax": f"{v['imax']:.2f}",
                       b + "_imaxpk": f"{v['imaxpk']:.2f}", b + "_falls": str(v["falls"]),
                       b + "_neps": str(v["neps"]), b + "_hops": f"{v['hops']:.1f}",
                       b + "_filt": f"{v['filt']:.1f}", b + "_infmax": f"{v['inf']:.3f}"})
        same = prev == (a_, s)
        if prev is not None and not same:
            pm_rows.append("\\midrule" if a_ != prev[0] else "\\addlinespace")
        prev = (a_, s)
        pm_rows.append(f"{'' if same else pm_slabel(a_, s)} & {PM_MLAB[m]} & {v['uv']:.3f} & {v['oc']:.3f} & {v['col']:.3f} & "
                       f"{v['imax']:.2f} & {v['falls']}/{v['neps']} & {v['hops']:.1f} & "
                       + " / ".join(f"{x:.3f}" for x in v["h"]) + f" & {v['filt']:.1f} & {v['inf']:.3f} \\\\")
    (GEN / "tab_perjoint_margin.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrrcrr}\\toprule\n"
        "Controller and margins & Policy & " + " & ".join([
            H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"), H("Collapse", "(s)"),
            H("Peak $\\Ib$", "(A)"), "Falls", "Hops", H("Max height (m)", "SOC 1.0 / 0.5 / 0.2"),
            H("Filter or limiter", "active (\\%)"), H("Infeasible", "(s)")])
        + " \\\\\\midrule\n" + "\n".join(pm_rows) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_perjoint_margin.tex: {len(pm_keys)} rows from {abl.relative_to(ROOT)}")
    pm_all = list(pm_agg.values())
    pm_nviol = sum(v["uv"] > 0 or v["oc"] > 0 or v["col"] > 0 for v in pm_all)
    # delta_V changes against the base filter_ws rows, and both margins zero against delta_I = 0 alone
    hb = pm_agg[("base", "base", "filter_ws")]["h"]
    dv_sets = [s for s in margin_order if s.startswith("delta_V") and "delta_I" not in s]
    di0 = [s for s in margin_order if s == "delta_I=0.0"]
    both = [s for s in margin_order if s.count("delta") == 2]
    sci = lambda x: f"\\num{{{x:.1e}}}" if x > 0 else "0"      # tiny differences, siunitx
    macros.update({
        "pm_nrows": str(len(pm_all)), "pm_nviol": str(pm_nviol),
        "pm_mv0": num(mv0), "pm_mi0": num(mi0), "pm_ilim": num(_p.i_bat_max),
        "pm_dv_dhmax": sci(max(abs(a - b) for s in dv_sets for a, b in zip(pm_agg[('margin', s, 'filter_ws')]['h'], hb))),
        "pm_dv_ddfilt": sci(max(abs(pm_agg[('margin', s, 'filter_ws')]['filt'] - pm_agg[('base', 'base', 'filter_ws')]['filt']) for s in dv_sets)),
        "pm_dvlist": join_and([num(float(s.split('=')[1])) for s in dv_sets]),
        "pm_dilist": join_and([num(float(s.split('=')[1])) for s in margin_order
                               if s.startswith("delta_I")]),
        **({"pm_both_vs_di0_dhmax": sci(max(abs(a - b) for a, b in zip(pm_agg[('margin', both[0], 'filter_ws')]['h'], pm_agg[('margin', di0[0], 'filter_ws')]['h'])))}
           if both and di0 else {}),
    })
    for key, (a_, s, m) in {"pjb": ("controller", "perjoint", "baseline_ws+perjoint@test"),
                            "ftb": ("base", "base", "baseline_ws+filter@test"),
                            "pjf": ("controller", "perjoint", "filter_ws+perjoint@test"),
                            "fwb": ("base", "base", "filter_ws")}.items():
        v = pm_agg[(a_, s, m)]
        for soc, nf in zip(("1", "0.5", "0.2"), v["falls_soc"]):
            macros[f"pm_{key}_soc{soc}_falls"] = str(nf)
        macros[f"pm_{key}_nepsoc"] = str(v["neps_soc"])

    # ---- CHANGELOG #42: ablation summary (one row per setting; main evaluation from per_seed.csv,
    # settings from ablations.csv). Violation times: largest over the states of charge of the mean
    # over seeds; falls: count over seeds and states of charge; max height at SOC 0.2: mean over
    # seeds; filter active: mean over states of charge.
    _psd = pd.read_csv(SUMMARY / "per_seed.csv")

    def _main(method):
        g = _psd[_psd.method == method]
        assert not g.empty, method
        return g.groupby("soc").agg(uv=("uv_s", "mean"), oc=("oc_s", "mean"), h=("height_max", "mean"),
                                    filt=("filt_frac", "mean"), falls=("n_fell", "sum"), neps=("n_episodes", "sum"))

    def _set(ablation, setting, method):
        src = abp if method == PEN_M else ab
        g = src[(src.ablation == ablation) & (src.setting == setting) & (src.method == method)].set_index("soc")
        assert len(g) == 3, (ablation, setting, method)
        return pd.DataFrame({"uv": g.uv_s_mean, "oc": g.oc_s_mean, "h": g.height_max_mean,
                             "filt": g.filt_frac_mean, "falls": g.n_fell, "neps": g.n_episodes})

    def _cells(d, filt=True, many=False):
        if many:      # d: list of per-setting frames (aggregate over settings)
            uv = max(x.uv.max() for x in d); oc = max(x.oc.max() for x in d)
            falls = f"{int(sum(x.falls.sum() for x in d))}/{int(sum(x.neps.sum() for x in d))}"
            hs = [x.h.loc[0.2] for x in d]
            h = f"{min(hs):.3f}--{max(hs):.3f}"
            fl = [100 * x.filt.mean() for x in d]
            f_ = f"{min(fl):.1f}--{max(fl):.1f}" if filt else "--"
        else:
            uv, oc = d.uv.max(), d.oc.max()
            falls = f"{int(d.falls.sum())}/{int(d.neps.sum())}"
            h = f"{d.h.loc[0.2]:.3f}"
            f_ = f"{100 * d.filt.mean():.1f}" if filt else "--"
        return f"{uv:.3f} & {oc:.3f} & {falls} & {h} & {f_}"

    S_ = []   # (factor, setting, policy, cells, reference)
    T_MAIN = "\\ref{tab:viol}, \\ref{tab:ws}"
    S_ += [("Initialisation", "From scratch", "Filter-trained", _cells(_main("filter")), T_MAIN),
           ("", "Warm start", "Filter-trained", _cells(_main("filter_ws")), "")]
    S_ += [("Filter placement", "None", "Baseline", _cells(_main("baseline_ws"), filt=False), "\\ref{tab:ws}"),
           ("", "At test time", "Baseline", _cells(_main("baseline_ws+filter@test")), "")]
    if PEN_M:      # CHANGELOG #57
        S_.append(("", "At test time", f"Penalty $w={pen_w:g}$", _cells(_main(PEN_M)), ""))
    S_ += [("", "In training", "Filter-trained", _cells(_main("filter_ws")), "")]
    gam = sorted([float(s.split("=")[1]) for s in ab[ab.ablation == "gamma"].setting.unique()] + [_env.gamma])
    for i, gv in enumerate(gam):
        d = _set("base", "base", "filter_ws") if gv == _env.gamma else _set("gamma", f"gamma={gv!r}", "filter_ws")
        S_.append(("Barrier rate" if i == 0 else "", f"$\\gamma={gv:g}$" + (" (base)" if gv == _env.gamma else ""),
                   "Filter-trained", _cells(d), "\\ref{tab:abl}" if i == 0 else ""))
    S_.append(("Margins", f"$\\delta_V={mv0:g}$~V, $\\delta_I={mi0:g}$~A (base)", "Filter-trained",
               _cells(_set("base", "base", "filter_ws")), "\\ref{tab:pjm}"))
    for s in margin_order:
        lab = pm_slabel("margin", s).replace("Joint filter, ", "").replace("~V", " V").replace("~A", " A")
        S_.append(("", lab, "Filter-trained", _cells(_set("margin", s, "filter_ws")), ""))
    S_ += [("Controller", "Joint filter (base)", "Baseline", _cells(_set("base", "base", "baseline_ws+filter@test")),
            "\\ref{tab:pjm}"),
           ("", "Per-joint limit", "Baseline", _cells(_set("controller", "perjoint", "baseline_ws+perjoint@test")), ""),
           ("", "Joint filter (base)", "Filter-trained", _cells(_set("base", "base", "filter_ws")), ""),
           ("", "Per-joint limit", "Filter-trained", _cells(_set("controller", "perjoint", "filter_ws+perjoint@test")), "")]
    EST_L = {"true": "true state", "ekf": "EKF", "ekfconf": "EKF, conf.\\ margins"}
    COND_L = {"nominal": "nominal", "r0_scale=0.75": "plant $R_0\\times0.75$", "r0_scale=2.0": "plant $R_0\\times2$",
              "est_noise=0.05": "noise $\\sigma=0.05$"}
    first = True
    for cond in ["nominal", "r0_scale=0.75", "r0_scale=2.0", "est_noise=0.05"]:
        for ek in ["true", "ekf", "ekfconf"]:
            S_.append(("Battery estimator" if first else "", f"{COND_L[cond]}, {EST_L[ek]}", "Filter-trained",
                       _cells(_set("estimator", f"{ek}, {cond}", "filter_ws")), "\\ref{tab:est}" if first else ""))
            first = False
    vars_ = list(ab[ab.ablation == "variant"].setting.unique())
    for i, (m, pol, filt) in enumerate([("baseline_ws", "Baseline", False),
                                        ("baseline_ws+filter@test", "Baseline + filter at test", True)]
                                       + ([(PEN_M, f"Penalty $w={pen_w:g}$ + filter at test", True)] if PEN_M else [])
                                       + [("filter_ws", "Filter-trained", True)]):
        S_.append(("Plant variants" if i == 0 else "", f"{len(vars_)} variants" if i == 0 else "", pol,
                   _cells([_set("variant", s, m) for s in vars_], filt=filt, many=True),
                   "\\ref{tab:abl}" if i == 0 else ""))
    s_rows = []
    for k, (fac, st_, pol, cells, ref) in enumerate(S_):
        if k and fac:
            s_rows.append("\\midrule")
        # the full table is in the supplementary material, where references to the manuscript's
        # tables do not resolve: no reference column (CHANGELOG #51)
        s_rows.append(f"{fac} & {st_} & {pol} & {cells} \\\\")
    (GEN / "tab_ablation_summary.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{lllrrrcr}\\toprule\n"
        "Factor & Setting & Policy & " + " & ".join([
            H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"), "Falls",
            H("Max height", "SOC 0.2 (m)"), H("Filter", "active (\\%)")])
        + " \\\\\\midrule\n" + "\n".join(s_rows) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_ablation_summary.tex: {len(S_)} rows")
    macros["abs_nvar"] = str(len(vars_))

    # estimator table (Step 8): true state, EKF with fixed margin, EKF with confidence margin
    CK = {"nominal": "nom", "est_noise=0.05": "noise", "r0_scale=2.0": "rbig", "r0_scale=0.75": "rsmall"}
    CLAB = {"nominal": "Nominal", "est_noise=0.05": "Noise $\\sigma=0.05$", "r0_scale=2.0": "Plant $R_0\\times2$",
            "r0_scale=0.75": "Plant $R_0\\times0.75$"}
    ELAB = {"true": "True state", "ekf": "EKF, fixed", "ekfconf": "EKF, conf."}
    es = ab[ab.ablation == "estimator"].copy()
    es["est"] = es.setting.str.split(", ").str[0]
    es["cond"] = es.setting.str.split(", ").str[1]
    cnt = lambda r: int(sum(r[c] for c in COUNTERS_ABL))
    COUNTERS_ABL = [c for c in ab.columns if c in ("a3_fail_n", "rem1_fail_n", "a4_fail_n", "a6_empty_n")]
    e_rows, prev = [], None
    for _, r in es.iterrows():
        ck, ek = CK[r.cond], r.est
        b = f"est_{ck}_{ek}_soc{r.soc:g}"
        isekf = not pd.isna(r.ekf_conf_mean_mean)
        conf = (f"{r.ekf_conf_mean_mean:.2f} [{r.ekf_conf_ep_min:.2f}, {r.ekf_conf_ep_max:.2f}]" if isekf else "--")
        rdiv = f"{int(r.n_r0_diverged)}/{int(r.n_episodes)}" if isekf else "--"
        macros.update({b + "_uv": f"{r.uv_s_mean:.3f}", b + "_oc": f"{r.oc_s_mean:.3f}",
                       b + "_ocstd": f"{r.oc_s_std:.3f}", b + "_imax": f"{r.i_max_mean:.2f}",
                       b + "_imaxstd": f"{r.i_max_std:.2f}", b + "_imaxpk": f"{r.i_max_peak:.2f}",
                       b + "_falls": str(int(r.n_fell)), b + "_neps": str(int(r.n_episodes)),
                       b + "_hops": f"{r.n_hops_mean:.1f}", b + "_hmax": f"{r.height_max_mean:.3f}",
                       b + "_infeas": f"{r.infeas_s_mean:.4f}"})
        if isekf:
            macros.update({b + "_conf": f"{r.ekf_conf_mean_mean:.2f}", b + "_confmin": f"{r.ekf_conf_ep_min:.2f}",
                           b + "_confmax": f"{r.ekf_conf_ep_max:.2f}", b + "_rdiv": str(int(r.n_r0_diverged)),
                           b + "_eabs": f"{r.ekf_e_abs_mean_mean:.3f}", b + "_r0rel": f"{r.ekf_r0_rel_final_mean:.3f}"})
        grp = (r.cond, r.est)
        if prev is not None and grp != prev:
            e_rows.append("\\midrule" if grp[0] != prev[0] else "\\addlinespace")
        clab = CLAB[r.cond] if prev is None or grp[0] != prev[0] else ""
        elab = ELAB[ek] if grp != prev else ""
        prev = grp
        e_rows.append(f"{clab} & {elab} & {r.soc:.1f} & {r.uv_s_mean:.3f} & "
                      f"{r.oc_s_mean:.3f} $\\pm$ {r.oc_s_std:.3f} & {r.i_max_mean:.2f} $\\pm$ {r.i_max_std:.2f} & "
                      f"{r.i_max_peak:.2f} & {int(r.n_fell)}/{int(r.n_episodes)} & {r.n_hops_mean:.1f} & "
                      f"{r.height_max_mean:.3f} & {r.infeas_s_mean:.4f} & {conf} & {rdiv} & {cnt(r)} \\\\")
    (GEN / "tab_estimator.tex").write_text(
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{llrrrrrrrrrlrr}\\toprule\n"
        "Condition & Estimator & SOC & " + " & ".join([
            H("$t(\\Vb<V_{\\min})$", "(s)"), H("$t(\\Ib>I_{\\lim})$", "(s)"), H("Peak $\\Ib$", "(A)"),
            H("Largest", "(A)"), "Falls", "Hops", H("Max", "height (m)"), H("Infeasible", "(s)"),
            H("Confidence $\\zeta$", "[min, max]"), H("$R_0$", "diverged"), "Counters"])
        + " \\\\\\midrule\n" + "\n".join(e_rows) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_estimator.tex: {len(es)} rows from {abl.relative_to(ROOT)}")
    # aggregates over SOC per condition and estimator
    for (cond, ek), g in es.groupby(["cond", "est"]):
        b = f"est_{CK[cond]}_{ek}"
        macros.update({b + "_nsococ": str(int((g.oc_s_mean > 0).sum())), b + "_nsoc": str(len(g)),
                       b + "_ocmax": f"{g.oc_s_mean.max():.3f}", b + "_imaxpk": f"{g.i_max_peak.max():.2f}",
                       b + "_falls": str(int(g.n_fell.sum())), b + "_neps": str(int(g.n_episodes.sum()))})
        if g.ekf_conf_mean_mean.notna().all():
            macros.update({b + "_rdiv": str(int(g.n_r0_diverged.sum())),
                           b + "_confmin": f"{g.ekf_conf_ep_min.min():.2f}",
                           b + "_confmax": f"{g.ekf_conf_ep_max.max():.2f}"})
    ekfrows = es[es.est != "true"]
    noise = ekfrows[ekfrows.cond == "est_noise=0.05"]
    clean = ekfrows[ekfrows.cond != "est_noise=0.05"]
    macros.update({"est_counters_total": str(int(es[COUNTERS_ABL].sum().sum())),
                   "est_falls_total": str(int(es.n_fell.sum())), "est_neps_total": str(int(es.n_episodes.sum())),
                   "est_noise_rdiv": str(int(noise.n_r0_diverged.sum())),
                   "est_noise_neps": str(int(noise.n_episodes.sum())),
                   "est_noise_confmax": f"{noise.ekf_conf_ep_max.max():.2f}",
                   "est_clean_confmin": f"{clean.ekf_conf_ep_min.min():.2f}",
                   "est_clean_rdiv": str(int(clean.n_r0_diverged.sum())),
                   "est_clean_r0rel_min": f"{clean.ekf_r0_rel_final_mean.min():.3f}",
                   "est_clean_r0rel_max": f"{clean.ekf_r0_rel_final_mean.max():.3f}",
                   "est_clean_eabs_max": f"{clean.ekf_e_abs_mean_mean.max():.3f}",
                   "est_noise_eabs_min": f"{noise.ekf_e_abs_mean_mean.min():.3f}",
                   "est_noise_eabs_max": f"{noise.ekf_e_abs_mean_mean.max():.3f}"})
    # ---- CHANGELOG #51: condensed main-text versions of the stress-test, estimator and ablation-
    # summary tables (full versions in the supplementary material)
    SM = "{\\footnotesize\\setlength{\\tabcolsep}{3pt}"
    # stress tests: one row per setting; per policy its key result
    def _mx(a_, s, m, col):
        g = ab[(ab.ablation == a_) & (ab.setting == s) & (ab.method == m)]
        return None if g.empty else g
    ab_rows, prev_a = [], None
    for (a_, s), _ in st.groupby(["ablation", "setting"], sort=False):
        b0, bt, fw = (_mx(a_, s, m, None) for m in ("baseline_ws", "baseline_ws+filter@test", "filter_ws"))
        if prev_a is not None and a_ != prev_a:
            ab_rows.append("\\midrule")
        prev_a = a_
        bcell = (f"{b0.uv_s_mean.max():.3f} & {b0.oc_s_mean.max():.3f}" if b0 is not None else "-- & --")
        tcell = (f"{int(bt.n_fell.sum())}/{int(bt.n_episodes.sum())}" if bt is not None else "--")

        def _fcell(x):     # violation (largest of the three times), falls, height at SOC 0.2, active
            if x is None or x.empty:
                return "-- & -- & -- & --"
            return (f"{max(x.uv_s_mean.max(), x.oc_s_mean.max(), x.collapse_s_mean.max()):.3f} & "
                    f"{int(x.n_fell.sum())}/{int(x.n_episodes.sum())} & "
                    f"{x[x.soc == 0.2].height_max_mean.iloc[0]:.3f} & {100 * x.filt_frac_mean.mean():.1f}")
        # CHANGELOG #57: penalty policies with the filter at test (variants, barrier rates and base only)
        pcell = _fcell(abp[(abp.ablation == a_) & (abp.setting == s)]) if PEN_M else None
        ab_rows.append(f"{slabel(s)} & {bcell} & {tcell} & " + (f"{pcell} & " if PEN_M else "") + f"{_fcell(fw)} \\\\")
    _pg = (" & \\multicolumn{4}{c}{\\begin{tabular}[b]{@{}c@{}}Penalty $w=" + num(pen_w) + "$ +\\\\{}test filter\\end{tabular}}"
           if PEN_M else "")
    _c4 = [H("Violation", "(s)"), "Falls", H("Height", "0.2 (m)"), H("Active", "(\\%)")]
    # with the penalty column group (12 columns) the table is wider than the text: scaled to the line
    # width, as the full tables are
    (GEN / "tab_ablation_short.tex").write_text(
        SM + ("\\resizebox{\\linewidth}{!}{" if PEN_M else "")
        + "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{" + ("0.15" if PEN_M else "0.22")
        + "\\linewidth}rrc" + ("rcrr" if PEN_M else "") + "rcrr}\\toprule\n"
        " & \\multicolumn{2}{c}{Baseline} & \\multicolumn{1}{c}{\\begin{tabular}[b]{@{}c@{}}Baseline +\\\\{}test filter\\end{tabular}}"
        + _pg + " & \\multicolumn{4}{c}{Filter-trained} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}"
        + ("\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}\n" if PEN_M else "\\cmidrule(lr){5-8}\n")
        + "Setting & " + " & ".join([H("Below", "$V_{\\min}$ (s)"), H("Above", "$I_{\\lim}$ (s)"), "Falls"]
                                    + (_c4 if PEN_M else []) + _c4)
        + " \\\\\\midrule\n" + "\n".join(ab_rows) + "\n\\bottomrule\\end{tabular}" + ("}" if PEN_M else "") + "}\n")
    print(f"tab_ablation_short.tex: {sum(not x.startswith('\\') for x in ab_rows)} rows")
    # estimator: one row per condition and estimator, aggregated over the states of charge
    e_short, prevc = [], None
    for cond in ["nominal", "r0_scale=0.75", "r0_scale=2.0", "est_noise=0.05"]:
        for ek in ["true", "ekf", "ekfconf"]:
            g = es[(es.cond == cond) & (es.est == ek)]
            if prevc is not None and cond != prevc:
                e_short.append("\\midrule")
            isekf = g.ekf_conf_mean_mean.notna().all()
            conf = f"{g.ekf_conf_ep_min.min():.2f}--{g.ekf_conf_ep_max.max():.2f}" if isekf else "--"
            rdiv = f"{int(g.n_r0_diverged.sum())}/{int(g.n_episodes.sum())}" if isekf else "--"
            e_short.append(f"{CLAB[cond] if cond != prevc else ''} & {ELAB[ek]} & {g.oc_s_mean.max():.3f} & "
                           f"{g.i_max_peak.max():.2f} & {int(g.n_fell.sum())}/{int(g.n_episodes.sum())} & "
                           f"{g[g.soc == 0.2].height_max_mean.iloc[0]:.3f} & {conf} & {rdiv} \\\\")
            prevc = cond
    (GEN / "tab_estimator_short.tex").write_text(
        SM + "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.17\\linewidth}lrrcrcc}\\toprule\n"
        "Condition & Estimator & " + " & ".join([H("Above", "$I_{\\lim}$ (s)"), H("Largest", "$\\Ib$ (A)"), "Falls",
                                                  H("Height", "0.2 (m)"), H("Confidence", "$\\zeta$"),
                                                  H("$R_0$", "diverged")])
        + " \\\\\\midrule\n" + "\n".join(e_short) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_estimator_short.tex: {sum(not x.startswith('\\') for x in e_short)} rows")
    # ablation summary: selected settings, one violation column (largest of the two times)
    def _cs(d, filt=True, many=False):
        if many:
            v = max(max(x.uv.max(), x.oc.max()) for x in d)
            falls = f"{int(sum(x.falls.sum() for x in d))}/{int(sum(x.neps.sum() for x in d))}"
            hs = [x.h.loc[0.2] for x in d]
            h = f"{min(hs):.3f}--{max(hs):.3f}"
            fl = [100 * x.filt.mean() for x in d]
            f_ = f"{min(fl):.1f}--{max(fl):.1f}" if filt else "--"
        else:
            v = max(d.uv.max(), d.oc.max())
            falls = f"{int(d.falls.sum())}/{int(d.neps.sum())}"
            h = f"{d.h.loc[0.2]:.3f}"
            f_ = f"{100 * d.filt.mean():.1f}" if filt else "--"
        return f"{v:.3f} & {falls} & {h} & {f_}"
    short = [("Initialisation", "from scratch", "filter-trained", _cs(_main("filter"))),
             ("", "warm start", "filter-trained", _cs(_main("filter_ws"))),
             ("Filter placement", "none", "baseline", _cs(_main("baseline_ws"), filt=False)),
             ("", "at test time", "baseline", _cs(_main("baseline_ws+filter@test")))]
    if PEN_M:      # CHANGELOG #57
        short.append(("", "at test time", f"penalty $w={pen_w:g}$", _cs(_main(PEN_M))))
    short += [("", "in training", "filter-trained", _cs(_main("filter_ws")))]
    for i, gv in enumerate(gam):
        d = _set("base", "base", "filter_ws") if gv == _env.gamma else _set("gamma", f"gamma={gv!r}", "filter_ws")
        short.append(("Barrier rate" if i == 0 else "", f"$\\gamma={gv:g}$" + (" (base)" if gv == _env.gamma else ""),
                      "filter-trained", _cs(d)))
    short.append(("Margins", f"$\\delta_V={mv0:g}$, $\\delta_I={mi0:g}$ (base)", "filter-trained", _cs(_set("base", "base", "filter_ws"))))
    for s in [x for x in margin_order if not x.startswith("delta_V=0.5") and x != "delta_V=0.0"]:
        short.append(("", pm_slabel("margin", s).replace("Joint filter, ", "").replace("~V", "").replace("~A", ""),
                      "filter-trained", _cs(_set("margin", s, "filter_ws"))))
    short += [("Controller", "joint filter (base)", "baseline", _cs(_set("base", "base", "baseline_ws+filter@test"))),
              ("", "per-joint limit", "baseline", _cs(_set("controller", "perjoint", "baseline_ws+perjoint@test"))),
              ("", "joint filter (base)", "filter-trained", _cs(_set("base", "base", "filter_ws"))),
              ("", "per-joint limit", "filter-trained", _cs(_set("controller", "perjoint", "filter_ws+perjoint@test")))]
    first = True
    for cond, ek in [("nominal", "true"), ("nominal", "ekf"), ("r0_scale=0.75", "true"), ("r0_scale=0.75", "ekf"),
                     ("est_noise=0.05", "true"), ("est_noise=0.05", "ekf"), ("est_noise=0.05", "ekfconf")]:
        short.append(("Estimator" if first else "", f"{COND_L[cond]}, {EST_L[ek]}", "filter-trained",
                      _cs(_set("estimator", f"{ek}, {cond}", "filter_ws"))))
        first = False
    for i, (m, pol, filt) in enumerate([("baseline_ws", "baseline", False),
                                        ("baseline_ws+filter@test", "baseline + filter at test", True),
                                        ("filter_ws", "filter-trained", True)]):
        short.append(("Plant variants" if i == 0 else "", f"{len(vars_)} variants" if i == 0 else "", pol,
                      _cs([_set("variant", s, m) for s in vars_], filt=filt, many=True)))
    s_rows = []
    for k, (fac, st_, pol, cells) in enumerate(short):
        if k and fac:
            s_rows.append("\\midrule")
        # policy folded into the setting: filter-trained unless marked (caption)
        lab = st_ if (pol == "filter-trained" and st_) else (f"{st_}, {pol}" if st_ else pol)
        s_rows.append(f"{fac} & {lab} & {cells} \\\\")
    (GEN / "tab_ablation_summary_short.tex").write_text(
        SM + "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.15\\linewidth}>{\\raggedright\\arraybackslash}p{0.33\\linewidth}rcrr}\\toprule\n"
        "Factor & Setting & " + " & ".join([H("Violation", "(s)"), "Falls", H("Height", "0.2 (m)"),
                                                     H("Active", "(\\%)")])
        + " \\\\\\midrule\n" + "\n".join(s_rows) + "\n\\bottomrule\\end{tabular}}\n")
    print(f"tab_ablation_summary_short.tex: {len(short)} rows")

    import importlib.util as _ilu                  # reporting threshold of the R0-divergence count
    _spec = _ilu.spec_from_file_location("aggabl", ROOT / "scripts" / "aggregate_ablations.py")
    _agg = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_agg)
    macros.update({"est_rdiv_lo": num(_agg.R0_DIV_BOUNDS[0]), "est_rdiv_hi": num(_agg.R0_DIV_BOUNDS[1])})
    # injected noise against the EKF's assumed sensor noise. est_noise multiplies each measurement
    # by (1 + sigma n): standard deviation sigma x the measured value. Voltage: at the pack
    # open-circuit voltage of the lowest and highest evaluated SOC; current: at I_lim - delta_I.
    sig = 0.05
    ocv = lambda s: _p.n_series * float(np.interp(s, _p.ocv_soc, _p.ocv_v))
    socs_ev = sorted(es.soc.unique())
    i_t = _p.i_bat_max - _env.margin_i
    macros.update({"est_sigma": num(sig), "par_sigv": num(_p.sigma_v_meas), "par_sigi": num(_p.sigma_i_meas),
                   "est_noisev_lo": f"{sig * ocv(socs_ev[0]):.2f}", "est_noisev_hi": f"{sig * ocv(socs_ev[-1]):.2f}",
                   "est_ratiov_lo": f"{sig * ocv(socs_ev[0]) / _p.sigma_v_meas:.0f}",
                   "est_ratiov_hi": f"{sig * ocv(socs_ev[-1]) / _p.sigma_v_meas:.0f}",
                   "est_itarget": num(i_t), "est_noisei": f"{sig * i_t:.3f}",
                   "est_ratioi": f"{sig * i_t / _p.sigma_i_meas:.1f}"})
# filter algorithm constants (busaware/safety_filter.py) and the random-state count of the rounding
# test (tests/test_budget_margin.py), for the filter section
import inspect
import busaware.safety_filter as SF
_eps_exp = np.log10(SF.EPS_REL)
macros.update({"filt_eps_exp": f"{_eps_exp:.0f}" if abs(_eps_exp - round(_eps_exp)) < 1e-12 else "?",
               "filt_maxdoublings": str(SF.MAX_DOUBLINGS),
               "filt_halvings": str(inspect.signature(SF.project_currents).parameters["iters"].default)})
_m = re.search(r"for _ in range\((\d+)\)", (ROOT / "tests" / "test_budget_margin.py").read_text(encoding="utf-8"))
if _m:
    macros["test_budget_nstates"] = thin(_m.group(1))
else:
    print("test_budget_nstates: loop count not found in tests/test_budget_margin.py; left pending")
# EKF tuning (busaware/ekf.py) for the filter section
macros.update({"ekf_window": str(EKFM.WINDOW), "ekf_window_ms": num(1e3 * EKFM.WINDOW * _env.dt_sim),
               "ekf_nislo": num(EKFM.NIS_LO), "ekf_nishi": num(EKFM.NIS_HI),
               "ekf_sdsoc0": num(EKFM.SD_SOC0), "ekf_sdr0rel": num(EKFM.SD_R0_REL0), "ekf_qr0": num(EKFM.Q_R0),
               "ekf_r0floor": num(EKFM.R0_FLOOR_REL)})

# ---- CHANGELOG #51: statements of the internal-review revision, checked here --------------------
# (f) Discussion "Performance cost of the limits": per SOC, the largest mean max height among the
#     main-evaluation methods (every training/test combination) with no violation, no collapse and
#     no fall, and which method it is. The text says a warm-started penalty policy with the filter
#     at test holds it at every SOC; checked, printed.
_psf6 = SUMMARY / "per_seed.csv"
if _psf6.exists():
    _d6 = pd.read_csv(_psf6)
    _g6 = _d6.groupby(["method", "soc"]).agg(uv=("uv_s", "mean"), oc=("oc_s", "mean"), col=("collapse_s", "mean"),
                                             h=("height_max", "mean"), fell=("n_fell", "sum")).reset_index()
    f_ok = True
    for _soc, _x in _g6.groupby("soc"):
        _ok = _x[(_x.uv == 0) & (_x.oc == 0) & (_x.col == 0) & (_x.fell == 0)].sort_values("h", ascending=False)
        _top = _ok.iloc[0]
        f_ok = f_ok and bool(re.fullmatch(r"penalty_w[0-9.]+_ws\+filter@test", _top.method))
        macros[f"lim_best_hmax_soc{_soc:g}"] = f"{_top.h:.3f}"
    print("claim (f) (best height within the limits, no fall: a penalty_ws policy with the filter at test):",
          "OK" if f_ok else "FAILS")
    # (f2) CHANGELOG #55: the Discussion now uses per-weight values of the weights with all seeds only:
    #      for each such weight, the warm-started penalty policies with the filter at test have no
    #      violation, collapse or fall and a larger mean max height than filter_ws at every SOC.
    _ns6 = _d6.groupby("method").seed.nunique()
    _full = sorted(float(m_.group(1)) for m_ in (re.fullmatch(r"penalty_w([0-9.]+)_ws\+filter@test", x) for x in _ns6.index)
                   if m_ and _ns6[m_.group(0)] == _ns6.max())
    f2_ok = bool(_full)
    for _w in _full:
        for _soc, _x in _g6[_g6.method.isin([f"penalty_w{_w:g}_ws+filter@test", "filter_ws"])].groupby("soc"):
            _p = _x[_x.method != "filter_ws"].iloc[0]
            _f = _x[_x.method == "filter_ws"].iloc[0]
            f2_ok = f2_ok and _p.uv == 0 and _p.oc == 0 and _p.col == 0 and _p.fell == 0 and _p.h > _f.h
    f2_ok = f2_ok and join_and([num(w_) for w_ in _full]) == macros.get("pen_ws_w_full")
    # Discussion, Training: "except the penalty weights <pen_ws_w_part>, trained with one seed each"
    _one = sorted(float(m_.group(1)) for m_ in (re.fullmatch(r"penalty_w([0-9.]+)_ws", x) for x in _ns6.index)
                  if m_ and _ns6[m_.group(0)] == 1)
    f2_ok = f2_ok and join_and([num(w_) for w_ in _one]) == macros.get("pen_ws_w_part")
    print(f"claim (f2) (each {_ns6.max()}-seed penalty weight with the filter at test: no violation/fall, higher "
          "than filter_ws at every SOC):", "OK" if f2_ok else "FAILS")
# (g) wording "unchanged" (V_min datasheet variant), "confidence is", "at every state of charge"
g_ok = (macros.get("vmin_dfalls") == "0" and float(macros.get("vmin_dhops", "nan")) == 0
        and float(macros.get("vmin_dviol", "nan")) == 0 and float(macros.get("est_noise_confmax", "nan")) == 0
        and all(len({macros.get(f"{b}_imax_soc{s}") for s in ("1", "0.5", "0.2")}) == 1
                for b in ("base_filttest", "filtws")))
print("claim (g) (falls/hops/violations unchanged; confidence 0; peak current equal over SOC):",
      "OK" if g_ok else "FAILS")


# (h) CHANGELOG #54 (page limit): qualitative sentences of Sections 6 and 7 that replaced restated
#     table values; each is checked here from the same macros the tables are built from.
def _mf(k):
    return float(macros[k].replace("\\,", ""))


def _h_checks():
    S3 = ("1", "0.5", "0.2")
    c = {}
    c["filter-trained hop more, lower (every SOC)"] = all(
        _mf(f"filtws_hops_soc{s}") > _mf(f"basews_hops_soc{s}") and _mf(f"filtws_hmax_soc{s}") < _mf(f"basews_hmax_soc{s}")
        for s in S3)
    c["Lagrangian seed 1 falls well before episode end"] = _mf("lag_s1_len") < 0.5 * _mf("env_maxsteps")
    c["test-filter falls at SOC 1: seeds 0 and 2 only"] = (
        _mf("ps_basews_filttest_s1_soc1_fell") == 0
        and _mf("ps_basews_filttest_s0_soc1_fell") + _mf("ps_basews_filttest_s2_soc1_fell") == _mf("basews_filttest_fallsn_soc1"))
    gs = ("g0.1", "base", "g0.6", "g1.0")            # increasing barrier rate
    act = [_mf(f"abl_{g}_filtws_filt") for g in gs]
    c["barrier rate: filter active decreasing in gamma"] = all(a > b for a, b in zip(act, act[1:]))
    c["barrier rate: max height barely changes (<= 0.005 m)"] = all(
        max(_mf(f"abl_{g}_filtws_soc{s}_hmax") for g in gs) - min(_mf(f"abl_{g}_filtws_soc{s}_hmax") for g in gs) <= 0.005
        for s in S3)
    c["barrier rate: same test-filter falls at every rate"] = len({macros[f"abl_{g}_basews_filttest_falls"] for g in gs}) == 1
    # CHANGELOG #57: infeasible time largest at the smallest rate for every filtered policy; for the
    # filter-trained policies zero at the other rates (the text said "only at the smallest rate" for all)
    for pk in ("filtws", "basews_filttest") + (("penwDws_filttest",) if "pst_w" in macros else ()):
        inf = {g: _mf(f"abl_{g}_{pk}_infmax") for g in gs}
        c[f"barrier rate: infeasible time largest at the smallest rate ({pk})"] = (
            inf["g0.1"] > 0 and all(inf["g0.1"] > v for g, v in inf.items() if g != "g0.1"))
    c["barrier rate: filter-trained infeasible only at the smallest rate"] = all(
        _mf(f"abl_{g}_filtws_infmax") == 0 for g in gs if g != "g0.1")
    c["per-joint: more baseline falls"] = (sum(_mf(f"pm_pjb_soc{s}_falls") for s in S3)
                                           > sum(_mf(f"pm_ftb_soc{s}_falls") for s in S3))
    c["per-joint: acts more, lower (every SOC), more infeasible"] = (
        _mf("pm_perjoint_filtws_pjtest_filt") > _mf("pm_base_filtws_filt")
        and all(_mf(f"abl_perjoint_filtws_pjtest_soc{s}_hmax") < _mf(f"abl_base_filtws_soc{s}_hmax") for s in S3)
        and _mf("pm_perjoint_basews_pjtest_infmax") > _mf("pm_base_basews_filttest_infmax"))
    c["margins: peak = I_lim - delta_I"] = all(
        abs(_mf(k) - (_mf("pm_ilim") - d)) <= 0.01
        for k, d in (("pm_deltaI0.0_filtws_imax", 0.0), ("pm_base_filtws_imax", _mf("pm_mi0")), ("pm_deltaI1.0_filtws_imax", 1.0)))
    c["margins: larger delta_I lowers max height (every SOC)"] = all(
        _mf(f"abl_deltaI0.0_filtws_soc{s}_hmax") > _mf(f"abl_base_filtws_soc{s}_hmax") > _mf(f"abl_deltaI1.0_filtws_soc{s}_hmax")
        for s in S3)
    c["EKF nominal: almost same peak (<= 0.1 A) and height (<= 0.002 m)"] = (
        abs(_mf("est_nom_ekf_soc1_imax") - _mf("est_nom_true_soc1_imax")) <= 0.1
        and all(abs(_mf(f"est_nom_ekf_soc{s}_hmax") - _mf(f"est_nom_true_soc{s}_hmax")) <= 0.002 for s in S3))
    c["R0 x small: true-state breaches at every SOC"] = all(_mf(f"est_rsmall_true_soc{s}_oc") > 0 for s in S3)
    c["R0 x big: EKF larger peak and height (every SOC)"] = all(
        _mf(f"est_rbig_ekf_soc{s}_imax") > _mf(f"est_rbig_true_soc{s}_imax")
        and _mf(f"est_rbig_ekf_soc{s}_hmax") > _mf(f"est_rbig_true_soc{s}_hmax") for s in S3)
    c["noise: confidence margins lower height (every SOC)"] = all(
        _mf(f"est_noise_ekfconf_soc{s}_hmax") < _mf(f"est_noise_ekf_soc{s}_hmax") for s in S3)
    vars_ = ("cold", "vmindatasheet", "wiring0mohm", "wiring20mohm", "paux0w", "paux10w", "rphasex0.5", "rphasex1.5",
             "currentlag", "geareff0.9")
    # CHANGELOG #57: for every policy under the filter in Table 8 (filter-trained, baseline and penalty
    # with the filter at test)
    for pk in ("filtws", "basews_filttest") + (("penwDws_filttest",) if "pst_w" in macros else ()):
        hv = {v: _mf(f"abl_{v}_{pk}_soc0.2_hmax") for v in vars_}
        c[f"variants at SOC 0.2: lowest cold, highest half phase R ({pk})"] = (
            len(vars_) == int(_mf("abl_nvar")) and min(hv, key=hv.get) == "cold" and max(hv, key=hv.get) == "rphasex0.5")
    c["plant R0 x 0.75: breach at every SOC"] = all(_mf(f"abl_r0.75_filtws_soc{s}_oc") > 0 for s in S3)
    c["plant R0 x 1.5, x 2: more active, lower (every SOC)"] = all(
        _mf(f"abl_{r}_filtws_filt") > _mf("abl_base_filtws_filt")
        and all(_mf(f"abl_{r}_filtws_soc{s}_hmax") < _mf(f"abl_base_filtws_soc{s}_hmax") for s in S3)
        for r in ("r1.5", "r2.0"))
    return c


try:
    _hc = _h_checks()
    for _k, _v in _hc.items():
        if not _v:
            print("   claim (h) fails:", _k)
    print("claim (h) (qualitative statements replacing table values, Sections 6-7):",
          "OK" if all(_hc.values()) else "FAILS")
except KeyError as _e:
    print("claim (h): macro missing", _e, "FAILS")

# (i) CHANGELOG #55: counts written as "all N" in the text, and the merged peak-current sentence of
#     Section 6.1 (one value for both methods).
try:
    _ic = {
        "Lagrangian cost above d in all seeds": macros["lag_ncostabove"] == macros["lag_nseeds"],
        "Lagrangian multiplier at its maximum in all seeds": macros["lag_nfinalmax"] == macros["lag_nseeds"],
        "sensitivity rule above the margin in all cases": macros["sens_nruleabovemargin"] == macros["sens_ncases"],
        "baseline violates in all variants": macros["abl_bws_nviol"] == macros["abl_nvar"],
        "penalty w=2 fell in all episodes at 0.5 and 0.2": (
            macros["penwEws_fallsn_soc0.5"] == macros["penwEws_neps_soc0.5"]
            == macros["penwEws_fallsn_soc0.2"] == macros["penwEws_neps_soc0.2"]),
        "no filter-mode row with a violation": macros["abl_fm_nviol"] == "0",
        "peak current equal for baseline+test filter and filter_ws": all(
            macros[f"base_filttest_imax_soc{s}"] == macros[f"filtws_imax_soc{s}"] for s in ("1", "0.5", "0.2")),
    }
    for _k, _v in _ic.items():
        if not _v:
            print("   claim (i) fails:", _k)
    print("claim (i) ('all N' wordings; merged peak-current sentence):", "OK" if all(_ic.values()) else "FAILS")
except KeyError as _e:
    print("claim (i): macro missing", _e, "FAILS")

# ---- CHANGELOG #39: settings quoted in the figure captions (written by scripts/make_figures.py) ----
_fm = ROOT / "results" / "figures" / "figures_meta.json"
if _fm.exists():
    _meta = json.loads(_fm.read_text())
    macros.update({"figbud_soc": num(_meta["budget"]["soc"]), "figbud_ifrac": num(_meta["budget"]["i_frac"]),
                   "figtrain_window": str(_meta["training"]["window"]),
                   "figtrain_penw": num(_meta["training"]["penalty_w"])})
else:
    print("figures_meta.json missing (run scripts/make_figures.py); figure-caption macros left pending")

# CHANGELOG #58 (claim (k), abstract and conclusion): filter-trained episode totals.
#  Episode totals of the policies trained with the filter, over every evaluation in the paper, each
#  episode once: the main evaluation (eval.csv) and every setting file that a table reports
#  (aggregate_ablations.setting_of: stress tests, model error, barrier rates, margins, per-joint
#  limit, EKF); files the paper does not report (eval_pre_eps.csv) are not counted.
import importlib.util as _ilu8
_sp8 = _ilu8.spec_from_file_location("aggabl_k", ROOT / "scripts" / "aggregate_ablations.py")
_ag8 = _ilu8.module_from_spec(_sp8); _sp8.loader.exec_module(_ag8)

def _episodes_of(run):
    n_, f_, files_ = 0, 0, 0
    for s_ in _ag8.SEEDS:
        d_run = RESULTS / f"{run}_s{s_}"
        for f in sorted(d_run.glob("eval*.csv")):
            if f.name != "eval.csv" and _ag8.setting_of(f.name) is None:
                continue
            d = pd.read_csv(f)
            n_ += len(d); f_ += int(d.fell.sum()); files_ += 1
    return n_, f_, files_
_nws, _fws, _kws = _episodes_of("filter_ws")
_nsc, _fsc, _ksc = _episodes_of("filter")
macros.update({"fws_neps_all": thin(_nws), "fws_falls_all": str(_fws),
               "filt_neps_all": thin(_nws + _nsc), "filt_falls_all": str(_fws + _fsc)})
print(f"filter-trained episodes: warm-started {_nws} ({_kws} files, {_fws} falls), from scratch {_nsc} "
      f"({_ksc} files, {_fsc} falls)")

# (l) CHANGELOG #59: conclusion, Corollary 1 wording. Over every filter-mode evaluation with exact
#     battery parameters (main evaluation of every policy with the filter, in training or at test;
#     base, variants, barrier rates and margins of the stress tests, incl. the penalty rows; the
#     true-state nominal estimator rows), the largest mean infeasible time per episode (mean over
#     seeds, per method, setting and SOC, as in the tables), and no violation or collapse in any of them.
_l_ok = False
if (SUMMARY / "per_seed.csv").exists() and abl.exists():
    _ps9 = pd.read_csv(SUMMARY / "per_seed.csv")                   # ps lacks infeas_s
    _mm = _ps9[_ps9.test_mode == "filter"].groupby(["method", "soc"])[["infeas_s", "uv_s", "oc_s", "collapse_s"]].mean()
    _ex = pd.concat([ab[ab.ablation.isin(["base", "variant", "gamma", "margin"]) & (ab.method != "baseline_ws")],
                     ab[ab.setting == "true, nominal"], abp], ignore_index=True)
    _inf = max(_mm.infeas_s.max(), _ex.infeas_s_mean.max())
    _viol = float(_mm[["uv_s", "oc_s", "collapse_s"]].sum().sum()
                  + _ex[["uv_s_mean", "oc_s_mean", "ov_s_mean", "collapse_s_mean"]].sum().sum())
    _ep_s = float(macros["env_episode_s"])
    macros.update({"infeas_exact_max": f"{_inf:.3f}", "infeas_exact_max_pct": f"{100 * _inf / _ep_s:.1f}"})
    _l_ok = _viol == 0 and 0 < _inf < 0.05 * _ep_s          # "rare": below 5 % of the episode
    print(f"claim (l) (exact parameters: infeasible time at most {_inf:.4f} s per {_ep_s:g} s episode, "
          f"{len(_mm) + len(_ex)} rows, no violation):", "OK" if _l_ok else "FAILS")

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
(GEN / "results.tex").write_text(f"% generated {stamp} by scripts/export_paper.py. Do not edit.\n"
    + "".join(f"\\resdef{{{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))

p = Params()
esc = lambda s: s.replace("_", "\\_")


def provenance_rows(path=ROOT / "params_provenance.md"):
    """Field -> {status, unit, source} from every markdown table in params_provenance.md that has a
    Field and a Status column (CLAUDE.md rule 3: every Params field has a row there). The source is
    the "Source / reason" or "Derivation and source" cell."""
    rows, cols = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            cols = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "Field" and "Status" in cells:
            cols = cells
            continue
        if cols and len(cells) > cols.index("Status") and re.fullmatch(r"`\w+`", cells[0]):
            get = lambda name: cells[cols.index(name)] if name in cols and cols.index(name) < len(cells) else ""
            rows[cells[0].strip("`")] = dict(status=get("Status"), unit=get("Unit"),
                                            source=get("Source / reason") or get("Derivation and source"))
    return rows


def tex_status(s):
    s = s.replace("`", "").replace("**", "")
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_")):
        s = s.replace(a, b)
    return s


# CHANGELOG #45: "Source" column. Source keys of params_provenance.md -> bibliography entries
# (CLAUDE.md rule 10: only refs.bib entries); the first printed page cited with the key is kept.
SRC_KEYS = [("Katz-thesis", "katz2018actuator", "Katz-thesis"), ("Go1-bat", "unitree2026go1battery", None),
            ("HG2-fit", "kollmeyer2020hg2", None), ("HG2-tech", "kollmeyer2020hg2", "HG2-tech"),
            ("HG2-data", "kollmeyer2020hg2", None)]
SRC_OVERRIDE = {"kt_out": "derived from \\cite{katz2018actuator}, p.~27",   # the cell says "the thesis"
                "v_min": "data range of \\cite{kollmeyer2020hg2}",           # p. 1 there is the 2.0 V datasheet value
                "i_chg_max": "\\cite{kollmeyer2020hg2} (test report, p.~1)"}  # "ASSUMED" refers to cold() only


# CHANGELOG #51: short reason for each ASSUMED field (from params_provenance.md), instead of a bare
# "assumption"; variant values read from Params
_vl = Params.variant("current_lag").i_loop_bw
_ge = Params.variant("gear_eff_0.9").gear_eff
ASSUMED_REASON = {
    "r_phase": "not reported by the actuator source",
    "p_idle": "no source for the idle power of the drive electronics",
    "gear_eff": f"lossless gear; {_ge:g} in variant \\texttt{{gear\\_eff\\_0.9}}",
    "i_loop_bw": f"ideal current loop, as in the averaged model; {_vl:g}~Hz in variant \\texttt{{current\\_lag}}",
    "v_max": "overvoltage trip above the full-charge voltage",
    "v_chop": "brake-chopper design, between full charge and $V_{\\max}$",
    "v_chop_off": "above the full-charge voltage, so a resting pack is not discharged",
    "r_chop": "brake-resistor design choice",
    "p_aux": "share of on-board computer and sensors",
    "sigma_v_meas": "a few steps of a 12-bit converter",
    "sigma_i_meas": f"{100 * Params().sigma_i_meas / Params().i_bat_max:g}\\,\\% of $I_{{\\lim}}$",
}


def source_cell(field, status, text):
    if field in SRC_OVERRIDE:
        return SRC_OVERRIDE[field]
    st = status.upper()
    if st.startswith("ASSUMED"):
        return ASSUMED_REASON[field]                 # the Status column already says ASSUMED
    if st.startswith("FIXED"):
        return "training setting"
    parts, done = [], set()
    for key, bib, pagekey in SRC_KEYS:
        if key in text and bib not in done:
            done.add(bib)
            page = re.search(re.escape(pagekey) + r"[^|;]*?p\.\s?(\d+)", text) if pagekey else None
            pre = "fit to " if key == "HG2-fit" else ""
            pg = (f" (test report, p.~{page.group(1)})" if key == "HG2-tech" else f", p.~{page.group(1)}") if page else ""
            parts.append(pre + f"\\cite{{{bib}}}" + pg)
    if "ASSUMED" in st:
        parts.append("wiring resistance assumed" if field == "r0" else "partly assumed")
    return "; ".join(parts) if parts else "--"


# paper symbol of each Params field (nomenclature); "--" where the text uses no symbol
SYMBOL = {"gear": "$N$", "kt_out": "$Nk_t$", "pole_pairs": "$p$", "r_phase": "$R$", "l_phase": "$L$",
          "i_max": "$I_{\\mathrm{ph}}$", "tau_max_out": "$\\tau_{\\max}$", "p_idle": "$P_0$", "gear_eff": "--",
          "i_loop_bw": "--", "n_series": "$n_s$", "q_ah": "--", "r0": "$R_0$", "r1": "$R_1$", "c1": "$C_1$",
          "c_bus": "$C$", "v_min": "$V_{\\min}$", "v_obs_ref": "--", "v_max": "$V_{\\max}$",
          "i_bat_max": "$I_{\\lim}$", "i_chg_max": "$I_{\\mathrm{chg}}$", "v_chop": "$V_{c,\\mathrm{on}}$",
          "v_chop_off": "$V_{c,\\mathrm{off}}$", "r_chop": "$R_c$", "p_aux": "$P_{\\mathrm{aux}}$",
          "sigma_v_meas": "$\\sigma_v$", "sigma_i_meas": "$\\sigma_i$"}
# CHANGELOG #58: the main-text table has no code field-name column (field names: params_provenance.md);
# a parameter without a symbol in the text is named in words instead
NAME = {"gear_eff": "Gear efficiency", "i_loop_bw": "Current-loop bandwidth", "q_ah": "Capacity (leg share)",
        "v_obs_ref": "Observation reference"}
prov = provenance_rows()
prow, missing = [], []
for k, v in p.__dict__.items():
    if np.isscalar(v):
        r = prov.get(k)
        if r is None:
            missing.append(k)
            r = dict(status="--", unit="", source="")
        unit = tex_status(r["unit"]).replace("Ω", "$\\Omega$").replace("·", "\\,").replace("σ", "$\\sigma$")
        unit = "--" if unit in ("", "-") else unit
        short = re.sub(r"\s*\(.*?\)|\s+from\b.*|;.*", "", r["status"].replace("`", "")).strip() or "--"
        val = "off (ideal)" if (k == "i_loop_bw" and v == 0) else f"{v:g}"   # 0 Hz means no lag
        unit = "--" if (k == "i_loop_bw" and v == 0) else unit
        sym = SYMBOL.get(k, "--")
        if sym == "--":
            if k not in NAME:
                print(f"tab_params: no symbol or name for {k}")
            sym = NAME.get(k, "--")
        prow.append(f"{sym} & {val} & {unit} & {tex_status(short)} & "
                    f"{source_cell(k, r['status'], r['source'])} \\\\")
if missing:
    print(f"tab_params: no provenance row for {missing} (CLAUDE.md rule 3)")
(GEN / "tab_params.tex").write_text(
    "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{>{\\raggedright\\arraybackslash}p{0.14\\linewidth}rll"
    ">{\\raggedright\\arraybackslash}p{0.39\\linewidth}}\\toprule\nQuantity & Value & Unit & Status & Source \\\\\\midrule\n"
    + "\n".join(prow) + "\n\\bottomrule\\end{tabular}}\n", encoding="utf-8")

n_fig = 0
for f in (ROOT / "results" / "figures").glob("*.pdf"):
    shutil.copy(f, GEN / "figures" / f.name); n_fig += 1

# ---- submission files (journal guide, Engineering Applications of Artificial Intelligence) ----
# Figures as separate files named Figure_1, Figure_2, ... in order of appearance in main.tex.
PAPER = ROOT / "paper"
SUB = GEN / "submission"
if SUB.exists():
    shutil.rmtree(SUB)                                  # never leave stale numbering
SUB.mkdir(parents=True)
order = []
for sec in re.findall(r"\\input\{(sections/[^}]+)\}", (PAPER / "main.tex").read_text(encoding="utf-8")):
    src = PAPER / (sec + ".tex")
    for name in re.findall(r"\\genfig\{([^}]+)\}", src.read_text(encoding="utf-8")):
        if name not in order:
            order.append(name)
fig_lines = []
for i, name in enumerate(order, 1):
    f = GEN / "figures" / f"{name}.pdf"
    if f.exists():
        shutil.copy(f, SUB / f"Figure_{i}.pdf")
        fig_lines.append(f"Figure_{i}.pdf  {name}.pdf")
    else:
        fig_lines.append(f"Figure_{i}  {name}: NOT GENERATED")
# graphical abstract: a separate file (Elsevier: >= 1328 x 531 px at >= 300 dpi; TIFF, EPS or PDF)
for ext in ("pdf", "tif"):
    ga = ROOT / "results" / "figures" / f"GA_graphical_abstract.{ext}"
    if ga.exists():
        shutil.copy(ga, SUB / f"graphical_abstract.{ext}")
        fig_lines.append(f"graphical_abstract.{ext}  GA_graphical_abstract.{ext}")
(SUB / "figure_list.txt").write_text("\n".join(fig_lines) + "\n")
print(f"submission figures: {', '.join(fig_lines)}")
# Highlights (separate editable file, "highlights" in its name): 3-5 bullets, <= 85 characters each
hl = PAPER / "highlights.txt"
if hl.exists():
    items = [h for h in hl.read_text(encoding="utf-8").splitlines() if h.strip()]
    bad = [h for h in items if len(h) > 85]
    ok = 3 <= len(items) <= 5 and not bad
    print(f"highlights.txt: {len(items)} bullets, longest {max(map(len, items))} characters: "
          + ("OK" if ok else f"VIOLATES the guide (3-5 bullets, <= 85 characters): {bad}"))
# Claims in highlights.txt, checked against the summaries (printed; CHANGELOG #34):
#  (a) without the filter, every policy that hopped breached a limit: per seed and SOC (main
#      evaluations) and per setting (baseline_ws under the variants), no row with hops > 0 and zero
#      time below V_min and above I_lim;
#  (b) with an exact battery model (no est_noise, no r0_scale), policies trained with the filter never
#      fell and never breached a limit (main evaluations, variants, barrier rates, EKF nominal);
#  (c) the filter added only at test time removed every violation, and some episodes of the
#      baseline policies (trained without the filter or a penalty) fell;
#  (d) the EKF rows have no breach under r0_scale, and do under est_noise.
if ps is not None and abl.exists():
    nf = ps[ps.test_mode != "filter"]
    a_bad = nf[(nf.n_hops > 0) & (nf.uv_s + nf.oc_s == 0)]
    abv = ab[(ab.method == "baseline_ws") & (ab.ablation == "variant")]
    a_bad2 = abv[(abv.n_hops_mean > 0) & (abv.uv_s_mean + abv.oc_s_mean == 0)]
    trained = ps[ps.method.isin(["filter", "filter_ws"])]
    exact = ab[(ab.method == "filter_ws") & ab.ablation.isin(["base", "variant", "gamma"])
               | ((ab.ablation == "estimator") & ab.setting.str.endswith("nominal"))]
    b_ok = (trained.n_fell.sum() == 0 and (trained.uv_s + trained.oc_s + trained.collapse_s).sum() == 0
            and exact.n_fell.sum() == 0 and (exact[["uv_s_mean", "oc_s_mean", "ov_s_mean", "collapse_s_mean"]].sum().sum() == 0))
    ft = ps[ps.method.str.endswith("+filter@test")]
    # CHANGELOG #56: highlight 4 names the baseline policies (no filter or penalty in training):
    # every test-time-filter row has no violation, and the baseline ones have falls.
    # CHANGELOG #57: highlight 4 now names baseline and penalty policies: also no violation or collapse
    # in the penalty stress tests (filter at test), and falls there.
    fb = ft[ft.method.isin(["baseline+filter@test", "baseline_ws+filter@test"])]
    c_ok = ((ft.uv_s + ft.oc_s + ft.collapse_s).sum() == 0 and fb.n_fell.sum() > 0
            and PEN_M is not None and macros.get("pst_nviol") == "0" and int(macros.get("pst_falls", "0")) > 0)
    er = ab[(ab.ablation == "estimator") & ~ab.setting.str.startswith("true")]
    d_ok = (er[er.setting.str.contains("r0_scale")].oc_s_mean.sum() == 0
            and er[er.setting.str.contains("est_noise")].oc_s_mean.sum() > 0)
    print("highlight claims: (a) hopped without filter -> breach:", "OK" if a_bad.empty and a_bad2.empty else
          f"FAILS {len(a_bad) + len(a_bad2)} rows", "| (b) filter-trained, exact model:", "OK" if b_ok else "FAILS",
          "| (c) filter at test:", "OK" if c_ok else "FAILS", "| (d) EKF:", "OK" if d_ok else "FAILS")

    # (k) CHANGELOG #58: abstract and conclusion (episode totals: computed before results.tex is
    #     written, below the figure-caption macros).
    # abstract (c): baseline policies with the filter at test fell in every stress-test setting in which
    # they were evaluated (variants and barrier rates); one of the n_seeds penalty seeds, in three
    # settings; the penalty policies hopped higher (main evaluation: every SOC; stress tests: most
    # rows). The largest relative difference is printed: up to 17 %, so the text does not say "slightly".
    _bst = ab[ab.ablation.isin(["variant", "gamma"]) & (ab.method == "baseline_ws+filter@test")]
    _bset = _bst.groupby("setting").n_fell.sum()
    _mn = ps[ps.method.isin([PEN_M, "filter_ws"])].groupby(["method", "soc"]).height_max.mean().unstack(0)
    _dmain = _mn[PEN_M] - _mn["filter_ws"]
    _pv8 = abp[abp.ablation.isin(["variant", "gamma"])].set_index(["setting", "soc"])
    _fv8 = ab[ab.ablation.isin(["variant", "gamma"]) & (ab.method == "filter_ws")].set_index(["setting", "soc"]).loc[_pv8.index]
    _rel = max(((_pv8.height_max_mean - _fv8.height_max_mean) / _fv8.height_max_mean).max(),
               (_dmain / _mn["filter_ws"]).max())
    _kc = {
        "(a) every policy run with the filter met both limits (exact parameters)": (
            b_ok and c_ok and macros.get("abl_fm_nviol") == "0" and macros.get("pst_nviol") == "0"),
        "(b) filter-trained: no fall in any evaluation episode": _fws == 0 and _fsc == 0 and _nws > 0,
        "(c) baseline + test filter fell in every variant and barrier rate": (
            len(_bset) == int(macros["abl_nvar"]) + ab[ab.ablation == "gamma"].setting.nunique()
            and (_bset > 0).all()),
        "(c) one of three penalty seeds, in three settings": (
            macros.get("n_seeds") == "3" and macros.get("pst_nfallset") == "3"
            and macros.get("pst_fall_seed") is not None),
        "(c) penalty higher: main evaluation every SOC, most stress rows": (
            (_dmain > 0).all() and int(macros["pst_nhigher"]) > int(macros["pst_nrows"]) / 2),
    }
    print(f"   largest relative height difference, penalty + test filter vs filter-trained: {100 * _rel:.1f} %")
    for _k, _v in _kc.items():
        if not _v:
            print("   claim (k) fails:", _k)
    print("claim (k) (abstract and conclusion: filter-trained episodes, test-time-filter falls):",
          "OK" if all(_kc.values()) else "FAILS")

# (e) Results text on Figure F6_tradeoff (CHANGELOG #39): at every SOC, every method with zero mean
#     time below V_min and above I_lim (each in its own training mode, or the baseline policies with
#     the filter at test, and the penalty w = 0.5 policies with the filter at test) has a lower mean max height than every baseline or penalty policy that
#     breaches a limit. Read from per_seed.csv, the figure's source.
_psf = SUMMARY / "per_seed.csv"
if _psf.exists():
    _t = pd.read_csv(_psf)
    # CHANGELOG #63: the dot plot also shows the penalty w = 0.5 policies with the test-time filter
    _t = _t[(_t.method == _t.train) | _t.method.isin(["baseline+filter@test", "baseline_ws+filter@test",
                                                      "penalty_w0.5_ws+filter@test"])]
    _g = _t.groupby(["method", "soc"])[["uv_s", "oc_s", "height_max"]].mean().reset_index()
    e_ok = True
    for _soc, _x in _g.groupby("soc"):
        _z = _x[(_x.uv_s == 0) & (_x.oc_s == 0)]
        _v = _x[((_x.uv_s > 0) | (_x.oc_s > 0)) & _x.method.str.match(r"(baseline|penalty)")]
        e_ok = e_ok and not _z.empty and not _v.empty and _z.height_max.max() < _v.height_max.min()
    print("trade-off claim (e) (no-violation methods lower than every breaching baseline/penalty policy):",
          "OK" if e_ok else "FAILS")

zpath = ROOT / "paper_overleaf.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((ROOT / "paper").rglob("*")):
        if (f.is_file() and f.suffix not in (".aux", ".log", ".out", ".bbl", ".blg", ".fls", ".fdb_latexmk", ".gz")
                and "submission" not in f.relative_to(ROOT / "paper").parts     # copies for upload only
                and f.name not in ("release_checklist.md", "ref_audit.md")):   # not part of the paper
            z.write(f, f.relative_to(ROOT / "paper"))
print(f"{len(macros)} macros, {n_fig} figures -> {zpath}")
