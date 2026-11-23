"""Fit OCV(SOC) and a one-RC Thevenin model (R0, R1, C1) to LG 18650HG2 cell data.

Data: Kollmeyer, Naguib, Skells, "LG 18650HG2 Li-ion Battery Data", Mendeley Data,
version 2, DOI 10.17632/b5mj79w5w9.2, CC BY 4.0. Files in data/raw/lg_hg2/, listed
with SHA-256 in data/raw/lg_hg2/MANIFEST.csv.

    python -I scripts/fit_battery.py

Per temperature (25 degC: cell test 549; 0 degC: cell test 585), all at CELL level:
  Q         charge removed by the C/20 discharge to the 2.8 V cut-off. SOC = 1 at
            the start of each test (full charge), SOC = 0 at Q removed.
  OCV(SOC)  mean of the C/20 discharge branch and the constant-current part of the
            C/20 charge branch on a common SOC grid. Averaging cancels the C/20
            resistive drop and most of the hysteresis. Checked against the relaxed
            voltage at the end of every 1 h rest in the HPPC test.
  R0,R1,C1  least-squares fit of
                V = OCV(SOC) - R0 I - v1,   dv1/dt = -v1/(R1 C1) + I/C1
            (I > 0 discharging; exact zero-order-hold update between samples) to
            every full 10 s HPPC discharge pulse (3, 6, 12, 18 A) plus the 40 s rest
            after it, for pulse-start SOC in [SOC_LO, 1]. v1 = 0 is assumed at each
            pulse start (the cell has rested about 10 min). Each pulse is also fitted
            on its own to show how the parameters vary with SOC and current.
  check     the fitted model is run through the whole HPPC test (all discharge and
            charge pulses and rests) from its first sample, with no re-fitting.

Writes results/battery_fit/: fit_summary.json, ocv_<T>.csv, ocv_check.csv,
pulse_fits.csv, and figures fig_*.png / fig_*.pdf.
"""
import csv
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "lg_hg2"
OUT = ROOT / "results" / "battery_fit"
TESTS = {  # label -> (HPPC csv, C/20 csv)
    "25degC": (RAW / "25degC" / "549_HPPC.csv", RAW / "25degC" / "549_C20DisCh.csv"),
    "0degC": (RAW / "0degC" / "585_HPPC.csv", RAW / "0degC" / "585_C20DisCh.csv"),
}
SOC_LO = 0.15            # lower SOC bound of the fit; matches leg_env's default soc_range
# Pulse currents used in the global fit, per temperature (A, nominal pulses 3/6/12/18 A).
# 0 degC: only >= 12 A pulses (decision 2026-11-18, CHANGELOG #8). At 0 degC the cell's
# resistance falls with current, so a linear 1-RC model fitted to all currents is wrong at
# the high currents that matter for bus sag. The all-currents fit is kept in the summary.
MIN_PULSE_A = {"25degC": 0.0, "0degC": 11.5}
GRID = np.round(np.concatenate([np.arange(0.0, 0.1, 0.025), np.arange(0.1, 1.0001, 0.05)]), 4)
PULSE_DUR = (9.5, 10.5)  # s, full-length HPPC discharge pulses only (truncated ones excluded)
PRE_S = 1.0              # s of pre-pulse rest included in each window
BOUNDS = ([1e-4, 1e-5, 0.05], [0.5, 0.5, 300.0])   # R0 [ohm], R1 [ohm], tau [s]

# ---- figure style (dataviz reference palette; first three categorical slots only) ----
SURF, INK, INK2, MUTED, GRIDC, AXISC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TCOL = {"25degC": "#2a78d6", "0degC": "#eb6834"}     # slot 1 blue, slot 2 orange
TNAME = {"25degC": "25 °C", "0degC": "0 °C"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.5, "font.family": "sans-serif",
})


# ------------------------------------------------------------------ data
def load(path):
    """Digatron CSV -> DataFrame with t [s] (from 'Prog Time', may exceed 24 h)."""
    with open(path, encoding="latin-1") as f:
        for i, line in enumerate(f):
            if line.startswith("Time Stamp,Step"):
                break
        else:
            raise ValueError(f"no header row in {path}")
    d = pd.read_csv(path, skiprows=list(range(i)) + [i + 1], encoding="latin-1")
    h = d["Prog Time"].str.split(":", expand=True).astype(float)
    d["t"] = h[0] * 3600 + h[1] * 60 + h[2]
    d["seg"] = ((d.Status != d.Status.shift()) | (d.Step != d.Step.shift())).cumsum()
    return d.reset_index(drop=True)


def segments(d):
    g = d.groupby("seg")
    s = pd.DataFrame({"st": g.Status.first(), "i0": g.apply(lambda x: x.index[0]),
                      "i1": g.apply(lambda x: x.index[-1]), "t0": g.t.first(), "t1": g.t.last(),
                      "I": g.Current.mean(), "proc": g.Procedure.first()})
    s["dur"] = s.t1 - s.t0
    return s.reset_index(drop=True)


def cum_ah(t, i):
    """Cumulative |charge| in Ah by the trapezoid rule."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(i[1:]) + np.abs(i[:-1])) * np.diff(t))]) / 3600.0


# ------------------------------------------------------------------ OCV
def c20_ocv(path):
    d = load(path); s = segments(d)
    dch = s[s.st == "DCH"].sort_values("dur").iloc[-1]
    chg = s[s.st == "CHA"].sort_values("dur").iloc[-1]
    a, b = int(dch.i0), int(dch.i1)
    ah_d = cum_ah(d.t.values[a:b + 1], d.Current.values[a:b + 1]); q = float(ah_d[-1])
    soc_d, v_d = 1.0 - ah_d / q, d.Voltage.values[a:b + 1]
    a, b = int(chg.i0), int(chg.i1)
    ah_c = cum_ah(d.t.values[a:b + 1], d.Current.values[a:b + 1]); q_c = float(ah_c[-1])
    soc_c, v_c = ah_c / q_c, d.Voltage.values[a:b + 1]
    cc = v_c < 4.195                                    # constant-current part only
    soc_c, v_c = soc_c[cc], v_c[cc]
    vd = np.interp(GRID, soc_d[::-1], v_d[::-1])        # discharge branch covers [0, 1]
    vc = np.interp(GRID, soc_c, v_c)
    both = GRID <= soc_c.max()
    ocv = np.where(both, 0.5 * (vd + vc), np.nan)
    hi = np.where(both)[0][-3:]                         # above the charge CC end: discharge
    half_gap = float(np.mean(0.5 * (vc[hi] - vd[hi])))  # branch + half the last measured gap
    ocv = np.where(both, ocv, vd + half_gap)
    return dict(q_ah=q, q_chg_ah=q_c, soc_cc_end=float(soc_c.max()), half_gap_hi_V=half_gap,
                ocv=ocv, soc_d=soc_d, v_d=v_d, soc_c=soc_c, v_c=v_c,
                t_mean_C=float(d.Temperature.iloc[int(dch.i0):int(dch.i1)].mean()))


# ------------------------------------------------------------------ RC model
def simulate(params, t, i_dis, ocv_t):
    """V = OCV - R0 I - v1 with exact ZOH update of v1 (I held at the interval mean)."""
    r0, r1, tau = params
    v1 = np.zeros_like(t)
    a = np.exp(-np.diff(t) / tau)
    i_mid = 0.5 * (i_dis[1:] + i_dis[:-1])
    for k in range(1, len(t)):
        v1[k] = a[k - 1] * v1[k - 1] + r1 * (1.0 - a[k - 1]) * i_mid[k - 1]
    return ocv_t - r0 * i_dis - v1


def pulse_windows(d, s, q, ocv_fn):
    wins = []
    pul = s[(s.st == "DCH") & (s.dur >= PULSE_DUR[0]) & (s.dur <= PULSE_DUR[1])
            & s.proc.str.startswith("HPPC")]
    for j, p in pul.iterrows():
        nxt = s.iloc[j + 1]
        if nxt.st != "PAU":
            continue
        t0 = p.t0
        a = int(d.index[(d.t >= t0 - PRE_S) & (d.index < p.i0)].min())
        b = int(nxt.i1)
        w = d.iloc[a:b + 1]
        t = w.t.values - t0
        soc = 1.0 + w.Capacity.values / q
        pre = t < 0
        v_pre = float(w.Voltage.values[pre].mean())
        soc_pre = float(soc[pre].mean())
        # baseline = measured pre-pulse voltage; OCV drift follows the charge removed
        ocv_t = v_pre + ocv_fn(soc) - ocv_fn(soc_pre)
        wins.append(dict(t=t, i=-w.Current.values, v=w.Voltage.values, ocv_t=ocv_t,
                         soc=soc_pre, i_pulse=float(-p.I), temp=float(w.Temperature.mean())))
    return wins


def fit(wins, x0):
    res = lambda x: np.concatenate([simulate(x, w["t"], w["i"], w["ocv_t"]) - w["v"] for w in wins])
    r = least_squares(res, x0, bounds=BOUNDS, x_scale=[0.01, 0.01, 5.0], method="trf")
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def by_current(x, ws):
    """Per nominal pulse current: pooled RMSE over the windows, and the median signed error
    at the last pulse sample (model - measured; > 0 means the model under-predicts sag)."""
    out = {}
    for cur in sorted({round(w["i_pulse"]) for w in ws}):
        g = [w for w in ws if round(w["i_pulse"]) == cur]
        res = [simulate(x, w["t"], w["i"], w["ocv_t"]) - w["v"] for w in g]
        end = [r[np.where(w["t"] <= 10.0)[0][-1]] for r, w in zip(res, g)]
        out[str(cur)] = dict(n=len(g), rmse_mV=1e3 * float(np.sqrt(np.mean(np.concatenate(res) ** 2))),
                             end_of_pulse_err_median_mV=1e3 * float(np.median(end)))
    return out


def x0_guess(w):
    k = np.argmax(w["t"] >= 0)                          # first pulse sample
    kend = np.where(w["t"] <= 10.0)[0][-1]
    di = max(w["i"][k], 1e-3)
    r0 = max((w["ocv_t"][k] - w["v"][k]) / di, 1e-3)
    r1 = max((w["v"][k] - w["v"][kend]) / di, 1e-4)
    return np.array([r0, r1, 5.0])


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {r["path"]: r["sha256"] for r in csv.DictReader(open(RAW / "MANIFEST.csv", encoding="utf-8-sig"))}
    summary = dict(dataset="Kollmeyer, Naguib, Skells, LG 18650HG2 Li-ion Battery Data, Mendeley Data v2, "
                           "DOI 10.17632/b5mj79w5w9.2, CC BY 4.0",
                   script="scripts/fit_battery.py", level="single cell (1S1P)", soc_fit_range=[SOC_LO, 1.0],
                   ocv_grid_soc=GRID.tolist(), temps={})
    pulse_rows, check_rows, fig_data = [], [], {}

    for lab, (f_hppc, f_c20) in TESTS.items():
        c = c20_ocv(f_c20)
        ocv_fn = lambda soc, c=c: np.interp(soc, GRID, c["ocv"])
        d = load(f_hppc); s = segments(d)
        q = c["q_ah"]

        # OCV check against relaxed HPPC rests (>= 1500 s) and the test start
        rests = s[(s.st == "PAU") & (s.dur > 1500) & s.proc.str.startswith("HPPC")]
        pts = [("start", 1.0 + d.Capacity.iloc[0] / q, float(d.Voltage.iloc[0]))]
        pts += [("rest_1h", 1.0 + d.Capacity.iloc[int(r.i1)] / q, float(d.Voltage.iloc[int(r.i1)]))
                for _, r in rests.iterrows()]
        errs = []
        for kind, soc, v in pts:
            e = 1e3 * (v - ocv_fn(soc)); errs.append(e)
            check_rows.append(dict(temp=lab, kind=kind, soc=round(soc, 5), v_meas=v,
                                   v_ocv_table=round(float(ocv_fn(soc)), 5), err_mV=round(e, 2)))
        errs = np.array(errs)

        # pulse windows, global and per-pulse fits
        wins = pulse_windows(d, s, q, ocv_fn)
        wins_rng = [w for w in wins if w["soc"] >= SOC_LO]
        sel = [w for w in wins_rng if w["i_pulse"] >= MIN_PULSE_A[lab]]
        x0 = np.median([x0_guess(w) for w in sel], axis=0)
        xg, rmse_g = fit(sel, x0)
        alt = None
        if len(sel) < len(wins_rng):                       # keep the all-currents fit for the record
            xa, rmse_a = fit(wins_rng, np.median([x0_guess(w) for w in wins_rng], axis=0))
            alt = dict(r0_ohm=float(xa[0]), r1_ohm=float(xa[1]), tau_s=float(xa[2]), c1_F=float(xa[2] / xa[1]),
                       n_pulses_fit=len(wins_rng), rmse_fit_windows_mV=1e3 * rmse_a,
                       by_pulse_current=by_current(xa, wins_rng))
        per_rmse = []
        for w in wins:
            xw, rw = fit([w], x0_guess(w))
            rg = float(np.sqrt(np.mean((simulate(xg, w["t"], w["i"], w["ocv_t"]) - w["v"]) ** 2)))
            per_rmse.append(rg)
            pulse_rows.append(dict(temp=lab, soc=round(w["soc"], 4), i_pulse_A=round(w["i_pulse"], 2),
                                   cell_temp_C=round(w["temp"], 2),
                                   in_fit=bool(w["soc"] >= SOC_LO and w["i_pulse"] >= MIN_PULSE_A[lab]),
                                   r0_ohm=xw[0], r1_ohm=xw[1], tau_s=xw[2], c1_F=xw[2] / xw[1],
                                   rmse_own_fit_mV=1e3 * rw, rmse_global_fit_mV=1e3 * rg))

        # whole-test check with the global parameters
        h = d[d.Procedure.str.startswith("HPPC")]
        soc_t = 1.0 + h.Capacity.values / q
        v_sim = simulate(xg, h.t.values, -h.Current.values, ocv_fn(soc_t))
        err = 1e3 * (v_sim - h.Voltage.values)
        loaded = np.abs(h.Current.values) > 0.1
        in_rng = soc_t >= SOC_LO
        rm = lambda m: float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")

        summary["temps"][lab] = dict(
            hppc_file=f_hppc.relative_to(ROOT).as_posix(), c20_file=f_c20.relative_to(ROOT).as_posix(),
            hppc_sha256=manifest.get(f_hppc.relative_to(ROOT).as_posix()),
            c20_sha256=manifest.get(f_c20.relative_to(ROOT).as_posix()),
            c20_mean_cell_temp_C=c["t_mean_C"], hppc_mean_cell_temp_C=float(h.Temperature.mean()),
            q_c20_discharge_Ah=q, q_c20_charge_Ah=c["q_chg_ah"], c20_charge_cc_end_soc=c["soc_cc_end"],
            ocv_V=c["ocv"].round(5).tolist(), ocv_hi_soc_half_gap_V=c["half_gap_hi_V"],
            ocv_check_vs_hppc_rests=dict(n=int(len(errs)), rmse_mV=float(np.sqrt(np.mean(errs ** 2))),
                                         mean_mV=float(errs.mean()), max_abs_mV=float(np.abs(errs).max())),
            r0_ohm=float(xg[0]), r1_ohm=float(xg[1]), tau_s=float(xg[2]), c1_F=float(xg[2] / xg[1]),
            # apparent resistance after a 10 s constant-current pulse, for comparison
            # with 10 s pulse-resistance figures such as HG2-tech p. 4
            r_10s_ohm=float(xg[0] + xg[1] * (1.0 - np.exp(-10.0 / xg[2]))),
            n_pulses_fit=len(sel), n_pulses_total=len(wins), fit_min_pulse_A=MIN_PULSE_A[lab],
            rmse_fit_windows_mV=1e3 * rmse_g,
            by_pulse_current_soc_ge_lo=by_current(xg, wins_rng),
            fit_all_currents=alt,
            rmse_per_pulse_global_params_mV=dict(median=float(np.median(per_rmse) * 1e3),
                                                 max=float(np.max(per_rmse) * 1e3)),
            fulltest_rmse_mV=dict(all=rm(np.ones_like(in_rng)), soc_ge_lo=rm(in_rng),
                                  soc_ge_lo_loaded=rm(in_rng & loaded), soc_ge_lo_rest=rm(in_rng & ~loaded)),
        )
        with open(OUT / f"ocv_{lab}.csv", "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["soc", "ocv_V"]); wr.writerows(zip(GRID, c["ocv"].round(5)))
        fig_data[lab] = dict(c=c, pts=pts, ocv_fn=ocv_fn, wins=wins, xg=xg, t=h.t.values, v=h.Voltage.values,
                             v_sim=v_sim, err=err, soc_t=soc_t)

    pd.DataFrame(pulse_rows).to_csv(OUT / "pulse_fits.csv", index=False, float_format="%.6g")
    pd.DataFrame(check_rows).to_csv(OUT / "ocv_check.csv", index=False)
    (OUT / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(fig_data, pd.DataFrame(pulse_rows))
    for lab, r in summary["temps"].items():
        print(f"{lab}: Q={r['q_c20_discharge_Ah']:.4f} Ah  R0={1e3*r['r0_ohm']:.2f} mOhm  "
              f"R1={1e3*r['r1_ohm']:.2f} mOhm  tau={r['tau_s']:.2f} s  C1={r['c1_F']:.0f} F  "
              f"fit RMSE={r['rmse_fit_windows_mV']:.2f} mV  "
              f"full-test RMSE (SOC>={SOC_LO})={r['fulltest_rmse_mV']['soc_ge_lo']:.2f} mV  "
              f"OCV vs rests RMSE={r['ocv_check_vs_hppc_rests']['rmse_mV']:.2f} mV")
    print("wrote", OUT)


# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(fd, pf):
    labs = list(fd)
    # F-a: OCV and its check
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    for j, lab in enumerate(labs):
        c, col = fd[lab]["c"], TCOL[lab]
        a = ax[0, j]
        a.plot(c["soc_d"], c["v_d"], color=MUTED, lw=1.0, ls="--", label="C/20 discharge")
        a.plot(c["soc_c"], c["v_c"], color=MUTED, lw=1.0, label="C/20 charge (CC part)")
        a.plot(GRID, c["ocv"], color=col, lw=1.5, marker="o", ms=3, label="OCV table (mean of branches)")
        sp = np.array([p[1] for p in fd[lab]["pts"]]); vp = np.array([p[2] for p in fd[lab]["pts"]])
        a.plot(sp, vp, ls="none", marker="D", ms=5, mfc=SURF, mec=INK, mew=1.0, label="HPPC, end of 1 h rest")
        a.set_title(f"{TNAME[lab]}: open-circuit voltage"); a.set_ylabel("cell voltage (V)")
        a.set_ylim(2.75, 4.25)
        if j == 0:
            a.legend(loc="lower right")
        b = ax[1, j]
        b.axhline(0, color=AXISC, lw=1.0)
        b.plot(sp, 1e3 * (vp - fd[lab]["ocv_fn"](sp)), ls="none", marker="D", ms=5, mfc=SURF, mec=INK, mew=1.0)
        b.set_ylabel("rest − table (mV)"); b.set_xlabel("state of charge")
    ax[0, 0].set_xlim(-0.02, 1.02)
    fig.align_ylabels(ax[:, 0]); save(fig, "fig_ocv")

    # F-b: example pulse fits, 18 A pulses at three SOC levels
    targets = [0.9, 0.5, SOC_LO + 0.05]
    fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.2), sharey="row")
    for r, lab in enumerate(labs):
        ws = [w for w in fd[lab]["wins"] if abs(w["i_pulse"] - 18.0) < 0.5]
        for cidx, st in enumerate(targets):
            w = min(ws, key=lambda w: abs(w["soc"] - st))
            vm = simulate(fd[lab]["xg"], w["t"], w["i"], w["ocv_t"])
            rms = 1e3 * np.sqrt(np.mean((vm - w["v"]) ** 2))
            a = ax[r, cidx]
            a.plot(w["t"], w["v"], ls="none", marker="o", ms=2.2, color=MUTED, label="measured")
            a.plot(w["t"], vm, color=TCOL[lab], lw=1.5, label=f"1-RC model ({TNAME[lab]} global fit)")
            a.set_title(f"{TNAME[lab]}, SOC {w['soc']:.2f}, {w['i_pulse']:.0f} A pulse\nRMSE {rms:.1f} mV",
                        fontsize=9)
            if cidx == 0:
                a.set_ylabel("cell voltage (V)"); a.legend(loc="lower right")
            if r == 1:
                a.set_xlabel("time from pulse start (s)")
    fig.tight_layout(); save(fig, "fig_pulse_fits")

    # F-c: per-pulse parameters vs SOC, with the global value of each temperature
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
    for lab in labs:
        p = pf[pf.temp == lab]
        g = fd[lab]["xg"]
        for a, key, gv, sc, yl in ((ax[0], "r0_ohm", g[0], 1e3, "R0 (mΩ)"), (ax[1], "r1_ohm", g[1], 1e3, "R1 (mΩ)"),
                                   (ax[2], "tau_s", g[2], 1.0, "τ = R1·C1 (s)")):
            a.plot(p.soc, p[key] * sc, ls="none", marker="o", ms=4, mfc=TCOL[lab], mec=SURF, mew=0.8,
                   label=f"{TNAME[lab]}, each pulse")
            a.axhline(gv * sc, color=TCOL[lab], lw=1.5, label=f"{TNAME[lab]}, global fit")
            a.set_ylabel(yl); a.set_xlabel("SOC at pulse start")
    for a in ax:
        a.axvspan(-0.02, SOC_LO, color=GRIDC, alpha=0.6, lw=0); a.set_xlim(-0.02, 1.02)
    ax[0].set_title("series resistance")
    ax[1].set_title("RC-pair resistance"); ax[2].set_title("RC time constant")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    hnd, lbl = ax[0].get_legend_handles_labels()
    fig.legend(hnd, lbl, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.01, -0.02, f"Shaded: SOC < {SOC_LO}, excluded from the global fit. Points: all four pulse currents.",
             color=INK2, fontsize=8)
    save(fig, "fig_rc_params")

    # F-d: whole HPPC test, model run open-loop with the global parameters
    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1]))
    for j, lab in enumerate(labs):
        f = fd[lab]; th = (f["t"] - f["t"][0]) / 3600.0
        a = ax[0, j]
        a.plot(th, f["v"], color=MUTED, lw=0.6, label="measured")
        a.plot(th, f["v_sim"], color=TCOL[lab], lw=0.8, label="1-RC model, open loop")
        a.set_title(f"{TNAME[lab]}: whole HPPC test"); a.set_ylabel("cell voltage (V)")
        if j == 0:
            a.legend(loc="lower left")
        b = ax[1, j]
        b.plot(th, f["err"], color=TCOL[lab], lw=0.6); b.axhline(0, color=AXISC, lw=1.0)
        lo = f["soc_t"] < SOC_LO
        if lo.any():
            b.axvspan(th[lo][0], th[-1], color=GRIDC, alpha=0.6, lw=0)
            a.axvspan(th[lo][0], th[-1], color=GRIDC, alpha=0.6, lw=0)
        b.set_ylabel("model − measured (mV)"); b.set_xlabel("time (h)")
    fig.text(0.01, -0.02, f"Shaded: SOC < {SOC_LO}.", color=INK2, fontsize=8)
    fig.tight_layout(); save(fig, "fig_fulltest")


if __name__ == "__main__":
    sys.exit(main())
