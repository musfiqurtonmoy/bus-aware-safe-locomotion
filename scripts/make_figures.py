"""Paper figures as vector PDFs in results/figures/ (names used by \\genfig in paper/), from
existing result files only (no simulation is run):

    python scripts/make_figures.py [--png_dir DIR] [--only F0_framework]

F0_framework         (CHANGELOG #62) overall framework, four panels: public sources -> electrically
                     coupled plant -> learning with the filter -> evaluation, and a strip mapping
                     them to contributions 1-4. Schematic only, no result numbers; drawn at the
                     text width (390 pt), fonts >= 7 pt, every text checked to lie inside its box.
F1_validation       Step 2: averaged vs switching-level model, scenarios S1-S3; bus voltage,
                     battery current, drive power; transient and steady windows marked.
                     Source: results/validation/S*.csv, metrics.json.
F2_traces            one episode at SOC 0.2 (reset seed 0), baseline_ws_s0 vs filter_ws_s0:
                     rail height above start, lowest bus voltage and highest battery current
                     in each control step with the limits, fraction of substeps with the filter
                     active. Source: results/<run>/rollout/rollout_<mode>_soc0.2.csv.
F3_violation_vs_soc  time below v_min and above i_bat_max per episode vs SOC for baseline_ws,
                     penalty_w*_ws (each weight) and filter_ws, each in its own training mode;
                     marker = mean over seeds, error bar = min-max over seeds. Methods with one
                     seed: hollow markers, dashed line, "(1 seed)" in the legend. Points where
                     any episode ended in a fall: black ring and "fell k/n" (episodes over seeds).
                     Source: results/summary/per_seed.csv (scripts/aggregate.py).
F4_architecture      schematic: policy -> PD -> filter (budget + projection) -> drives -> bus/
                     battery and MuJoCo, the EKF loop, and the signals each block uses (as in
                     busaware/leg_env.py). Time steps from HoppingLegEnv.
F5_budget            Lemma 1 / Proposition 1 at one illustrative state (SOC 0.2, battery at rest
                     with battery current 0.8 I_lim, bus at E - R0 I): F(V'), x*, the two
                     barrier bounds, V_req and P_bar, computed from Params and the env's gamma
                     and margins; P_bar is checked against safety_filter.power_budget.
F6_tradeoff          (dot plot, CHANGELOG #63) main evaluation, warm-started policies: max height per
                     method, one panel per SOC; rows grouped into "violates a limit" (hollow) and
                     "within both limits" (filled), the grouping asserted from the data at every
                     SOC; mean over seeds, bars = std over seeds; "fell k/n" where any episode fell.
                     Drawn at the text width, fonts >= 7 pt. Source: results/summary/per_seed.csv.
F8_settings          (split from F6, CHANGELOG #49) the 10-episode settings of the warm-started
                     policies (joint filter base, per-joint limit, margins): max height against the
                     fraction of substeps with the filter or limiter active, mean and std over seeds,
                     falls as "fell k/n". Source: results/summary/ablations.csv.
F7_training          training curves of the warm-started baseline, filter, penalty (w = 0.5) and
                     SAC-Lagrangian runs, each seed: moving mean of episode return (each run's own
                     training reward) and episode length. Source: results/<run>/monitor.monitor.csv.
GA_graphical_abstract  problem -> method -> result, 13.28 x 5.31 cm (Elsevier: >= 1328 x 531 px
                     at >= 300 dpi, 500:200 ratio), Arial. Numbers from per_seed.csv and the
                     baseline_ws_s0 rollout at SOC 0.2.
Colours: dataviz reference palette (light), categorical slots in fixed order; every series also
has its own marker and line style, so the figures read in greyscale. F6/F7 (scatter, small
multiples): only the first three slots (all-pairs safe) plus ink, and a grey lightness ramp for
the penalty weights; markers carry the method as well.
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from busaware.params import Params  # noqa: E402
from busaware.leg_env import HoppingLegEnv  # noqa: E402
from busaware.battery import Battery  # noqa: E402
from busaware.bus import chopper_conductance  # noqa: E402
from busaware.safety_filter import power_budget  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = ROOT / "results" / "figures"
# dataviz reference palette, light mode
SURF, INK, INK2, MUTED, GRIDC, AXISC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7.5,
    "font.family": "sans-serif", "pdf.fonttype": 42, "lines.linewidth": 1.2,
})
P = Params()
META = {}   # figure settings quoted in captions -> results/figures/figures_meta.json (export_paper macros)


def save(fig, name, png_dir):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    if png_dir:
        pathlib.Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(pathlib.Path(png_dir) / f"{name}.png", dpi=170)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def f1_validation(png_dir):
    vdir = ROOT / "results" / "validation"
    m = json.loads((vdir / "metrics.json").read_text())
    t_step, t_steady, t_end = 1e3 * m["t_step_s"], 1e3 * m["t_steady_from_s"], 1e3 * m["t_end_s"]
    scen = [("S1_standstill_step", "S1: step at standstill"),
            ("S2_high_speed_step", "S2: step near the voltage limit"),
            ("S3_braking", "S3: braking step")]
    rows = [("v_bus", "bus voltage (V)"), ("i_bat", "battery current (A)"), ("p_elec", "drive power (W)")]
    fig, ax = plt.subplots(3, 3, figsize=(7.0, 5.6), sharex=True)
    for c, (key, title) in enumerate(scen):
        d = pd.read_csv(vdir / f"{key}.csv", comment="#")
        t = 1e3 * d["t_s"]
        for r, (sig, lab) in enumerate(rows):
            a = ax[r, c]
            a.axvspan(t_step, t_steady, color=GRIDC, alpha=0.7, lw=0, zorder=0)
            a.axvline(t_step, color=AXISC, lw=0.8, ls=":", zorder=1)
            a.plot(t, d[f"{sig}_switching_{'V' if sig == 'v_bus' else 'A' if sig == 'i_bat' else 'W'}"],
                   color=INK, lw=1.4, label="switching-level reference", zorder=2)
            a.plot(t, d[f"{sig}_averaged_{'V' if sig == 'v_bus' else 'A' if sig == 'i_bat' else 'W'}"],
                   color=SLOT[0], lw=1.2, ls="--", label="averaged model (plant)", zorder=3)
            if c == 0:
                a.set_ylabel(lab)
            if r == 0:
                a.set_title(title, loc="left", pad=13)
                y0, y1 = a.get_ylim()
                a.text((t_step + t_steady) / 2, y1, "transient", ha="center", va="bottom", fontsize=7, color=INK2)
                a.text((t_steady + t_end) / 2, y1, "steady", ha="center", va="bottom", fontsize=7, color=INK2)
            if r == 2:
                a.set_xlabel("time (ms)")
            a.set_xlim(0, t_end)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(h_pad=1.2, rect=(0, 0.04, 1, 1))
    save(fig, "F1_validation", png_dir)


def f2_traces(png_dir):
    runs = [("baseline_ws_s0", "baseline", "warm start, baseline mode (baseline_ws_s0)", SLOT[0], "-"),
            ("filter_ws_s0", "filter", "warm start, filter mode (filter_ws_s0)", SLOT[1], "--")]
    tr = {r: pd.read_csv(ROOT / "results" / r / "rollout" / f"rollout_{m}_soc0.2.csv") for r, m, *_ in runs}
    n_sub = HoppingLegEnv().substeps                         # substeps per control step (filt counts them)
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1.1, 1.1, 0.6]))
    for r, m, lab, col, ls in runs:
        d = tr[r]
        ax[0].plot(d["t"], d["z"], color=col, ls=ls, label=lab)
        ax[1].plot(d["t"], d["v_bus_min"], color=col, ls=ls, lw=1.0, label=lab)
        ax[2].plot(d["t"], d["i_bat_max"], color=col, ls=ls, lw=1.0, label=lab)
    ax[1].axhline(P.v_min, color=INK, lw=0.9, ls=":")
    ax[1].text(1.01, P.v_min, f"$V_{{\\min}}$\n{P.v_min:g} V", transform=ax[1].get_yaxis_transform(),
               ha="left", va="center", fontsize=7, color=INK)
    ax[2].axhline(P.i_bat_max, color=INK, lw=0.9, ls=":")
    ax[2].text(1.01, P.i_bat_max, f"$I_{{\\lim}}$\n{P.i_bat_max:g} A", transform=ax[2].get_yaxis_transform(),
               ha="left", va="center", fontsize=7, color=INK)
    d = tr["filter_ws_s0"]
    assert d["filt"].max() <= n_sub
    ax[3].fill_between(d["t"], 0, d["filt"] / n_sub, step="pre", color=SLOT[1], alpha=0.35, lw=0)
    ax[3].step(d["t"], d["filt"] / n_sub, where="pre", color=SLOT[1], lw=0.8, ls="--",
               label="filter_ws_s0")
    ax[0].set_ylabel("rail height\nabove start (m)")
    ax[1].set_ylabel("bus voltage, lowest\nin control step (V)")
    ax[2].set_ylabel("battery current, highest\nin control step (A)")
    ax[3].set_ylabel("filter active\n(fraction of substeps)")
    ax[3].set_ylim(0, 1.05)
    ax[3].set_xlabel("time (s)")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout(h_pad=0.8, rect=(0, 0, 0.93, 1))
    save(fig, "F2_traces", png_dir)


def f3_violation_vs_soc(png_dir):
    d = pd.read_csv(ROOT / "results" / "summary" / "per_seed.csv")
    own = d[d["method"] == d["train"]]                       # each run in its own training mode
    pen = sorted({m for m in own["method"] if re.fullmatch(r"penalty_w[0-9.]+_ws", m)},
                 key=lambda m: float(m[len("penalty_w"):-len("_ws")]))
    methods = [m for m in ["baseline_ws"] + pen + ["filter_ws"] if m in set(own["method"])]
    colour = {"baseline_ws": SLOT[0], "filter_ws": SLOT[1]}
    for i, m in enumerate(pen):                              # fixed order by weight, never cycled
        colour[m] = SLOT[2 + i]
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    label = {"baseline_ws": "baseline, warm start", "filter_ws": "filter, warm start"}
    for m in pen:
        label[m] = f"penalty w = {float(m[len('penalty_w'):-len('_ws')]):g}, warm start"
    socs = sorted(own["soc"].unique())
    dodge = np.linspace(-0.02, 0.02, len(methods))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.7), sharex=True)
    for k, (col, ylab) in enumerate([("uv_s", f"time below $V_{{\\min}}$ = {P.v_min:g} V (s per episode)"),
                                     ("oc_s", f"time above $I_{{\\lim}}$ = {P.i_bat_max:g} A (s per episode)")]):
        a = ax[k]
        for j, m in enumerate(methods):
            g = own[own["method"] == m].groupby("soc")[col]
            mean, lo, hi, n = g.mean(), g.min(), g.max(), g.size()
            x = np.array(mean.index) + dodge[j]
            single = bool((n == 1).all())
            a.errorbar(x, mean.values, yerr=[mean.values - lo.values, hi.values - mean.values],
                       color=colour[m], marker=markers[j], ms=5.5, lw=1.0, capsize=2.5, elinewidth=0.9,
                       ls="--" if single else "-", mfc=SURF if single else colour[m], mew=1.2,
                       label=label[m] + (" (1 seed)" if single else f" ({int(n.max())} seeds)"))
            gf = own[own["method"] == m].groupby("soc")[["n_fell", "n_episodes"]].sum()
            for xi, (soc, nf, ne) in zip(x, gf.reindex(mean.index).itertuples()):
                if nf > 0:                                   # falls are not visible in violation times
                    a.plot([xi], [mean[soc]], ls="none", marker="o", ms=13, mfc="none", mec=INK,
                           mew=1.0, zorder=5, label="_fall")
                    short = {"baseline_ws": "baseline", "filter_ws": "filter"}.get(
                        m, label[m].split(",")[0].replace("penalty ", ""))
                    a.annotate(f"{short}\nfell {int(nf)}/{int(ne)}", (xi, mean[soc]), xytext=(0, 9),
                               textcoords="offset points", ha="center", va="bottom", fontsize=6.8,
                               color=INK, zorder=6)
        a.set_xticks(socs)
        a.set_xlabel("state of charge at episode start")
        a.set_ylabel(ylab)
        a.set_ylim(bottom=-0.03 * a.get_ylim()[1])
    h, lab = ax[0].get_legend_handles_labels()
    if (own[own["method"].isin(methods)]["n_fell"] > 0).any():
        h.append(plt.Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec=INK, mew=1.0))
        lab.append("episodes ended in a fall (fell k of n)")
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.005, "markers: mean over seeds; bars: min to max over seeds; "
             "hollow markers and dashed lines: one seed; ring: some episodes ended in a fall",
             ha="center", fontsize=6.8, color=INK2)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    save(fig, "F3_violation_vs_soc", png_dir)


# ---- CHANGELOG #39: schematics, trade-off, training curves, graphical abstract ----------------
# grey lightness ramp for the penalty weights (ordinal, light = small weight); every step >= 2:1
# against the surface
GREY_RAMP = ["#a3a29a", "#8a8882", "#6f6e69", "#52514e", "#2f2e2c"]
VB, IB = r"V_{\mathrm{bus}}", r"I_{\mathrm{bat}}"


def _pen_weights():
    """Warm-started penalty weights with a run directory, ascending."""
    ws = {float(m.group(1)) for d in (ROOT / "results").iterdir()
          if (m := re.fullmatch(r"penalty_w([0-9.]+)_ws_s\d+", d.name))}
    assert len(ws) <= len(GREY_RAMP), "more penalty weights than ramp steps"
    return sorted(ws)


def _pen_colour(w):
    ws = _pen_weights()
    return GREY_RAMP[ws.index(w)] if w in ws else GREY_RAMP[len(GREY_RAMP) // 2]


def _box(ax, x, y, w, h, title, sub=None, ec=INK2, lw=1.0, ls="-", fs=8.0, fc=SURF, sub_ls=1.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    dy = 0.17 * h if sub else 0.0
    ax.text(x + w / 2, y + h / 2 + dy, title, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2 * h, sub, ha="center", va="center", fontsize=6.6,
                color=INK2, zorder=3, linespacing=sub_ls)


def _arrow(ax, p0, p1, ls="-", color=INK2, conn="arc3"):
    ax.annotate("", xy=p1, xytext=p0, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=8, connectionstyle=conn))


def _lab(ax, x, y, s, ha="center", va="bottom", fs=6.8, color=INK):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=5)


def _line(ax, pts, arrow=True, ls="-"):
    """Polyline through pts (data coordinates), with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if arrow else xs, ys[:-1] if arrow else ys, color=INK2, lw=0.9, ls=ls, zorder=4,
            solid_capstyle="butt")
    if arrow:
        _arrow(ax, pts[-2], pts[-1], ls=ls)


def f4_architecture(png_dir):
    env = HoppingLegEnv()
    dt_ms, ctrl_ms = 1e3 * env.dt_sim, 1e3 * env.dt_sim * env.substeps
    # drawn at 5.9 in and placed at text width: all text about 1.2 times larger (CHANGELOG #51)
    fig, ax = plt.subplots(figsize=(5.9, 3.45))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.7)
    ax.axis("off")
    # substep frame: everything except the policy runs every substep
    ax.add_patch(Rectangle((2.8, 0.2), 11.15, 6.8, fc="none", ec=AXISC, lw=0.9, ls=(0, (4, 3)), zorder=1))
    _lab(ax, 2.92, 0.28, f"every $\\Delta t$ = {dt_ms:g} ms", ha="left", va="bottom", color=INK2)
    # blocks
    _box(ax, 0.1, 4.6, 2.2, 1.3, "Policy", f"SAC actor,\nevery {ctrl_ms:g} ms")
    _lab(ax, 1.2, 4.45, "observation:\n" + r"$q,\ \dot q$, previous action," + "\n" + f"contact, ${VB}$, SOC",
         va="top", fs=6.3, color=INK2)
    _box(ax, 3.05, 4.6, 1.8, 1.3, "PD law", r"$q^{\mathrm{des}}-q,\ \dot q$" + "\n" + r"$\to$ torque $\to i_q^{\mathrm{des}}$")
    ax.add_patch(FancyBboxPatch((5.35, 2.9), 3.95, 3.75, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#f3f2ee", ec=SLOT[1], lw=1.4, zorder=1))
    _lab(ax, 7.33, 6.28, "Barrier power filter", fs=8.0)
    _box(ax, 5.5, 4.75, 3.65, 1.4, r"Budget $\bar P$",
         "one bus step (Prop. 1):\n" + r"$V_{\mathrm{req}}\to\bar P=F(V_{\mathrm{req}})$")
    _box(ax, 5.5, 3.05, 3.65, 1.2, "Projection",
         r"nearest $i_q\in[l_i,u_i]$" + "\n" + r"with $\Sigma_i\,(a_i i_{q,i}+b\,i_{q,i}^2)\leq B$")
    _box(ax, 9.85, 4.6, 2.35, 1.3, "Drives ×2",
         r"$P_i=a_i i_{q,i}+b\,i_{q,i}^2+P_0$" + "\n" + r"torque $\propto i_{q,i}$", sub_ls=1.45)
    _box(ax, 12.5, 4.6, 1.4, 1.3, "MuJoCo", "hip, knee,\ncontacts")
    _box(ax, 9.85, 0.45, 4.05, 1.45, "DC bus + battery",
         r"bus capacitor $C$, chopper $g(V)$;" + "\n" + r"OCV(SOC), $R_0$, RC pair")
    _box(ax, 5.5, 0.45, 3.0, 1.45, "Battery EKF",
         r"state $[\mathrm{SOC}, v_1, R_0]$;" + "\n" + "NIS confidence $\\zeta$")
    # forward path
    _arrow(ax, (2.3, 5.25), (3.05, 5.25))
    _lab(ax, 2.68, 5.32, r"$q^{\mathrm{des}}$")
    _line(ax, [(4.85, 5.0), (5.12, 5.0), (5.12, 3.65), (5.5, 3.65)])
    _lab(ax, 5.05, 4.25, r"$i_q^{\mathrm{des}}$", ha="right", va="center")
    _arrow(ax, (7.33, 4.75), (7.33, 4.25))
    _lab(ax, 7.45, 4.5, r"$B$", ha="left", va="center", fs=6.8)
    _line(ax, [(9.15, 3.65), (10.35, 3.65), (10.35, 4.6)])
    _lab(ax, 9.75, 3.72, r"$i_q$")
    _arrow(ax, (12.2, 5.25), (12.5, 5.25))
    _lab(ax, 12.35, 5.32, r"$\tau$")
    # electrical path
    _arrow(ax, (10.95, 4.6), (10.95, 1.9))
    _lab(ax, 10.88, 2.75, r"$\Sigma_i P_i$" + "\n" + r"$+P_{\mathrm{aux}}$", ha="right", va="center", fs=6.3)
    _arrow(ax, (11.65, 1.9), (11.65, 4.6))
    _lab(ax, 11.72, 3.25, f"${VB}$: voltage\ncircle" + r"$\to[l_i,u_i]$", ha="left", va="center", fs=6.3)
    _line(ax, [(9.85, 1.55), (9.5, 1.55), (9.5, 5.45), (9.15, 5.45)])
    ax.text(9.4, 2.45, f"${VB}^k,\\ {IB}^k$", rotation=90, ha="right", va="center", fontsize=6.3, color=INK)
    _arrow(ax, (9.85, 0.95), (8.5, 0.95))
    _lab(ax, 9.17, 0.88, f"${VB},{IB}$\n+ noise", va="top", fs=6.0, color=INK)
    _arrow(ax, (7.0, 1.9), (7.0, 2.9))
    _lab(ax, 6.9, 2.4, r"$\hat E,\hat R_0$; margins $\delta_V,\delta_I$ from $\zeta$" + "\n" +
         r"(true-state filter: $E,R_0$)", ha="right", va="center", fs=6.3)
    # mechanical feedback over the top
    ax.plot([13.2, 13.2, 1.2], [5.9, 7.35, 7.35], color=INK2, lw=0.9, zorder=4)
    for xx, y_end in ((1.2, 5.9), (3.95, 5.9), (7.33, 6.65), (11.0, 5.9)):
        _arrow(ax, (xx, 7.35), (xx, y_end))
    _lab(ax, 5.6, 7.42, r"joint angles and speeds $q,\ \dot q$")
    fig.tight_layout(pad=0.2)
    save(fig, "F4_architecture", png_dir)


def f0_framework(png_dir):
    """Overall framework (CHANGELOG #62): public sources -> electrically coupled plant -> learning
    with the filter -> evaluation, with the contributions in a strip below. A schematic with no
    result numbers. Drawn at the manuscript text width (390 pt), so font sizes are print sizes;
    every text is checked to be at least 7 pt and to lie inside its panel or box."""
    W = 390 / 72.27 * 2.54            # text width of elsarticle preprint, 12 pt (cm)
    H = 6.5                           # cm
    FS, FT = 7.0, 7.5                 # body and title font sizes (pt)
    LH = FS * 1.12 / 72.27 * 2.54     # line pitch (cm)
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    checks = []                       # (text, (x0, y0, x1, y1)) that must lie inside the box

    def txt(x, y, s, box, fs=FS, color=INK, weight="normal", ha="left", va="top", style="normal"):
        t = ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va, style=style,
                    linespacing=1.12, zorder=5)
        checks.append((t, box))
        return t

    def card(x, ytop, w, head, body=(), ec=INK2, lw=0.8, ls="-", fc=SURF, ha="left"):
        """Rounded box with a bold head line and body lines; returns its bottom y."""
        pad = 0.09
        n = len(head) + len(body)
        y0 = ytop - (n * LH + 2 * pad)
        ax.add_patch(FancyBboxPatch((x, y0), w, ytop - y0, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        box = (x, y0, x + w, ytop)
        xt = x + w / 2 if ha == "center" else x + 0.07
        if head:
            txt(xt, ytop - pad, "\n".join(head), box, weight="bold", ha=ha)
        if body:
            txt(xt, ytop - pad - len(head) * LH, "\n".join(body), box, color=INK2, ha=ha)
        return y0

    # panels and the strip below them
    gap, m, PI = 0.3, 0.03, 0.07     # panel gap, outer margin, panel inset (cm)
    widths = [3.15, 2.8, 3.55, 3.3]
    widths[-1] = W - 2 * m - 3 * gap - sum(widths[:-1])
    xs = [m + sum(widths[:k]) + k * gap for k in range(4)]
    ys0, ys1 = 0.03, 0.80             # strip
    yp0, yp1 = 0.95, H - 0.03         # panels
    heads = [("(a) Public sources", "Section 3, Table 2"),
             ("(b) Electrically\ncoupled plant", "Section 3"),
             ("(c) Learning with\nthe filter", "Sections 4 and 5"),
             ("(d) Evaluation", "Sections 6 and 7")]
    tops = []
    for x, w, (t, s) in zip(xs, widths, heads):
        ax.add_patch(FancyBboxPatch((x, yp0), w, yp1 - yp0, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#f3f2ee", ec=AXISC, lw=0.8, zorder=1))
        box = (x, yp0, x + w, yp1)
        txt(x + PI, yp1 - 0.1, t, box, fs=FT, weight="bold")
        nl = t.count("\n") + 1
        y_sub = yp1 - 0.1 - nl * FT * 1.12 / 72.27 * 2.54
        txt(x + PI, y_sub, s, box, color=INK2)
        tops.append(y_sub - LH - 0.1)
    ymid = (yp0 + yp1) / 2
    for k in range(3):
        _arrow(ax, (xs[k] + widths[k] + 0.03, ymid), (xs[k + 1] - 0.03, ymid))

    # (a) public sources -> model parameters
    x, w = xs[0] + PI, widths[0] - 2 * PI
    y = tops[0]
    for head, body, ls in [(["LG 18650HG2 cell", "data"], ["→ battery fit:", "Thevenin, one RC pair"], "-"),
                           (["MuJoCo Menagerie", "Go1"], ["→ leg masses and", "inertias"], "-"),
                           (["Actuator thesis"], ["→ drive parameters"], "-"),
                           (["Assumed values"], ["each with a reason", "(Table 2)"], (0, (3, 2)))]:
        y = card(x, y, w, head, body, ls=ls) - 0.12

    # (b) the electrically coupled plant: leg, drives, bus, battery
    x, w = xs[1] + PI, widths[1] - 2 * PI
    y = tops[1]
    chain = [(["Leg on a rail"], ["MuJoCo"]),
             (["Two drives"], []),
             (["DC bus"], ["capacitor, brake", "chopper"]),
             (["Battery"], ["feeds both drives"])]
    for k, (head, body) in enumerate(chain):
        yb = card(x, y, w, head, body, ha="center")
        if k < len(chain) - 1:
            ax.plot([x + w / 2] * 2, [yb, yb - 0.13], color=INK2, lw=1.4, zorder=1.5)
        y = yb - 0.13
    txt(x + w / 2, y + 0.06, "validated against\na switching-level\nmodel (Section 3.4)",
        (xs[1], yp0, xs[1] + widths[1], yp1), color=INK2, ha="center", style="italic")

    # (c) learning with the filter
    x, w = xs[2] + PI, widths[2] - 2 * PI
    y = tops[2]
    wl = w - 1.38                      # left column: policy -> filter -> plant
    y1 = card(x, y, wl, ["SAC policy"], ha="center")
    _arrow(ax, (x + wl / 2, y1), (x + wl / 2, y1 - 0.2))
    y2 = card(x, y1 - 0.2, wl, ["Barrier power", "filter"], ["power budget,", "then projection"],
              ec=SLOT[1], lw=1.4, fc="#fbe6dc", ha="center")
    _arrow(ax, (x + wl / 2, y2), (x + wl / 2, y2 - 0.2))
    y3 = card(x, y2 - 0.2, wl, ["Plant (b)"], ha="center")
    xe, yt = x + wl + 0.1, y1 - 0.2 - 0.14      # EKF beside the filter, within its height
    ye = card(xe, yt, w - wl - 0.1, ["EKF"], ["optional", "$E$, $R_0$"], ls=(0, (3, 2)), ha="center")
    assert ye > y2, "EKF box must not reach below the filter box"
    _arrow(ax, (xe, (ye + yt) / 2), (x + wl, (ye + yt) / 2), ls=(0, (3, 2)))
    card(x, y3 - 0.14, w, ["Training modes"],
         ["baseline; reward penalty;", "SAC-Lagrangian; filter in", "the loop (filter-trained", "policies); warm start"])

    # (d) evaluation
    x, w = xs[3] + PI, widths[3] - 2 * PI
    y = tops[3]
    for head, body in [(["Main comparison"], ["three states of charge"]),
                       (["Ablations"], ["initialisation, filter", "placement (test-time", "filter), barrier rate,",
                                        "margins, per-joint limit"]),
                       (["Stress tests"], ["parameter variants,", "model error (true-state", "filter), EKF filter"]),
                       (["Metrics"], ["violation time, falls,", "max height"])]:
        y = card(x, y, w, head, body) - 0.1

    # strip: panels -> contributions (numbered as in the Introduction)
    strip = [((0, 1), "Contribution 1", "electrically coupled simulation layer"),
             ((2, 2), "Contribution 2", "barrier power filter"),
             ((3, 3), "Contributions 3, 4", "comparison, robustness")]
    for (a, b), t, s in strip:
        x0, x1 = xs[a], xs[b] + widths[b]
        ax.add_patch(FancyBboxPatch((x0, ys0), x1 - x0, ys1 - ys0, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=SURF, ec=INK2, lw=0.8, zorder=1))
        box = (x0, ys0, x1, ys1)
        txt((x0 + x1) / 2, ys1 - 0.08, t, box, weight="bold", ha="center")
        txt((x0 + x1) / 2, ys1 - 0.08 - LH, s, box, color=INK2, ha="center")
        for k in range(a, b + 1):
            xx = xs[k] + widths[k] / 2
            ax.plot([xx, xx], [yp0, ys1], color=INK2, lw=0.8, zorder=0.5)

    # the section and table numbers drawn above must match the last LaTeX build of the manuscript
    drawn = {"sec:model": "3", "sec:validation": "3.4", "tab:params": "2", "sec:filter": "4",
             "sec:setup": "5", "sec:results": "6", "sec:ablation": "7"}
    aux = ROOT / "paper" / "main.aux"
    if aux.exists():
        got = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([0-9.]+)\}", aux.read_text(encoding="utf-8", errors="replace")))
        wrong = {k: (v, got.get(k)) for k, v in drawn.items() if got.get(k) != v}
        assert not wrong, f"framework figure: numbers drawn vs paper/main.aux: {wrong}"
    else:
        print("F0_framework: paper/main.aux missing, section and table numbers in the figure not checked")

    # checks: every text at least 7 pt and inside its box
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for t, (bx0, by0, bx1, by1) in checks:
        assert t.get_fontsize() >= 7.0, t.get_text()
        e = t.get_window_extent(r).transformed(ax.transData.inverted())
        if e.x0 < bx0 + 0.02 or e.x1 > bx1 - 0.02 or e.y0 < by0 + 0.01 or e.y1 > by1 - 0.01:
            bad.append(f"{t.get_text()!r}: text ({e.x0:.2f},{e.y0:.2f})-({e.x1:.2f},{e.y1:.2f}) "
                       f"box ({bx0:.2f},{by0:.2f})-({bx1:.2f},{by1:.2f})")
    assert not bad, "framework figure: text outside its box:\n" + "\n".join(bad)
    save(fig, "F0_framework", png_dir)


def f5_budget(png_dir):
    env = HoppingLegEnv()
    p, dt, gam, mv, mi = env.p, env.dt_sim, env.gamma, env.margin_v, env.margin_i
    soc, i_frac = 0.2, 0.8                           # illustrative state
    e = Battery(p, soc).emf()                       # battery at rest: E = OCV(SOC)
    i_k = i_frac * p.i_bat_max
    META["budget"] = dict(soc=soc, i_frac=i_frac)
    v_k = e - p.r0 * i_k
    alpha, beta = p.c_bus / dt + 1.0 / p.r0, p.c_bus * v_k / dt + e / p.r0
    x_star = beta / (2.0 * alpha)
    vmin_m, ilim_m = p.v_min + mv, p.i_bat_max - mi
    v_uv = vmin_m + (1.0 - gam) * (v_k - vmin_m)
    v_oc = e - p.r0 * (ilim_m - (1.0 - gam) * (ilim_m - i_k))
    v_req = max(v_uv, v_oc, x_star)
    chop = (p.v_chop_off, p.v_chop, p.r_chop)
    F = lambda x: x * (beta - alpha * x) - chopper_conductance(x, *chop) * x * x
    p_bar = F(v_req)
    p_code = power_budget(v_k, e, p.r0, p.c_bus, dt, p.v_min, p.i_bat_max, i_k, gam, mv, mi, chop, eps_rel=0.0)
    assert abs(p_code - p_bar) <= 1e-9 * max(1.0, abs(p_bar)), (p_code, p_bar)
    p_max = beta * beta / (4.0 * alpha)
    assert abs(F(x_star) - p_max) <= 1e-9 * p_max
    # stated in the caption: the current barrier sets V_req here, and F(V^k) = V^k I^k (no chopper)
    assert v_oc > v_uv and v_oc > x_star
    assert abs(F(v_k) - v_k * i_k) <= 1e-9 * v_k * i_k
    print(f"F5: E={e:.3f} V, V^k={v_k:.3f} V, I^k={i_k:.2f} A, x*={x_star:.3f} V, V_uv={v_uv:.3f} V, "
          f"V_oc={v_oc:.3f} V, P_bar={p_bar:.1f} W, max={p_max:.1f} W")
    x_hi = 1.04 * max(e, v_k)
    x = np.linspace(1e-3, x_hi, 1500)
    fx = np.array([F(xi) for xi in x])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    # left: whole curve
    a = ax[0]
    a.plot(x, fx, color=INK, lw=1.4)
    a.axhline(0, color=AXISC, lw=0.8)
    a.plot([x_star], [p_max], "o", ms=5, color=INK)
    a.annotate(r"$x^\star=\beta/2\alpha$" + "\n" + r"$F(x^\star)=\beta^2/4\alpha$", (x_star, p_max), xytext=(0, -14),
               textcoords="offset points", ha="center", va="top", fontsize=7, color=INK)
    lo, hi = v_uv - 0.6, v_k + 0.35
    a.axvspan(lo, hi, color=GRIDC, alpha=0.8, lw=0)
    a.text((lo + hi) / 2, 0.06 * p_max, "zoom\n(right)", ha="center", fontsize=6.8, color=INK2)
    a.set_xlim(0, x_hi)
    a.set_ylim(min(fx.min(), -0.05 * p_max), 1.15 * p_max)
    a.set_xlabel(r"end-of-step bus voltage $V'$ (V)")
    a.set_ylabel(r"load power $P=F(V')$ (W)")
    a.set_title("one bus step (Lemma 1)", loc="left")
    # right: zoom on the barrier bounds
    a = ax[1]
    m = (x >= lo) & (x <= hi)
    a.plot(x[m], fx[m], color=INK, lw=1.4, label=r"$F(V')$")
    adm = m & (x >= v_req)
    a.fill_between(x[adm], fx[adm], fx[m].min() - 1e3, color=GRIDC, alpha=0.9, lw=0)
    a.axvline(v_uv, color=SLOT[0], lw=1.1, ls="--")
    a.axvline(v_oc, color=SLOT[1], lw=1.1, ls="-.")
    a.axhline(p_bar, color=INK, lw=0.9, ls=":")
    a.plot([v_req], [p_bar], "o", ms=5.5, color=INK, zorder=5)
    a.plot([v_k], [F(v_k)], "s", ms=4.5, mfc=SURF, mec=INK2, mew=1.0, zorder=5)
    y0, y1 = fx[m].min(), fx[m].max()
    a.set_ylim(y0 - 0.05 * (y1 - y0), y1 + 0.3 * (y1 - y0))
    ytop = a.get_ylim()[1]
    a.text(v_uv, ytop, "voltage barrier\n" + r"$V_{\min}+\delta_V+(1-\gamma)(h_1^k-\delta_V)$", ha="right", va="top",
           fontsize=6.6, color=INK, rotation=0, bbox=dict(fc=SURF, ec="none", pad=0.5))
    a.text(v_oc, ytop, "current barrier\n" + r"$E-R_0 I_{\mathrm{allow}}$", ha="left", va="top", fontsize=6.6,
           color=INK, bbox=dict(fc=SURF, ec="none", pad=0.5))
    a.annotate(r"$V_{\mathrm{req}}$, $\bar P=F(V_{\mathrm{req}})$", (v_req, p_bar), xytext=(-8, -7),
               textcoords="offset points", ha="right", va="top", fontsize=7, color=INK,
               bbox=dict(fc=SURF, ec="none", pad=0.6))
    a.annotate(f"no change:\n$V'={VB}^k$", (v_k, F(v_k)), xytext=(6, 16),
               textcoords="offset points", ha="left", va="bottom", fontsize=6.6, color=INK2,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=3))
    a.text(0.98, 0.70, r"shaded: $P\leq\bar P\ \Leftrightarrow\ V'\geq V_{\mathrm{req}}$" + "\n(both barriers hold)",
           transform=a.transAxes, ha="right", va="top", fontsize=7, color=INK)
    a.set_xlim(lo, hi)
    a.set_xlabel(r"end-of-step bus voltage $V'$ (V)")
    a.set_title("barrier bounds and budget (Prop. 1)", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "F5_budget", png_dir)


def f6_tradeoff(png_dir):
    """CHANGELOG #63: dot plot of max height, main evaluation, warm-started policies, one panel per
    state of charge. Rows in two groups; the grouping is asserted from the data at every state of
    charge (violates: mean time below V_min or above I_lim > 0; within: zero in every seed). Drawn at
    the text width (390 pt x 6 cm), so font sizes are print sizes (>= 7 pt, checked)."""
    from matplotlib.transforms import blended_transform_factory
    d = pd.read_csv(ROOT / "results" / "summary" / "per_seed.csv")
    d = d[d["method"].str.split("+").str[0] == d["train"]]          # each policy with its own training run
    groups = [("violates a limit", SLOT[0], False,
               [("baseline_ws", "Baseline"), ("penalty_w0.05_ws", "Penalty $w$ = 0.05"),
                ("penalty_w0.1_ws", "Penalty $w$ = 0.1"), ("penalty_w0.5_ws", "Penalty $w$ = 0.5"),
                ("lagrangian_ws", "SAC-Lagrangian")]),
              ("within both limits", SLOT[1], True,
               [("baseline_ws+filter@test", "Baseline + test-time filter"),
                ("penalty_w0.5_ws+filter@test", "Penalty $w$ = 0.5 + test-time filter"),
                ("filter_ws", "Filter-trained")])]
    socs = sorted(d["soc"].unique(), reverse=True)
    n_seeds = d[d["method"] == "filter_ws"]["seed"].nunique()
    eps = set(d[d["method"] == "filter_ws"]["n_episodes"])
    # row positions (top to bottom): group header, its rows; a rule between the groups
    yrow, yhead, ylab, y = {}, [], [], 0.0
    for gi, (gl, _, _, rows) in enumerate(groups):
        if gi:                         # room below the last row for a "fell k/n" label, then the rule
            y_rule = y + 1.15
            y += 1.75
        yhead.append(y)
        for m, lab in rows:
            y += 1.0
            yrow[m] = y
            ylab.append(lab)
        y += 0.0
    W, H = 390 / 72.27, 6.0 / 2.54
    FS = 7.0
    fig = plt.figure(figsize=(W, H))
    r = fig.canvas.get_renderer()
    lab_w = 0.0
    for s in ylab + [g[0] for g in groups]:
        t = fig.text(0, 0, s, fontsize=FS + (0.5 if s in [g[0] for g in groups] else 0), fontweight="bold"
                     if s in [g[0] for g in groups] else "normal")
        lab_w = max(lab_w, t.get_window_extent(r).width / fig.dpi / W)
        t.remove()
    left, right, gap, bottom, top = lab_w + 0.025, 0.995, 0.03, 0.16, 0.90
    pw = (right - left - 2 * gap) / len(socs)
    axs = [fig.add_axes([left + k * (pw + gap), bottom, pw, top - bottom]) for k in range(len(socs))]
    xmax = 0.0
    stats = {}
    for soc in socs:
        for gl, colr, filled, rows in groups:
            for m, _ in rows:
                g = d[(d["method"] == m) & (d["soc"] == soc)]
                assert g["seed"].nunique() == n_seeds and set(g["n_episodes"]) == eps, (m, soc)
                viol = g["uv_s"] + g["oc_s"]
                if filled:
                    assert (viol == 0).all() and (g["collapse_s"] == 0).all(), f"{m} at SOC {soc} is not within both limits"
                else:
                    assert viol.mean() > 0, f"{m} at SOC {soc} does not violate a limit"
                stats[m, soc] = (g["height_max"].mean(), g["height_max"].std(), int(g["n_fell"].sum()),
                                 int(g["n_episodes"].sum()))
                xmax = max(xmax, stats[m, soc][0] + stats[m, soc][1])
    xmax = np.ceil(xmax / 0.05) * 0.05 + 0.05
    falls = []
    for k, (ax, soc) in enumerate(zip(axs, socs)):
        ax.set_xlim(0, xmax)
        ax.set_ylim(y + 0.6, -0.6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=FS, length=2, pad=1.5)
        for m, yy in yrow.items():
            ax.axhline(yy, color=GRIDC, lw=0.4, zorder=0)
        ax.axhline(y_rule, color=INK2, lw=0.6, zorder=1)
        for gl, colr, filled, rows in groups:
            for m, _ in rows:
                xm, sd, kf, nf = stats[m, soc]
                ax.errorbar(xm, yrow[m], xerr=sd, fmt="none", ecolor=colr, elinewidth=0.9, capsize=1.8,
                            capthick=0.9, zorder=2)
                ax.plot(xm, yrow[m], ls="none", marker="o", ms=5.0, mfc=colr if filled else SURF, mec=colr,
                        mew=1.2, zorder=3)
                if kf:
                    falls.append((ax, xm, sd, yrow[m], f"fell {kf}/{nf}"))
        ax.set_yticks(list(yrow.values()))
        ax.set_yticklabels(ylab if k == 0 else [], fontsize=FS, color=INK)
        if k:
            ax.spines["left"].set_visible(False)
        ax.set_title(f"state of charge {soc:.1f}", loc="left", fontsize=FS + 0.5, pad=3)
        ax.set_xlabel("max height (m)", fontsize=FS, labelpad=1.5)
    tr = blended_transform_factory(fig.transFigure, axs[0].transData)
    for (gl, _, _, _), yy in zip(groups, yhead):
        fig.text(0.004, yy, gl, transform=tr, fontsize=FS + 0.5, fontweight="bold", color=INK, va="center")
    fig.add_artist(plt.Line2D([0.004, left], [y_rule, y_rule], transform=tr, color=INK2, lw=0.6))
    # "fell k/n" beside the marker: right of the error bar, else left of it, else just below the row
    fig.canvas.draw()
    inv = None
    for ax, xm, sd, yy, s in falls:
        inv = ax.transData.inverted()
        pad = 0.012 * xmax
        placed = False
        for x, ha, dy in ((xm + sd + pad, "left", 0.0), (xm - sd - pad, "right", 0.0), (xm + pad, "left", 0.62)):
            t = ax.text(x, yy + dy, s, fontsize=FS, color=INK, ha=ha, va="center", zorder=5)
            e = t.get_window_extent(r).transformed(inv)
            if e.x0 >= 0 and e.x1 <= xmax:
                placed = True
                break
            t.remove()
        assert placed, f"no room for {s!r} at {xm:.3f}"
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip():
            assert t.get_fontsize() >= 7.0, (t.get_text(), t.get_fontsize())
    save(fig, "F6_tradeoff", png_dir)


def f8_settings(png_dir):
    """CHANGELOG #49 (split from F6, #48): the 10-episode settings of the warm-started policies
    under the joint filter, the per-joint limit and the margins; max height against the fraction of
    substeps with the filter or limiter active (all of them have zero violation time)."""
    ab = pd.read_csv(ROOT / "results" / "summary" / "ablations.csv")
    env = HoppingLegEnv()
    mv0, mi0 = env.margin_v, env.margin_i

    def mlab(s):
        parts = dict(p.split("=") for p in s.split(", "))
        return (f"$\\delta_V$ = {float(parts.get('delta_V', mv0)):g} V, "
                f"$\\delta_I$ = {float(parts.get('delta_I', mi0)):g} A")
    MMK = {"delta_I=0.0": "^", "delta_I=1.0": "v", "delta_V=0.0": "<", "delta_V=0.5": ">",
           "delta_V=0.0, delta_I=0.0": "*"}
    srows = [("base", "base", "baseline_ws+filter@test", "baseline + joint filter at test", SLOT[2], "D"),
             ("controller", "perjoint", "baseline_ws+perjoint@test", "baseline + per-joint limit at test", SLOT[2], "X"),
             ("base", "base", "filter_ws", f"filter-trained, joint filter, {mlab('base=0')}", SLOT[1], "s"),
             ("controller", "perjoint", "filter_ws+perjoint@test", "filter-trained, per-joint limit", SLOT[1], "X")]
    for s in sorted(ab[ab.ablation == "margin"].setting.unique(), key=lambda s: list(MMK).index(s)):
        srows.append(("margin", s, "filter_ws", f"filter-trained, {mlab(s)}", SLOT[1], MMK[s]))
    socs = sorted(ab[ab.ablation == "controller"].soc.unique(), reverse=True)
    fig, ax = plt.subplots(1, len(socs), figsize=(7.0, 3.6), sharey=True, sharex=True)
    eps_set = None
    for c, soc in enumerate(socs):
        a = ax[c]
        for abl, s, m, lab, colr, mk in srows:
            g = ab[(ab.ablation == abl) & (ab.setting == s) & (ab.method == m) & (ab.soc == soc)]
            assert len(g) == 1, (abl, s, m, soc)
            g = g.iloc[0]
            eps_set = int(g.n_episodes / g.n_seeds)
            xm, ym = 100 * g.filt_frac_mean, g.height_max_mean
            a.errorbar(xm, ym, xerr=100 * g.filt_frac_std, yerr=g.height_max_std, fmt="none", ecolor=colr,
                       elinewidth=0.7, capsize=0, zorder=2)
            a.plot(xm, ym, ls="none", marker=mk, ms=7.5 if mk in "X*" else 6.5, mfc=colr, mec=colr, zorder=3)
            if g.n_fell > 0:
                a.plot(xm, ym, ls="none", marker="o", ms=14, mfc="none", mec=INK, mew=0.9, zorder=4)
                # label well clear of the markers, with a leader line (joint filter below left,
                # per-joint limit above right)
                below = mk == "D"
                a.annotate(f"fell {int(g.n_fell)}/{int(g.n_episodes)}", (xm, ym),
                           xytext=(-30, -26) if below else (24, 24), textcoords="offset points",
                           ha="center", va="top" if below else "bottom", fontsize=6.8, color=INK, zorder=6,
                           bbox=dict(fc=SURF, ec="none", pad=0.6),
                           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=7))
        a.set_title(f"state of charge {soc:g}", loc="left")
        if c == len(socs) // 2:
            a.set_xlabel("filter or per-joint limit active (% of substeps)")
        if c == 0:
            a.set_ylabel("max height (m)")
    META["settings"] = dict(eps_set=eps_set)
    hs = [plt.Line2D([], [], ls="none", marker=mk, ms=7 if mk in "X*" else 6, mfc=colr, mec=colr)
          for *_, colr, mk in srows]
    hs.append(plt.Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=0.9))
    ls = [r[3] for r in srows] + ["ring: some episode ended in a fall"]
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save(fig, "F8_settings", png_dir)


def _monitor(run):
    lines = (ROOT / "results" / run / "monitor.monitor.csv").read_text().splitlines()
    head = lines[1].split(",")
    rows = np.array([[float(v) for v in ln.split(",")] for ln in lines[2:] if ln.strip()])
    return {h: rows[:, i] for i, h in enumerate(head)}


def f7_training(png_dir, window=20):
    from busaware.runs import run_config, DEFAULT_PENALTY_W
    META["training"] = dict(window=window, penalty_w=DEFAULT_PENALTY_W)
    pw = f"penalty_w{float(DEFAULT_PENALTY_W)!r}_ws"
    groups = [("baseline_ws", "baseline", "baseline reward", SLOT[0]),
              ("filter_ws", "filter", "filter-mode reward", SLOT[1]),
              (pw, f"penalty, w = {DEFAULT_PENALTY_W:g}", "penalty-mode reward", _pen_colour(DEFAULT_PENALTY_W)),
              ("lagrangian_ws", "SAC-Lagrangian", "baseline reward", INK)]
    seeds = [0, 1, 2]
    styles = ["-", "--", ":"]
    # returns: each panel its own scale (the rewards differ between modes); lengths: one scale
    fig, ax = plt.subplots(2, 4, figsize=(7.0, 4.3), sharex=True)
    for c in range(1, 4):
        ax[1, c].sharey(ax[1, 0])
    for c, (base, title, rew, colr) in enumerate(groups):
        for s, lsty in zip(seeds, styles):
            run = f"{base}_s{s}"
            if not (ROOT / "results" / run / "monitor.monitor.csv").exists():
                continue
            mon = _monitor(run)
            if base == "lagrangian_ws":
                # the Monitor wraps the cost wrapper, so it logs the shaped return r - mu c; the
                # unshaped (baseline) return of every episode is in lagrangian.csv
                lag = pd.read_csv(ROOT / "results" / run / "lagrangian.csv")
                assert len(lag) == len(mon["r"]) and np.array_equal(lag["episode_len"].values, mon["l"])
                mon["r"] = lag["episode_return_unshaped"].values
            steps = np.cumsum(mon["l"]) / 1e3
            mov = lambda v: pd.Series(v).rolling(window, min_periods=1).mean().values   # first episodes: mean so far
            ax[0, c].plot(steps, mov(mon["r"]), color=colr, ls=lsty, lw=1.1)
            ax[1, c].plot(steps, mov(mon["l"]), color=colr, ls=lsty, lw=1.1)
            ls0 = run_config(ROOT / "results" / run).get("learning_starts")
            if ls0 and s == 0:
                for r in (0, 1):
                    ax[r, c].axvspan(0, ls0 / 1e3, color=GRIDC, alpha=0.9, lw=0, zorder=0)
        ax[0, c].set_title(f"{title}\n({rew})", loc="left", fontsize=7.5)
        ax[1, c].set_xlabel("steps (thousands)")
    ax[0, 0].set_ylabel(f"episode return\n(mean of last {window} episodes)")
    ax[1, 0].set_ylabel(f"episode length (control\nsteps, mean of last {window})")
    hs = [plt.Line2D([], [], color=INK2, ls=st, lw=1.1) for st in styles]
    hs.append(plt.Rectangle((0, 0), 1, 1, fc=GRIDC, ec="none"))
    fig.legend(hs, [f"seed {s}" for s in seeds] + ["replay buffer filled by the loaded policy, no gradient steps"],
               loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.8, w_pad=0.6)
    save(fig, "F7_training", png_dir)


def ga_graphical_abstract(png_dir):
    soc = 0.2
    d = pd.read_csv(ROOT / "results" / "summary" / "per_seed.csv")
    sel = {m: d[(d["method"] == m) & (d["train"] == m) & (d["soc"] == soc)] for m in ("baseline_ws", "filter_ws")}
    n_seeds = {m: g["seed"].nunique() for m, g in sel.items()}
    assert n_seeds["baseline_ws"] == n_seeds["filter_ws"] > 0
    tr = pd.read_csv(ROOT / "results" / "baseline_ws_s0" / "rollout" / f"rollout_baseline_soc{soc:g}.csv")
    W, H = 13.28 / 2.54, 5.31 / 2.54                 # Elsevier: >= 1328 x 531 px, ratio 500:200
    T = fig_transform = None  # noqa: F841 (figure coordinates below)
    with plt.rc_context({"font.family": "Arial", "axes.titlesize": 8, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": False, "hatch.linewidth": 0.8}):
        fig = plt.figure(figsize=(W, H))
        T = fig.transFigure
        cols = [(0.012, 0.29), (0.335, 0.28), (0.65, 0.338)]
        for (x0, w), head in zip(cols, ["Problem", "Method", f"Result at SOC {soc:g}"]):
            fig.patches.append(FancyBboxPatch((x0, 0.03), w, 0.94, boxstyle="round,pad=0,rounding_size=0.02",
                                              transform=T, fc=SURF, ec=AXISC, lw=0.8, zorder=-1))
            fig.text(x0 + 0.015, 0.895, head, fontsize=10.5, fontweight="bold", color=INK, va="center")
        for xa in (0.305, 0.618):
            fig.patches.append(matplotlib.patches.FancyArrow(xa, 0.5, 0.027, 0, width=0.03, head_width=0.09,
                                                             head_length=0.013, length_includes_head=True,
                                                             transform=T, fc=INK2, ec="none"))
        # problem: the baseline policy's lowest bus voltage per control step, one episode
        fig.text(0.027, 0.80, "Learned hopping on a\nlow battery pulls the\nbus below its limit",
                 fontsize=8, color=INK2, va="top", linespacing=1.1)
        a = fig.add_axes([0.07, 0.17, 0.185, 0.33])
        a.plot(tr["t"], tr["v_bus_min"], color=SLOT[0], lw=0.6)
        a.axhline(P.v_min, color=INK, lw=1.1, ls="--")
        a.text(1.02, P.v_min, r"$V_{\min}$", transform=a.get_yaxis_transform(), fontsize=8, color=INK,
               va="center", ha="left")
        a.set_ylabel("bus voltage (V)", labelpad=1, fontsize=6.8)
        a.set_xlabel("time (s)", labelpad=0.5, fontsize=6.8)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=2, pad=1)
        # method
        x0 = 0.35
        for k, (t, s) in enumerate([("RL policy", "joint targets"),
                                    ("Barrier power filter", "one-step power budget\n+ nearest safe currents"),
                                    ("Drives, bus, battery", None)]):
            y = 0.70 - 0.25 * k
            hgt = 0.18 if s and "\n" in s else 0.13
            fig.patches.append(FancyBboxPatch((x0, y - hgt / 2), 0.25, hgt, boxstyle="round,pad=0,rounding_size=0.015",
                                              transform=T, fc="#f3f2ee" if k == 1 else SURF,
                                              ec=SLOT[1] if k == 1 else INK2, lw=1.3 if k == 1 else 0.8))
            fig.text(x0 + 0.125, y + (0.04 if s and "\n" in s else 0.025 if s else 0), t, fontsize=8.5,
                     fontweight="bold", ha="center", va="center", color=INK)
            if s:
                fig.text(x0 + 0.125, y - 0.04 if "\n" in s else y - 0.033, s, fontsize=7, ha="center",
                         va="center", color=INK2, linespacing=1.05)
            if k < 2:
                fig.patches.append(matplotlib.patches.FancyArrow(x0 + 0.125, y - hgt / 2 - 0.005, 0, -0.04,
                                                                 width=0.003, head_width=0.012, head_length=0.022,
                                                                 length_includes_head=True, transform=T,
                                                                 fc=INK2, ec="none"))
        # result: warm-started policies in their own modes, mean over seeds
        fig.text(0.665, 0.81, f"warm-started policies,\nmean over {n_seeds['filter_ws']} seeds",
                 fontsize=7, color=INK2, va="top", linespacing=1.05)
        mets = [("uv_s", "under-\nvoltage (s)"), ("oc_s", "over-\ncurrent (s)"), ("height_max", "max\nheight (m)")]
        for j, (col, lab) in enumerate(mets):
            a = fig.add_axes([0.675 + 0.108 * j, 0.25, 0.075, 0.30])
            vals = [sel[m][col].mean() for m in ("baseline_ws", "filter_ws")]
            a.bar([0], [vals[0]], width=0.75, color=SLOT[0], lw=0)
            a.bar([1], [vals[1]], width=0.75, color=SLOT[1], lw=0, hatch="////", edgecolor=SURF)
            top = max(vals) * 1.3 if max(vals) > 0 else 1.0
            a.set_ylim(0, top)
            for i, v in enumerate(vals):
                a.text(i, v + 0.03 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
            a.set_xlim(-0.6, 1.6)
            a.set_xticks([])
            a.set_yticks([])
            a.spines[["top", "right", "left"]].set_visible(False)
            a.spines["bottom"].set_color(AXISC)
            a.set_title(lab, fontsize=7, pad=2, linespacing=1.0, color=INK)
        for k, (lab, colr, hatch) in enumerate([("baseline", SLOT[0], None), ("trained with filter", SLOT[1], "////")]):
            xk = 0.675 + 0.14 * k
            fig.patches.append(Rectangle((xk, 0.10), 0.016, 0.045, transform=T, fc=colr, ec=SURF if hatch else colr,
                                         lw=0, hatch=hatch))
            fig.text(xk + 0.022, 0.1225, lab, fontsize=7, color=INK, va="center")
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "GA_graphical_abstract.pdf")
        fig.savefig(OUT / "GA_graphical_abstract.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
        if png_dir:
            pathlib.Path(png_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(pathlib.Path(png_dir) / "GA_graphical_abstract_500x200.png", dpi=500 / W)
            fig.savefig(pathlib.Path(png_dir) / "GA_graphical_abstract.png", dpi=300)
        plt.close(fig)
    print("wrote", OUT / "GA_graphical_abstract.pdf", "and .tif (300 dpi)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png_dir", default=None, help="also write PNG previews here")
    ap.add_argument("--only", default=None, help="write only this figure (e.g. F0_framework); "
                    "figures_meta.json is then left unchanged")
    a = ap.parse_args()
    if a.only:
        fns = {"F0_framework": f0_framework, "F4_architecture": f4_architecture, "F6_tradeoff": f6_tradeoff}
        assert a.only in fns, f"--only supports {sorted(fns)}"
        fns[a.only](a.png_dir)
        return
    f0_framework(a.png_dir)
    f1_validation(a.png_dir)
    f2_traces(a.png_dir)
    f3_violation_vs_soc(a.png_dir)
    f4_architecture(a.png_dir)
    f5_budget(a.png_dir)
    f6_tradeoff(a.png_dir)
    f8_settings(a.png_dir)
    f7_training(a.png_dir)
    ga_graphical_abstract(a.png_dir)
    (OUT / "figures_meta.json").write_text(json.dumps(META, indent=2))
    print("wrote", OUT / "figures_meta.json")


if __name__ == "__main__":
    main()
