"""Step 2: averaged drive + bus model against the switching-level reference.

    python scripts/validate_drive.py

Drive values: gear, kt_out (derived), pole pairs, L, i_max and the 17 N m torque cap confirmed
in Katz 2018 (Step 1(a)); c_bus derived from it; r_phase ASSUMED; p_idle PLACEHOLDER. Every CSV and figure
states this in its header.

One drive on one leg's bus (battery, DC link, averaged chopper, P_idle), auxiliary load off
(variant paux_0w) in BOTH models. Mechanical speed imposed. Scenarios (0.1 s each, current
command stepped at 10 ms; speeds chosen by a rule stated here, before any result):
  S1  torque step at standstill: i_q* 0 -> 30 A, joint speed 0.
  S2  torque step at high speed near the voltage limit: i_q* 0 -> 20 A at the joint speed
      where the steady drive voltage |v_dq| is 95 % of V_bus/sqrt(3) (quasi-static V_bus).
  S3  braking: i_q* 0 -> -20 A at the S2 speed (regeneration).
Averaged model: busaware/drive.py + bus.py at the plant step (0.5 ms), as in leg_env.
Switching model: busaware/switching_ref.py at 1 us, 40 kHz SVPWM (Katz-fw loop period).
Compared signals: bus voltage, battery current, electrical power drawn by the drive.
I_bat and P: mean of the switching signal over each 0.5 ms plant step (the averaged model
holds them constant over the step). V_bus: the averaged model's end-of-step value against
the switching model's last PWM-period mean in the step. Errors: RMSE and peak |error|,
absolute and in % of the peak |switching signal|, over the whole run, the transient
(10 ms <= t < 30 ms: from the current step until steady) and the steady part (t >= 30 ms). Writes results/validation/.
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from busaware import Params, Battery, Bus, Drive  # noqa: E402
from busaware.switching_ref import simulate, F_PWM, DT, DEAD_TIME, I_BW  # noqa: E402

OUT = ROOT / "results" / "validation"
NOTE = ("Drive values: gear, kt_out (derived), pole pairs, L, i_max and 17 N m torque cap from Katz 2018; "
        "c_bus derived from Katz 2018; r_phase ASSUMED; p_idle PLACEHOLDER; see params_provenance.md")
DT_AVG = 5e-4
T_END, T_STEP, T_STEADY = 0.1, 0.010, 0.030
SOC = 0.5
SIGNALS = (("v_bus", "bus voltage", "V"), ("i_bat", "battery current", "A"),
           ("p_elec", "drive electrical power", "W"))

# dataviz reference palette (light): averaged model = slot 1 blue, switching = muted ink
SURF, INK, INK2, MUTED, GRIDC, AXISC, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
})


def speed_near_voltage_limit(p, iq, soc, frac=0.95):
    """Joint speed at which |v_dq| = frac * V_bus/sqrt(3) in steady state, with V_bus from a
    quasi-static battery (E - R0 I, RC pair at rest)."""
    d, e = Drive(p), Battery(p, soc).ocv()
    v = e
    for _ in range(100):
        vlim = frac * v / math.sqrt(3.0)
        qa = p.psi ** 2 + (p.l_phase * iq) ** 2
        qb = 2.0 * p.r_phase * iq * p.psi
        qc = (p.r_phase * iq) ** 2 - vlim ** 2
        we = (-qb + math.sqrt(qb * qb - 4.0 * qa * qc)) / (2.0 * qa)
        w = we / (p.pole_pairs * p.gear)
        v = e - p.r0 * float(d.power(np.array([iq]), np.array([w]))[0]) / v
    return w, v


def run_averaged(p, w, iq_fn, soc):
    bat = Battery(p, soc); bus = Bus(p, bat); d = Drive(p)
    n = int(round(T_END / DT_AVG))
    rec = {k: np.zeros(n) for k in ("t", "v_bus", "i_bat", "p_elec", "i_q")}
    ww = np.array([w])
    for k in range(n):
        iq = d.limit(np.array([iq_fn(k * DT_AVG)]), ww, bus.v)
        iq, _ = d.scale_to_power(iq, ww, bus.max_power(DT_AVG))
        pw = min(float(d.power(iq, ww)[0]), bus.max_power(DT_AVG))
        s = bus.step(pw, DT_AVG)
        for key, val in (("t", (k + 1) * DT_AVG), ("v_bus", s["v"]), ("i_bat", s["i_bat"]),
                         ("p_elec", pw), ("i_q", float(iq[0]))):
            rec[key][k] = val
    return rec


def to_plant_grid(sw):
    m = int(round(DT_AVG * F_PWM))                           # PWM periods per plant step
    g = {k: sw[k].reshape(-1, m).mean(axis=1) for k in ("i_bat", "p_elec", "i_q")}
    g["v_bus"] = sw["v_bus"].reshape(-1, m)[:, -1]           # last period of each step
    g["t"] = sw["t"].reshape(-1, m)[:, -1]
    return g


def metrics(ref, avg, t):
    out = {}
    for key, _, unit in SIGNALS:
        err = avg[key] - ref[key]
        scale = float(np.max(np.abs(ref[key])))
        for part, mask in (("all", np.ones_like(t, bool)), ("transient", (t > T_STEP) & (t < T_STEADY)),
                           ("steady", t >= T_STEADY)):
            rmse, peak = float(np.sqrt(np.mean(err[mask] ** 2))), float(np.max(np.abs(err[mask])))
            out[f"{key}_{part}"] = dict(unit=unit, rmse=rmse, peak=peak, ref_peak=scale,
                                        rmse_pct=100 * rmse / scale, peak_pct=100 * peak / scale)
    return out


def main():
    p = Params.variant("paux_0w")
    w_hi, v_q = speed_near_voltage_limit(p, 20.0, SOC)
    scen = {
        "S1_standstill_step": (0.0, lambda t: 0.0 if t < T_STEP else 30.0),
        "S2_high_speed_step": (w_hi, lambda t: 0.0 if t < T_STEP else 20.0),
        "S3_braking": (w_hi, lambda t: 0.0 if t < T_STEP else -20.0),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    summary = dict(note=NOTE, soc=SOC, t_end_s=T_END, t_step_s=T_STEP, t_steady_from_s=T_STEADY,
                   switching=dict(dt_s=DT, f_pwm_Hz=F_PWM, dead_time_s=DEAD_TIME, current_bw_Hz=I_BW),
                   averaged_dt_s=DT_AVG, s2_s3_joint_speed_rad_s=w_hi,
                   s2_quasi_static_bus_V=v_q, scenarios={})
    for name, (w, fn) in scen.items():
        sw = simulate(p, w, fn, T_END, SOC)
        ref, avg = to_plant_grid(sw), run_averaged(p, w, fn, SOC)
        summary["scenarios"][name] = dict(joint_speed_rad_s=w, metrics=metrics(ref, avg, avg["t"]))
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            f.write(f"# {NOTE}\n# scenario {name}: joint speed {w:.6g} rad/s, SOC {SOC}\n")
            f.write("t_s,v_bus_switching_V,v_bus_averaged_V,i_bat_switching_A,i_bat_averaged_A,"
                    "p_elec_switching_W,p_elec_averaged_W,i_q_switching_A,i_q_averaged_A\n")
            for r in zip(avg["t"], ref["v_bus"], avg["v_bus"], ref["i_bat"], avg["i_bat"],
                         ref["p_elec"], avg["p_elec"], ref["i_q"], avg["i_q"]):
                f.write(",".join(f"{x:.6g}" for x in r) + "\n")
        fig, ax = plt.subplots(3, 1, figsize=(7.5, 6.6), sharex=True)
        for a, (key, label, unit) in zip(ax, SIGNALS):
            a.plot(1e3 * sw["t"], sw[key], color=MUTED, lw=0.8, label="switching reference (PWM-period mean)")
            a.step(1e3 * avg["t"], avg[key], where="pre", color=BLUE, lw=1.5, label="averaged model (0.5 ms step)")
            a.set_ylabel(f"{label} ({unit})")
        ax[0].legend(loc="best")
        ax[-1].set_xlabel("time (ms)")
        fig.suptitle(f"{name.replace('_', ' ')}  (joint speed {w:.1f} rad/s)", color=INK, fontsize=10)
        fig.text(0.01, 0.005, NOTE, color=INK2, fontsize=6, wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig_{name}.{ext}", dpi=150)
        plt.close(fig)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(NOTE)
    print(f"S2/S3 joint speed {w_hi:.3f} rad/s (quasi-static bus {v_q:.3f} V)")
    for name, sres in summary["scenarios"].items():
        for key, label, unit in SIGNALS:
            for part in ("transient", "steady"):
                m = sres["metrics"][f"{key}_{part}"]
                print(f"{name:20s} {label:23s} {part:9s}: RMSE {m['rmse']:9.4f} {unit} ({m['rmse_pct']:6.2f} %), "
                      f"peak {m['peak']:9.4f} {unit} ({m['peak_pct']:6.2f} %)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
