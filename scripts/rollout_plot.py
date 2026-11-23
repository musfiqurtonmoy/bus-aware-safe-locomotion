"""Step 3: one-episode rollouts of a trained policy at SOC 1.0 and 0.2, plotted and counted.

    python scripts/rollout_plot.py --run baseline_s0
    python scripts/rollout_plot.py --run baseline_s0 --mode filter --socs 1.0 0.5 0.2

Loads results/<run>/model.zip (SAC, CPU) and runs one deterministic episode per SOC in the
environment of the run's mode (or --mode). Writes to results/<run>/rollout/:
  rollout_<mode>_soc<soc>.csv   per-env-step traces (4 ms): torso height, applied joint
                                torques, bus voltage (end of step and worst substep),
                                battery current (end of step and worst substep), contact
  summary_<mode>.json           per SOC: hops (complete flights) with apex heights, return,
                                violation counts, whether the episode terminated (fell)
  fig_rollout_<mode>.png/.pdf   4 x n_soc panels
Hop = complete flight >= 20 ms with apex above the start height (busaware/rollout.py).
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from busaware.leg_env import HoppingLegEnv  # noqa: E402
from busaware.rollout import rollout, count_hops, MIN_FLIGHT_S  # noqa: E402
from busaware.runs import run_config, env_mode  # noqa: E402

# dataviz reference palette (light): slots 1-2 for hip/knee, ink for single series
SURF, INK, INK2, MUTED, GRIDC, AXISC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXISC, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<run>/model.zip, e.g. baseline_s0")
    ap.add_argument("--mode", default=None, help="env mode; default: the run's training mode")
    ap.add_argument("--socs", type=float, nargs="+", default=[1.0, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from stable_baselines3 import SAC
    cfg = run_config(ROOT / "results" / a.run)
    mode = a.mode or cfg["mode"]
    model = SAC.load(ROOT / "results" / a.run / "model", device="cpu")
    env = HoppingLegEnv(mode=env_mode(mode), seed=a.seed, penalty_w=cfg["penalty_w"], w_energy=cfg["w_energy"])
    policy = lambda obs: model.predict(obs, deterministic=True)[0]
    out_dir = ROOT / "results" / a.run / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = env.p
    summary = dict(run=a.run, mode=mode, seed=a.seed, hop_definition=f"complete flight (no contact) >= {MIN_FLIGHT_S} s, ended by contact before the "
                                  "episode end, apex above the start height",
                   episode_s=env.max_steps * env.dt_sim * env.substeps, socs={})
    traces = {}
    for soc in a.socs:
        r = rollout(env, policy, soc, a.seed)
        traces[soc] = r
        hops = count_hops(r["t"], r["contact"], r["z"], r["z0"])
        summary["socs"][f"{soc:g}"] = dict(
            n_hops=len(hops), hops=hops, fell=r["terminated"], steps=len(r["t"]),
            ret=float(r["reward"].sum()), z_max=float(r["z"].max()),
            height_max=float(r["z"].max() - r["z0"]),          # max torso height above the start
            uv_substeps=int(r["uv"].sum()), oc_substeps=int(r["oc"].sum()),
            collapse_substeps=int(r["collapse"].sum()),
            # A6 (empty voltage-circle / current-box intersection) substeps, and those that are
            # also collapse substeps (plant scaled the drive currents or the bus collapsed)
            a6_substeps=int(r["a6_empty"].sum()), a6_collapse_substeps=int(r["a6_collapse"].sum()),
            a6_after_collapse_substeps=int(r["a6_prev_collapse"].sum()), a6_below_vmin_substeps=int(r["a6_uv"].sum()),
            # time = substeps x simulation step (bus/drive substep, 0.5 ms)
            uv_s=int(r["uv"].sum()) * env.dt_sim, oc_s=int(r["oc"].sum()) * env.dt_sim,
            ov_s=int(r["ov"].sum()) * env.dt_sim, collapse_s=int(r["collapse"].sum()) * env.dt_sim,
            filt_s=int(r["filt"].sum()) * env.dt_sim, infeas_s=int(r["infeas"].sum()) * env.dt_sim,
            v_bus_min=float(r["v_bus_min"].min()), i_bat_max=float(r["i_bat_max"].max()))
        cols = [k for k in r if k not in ("terminated", "z0")]
        np.savetxt(out_dir / f"rollout_{mode}_soc{soc:g}.csv", np.column_stack([r[k] for k in cols]),
                   delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
    (out_dir / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2))

    n = len(a.socs)
    fig, ax = plt.subplots(4, n, figsize=(5.2 * n, 9.0), sharex=True, sharey="row", squeeze=False)
    for c, soc in enumerate(a.socs):
        r, s = traces[soc], summary["socs"][f"{soc:g}"]
        ax[0, c].plot(r["t"], r["z"], color=INK, lw=1.2)
        for h in s["hops"]:
            ax[0, c].axvspan(h["start"], h["end"], color=GRIDC, alpha=0.6, lw=0)
        ax[0, c].set_title(f"SOC {soc:g}: {s['n_hops']} hops, {'fell' if s['fell'] else 'did not fall'}")
        ax[1, c].plot(r["t"], r["tau_hip"], color=BLUE, lw=1.0, label="hip")
        ax[1, c].plot(r["t"], r["tau_knee"], color=ORANGE, lw=1.0, label="knee")
        ax[2, c].plot(r["t"], r["v_bus"], color=INK, lw=1.0, label="end of step")
        ax[2, c].plot(r["t"], r["v_bus_min"], color=MUTED, lw=0.8, label="lowest in step")
        ax[2, c].axhline(p.v_min, color=AXISC, lw=1.0, ls="--", label=f"v_min {p.v_min:g} V")
        ax[3, c].plot(r["t"], r["i_bat"], color=INK, lw=1.0, label="end of step")
        ax[3, c].plot(r["t"], r["i_bat_max"], color=MUTED, lw=0.8, label="highest in step")
        ax[3, c].axhline(p.i_bat_max, color=AXISC, lw=1.0, ls="--", label=f"i_bat_max {p.i_bat_max:g} A")
        ax[3, c].set_xlabel("time (s)")
    ax[0, 0].set_ylabel("torso height\nabove stand (m)")
    ax[1, 0].set_ylabel("joint torque (N m)")
    ax[2, 0].set_ylabel("bus voltage (V)")
    ax[3, 0].set_ylabel("battery current (A)")
    for row in (1, 2, 3):
        ax[row, 0].legend(loc="best")
    fig.suptitle(f"{a.run}, {mode} mode, seed {a.seed}  (shaded: hops)",
                 color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_rollout_{mode}.{ext}", dpi=150)
    plt.close(fig)
    for soc, s in summary["socs"].items():
        apex = ", ".join(f"{h['apex']:.3f}" for h in s["hops"]) or "-"
        print(f"SOC {soc}: {s['n_hops']} hops (apex m: {apex}); fell={s['fell']} after {s['steps']} steps; "
              f"return {s['ret']:.1f}; max height above start {s['height_max']:.3f} m; below v_min "
              f"{s['uv_s']:.4f} s, above i_bat_max {s['oc_s']:.4f} s, above v_max {s['ov_s']:.4f} s, collapse "
              f"{s['collapse_s']:.4f} s of {summary['episode_s']:g} s; filter active {s['filt_s']:.3f} s, "
              f"infeasible {s['infeas_s']:.4f} s; V_min {s['v_bus_min']:.2f} V; I_max {s['i_bat_max']:.2f} A")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
