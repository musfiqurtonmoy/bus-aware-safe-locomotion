"""One-episode rollouts of a policy in HoppingLegEnv, and hop detection (Step 3).

A hop is a complete flight phase: a run of consecutive environment steps without any contact
that (1) lasts at least MIN_FLIGHT_S, (2) ends in contact before the episode ends (a flight cut
off by the episode end, including a fall, does not count), and (3) has an apex above the torso
height at the start of the episode. Its apex is the largest torso rail displacement during the
flight (relative to the stand keyframe). Rule (1) was fixed before any trained policy existed;
(2) and (3) were added after seeing baseline_s0 (CHANGELOG #18).
"""
import numpy as np

MIN_FLIGHT_S = 0.020
# proof-assumption counters from HoppingLegEnv info, substeps per episode (paper Sec. filter):
# A3 physical bus state, Remark 1 bound, A4 plant root below x*, A6 empty voltage-circle /
# current-box intersection on some joint. Counted only; they never act on control or plant.
PROOF_COUNTERS = ("a3_fail", "rem1_fail", "a4_fail", "a6_empty")
# EKF diagnostics from HoppingLegEnv info (estimator="ekf" only; Step 8): per env step, the mean
# |E_hat - E| over its substeps (V), R0_hat / plant R0, SOC_hat - SOC and the EKF confidence at
# its end, and the mean margins the filter used.
EKF_KEYS = ("ekf_e_abs", "ekf_r0_rel", "ekf_soc_err", "ekf_conf", "margin_v", "margin_i")


def rollout(env, policy, soc, seed):
    """Run one episode with policy(obs) -> action from SOC soc. Returns per-env-step arrays."""
    obs, _ = env.reset(seed=seed, options=dict(soc=soc))
    z0 = float(env.data.qpos[0])                 # torso height at the start of the episode
    dt = env.dt_sim * env.substeps
    keys = ("t", "z", "tau_hip", "tau_knee", "v_bus", "v_bus_min", "i_bat", "i_bat_max",
            "contact", "uv", "oc", "ov", "collapse", "filt", "infeas", "a3_fail", "rem1_fail",
            "a4_fail", "a6_empty", "a6_collapse", "a6_prev_collapse", "a6_uv", "e_elec", "reward")
    rec = {k: [] for k in keys}
    done, k, term = False, 0, False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        done = term or trunc
        k += 1
        for key, val in (("t", k * dt), ("z", float(env.data.qpos[0])),
                         ("tau_hip", float(env.data.ctrl[0])), ("tau_knee", float(env.data.ctrl[1])),
                         ("v_bus", env.bus.v), ("v_bus_min", info["v_min"]), ("i_bat", env.bus.i_bat),
                         ("i_bat_max", info["i_max"]), ("contact", int(env.data.ncon > 0)),
                         ("uv", info["uv"]), ("oc", info["oc"]), ("ov", info["ov"]),
                         ("collapse", info["collapse"]), ("filt", info["filt"]),
                         ("infeas", info["infeas"]), ("a3_fail", info["a3_fail"]),
                         ("rem1_fail", info["rem1_fail"]), ("a4_fail", info["a4_fail"]),
                         ("a6_empty", info["a6_empty"]), ("a6_collapse", info["a6_collapse"]),
                         ("a6_prev_collapse", info["a6_prev_collapse"]), ("a6_uv", info["a6_uv"]),
                         ("e_elec", info["e_elec"]), ("reward", r)):
            rec[key].append(val)
        for key in EKF_KEYS:
            if key in info:
                rec.setdefault(key, []).append(info[key])
    out = {key: np.asarray(v) for key, v in rec.items()}
    out["terminated"] = bool(term)
    out["z0"] = z0
    return out


def count_hops(t, contact, z, z_start, min_flight_s=MIN_FLIGHT_S):
    """Complete flight phases (contact == 0): lasting >= min_flight_s, followed by a contact step
    within the record, and with apex (max z during the flight) > z_start. Returns a list of dicts
    with start, end, duration and apex of each."""
    t, contact, z = np.asarray(t), np.asarray(contact), np.asarray(z)
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
    hops, i, n = [], 0, len(t)
    while i < n:
        if contact[i] == 0:
            j = i
            while j + 1 < n and contact[j + 1] == 0:
                j += 1
            dur = (j - i + 1) * dt
            apex = float(z[i:j + 1].max())
            complete = j + 1 < n                  # contact follows within the episode
            if dur >= min_flight_s - 1e-12 and complete and apex > z_start:
                hops.append(dict(start=float(t[i]), end=float(t[j]), duration=dur,
                                 apex=apex))
            i = j + 1
        else:
            i += 1
    return hops


def episode_metrics(r, dt_sim, substeps, max_steps):
    """Per-episode metrics from a rollout() result. Counters (uv, oc, ov, collapse, filt, infeas)
    are per simulation substep; times are substeps x dt_sim; fractions are over the full episode
    length (max_steps x substeps), so an early fall does not inflate them. Hops use count_hops."""
    hops = count_hops(r["t"], r["contact"], r["z"], r["z0"])
    apex = [h["apex"] for h in hops]
    n_sub = max_steps * substeps
    out = dict(fell=int(r["terminated"]), steps=int(len(r["t"])), ret=float(r["reward"].sum()),
               n_hops=len(hops), apex_mean=float(np.mean(apex)) if apex else float("nan"),
               apex_max=float(np.max(apex)) if apex else float("nan"), z_max=float(r["z"].max()),
               z0=float(r["z0"]), height_max=float(r["z"].max() - r["z0"]),   # defined for every episode
               v_min=float(r["v_bus_min"].min()), i_max=float(r["i_bat_max"].max()),
               e_elec=float(r["e_elec"].sum()))
    for k in ("uv", "oc", "ov", "collapse", "filt", "infeas") + PROOF_COUNTERS:
        c = int(r[k].sum())
        out[f"{k}_s"] = c * dt_sim
        out[f"{k}_frac"] = c / n_sub
        if k in PROOF_COUNTERS:                      # proof-assumption counters: raw substep counts
            out[f"{k}_n"] = c
    # A6 diagnostics: A6 substeps that are collapse substeps, that follow a collapse substep, and
    # whose start-of-substep bus voltage is below v_min
    for k in ("a6_collapse", "a6_prev_collapse", "a6_uv"):
        out[f"{k}_n"] = int(r[k].sum())
    if "ekf_e_abs" in r:                             # EKF evaluations only (Step 8)
        out.update(ekf_e_abs_mean=float(np.mean(r["ekf_e_abs"])), ekf_e_abs_max=float(np.max(r["ekf_e_abs"])),
                   ekf_r0_rel_final=float(r["ekf_r0_rel"][-1]), ekf_soc_err_final=float(r["ekf_soc_err"][-1]),
                   ekf_conf_mean=float(np.mean(r["ekf_conf"])), ekf_conf_min=float(np.min(r["ekf_conf"])),
                   margin_v_mean=float(np.mean(r["margin_v"])), margin_i_mean=float(np.mean(r["margin_i"])))
    return out
