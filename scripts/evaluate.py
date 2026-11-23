"""Evaluate a trained policy across SOC levels, in one or more test modes.

  python scripts/evaluate.py --run baseline_s0 --episodes 20 --test_modes baseline filter
Writes results/<run>/eval.csv with one row per (test_mode, soc, episode).

Each episode is one deterministic rollout (busaware/rollout.py). Episode ep uses reset seed
1000 + ep in every test mode and at every SOC (fixed seeds, the same initial joint offsets
across modes). Per episode: return in the test mode's reward, falls, hops (complete flights >= 20 ms
with apex above the start height, busaware/rollout.count_hops), mean and max hop apex, max torso height, time below v_min
(uv_s), above i_bat_max (oc_s), above v_max (ov_s), in bus collapse (collapse_s), filter active
and infeasible time, all in seconds of simulated time, plus the same as fractions of the full
episode, minimum bus voltage, peak battery current and drive electrical energy, and the
proof-assumption counters (substeps where A3, the Remark 1 bound or A4 fails, or where some
joint's voltage circle and current box do not intersect, A6): a3_fail_n, rem1_fail_n,
a4_fail_n, a6_empty_n.
Evaluation settings (Step 9; busaware/runs.py): --variant <name> evaluates under a Params variant
(cold, or one of Params.VARIANTS) for plant and filter alike; --gamma sets the filter's barrier
rate (default: the env's 0.3); --r0_scale scales the plant's R0 (aged pack); --est_noise perturbs
the filter's EMF and R0 estimates (with --estimator ekf: the EKF's two measurements instead).
Step 8: --estimator ekf gives the filter the battery EKF's EMF and R0 (busaware/ekf.py) instead
of the true EMF and the model's R0; --margin confidence scales its margins with the EKF's
confidence (filter test mode only). CHANGELOG #38: --test_modes perjoint evaluates the
joint-level limiter (the filter's budget split equally between the drives, each joint limited on
its own; busaware/safety_filter.project_per_joint), on its own, into eval_ctl-perjoint*.csv;
--margin_v / --margin_i set the fixed margins (files _mv<v>, _mi<i>). Each setting writes its own file (runs.eval_filename), e.g.
eval_var-cold.csv, eval_g0.6.csv, eval_r1.5_n0.0.csv, eval_est-ekf.csv; the default setting
writes eval.csv. EKF evaluations add the columns ekf_* and margin_*_mean (rollout.episode_metrics). The
file is written only when every episode has finished (via a temporary file), so an interrupted
evaluation leaves no file. With --skip_done the run is skipped (before loading the model) when
its file is already complete and written by the current code (busaware/runs.eval_done).
The reward weights (penalty_w, w_energy) are the run's own (busaware/runs.run_config), so the
return in the run's training mode is the reward it was trained on. Test mode "lagrangian" runs
the baseline environment (busaware/runs.env_mode); its return is the baseline reward, without
the multiplier term.
"""
import sys, os, pathlib, argparse, csv
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import SAC
from busaware.leg_env import HoppingLegEnv
from busaware.rollout import rollout, episode_metrics, PROOF_COUNTERS
from busaware.runs import run_config, env_mode, EVAL_VARIANTS, variant_params, eval_filename, eval_done

SEED_BASE = 1000

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--socs", type=float, nargs="+", default=[1.0, 0.5, 0.2])
ap.add_argument("--test_modes", nargs="+", default=["baseline", "filter"])
ap.add_argument("--r0_scale", type=float, default=1.0)
ap.add_argument("--est_noise", type=float, default=0.0)
ap.add_argument("--variant", default=None, choices=EVAL_VARIANTS, help="Params variant (default: base Params)")
ap.add_argument("--gamma", type=float, default=None, help="filter barrier rate (default: env default 0.3)")
ap.add_argument("--estimator", default="true", choices=["true", "ekf"], help="filter's EMF/R0 source (Step 8)")
ap.add_argument("--margin", default="fixed", choices=["fixed", "confidence"], help="filter margins (Step 8)")
ap.add_argument("--margin_v", type=float, default=None, help="fixed voltage margin, V (default: env's 0.2)")
ap.add_argument("--margin_i", type=float, default=None, help="fixed current margin, A (default: env's 0.5)")
ap.add_argument("--skip_done", action="store_true", help="skip if the output file is already complete")
a = ap.parse_args()
# CHANGELOG #38: test mode "perjoint" (joint-level limiter, evaluation only) gets its own file
if "perjoint" in a.test_modes and a.test_modes != ["perjoint"]:
    ap.error("--test_modes perjoint must be evaluated on its own")
if "perjoint" in a.test_modes and a.estimator != "true":
    ap.error("the per-joint limiter uses the true state only")
controller = "perjoint" if a.test_modes == ["perjoint"] else "filter"
if a.estimator == "ekf" and set(a.test_modes) != {"filter"}:
    ap.error("--estimator ekf applies to the filter test mode only: use --test_modes filter")
if a.margin == "confidence" and a.estimator != "ekf":
    ap.error("--margin confidence needs --estimator ekf")
path = ROOT / "results" / a.run / eval_filename(a.variant, a.gamma, a.r0_scale, a.est_noise, a.estimator, a.margin,
                                               controller, a.margin_v, a.margin_i)
if a.skip_done and eval_done(path, a.test_modes, a.socs, a.episodes, PROOF_COUNTERS):
    print("done already, skipped:", path)
    sys.exit(0)

cfg = run_config(ROOT / "results" / a.run)
model = SAC.load(ROOT / "results" / a.run / "model", device="cpu")
policy = lambda obs: model.predict(obs, deterministic=True)[0]
rows = []
for tm in a.test_modes:
    kw = {} if a.gamma is None else dict(gamma=a.gamma)
    kw.update({k: v for k, v in (("margin_v", a.margin_v), ("margin_i", a.margin_i)) if v is not None})
    env = HoppingLegEnv(mode=env_mode(tm), r0_scale=a.r0_scale, est_noise=a.est_noise, seed=123,
                        penalty_w=cfg["penalty_w"], w_energy=cfg["w_energy"],
                        params=variant_params(a.variant), estimator=a.estimator, margin_mode=a.margin, **kw)
    for soc in a.socs:
        for ep in range(a.episodes):
            r = rollout(env, policy, soc, SEED_BASE + ep)
            m = episode_metrics(r, env.dt_sim, env.substeps, env.max_steps)
            rows.append(dict(run=a.run, train_mode=cfg["mode"], test_mode=tm, soc=soc, episode=ep, reset_seed=SEED_BASE + ep,
                             variant=a.variant or "base", gamma=env.gamma, estimator=a.estimator, margin_mode=a.margin,
                             margin_v_fixed=env.margin_v, margin_i_fixed=env.margin_i,
                             r0_scale=a.r0_scale, est_noise=a.est_noise, **m))
tmp = path.with_name(path.name + ".tmp")
with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
os.replace(tmp, path)                     # complete file or none
print("wrote", path)
for tm in a.test_modes:
    for soc in a.socs:
        r = [x for x in rows if x["test_mode"] == tm and x["soc"] == soc]
        m = lambda k: np.nanmean([x[k] for x in r])
        print(f"{tm:9s} soc={soc:.1f} return={m('ret'):7.1f} falls={sum(x['fell'] for x in r)}/{len(r)} "
              f"hops={m('n_hops'):.1f} apex={m('apex_mean'):.3f} uv={m('uv_s'):.4f}s oc={m('oc_s'):.4f}s "
              f"collapse={m('collapse_s'):.4f}s filt={m('filt_s'):.3f}s infeas={m('infeas_s'):.4f}s "
              f"Vmin={m('v_min'):.2f} Imax={m('i_max'):.1f}")
        print("          proof-assumption counters (substeps, all episodes): "
              + " ".join(f"{k}={sum(x[k + '_n'] for x in r)}" for k in PROOF_COUNTERS))
        if "ekf_e_abs_mean" in r[0]:
            print(f"          EKF: |E err| mean {m('ekf_e_abs_mean'):.4f} V, R0/true end {m('ekf_r0_rel_final'):.3f}, "
                  f"SOC err end {m('ekf_soc_err_final'):+.4f}, confidence mean {m('ekf_conf_mean'):.2f}, "
                  f"margins {m('margin_v_mean'):.2f} V / {m('margin_i_mean'):.2f} A")
