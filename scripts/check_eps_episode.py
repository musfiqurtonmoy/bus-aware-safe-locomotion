"""CHANGELOG #27. One-episode check: filter_ws_s0, reset seed 1000, SOC 1.0, filter mode, with eps_rel = 0
(compare to eval_pre_eps.csv) and with the current epsilon (compare to eval.csv)."""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from stable_baselines3 import SAC
import busaware.leg_env as le
from busaware.leg_env import HoppingLegEnv
from busaware.rollout import rollout, episode_metrics
from busaware.runs import run_config, env_mode
run = "filter_ws_s0"
cfg = run_config(ROOT / "results" / run)
model = SAC.load(ROOT / "results" / run / "model", device="cpu")
policy = lambda o: model.predict(o, deterministic=True)[0]
orig = le.power_budget
for label, eps, ref in (("eps_rel = 0", 0.0, "eval_pre_eps.csv"), ("eps_rel = 1e-9 (current)", None, "eval.csv")):
    le.power_budget = orig if eps is None else (lambda *a, **k: orig(*a, **{**k, "eps_rel": 0.0}))
    env = HoppingLegEnv(mode=env_mode("filter"), seed=123, penalty_w=cfg["penalty_w"], w_energy=cfg["w_energy"])
    t0 = time.perf_counter()
    r = rollout(env, policy, 1.0, 1000)
    dt = time.perf_counter() - t0
    m = episode_metrics(r, env.dt_sim, env.substeps, env.max_steps)
    d = pd.read_csv(ROOT / "results" / run / ref)
    row = d[(d.test_mode == "filter") & (d.soc == 1.0) & (d.episode == 0)].iloc[0]
    cols = [c for c in d.columns if c in m and pd.api.types.is_numeric_dtype(d[c])]
    diff = {c: (m[c], row[c]) for c in cols if not (m[c] == row[c] or (np.isnan(m[c]) and np.isnan(row[c])))}
    print(f"{label}: vs {ref}: {len(cols)} columns compared, {len(diff)} differ; episode took {dt:.1f} s")
    for c, (a, b) in diff.items():
        print(f"    {c}: rerun {a!r}  file {b!r}")
le.power_budget = orig
