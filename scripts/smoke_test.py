"""Random-action rollout in each mode. Run first: python scripts/smoke_test.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from busaware.leg_env import HoppingLegEnv

for mode in ("baseline", "filter"):
    env = HoppingLegEnv(mode=mode, seed=0)
    env.action_space.seed(0)        # same random actions in every run and in both modes
    tot = dict(uv=0, oc=0, ov=0, collapse=0, filt=0, infeas=0); steps = 0; vmin = 99; imax = 0
    for ep in range(5):
        env.reset(options=dict(soc=0.3)); done = False
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc; steps += env.substeps
            for k in tot: tot[k] += info[k]
            vmin = min(vmin, info["v_min"]); imax = max(imax, info["i_max"])
    tot["active"] = tot["filt"] - tot["infeas"]      # filt = active + infeasible
    t_uv = tot["uv"] * env.dt_sim                    # time below v_min
    print(f"{mode:9s} substeps={steps} " + " ".join(f"{k}={v}" for k, v in tot.items())
          + f" Vmin={vmin:.2f} Imax={imax:.2f}"
          + f" t_below_{env.p.v_min:g}V={t_uv:.3f}s ({100 * tot['uv'] / steps:.1f}%)")
