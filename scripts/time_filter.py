"""Step 5(c): filter compute time per call, measured on the states a trained filter policy visits.

    python scripts/time_filter.py --run filter_s0

One call = power_budget (barriers -> power budget) + project_currents (QCQP projection), i.e.
everything the filter adds per 0.5 ms control substep. Both functions are wrapped inside
busaware.leg_env with time.perf_counter_ns; the plant is not timed. One deterministic episode
per SOC (reset seed 1000), the first --warmup calls are discarded. Single Python process, CPU.
Writes results/summary/filter_timing.json (mean, median, 99th percentile, max in microseconds,
for the budget, the projection and their sum, plus the projection status counts).
"""
import argparse
import json
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import busaware.leg_env as le  # noqa: E402
from busaware.leg_env import HoppingLegEnv  # noqa: E402
from busaware.rollout import rollout  # noqa: E402

T_BUDGET, T_PROJ, STATUS = [], [], []
_budget, _project = le.power_budget, le.project_currents


def timed_budget(*args, **kw):
    t0 = time.perf_counter_ns()
    out = _budget(*args, **kw)
    T_BUDGET.append(time.perf_counter_ns() - t0)
    return out


def timed_project(*args, **kw):
    t0 = time.perf_counter_ns()
    out = _project(*args, **kw)
    T_PROJ.append(time.perf_counter_ns() - t0)
    STATUS.append(out[1])
    return out


def stats_us(ns):
    us = np.asarray(ns, float) / 1e3
    return dict(mean=float(us.mean()), median=float(np.median(us)),
                p99=float(np.percentile(us, 99)), max=float(us.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="filter_s0")
    ap.add_argument("--socs", type=float, nargs="+", default=[1.0, 0.5, 0.2])
    ap.add_argument("--warmup", type=int, default=200)
    a = ap.parse_args()
    from stable_baselines3 import SAC
    model = SAC.load(ROOT / "results" / a.run / "model", device="cpu")
    policy = lambda obs: model.predict(obs, deterministic=True)[0]
    le.power_budget, le.project_currents = timed_budget, timed_project
    env = HoppingLegEnv(mode="filter", seed=123)
    for soc in a.socs:
        rollout(env, policy, soc, 1000)
    le.power_budget, le.project_currents = _budget, _project
    assert len(T_BUDGET) == len(T_PROJ)
    w = a.warmup
    tb, tp = np.asarray(T_BUDGET[w:]), np.asarray(T_PROJ[w:])
    st = STATUS[w:]
    res = dict(run=a.run, socs=a.socs, n_calls=int(len(tb)), warmup_discarded=w,
               budget_us=stats_us(tb), projection_us=stats_us(tp), total_us=stats_us(tb + tp),
               status_counts={s: st.count(s) for s in sorted(set(st))},
               machine=dict(processor=platform.processor(), python=platform.python_version(),
                            numpy=np.__version__, timer="time.perf_counter_ns"))
    out = ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "filter_timing.json").write_text(json.dumps(res, indent=2))
    for k in ("budget_us", "projection_us", "total_us"):
        s = res[k]
        print(f"{k:14s} mean {s['mean']:7.1f}  median {s['median']:7.1f}  p99 {s['p99']:7.1f}  "
              f"max {s['max']:8.1f}")
    print(f"{res['n_calls']} calls; status {res['status_counts']}; wrote {out / 'filter_timing.json'}")


if __name__ == "__main__":
    main()
