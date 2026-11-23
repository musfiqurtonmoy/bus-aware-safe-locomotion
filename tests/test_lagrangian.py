"""SAC-Lagrangian wrapper and multiplier update (PROMPTS Step 6, CHANGELOG #22)."""
import csv

import numpy as np

from busaware.lagrangian import LagrangianCostWrapper, LagrangianCallback
from busaware.leg_env import HoppingLegEnv
from busaware.runs import env_mode


def test_cost_is_the_union_fraction_and_reward_is_shaped():
    base = HoppingLegEnv(mode="baseline", seed=0, episode_s=0.2)
    env = LagrangianCostWrapper(base, lam=3.0)
    env.reset(seed=0, options=dict(soc=0.2))
    costs, seen_both = [], False
    for _ in range(base.max_steps):
        _, r, term, trunc, info = env.step(np.array([-1.0, 1.0]))   # hard push at low SOC
        n_uv, n_oc, n_u = info["uv"], info["oc"], info["uvoc"]
        assert max(n_uv, n_oc) <= n_u <= n_uv + n_oc                # union of the two counts
        seen_both |= n_u < n_uv + n_oc
        assert info["cost"] == n_u / base.substeps
        assert np.isclose(r, info["reward_unshaped"] - 3.0 * info["cost"])
        costs.append(info["cost"])
        if term or trunc:
            break
    assert max(costs) > 0
    assert np.isclose(info["episode_cost"], np.mean(costs)) and info["episode_len"] == len(costs)
    assert env_mode("lagrangian") == "baseline"                      # plant/env unchanged


def test_multiplier_update_and_log(tmp_path):
    env = LagrangianCostWrapper(HoppingLegEnv(mode="baseline", seed=0), lam=0.0)
    cb = LagrangianCallback(env, cost_limit=0.001, lr=0.1, csv_path=tmp_path / "lag.csv")
    assert cb.update(0.301) == (0.0, 0.1 * 0.3)                      # ascent above the limit
    assert np.isclose(cb.update(0.0)[1], 0.03 - 0.1 * 0.001)         # descent below it
    env.lam = 1e-5
    assert cb.update(0.0)[1] == 0.0                                  # projected onto lam >= 0
    cb._on_training_start()
    cb.num_timesteps = 1000
    cb.locals = {"infos": [dict(episode_cost=0.2, episode_len=1000, episode_return_unshaped=900.0),
                           dict(cost=0.1)]}
    cb._on_step()
    rows = list(csv.DictReader(open(tmp_path / "lag.csv")))
    assert len(rows) == 1 and rows[0]["episode"] == "1" and rows[0]["timesteps"] == "1000"
    assert np.isclose(float(rows[0]["lam_new"]), 0.1 * (0.2 - 0.001))
