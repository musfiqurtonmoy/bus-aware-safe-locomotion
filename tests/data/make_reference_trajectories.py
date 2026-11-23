"""Record reference trajectories of HoppingLegEnv for tests/test_counters.py.

    python tests/data/make_reference_trajectories.py

Run ONCE at commit 7a895f5, before the empty-intersection and A4 counters were added
(CHANGELOG #25), so the test can check that those counters do not change any control or plant
behaviour. Do not rerun it to make a failing test pass (CLAUDE.md rule 7): a mismatch means
the plant or the controller changed.

Cases cover every env mode, low and high SOC, estimator noise (which draws from the env's
random generator) and a scaled plant R0. Actions are uniform random from a fixed seed; each case
runs until termination or N_STEPS. Per step it stores the observation, reward, termination
flags, the env's info counters, and the plant state (rail and joint positions and velocities,
bus voltage, battery current, SOC, drive currents).
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from busaware.leg_env import HoppingLegEnv  # noqa: E402

N_STEPS = 300
CASES = [  # name, env kwargs, soc, action seed
    ("baseline_soc1.0", dict(mode="baseline"), 1.0, 11),
    ("baseline_soc0.2", dict(mode="baseline"), 0.2, 12),
    ("penalty_soc0.2", dict(mode="penalty", penalty_w=2.0), 0.2, 13),
    ("filter_soc1.0", dict(mode="filter"), 1.0, 14),
    ("filter_soc0.2", dict(mode="filter"), 0.2, 15),
    ("filter_noise_soc0.5", dict(mode="filter", est_noise=0.05), 0.5, 16),
    ("baseline_r0x1.5_soc0.2", dict(mode="baseline", r0_scale=1.5), 0.2, 17),
]
INFO_KEYS = ("uv", "ov", "oc", "uvoc", "collapse", "filt", "infeas", "a3_fail", "rem1_fail",
             "e_elec", "dcorr", "v_min", "v_max", "i_max", "p_peak")


def run_case(kw, soc, action_seed, n_steps=N_STEPS):
    env = HoppingLegEnv(seed=123, **kw)
    obs, _ = env.reset(seed=0, options=dict(soc=soc))
    rng = np.random.default_rng(action_seed)
    rec = {k: [] for k in ("obs", "reward", "terminated", "truncated", "qpos", "qvel", "v_bus",
                           "i_bat", "soc", "iq_act", *("info_" + k for k in INFO_KEYS))}
    rec["obs0"] = obs
    for _ in range(n_steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1.0, 1.0, 2))
        for k, v in (("obs", obs), ("reward", r), ("terminated", term), ("truncated", trunc),
                     ("qpos", env.data.qpos.copy()), ("qvel", env.data.qvel.copy()),
                     ("v_bus", env.bus.v), ("i_bat", env.bus.i_bat), ("soc", env.bat.soc),
                     ("iq_act", env.iq_act.copy())):
            rec[k].append(v)
        for k in INFO_KEYS:
            rec["info_" + k].append(info[k])
        if term or trunc:
            break
    return {k: np.asarray(v) for k, v in rec.items()}


def main():
    out = {}
    for name, kw, soc, seed in CASES:
        for k, v in run_case(kw, soc, seed).items():
            out[f"{name}/{k}"] = v
    path = pathlib.Path(__file__).with_name("reference_trajectories.npz")
    np.savez_compressed(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
